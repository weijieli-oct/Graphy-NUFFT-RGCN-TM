
from dataclasses import dataclass
from pathlib import Path
import random
import time as clock
import numpy as np
import torch
from model import (FNO, AirfoilPhysics, get_device, validate, append_csv, write_json,
                   checkpoint, record)


@dataclass
class StepResult:
    velocity: torch.Tensor
    pressure: torch.Tensor
    metrics: dict
    updates: int
    status: str
    elapsed: float

    @property
    def accepted(self):
        return self.status == 'accepted'


def solve_step(model, optimizer, physics, previous, re, time, replay, training, log=None):
    """Never advance time here. Passing candidates perform exactly zero updates."""
    started = clock.perf_counter()
    previous = previous.detach()
    model.train()
    def evaluate():
        with torch.no_grad():
            v, p = physics.predict(model, previous, re, time)
            _, metrics = physics.residual(previous, v, p, re)
        return v, p, metrics
    velocity, pressure, metrics = evaluate()
    updates = 0
    status = 'residual_gate_failed'
    if log:
        log(updates, metrics)
    while not physics.accepted(metrics) and updates < training['iterations_per_step']:
        optimizer.zero_grad(set_to_none=True)
        velocity, pressure = physics.predict(model, previous, re, time)
        loss, _ = physics.residual(previous, velocity, pressure, re)
        if replay and training['replay_weight'] > 0:
            old, old_time = replay[updates % len(replay)]
            old = old.to(previous.device).detach()
            ov, op = physics.predict(model, old, re, old_time)
            old_loss, _ = physics.residual(old, ov, op, re)
            loss = loss + training['replay_weight']*old_loss
        if not bool(torch.isfinite(loss)):
            status = 'nonfinite_loss'; break
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training['clip_grad'])
        if not bool(torch.isfinite(norm)):
            optimizer.zero_grad(set_to_none=True)
            status = 'nonfinite_gradient'; break
        optimizer.step()
        updates += 1
        velocity, pressure, metrics = evaluate()
        if log and (updates % training['check_every'] == 0 or physics.accepted(metrics)
                    or updates == training['iterations_per_step']):
            log(updates, metrics)
    if physics.accepted(metrics):
        status = 'accepted'
    return StepResult(velocity.detach(), pressure.detach(), metrics, updates, status,
                      clock.perf_counter()-started)


def run(c, out, saved=None, initialization=None):
    """Exact resume restores state/replay/optimizer; initialization loads weights only."""
    validate(c)
    device = get_device(c['device'])
    torch.manual_seed(c['seed']); np.random.seed(c['seed']); random.seed(c['seed'])
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    physics = AirfoilPhysics(c, device)
    model = FNO(**c['model']).to(device)
    tr = c['training']
    optimizer = torch.optim.Adam(model.parameters(), lr=tr['lr'])
    re = torch.tensor(c['cases']['re'], dtype=torch.float32, device=device)
    state = physics.initial(torch.tensor(c['cases']['amplitude'], device=device))
    replay, committed = [], 0
    if saved is not None:
        if saved['mesh_signature'] != physics.signature or tuple(saved['state'].shape) != tuple(state.shape):
            raise ValueError('Resume requires the identical mesh and case batch; use initialization for transfer')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        state = saved['state'].to(device).detach()
        replay, committed = saved['replay'], saved['committed_steps']
        torch.set_rng_state(saved['rng_torch']); np.random.set_state(saved['rng_numpy']); random.setstate(saved['rng_python'])
        if device.type == 'cuda' and saved['rng_cuda'] is not None:
            torch.cuda.set_rng_state_all(saved['rng_cuda'])
    elif initialization is not None:
        if initialization['config']['model'] != c['model']:
            raise ValueError('Shared initialization requires the same model architecture')
        model.load_state_dict(initialization['model'])
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    write_json(out/'config_used.json', c)
    print('Output:', out, '| nodes:', physics.n, '| Re:', c['cases']['re'], '| start:', committed, flush=True)
    for step in range(committed, tr['steps']):
        time = step*c['physics']['dt']
        def log(updates, metrics):
            append_csv(out/'loss.csv', {'step': step+1, 'iteration': updates, **metrics})
            print('step=%d updates=%d %s' % (step+1, updates, metrics), flush=True)
        result = solve_step(model, optimizer, physics, state, re, time, replay, tr, log)
        append_csv(out/'residuals.csv', {'step': step+1, 'time': time+c['physics']['dt'],
                   'updates': result.updates, 'passed': result.accepted, 'seconds': result.elapsed, **result.metrics})
        if not result.accepted:
            checkpoint(out/'checkpoint.pt', model, optimizer, c, state, step, replay, result.status, physics)
            write_json(out/'status.json', {'status': result.status, 'committed_steps': step,
                       'attempted_time': time+c['physics']['dt'],
                       'metrics': {k: v if np.isfinite(v) else str(v) for k, v in result.metrics.items()},
                       'tolerances': c['tolerances'], 'physical_validation': False})
            print('STOP: rejected candidate; accepted state/time preserved.', flush=True)
            return 2, out
        if tr['replay_capacity']:
            replay.append((state.detach().cpu(), time))
            replay = replay[-tr['replay_capacity']:]
        previous, state = state, result.velocity
        committed = step + 1
        record(out, physics, state, result.pressure, re, committed*c['physics']['dt'], committed,
               committed % tr['save_every'] == 0 or committed == tr['steps'], previous,
               result.updates, result.elapsed)
        checkpoint(out/'checkpoint.pt', model, optimizer, c, state, committed, replay, 'running', physics)
    write_json(out/'status.json', {'status': 'complete', 'committed_steps': committed, 'physical_validation': False})
    checkpoint(out/'checkpoint.pt', model, optimizer, c, state, committed, replay, 'complete', physics)
    return 0, out
