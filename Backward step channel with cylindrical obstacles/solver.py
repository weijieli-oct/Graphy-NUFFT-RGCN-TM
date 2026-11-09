
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
import argparse
import json
import random
import time as clock
import numpy as np
import torch
import params as settings
from model import (FNO, FlowPhysics, get_device, validate, append_csv, write_json,
                   checkpoint, record, load_checkpoint)

PREDICT_RE = [50., 70., 90., 110., 130., 150.]


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
    physics = FlowPhysics(c, device)
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


# Training, exact resume and new-condition entry points
def train(config=None, output=None, resume=None, initialize=None):
    resume = resume or settings.RESUME_CHECKPOINT
    if resume and initialize:
        raise ValueError('Choose resume OR shared initialization')
    saved = load_checkpoint(settings.resolve_path(resume)) if resume else None
    if saved is not None:
        if config is not None:
            raise ValueError('Exact resume uses the saved config; use --initialize for new cases/meshes')
        c = deepcopy(saved['config'])
        out = settings.resolve_path(output) if output else settings.resolve_path(resume).parent
    else:
        c = deepcopy(config or settings.CONFIG)
        out = settings.fresh_output_directory(output or settings.TRAIN_OUTPUT)
    initialization = load_checkpoint(settings.resolve_path(initialize)) if initialize else None
    result = run(c, out, saved=saved, initialization=initialization)
    settings.remember_training_directory(out)
    return result


def predict(checkpoint_path, re_values=None, config=None, output=None):
    """Each target starts from the same saved weights with fresh state/replay."""
    saved = load_checkpoint(settings.resolve_path(checkpoint_path))
    base = deepcopy(saved['config'])
    if config:
        target = json.loads(settings.resolve_path(config).read_text(encoding='utf8'))
        for key in ('model', 'tolerances', 'loss', 'training', 'physics'):
            if target[key] != base[key]:
                raise ValueError(f'Controlled transfer requires identical {key}')
        base = target
    root = settings.fresh_output_directory(output or settings.ROOT/'runs'/'unseen')
    failed = False
    for reynolds in (PREDICT_RE if re_values is None else re_values):
        if reynolds <= 0:
            raise ValueError('Re must be positive')
        if reynolds in saved['config']['cases']['re'] and not config:
            raise ValueError('Use a new Re or specify a target mesh for cross-mesh evaluation')
        c = deepcopy(base)
        c['cases'] = {'re': [reynolds], 'amplitude': [base['cases']['amplitude'][0]]}
        code, _ = run(c, root/f'Re_{reynolds:g}', initialization=saved)
        failed |= code != 0
    return 2 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    training = actions.add_parser('train', help='Train, initialize from weights or resume exactly')
    training.add_argument('--config', help='Experiment JSON configuration')
    training.add_argument('--output')
    training.add_argument('--resume')
    training.add_argument('--initialize', help='Load v2 weights; reset state/replay/optimizer')
    prediction = actions.add_parser('predict', help='Solve new Re or transfer to another mesh')
    prediction.add_argument('--checkpoint')
    prediction.add_argument('--re', type=float, nargs='+', default=PREDICT_RE)
    prediction.add_argument('--config', help='Optional target mesh/configuration')
    prediction.add_argument('--output', default=str(settings.ROOT/'runs'/'unseen'))
    args = parser.parse_args(argv)
    if args.action == 'train':
        c = json.loads(settings.resolve_path(args.config).read_text(encoding='utf8')) if args.config else None
        code, out = train(c, args.output, args.resume, args.initialize)
        print('结果目录:', out)
        return code
    source = args.checkpoint or settings.latest_training_directory()/'checkpoint.pt'
    return predict(source, args.re, args.config, args.output)


if __name__ == '__main__':
    raise SystemExit(main())
