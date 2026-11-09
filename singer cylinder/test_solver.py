
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from model import FNO, CHECKPOINT_VERSION, checkpoint, load_checkpoint
from solver import solve_step


class ScalarPhysics:
    def __init__(self, tolerance=.01):
        self.tolerance = tolerance
        self.times = []

    def predict(self, model, previous, re, time):
        self.times.append(time)
        return model.weight.expand_as(previous), previous[:, :1]*0

    def residual(self, previous, velocity, pressure, re):
        loss = (velocity-1).square().mean()
        return loss, {'momentum': float(loss.detach().sqrt())}

    def accepted(self, metrics):
        return metrics['momentum'] <= self.tolerance


def make_case(weight=0., tolerance=.01, lr=.5, iterations=3):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    physics = ScalarPhysics(tolerance)
    tr = dict(iterations_per_step=iterations, replay_weight=.25, clip_grad=100., check_every=50)
    return model, optimizer, physics, tr


def invoke(parts, replay=()):
    model, opt, physics, tr = parts
    previous = torch.zeros(1, 2, 4)
    old = previous.clone()
    result = solve_step(model, opt, physics, previous, torch.tensor([40.]), 1., replay, tr)
    assert torch.equal(previous, old), 'A candidate must not modify the accepted input state'
    return result


def test_passing_candidate_never_updates():
    parts = make_case(weight=1.)
    before = deepcopy(parts[0].state_dict())
    result = invoke(parts, [(torch.zeros(1, 2, 4), .5)])
    assert result.accepted and result.updates == 0
    assert torch.equal(before['weight'], parts[0].weight)
    assert .5 not in parts[2].times
    assert not parts[1].state


def test_last_update_checked_and_budget_not_multiple_of_check_interval():
    result = invoke(make_case(lr=.25, tolerance=.26, iterations=2))
    assert result.accepted and result.updates == 2
    assert result.metrics['momentum'] == pytest.approx(.25)


def test_failed_candidate_does_not_advance():
    result = invoke(make_case(lr=.01, iterations=1))
    assert not result.accepted and result.updates == 1
    result = invoke(make_case(iterations=0))
    assert not result.accepted and result.updates == 0


def test_replay_is_used_during_correction():
    parts = make_case(iterations=1)
    invoke(parts, [(torch.zeros(1, 2, 4), .5)])
    assert .5 in parts[2].times


def test_nonfinite_loss_never_steps_optimizer():
    parts = make_case(weight=float('nan'))
    result = invoke(parts)
    assert result.status == 'nonfinite_loss' and result.updates == 0


def test_checkpoint_version_roundtrip_and_legacy_rejected(tmp_path):
    model, opt, _, _ = make_case()
    path = tmp_path/'checkpoint.pt'
    checkpoint(path, model, opt, {}, torch.zeros(1, 2, 4), 3, [], 'running', SimpleNamespace(signature='mesh'))
    saved = load_checkpoint(path)
    assert saved['version'] == CHECKPOINT_VERSION and saved['committed_steps'] == 3
    assert saved['mesh_signature'] == 'mesh' and not path.with_suffix('.tmp').exists()
    torch.save({'version': 'rgcn_tm_triangular_airfoil_v1'}, path)
    with pytest.raises(ValueError, match='Legacy'):
        load_checkpoint(path)


@pytest.mark.parametrize('operator', ['graph_nufft', 'nufft_only', 'graph_only'])
def test_operator_backward_and_node_permutation(operator):
    torch.manual_seed(4)
    model = FNO(width=4, layers=2, modes_y=3, modes_x=3, backend='direct', operator=operator)
    n = 9
    x = torch.randn(1, 9, n, requires_grad=True)
    points = torch.rand(2, n)
    w = torch.ones(n)/n
    src = torch.arange(n).repeat(2)
    dst = torch.cat(((torch.arange(n)+1)%n, (torch.arange(n)-1)%n))
    relative = (points[:, dst]-points[:, src]).T
    graph = src, dst, torch.cat((relative, relative.norm(dim=-1, keepdim=True)), -1), torch.ones(n)*2
    y = model(x, points, w, graph)
    y.square().mean().backward()
    assert y.shape == (1, 3, n) and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    perm = torch.randperm(n); inverse = torch.argsort(perm)
    other = inverse[src], inverse[dst], graph[2], graph[3][perm]
    yp = model(x.detach()[:, :, perm], points[:, perm], w[perm], other)
    assert torch.allclose(y.detach()[:, :, perm], yp, atol=2e-6)


def test_finite_volume_backward():
    from sanity_check import analytic_mesh
    from finite_volume import FiniteVolume
    fv = FiniteVolume(*analytic_mesh(5), dtype=torch.float64)
    u = torch.randn(1, 2, len(fv.xy), dtype=torch.float64, requires_grad=True)
    p = torch.randn(1, 1, len(fv.xy), dtype=torch.float64, requires_grad=True)
    m, d, _ = fv.flow_terms(u, p, torch.tensor([40.], dtype=torch.float64))
    objective = (m.square()*fv.volume).sum()+(d.square()*fv.volume).sum()
    objective.backward()
    assert torch.isfinite(u.grad).all() and torch.isfinite(p.grad).all()
    eps = 1e-6
    idx = (0, 0, 12)
    with torch.no_grad():
        a = u.clone(); b = u.clone(); a[idx] += eps; b[idx] -= eps
        def value(v):
            m, d, _ = fv.flow_terms(v, p, torch.tensor([40.], dtype=torch.float64))
            return (m.square()*fv.volume).sum()+(d.square()*fv.volume).sum()
        fd = (value(a)-value(b))/(2*eps)
    assert float(u.grad[idx]) == pytest.approx(float(fd), rel=2e-6)
