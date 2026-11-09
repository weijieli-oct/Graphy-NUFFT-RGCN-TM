
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
from finite_volume import BOUNDARIES, DISCRETIZATION, FiniteVolume, mesh_arrays, mesh_signature

try:
    from nufft_compat import finufft_type1, finufft_type2
except ImportError:
    finufft_type1 = finufft_type2 = None

CHECKPOINT_VERSION = 'rgcn_tm_graph_nufft_fv_v2'


def load_mesh(path, device):
    """Compatibility API; full dual geometry is constructed by FiniteVolume."""
    xy, tri, bounds = mesh_arrays(path)
    fv = FiniteVolume(xy, tri, bounds, device=device)
    edges = fv.edges
    return (fv.xy, torch.cat((edges[:, 0], edges[:, 1])),
            torch.cat((edges[:, 1], edges[:, 0])), fv.volume,
            {k: torch.as_tensor(v, dtype=torch.long, device=device) for k, v in bounds.items()})


class MeshGradient(nn.Module):

    def __init__(self, xy, src, dst):
        super().__init__()
        self.register_buffer('src', src); self.register_buffer('dst', dst)
        dd = xy[dst] - xy[src]
        w = 1 / dd.square().sum(-1).clamp_min(torch.finfo(xy.dtype).tiny)
        x, y = dd.unbind(-1)
        def sums(v):
            out = xy.new_zeros(len(xy)); out.index_add_(0, src, v); return out
        a, b, c = sums(w*x*x), sums(w*x*y), sums(w*y*y)
        det = a*c - b*b
        if bool((det <= 1e-12).any()):
            raise ValueError('Rank-deficient linear gradient stencil')
        self.register_buffer('coef_x', (c[src]*w*x-b[src]*w*y)/det[src])
        self.register_buffer('coef_y', (a[src]*w*y-b[src]*w*x)/det[src])

    def forward(self, f):
        change = f[:, :, self.dst] - f[:, :, self.src]
        gx, gy = torch.zeros_like(f), torch.zeros_like(f)
        gx.index_add_(2, self.src, change*self.coef_x)
        gy.index_add_(2, self.src, change*self.coef_y)
        return gx, gy


def direct_basis(points, my, mx):

    ky = torch.fft.fftfreq(my, d=1/my, device=points.device, dtype=points.dtype)
    kx = torch.fft.fftfreq(mx, d=1/mx, device=points.device, dtype=points.dtype)
    yy, xx = torch.meshgrid(ky, kx, indexing='ij')
    return torch.exp(1j * (points[0, :, None]*yy.reshape(-1) + points[1, :, None]*xx.reshape(-1)))


class SpectralConv(nn.Module):
    def __init__(self, width, my, mx, eps, backend='finufft'):
        super().__init__()
        self.my, self.mx, self.eps, self.backend = my, mx, eps, backend
        self.weight = nn.Parameter(torch.randn(width, width, my, mx, dtype=torch.cfloat)/width)

    def forward(self, x, points, quadrature):
        complex_dtype = torch.complex128 if x.dtype == torch.float64 else torch.complex64
        values = x.to(complex_dtype) * quadrature[None, None, :]
        weight = self.weight.to(complex_dtype)
        if self.backend == 'direct':
            if points.shape[1]*self.my*self.mx > 4_000_000:
                raise ValueError('Direct DFT is limited to small verification meshes; use finufft')
            basis = direct_basis(points, self.my, self.mx)
            z = torch.einsum('bcn,nk->bck', values, basis.conj())
            z = torch.einsum('bik,iok->bok', z, weight.reshape(weight.shape[0], weight.shape[1], -1))
            return torch.einsum('bck,nk->bcn', z, basis).real
        if finufft_type1 is None:
            raise ImportError('Install a pytorch-finufft version compatible with this PyTorch/device')
        z = finufft_type1(points, values, (self.my, self.mx), eps=self.eps, modeord=1, isign=-1)
        z = torch.einsum('bihw,iohw->bohw', z, weight)
        return finufft_type2(points, z, eps=self.eps, modeord=1, isign=1).real


class GraphConv(nn.Module):

    def __init__(self, width, chunk_size=8192):
        super().__init__()
        self.neighbor = nn.Linear(width, width, bias=False)
        self.difference = nn.Linear(width, width, bias=False)
        self.geometry = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width), nn.Sigmoid())
        self.chunk_size = chunk_size

    def forward(self, x, graph):
        src, dst, features, degree = graph
        nodes = x.transpose(1, 2).contiguous()
        neighbor = self.neighbor(nodes).transpose(1, 2).contiguous()
        delta = self.difference(nodes).transpose(1, 2).contiguous()
        out = torch.zeros_like(x)
        for start in range(0, len(src), self.chunk_size):
            s, d = src[start:start+self.chunk_size], dst[start:start+self.chunk_size]
            gate = self.geometry(features[start:start+self.chunk_size]).T
            message = (neighbor[:, :, d] + delta[:, :, d] - delta[:, :, s]) * gate
            out.index_add_(2, s, message)
        return out / degree.clamp_min(1)[None, None, :]


class FNO(nn.Module):
    def __init__(self, width=32, layers=4, modes_y=20, modes_x=32, backend='finufft',
                 nufft_eps=1e-5, operator='graph_nufft', graph_chunk_size=8192, activation_checkpointing=False):
        super().__init__()
        if operator not in ('graph_nufft', 'nufft_only', 'graph_only') or backend not in ('finufft', 'direct'):
            raise ValueError('Unknown operator/backend')
        self.operator = operator
        self.activation_checkpointing = activation_checkpointing
        self.lift = nn.Conv1d(9, width, 1)
        self.spec = nn.ModuleList([SpectralConv(width, modes_y, modes_x, nufft_eps, backend)
                                   for _ in range(layers)] if operator != 'graph_only' else [])
        self.graph = nn.ModuleList([GraphConv(width, graph_chunk_size) for _ in range(layers)]
                                  if operator != 'nufft_only' else [])
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.head = nn.Sequential(nn.Conv1d(width, 64, 1), nn.GELU(), nn.Conv1d(64, 3, 1))

    def forward(self, inputs, points, weights, graph=None):
        x = self.lift(inputs)
        for i in range(len(self.local)):
            def layer(z, index=i):
                y = self.local[index](z)
                if self.spec:
                    y = y + self.spec[index](z, points, weights)
                if self.graph:
                    if graph is None:
                        raise ValueError('Graph branch requires mesh edges and geometry')
                    y = y + self.graph[index](z, graph)
                return torch.nn.functional.gelu(y)
            if self.activation_checkpointing and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint as recompute

                x = recompute(layer, x, use_reentrant=True)
            else:
                x = layer(x)
        return self.head(x)


class FlowPhysics:
    def __init__(self, config, device):
        self.c, self.device = config, device
        xy, tri, bounds = mesh_arrays(config['grid']['mesh_path'])
        self.signature = mesh_signature(xy, tri, bounds)
        self.fv = FiniteVolume(xy, tri, bounds, device=device)
        self.xy, self.area = self.fv.xy, self.fv.volume
        self.bounds = {k: torch.as_tensor(v, device=device, dtype=torch.long) for k, v in bounds.items()}
        edges = self.fv.edges
        self.src, self.dst = torch.cat((edges[:, 0], edges[:, 1])), torch.cat((edges[:, 1], edges[:, 0]))
        self.gradient = MeshGradient(self.xy, self.src, self.dst).to(device)
        self.n = len(xy); self.x, self.y = self.xy.unbind(-1)
        g = config['grid']
        self.points = torch.stack((2*math.pi*(self.y-g['ymin'])/(g['ymax']-g['ymin'])-math.pi,
                                   2*math.pi*(self.x-g['xmin'])/(g['xmax']-g['xmin'])-math.pi)).contiguous()
        self.quad = self.area / self.area.sum()
        self.boundary_flag = self.x.new_zeros(self.n)
        for ids in self.bounds.values():
            self.boundary_flag[ids] = 1
        self.active = self.boundary_flag == 0
        if not bool(self.active.any()):
            raise ValueError('No interior control volumes')
        length = float(config['geometry'].get('chord', config['geometry'].get('diameter', 1.)))
        relative = (self.xy[self.dst] - self.xy[self.src]) / length
        features = torch.cat((relative, relative.norm(dim=-1, keepdim=True)), -1)
        degree = self.x.new_zeros(self.n); degree.index_add_(0, self.src, torch.ones_like(self.src, dtype=self.x.dtype))
        self.graph = self.src, self.dst, features, degree
        solid_ids = np.unique(np.concatenate([bounds[k] for k in ('wall','top','bottom')]))
        self.solid_nodes = torch.as_tensor(solid_ids, dtype=torch.long, device=device)
        distance = cKDTree(xy[solid_ids]).query(xy)[0]
        self.near_wall = torch.as_tensor(distance < config['physics']['near_wall_distance'], device=device) & self.active
        wx0, wx1, wy0, wy1 = config['physics']['wake_box']
        self.wake = (self.x > wx0) & (self.x < wx1) & (self.y > wy0) & (self.y < wy1) & self.active
        if not bool(self.near_wall.any()) or not bool(self.wake.any()):
            raise ValueError('Near-wall and wake checks require interior nodes; refine mesh or correct region settings')
        self.length = length

        from mesh import validate_geometry
        centers, radius = validate_geometry(config)
        self.body_names = ('cylinder_1', 'cylinder_2', 'cylinder_3')
        self.body_face_masks = {}
        wall = self.fv.boundary_tag == 0
        owner_xy = self.xy[self.fv.boundary_owner]
        for name, center in zip(self.body_names, centers):
            distance = (owner_xy-owner_xy.new_tensor(center)).norm(dim=-1)
            mask = wall & ((distance-radius).abs() < 1e-5*length)
            if not bool(mask.any()):
                raise ValueError('No wall faces found for '+name)
            self.body_face_masks[name] = mask
        coverage = torch.stack(list(self.body_face_masks.values())).to(torch.int64).sum(0)
        if not torch.equal(coverage, wall.to(torch.int64)):
            raise ValueError('Body face masks must partition every wall face exactly once')

    def body_coefficients(self, flux):

        return {name: flux['boundary_momentum'][:, :, mask].sum(-1)/(.5*self.length)
                for name, mask in self.body_face_masks.items()}

    def inlet(self, time):
        return 1 - math.exp(-(time/self.c['physics']['ramp_time'])**2)

    def initial(self, amplitudes):
        cx, cy = self.c['physics']['perturbation_center']
        radius = self.c['physics']['perturbation_radius']
        psi = torch.clamp(1-((self.x-cx)/radius)**2-((self.y-cy)/radius)**2, min=0)**4
        px, py = self.gradient(psi[None, None, :])
        velocity = amplitudes[:, None, None] * torch.cat((py, -px), 1)
        return self.velocity_bc(velocity, 0.)

    def velocity_bc(self, velocity, speed):
        b = velocity.shape[0]
        ids = self.bounds['inlet']
        values = velocity.new_zeros((b, 2, len(ids)))
        amplitude = torch.as_tensor(speed, dtype=velocity.dtype, device=velocity.device)
        values[:, 0] = amplitude.reshape(-1, 1)*self.inlet_shape(self.y[ids])
        velocity = torch.index_copy(velocity, 2, ids, values)
        ids = self.solid_nodes
        return torch.index_copy(velocity, 2, ids, velocity.new_zeros((b, 2, len(ids))))

    def inlet_shape(self, y):

        h = self.c['geometry']['inlet_half_height']
        if self.c['physics']['inlet_profile'] != 'parabolic_mean_one':
            raise ValueError('This case uses a mean-one parabolic inlet')
        return 1.5*(1-(y/h).square()).clamp_min(0)

    def inlet_amplitude(self, velocity):

        ids = self.bounds['inlet']
        shape = self.inlet_shape(self.y[ids])
        return (velocity[:, 0, ids]*shape).sum(-1)/shape.square().sum()


    def predict(self, model, previous, re, time):
        dt = self.c['physics']['dt']; b, _, n = previous.shape
        g = self.c['grid']
        def expand(v):
            return v.reshape(1, 1, n).expand(b, 1, n)
        speed = self.inlet(time + dt)
        inputs = torch.cat((previous, expand(2*(self.x-g['xmin'])/(g['xmax']-g['xmin'])-1),
                            expand(2*(self.y-g['ymin'])/(g['ymax']-g['ymin'])-1), expand(self.boundary_flag),
                            (re/100)[:, None, None].expand(b, 1, n), previous.new_full((b, 1, n), speed),
                            previous.new_full((b, 1, n), dt), expand(self.area/self.area.mean())), 1)
        raw = model(inputs, self.points, self.quad, self.graph)
        velocity = self.velocity_bc(previous + dt*raw[:, :2], speed)
        pressure = raw[:, 2:3] - raw[:, 2:3, self.bounds['outlet']].mean(-1, keepdim=True)
        ids = self.bounds['outlet']
        pressure = torch.index_copy(pressure, 2, ids, pressure.new_zeros((b, 1, len(ids))))
        return velocity, pressure

    def boundary_flux(self, speed):
        def apply(points, normals, tags, velocity, pressure, gradient):
            prescribed = velocity.new_zeros(velocity.shape)
            h = self.c['geometry']['inlet_half_height']

            shape = self.inlet_shape(points[:, 1]) - normals[:, 0].square()/(8*h*h)
            prescribed[:, 0] = speed[:, None]*shape[None, :]*(tags == 1)[None, :]
            outlet = tags == 2
            velocity = torch.where(outlet[None, None, :], velocity, prescribed)
            pressure = torch.where(outlet[None, None, :], torch.zeros_like(pressure), pressure)
            unit = normals / normals.norm(dim=-1, keepdim=True)
            normal_grad = (gradient*unit).sum(-1, keepdim=True)*unit
            gradient = gradient - normal_grad*outlet[None, None, :, None]
            return velocity, pressure, gradient
        return apply

    def residual(self, previous, velocity, pressure, re):
        speed = self.inlet_amplitude(velocity)
        spatial, div, flux = self.fv.flow_terms(velocity, pressure, re, self.boundary_flux(speed))
        momentum = (velocity-previous)/self.c['physics']['dt'] + spatial
        def mse(f, mask=None):
            weight = self.area if mask is None else self.area*mask
            if not bool(weight.sum() > 0):
                return f.new_zeros(f.shape[0])
            return (f.square()*weight).sum((1, 2))/(f.shape[1]*weight.sum())
        cv = self.fv.reconstruction(velocity)
        _, grad = self.fv.evaluate(velocity, cv, self.fv.boundary_owner, self.fv.boundary_face)
        outlet = self.fv.boundary_tag == 2
        normals = self.fv.boundary_normal[outlet]
        length = normals.norm(dim=-1)
        dn = (grad[:, :, outlet]*(normals/length[:, None])).sum(-1)
        boundary = (dn.square()*length).sum((1, 2))/(2*length.sum())

        inlet_length = self.fv.boundary_normal[self.fv.boundary_tag == 1].norm(dim=-1).sum()
        imbalance = flux['boundary_mass'].sum((1, 2))/inlet_length
        terms = {'momentum': mse(momentum, self.active), 'divergence': mse(div), 'boundary': boundary,
                 'near_wall_momentum': mse(momentum, self.near_wall), 'near_wall_divergence': mse(div, self.near_wall),
                 'wake_momentum': mse(momentum, self.wake), 'wake_divergence': mse(div, self.wake),
                 'mass_balance': imbalance.square()}
        loss = sum(self.c['loss'][k]*v.mean() for k, v in terms.items())
        metrics = {k: float(v.detach().max().sqrt().cpu()) for k, v in terms.items()}
        return loss, metrics

    def accepted(self, metrics):
        return all(k in metrics and math.isfinite(metrics[k]) and metrics[k] <= limit
                   for k, limit in self.c['tolerances'].items())

    @torch.no_grad()
    def diagnostics(self, velocity):
        gx, gy = self.gradient(velocity)
        return gx[:, 1:2]-gy[:, 0:1]




def get_device(name):
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; select device=cpu explicitly for verification')
    return torch.device(name)


def source_signature():
    digest = hashlib.sha256()
    for name in ('finite_volume.py', 'model.py', 'nufft_compat.py'):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def validate(c):
    from mesh import quality_metrics, validate_geometry, channel_vertices
    validate_geometry(c)
    p = Path(c['grid']['mesh_path'])
    xy, tri, bounds = mesh_arrays(p)
    report = json.loads(p.with_suffix('.json').read_text(encoding='utf8'))
    if report.get('geometry') != c['geometry']:
        raise ValueError('Mesh geometry differs from this case; rebuild mesh')
    if not np.array_equal(np.asarray(report.get('channel_polygon')), channel_vertices(c)):
        raise ValueError('Mesh channel dimensions differ from config; rebuild mesh')
    if report.get('mesh_signature') != mesh_signature(xy, tri, bounds):
        raise ValueError('Mesh file differs from its quality report; rebuild mesh')
    if c['physics']['perturbation_radius'] <= 0:
        raise ValueError('perturbation_radius must be positive')
    if c['physics'].get('inlet_profile') != 'parabolic_mean_one':
        raise ValueError('Use parabolic_mean_one inlet for this channel case')
    if not report.get('passed'):
        raise ValueError('Mesh quality/size gates failed; regenerate mesh.py')
    q = quality_metrics(xy, tri)
    g = c['grid']
    if not (q['all_positive_area'] and q['min_angle_deg'] >= g['min_triangle_angle_deg']
            and q['min_scaled_jacobian'] >= g['min_scaled_jacobian'] and q['max_edge_ratio'] <= g['max_aspect']
            and abs(q['cells']-g['target_cells']) <= g['tolerance']*g['target_cells']):
        raise ValueError('Actual mesh fails configured quality/size gates')
    if min(c['cases']['re']) <= 0 or len(c['cases']['re']) != len(c['cases']['amplitude']):
        raise ValueError('Invalid case list')
    if c['physics']['dt'] <= 0 or c['physics']['ramp_time'] <= 0:
        raise ValueError('dt and ramp_time must be positive')
    reference_length = c['geometry'].get('chord', c['geometry'].get('diameter', 1.))
    if not math.isclose(reference_length, 1.):
        raise ValueError('Use coordinates nondimensionalized by D/chord=1 for nu=1/Re')
    tr = c['training']
    if min(tr['steps'], tr['iterations_per_step'], tr['replay_capacity']) < 0 or tr['check_every'] <= 0 or tr['save_every'] <= 0:
        raise ValueError('Invalid optimization/checkpoint budget')
    if set(c['loss']) != set(c['tolerances']) or any(v <= 0 for v in c['tolerances'].values()):
        raise ValueError('Every residual needs a positive gate and a loss weight')
    if c.get('verification_report'):
        v = json.loads(Path(c['verification_report']).read_text(encoding='utf8'))
        if v.get('source_signature') != source_signature() or not v.get('finite_volume_passed'):
            raise ValueError('Run sanity_check.py after changing physics/operator code')
        if c['model']['operator'] != 'graph_only' and not v.get(c['model']['backend']+'_passed'):
            raise ValueError('Requested spectral backend has not passed sanity_check.py')
        if c['model']['operator'] != 'graph_only' and c['model']['backend'] == 'finufft':
            if torch.device(v.get('spectral_device', 'cpu')).type != torch.device(c['device']).type:
                raise ValueError('Verify the requested device backend: python sanity_check.py --device '+c['device'])


def append_csv(path, row):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline='', encoding='utf8') as f:
            if next(csv.reader(f)) != list(row):
                raise ValueError(f'CSV schema mismatch: {path}; use a new run directory')
    with path.open('a', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, default=str, allow_nan=False), encoding='utf8')


def load_checkpoint(path):
    kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    if not Path(path).is_file():
        raise FileNotFoundError('Select the checkpoint.pt FILE, not its train folder: '+str(path))
    saved = torch.load(path, **kwargs)  # Only load checkpoints from trusted local runs.
    if saved.get('version') != CHECKPOINT_VERSION or saved.get('discretization') != DISCRETIZATION:
        raise ValueError('Legacy/foreign checkpoint: v1 LS physics cannot resume v2 FV trajectories; retrain')
    return saved


def checkpoint(path, model, optimizer, c, state, committed, replay, status, physics=None):
    import random
    path = Path(path); temporary = path.with_suffix('.tmp')
    payload = {'version': CHECKPOINT_VERSION, 'discretization': DISCRETIZATION,
               'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': c,
               'state': state.detach().cpu(), 'committed_steps': committed,
               'replay': [(s.detach().cpu(), t) for s, t in replay], 'status': status,
               'mesh_signature': physics.signature if physics is not None else mesh_signature(*mesh_arrays(c['grid']['mesh_path'])),
               'rng_torch': torch.get_rng_state(), 'rng_numpy': np.random.get_state(), 'rng_python': random.getstate(),
               'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    torch.save(payload, temporary); temporary.replace(path)


@torch.no_grad()
def record(out, physics, state, pressure, re, time, step, save_field, previous=None, updates=0, elapsed=0.):
    omega = physics.diagnostics(state)
    speed = physics.inlet_amplitude(state)
    _, div, flux = physics.fv.flow_terms(state, pressure, re, physics.boundary_flux(speed))
    coefficients = flux['wall_force'] / (.5*physics.length)  # ONLY three cylinders; rho=U_bulk,in=D=1
    channel_mask = (physics.fv.boundary_tag == 3) | (physics.fv.boundary_tag == 4)
    channel_force = flux['boundary_momentum'][:, :, channel_mask].sum(-1)
    inlet_flux = flux['boundary_mass'][:, :, physics.fv.boundary_tag == 1].sum((1,2))
    outlet_flux = flux['boundary_mass'][:, :, physics.fv.boundary_tag == 2].sum((1,2))
    body_coefficients = physics.body_coefficients(flux)
    for i in range(len(re)):
        append_csv(Path(out)/'history.csv', {'step': step, 'time': time, 'case': i, 'Re': float(re[i]),
                   'Cd': float(coefficients[i, 0]), 'Cl': float(coefficients[i, 1]),
                   'mass_flux': float(flux['boundary_mass'][i].sum()),
                   'inlet_flux': float(inlet_flux[i]), 'outlet_flux': float(outlet_flux[i]),
                   'channel_wall_force_x': float(channel_force[i,0]),
                   'channel_wall_force_y': float(channel_force[i,1]),
                   'divergence_rms': float((div[i].square()*physics.area).sum().div(physics.area.sum()).sqrt()),
                   'max_speed': float(state[i].square().sum(0).sqrt().max()),
                   'vorticity_rms': float((omega[i].square()*physics.quad).sum().sqrt()),
                   'updates': updates, 'step_seconds': elapsed})
    for name, values in body_coefficients.items():
        for i in range(len(re)):
            append_csv(Path(out)/'body_history.csv', {'step': step, 'time': time, 'case': i,
                       'Re': float(re[i]), 'body': name,
                       'Cd': float(values[i, 0]), 'Cl': float(values[i, 1])})
    if save_field:
        np.savez_compressed(Path(out)/('field_%06d.npz' % step), velocity=state.cpu().numpy(),
                            pressure=pressure.cpu().numpy(), vorticity=omega.cpu().numpy(),
                            re=re.cpu().numpy(), time=time, xy=physics.xy.cpu().numpy(),
                            triangles=physics.fv.triangles.cpu().numpy(), volume=physics.area.cpu().numpy())
