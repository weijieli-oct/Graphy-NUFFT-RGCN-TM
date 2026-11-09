
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch import nn

BOUNDARIES = ('wall', 'inlet', 'outlet', 'top', 'bottom')
DISCRETIZATION = 'median_dual_quadratic_flux_v2'


def mesh_arrays(path):
    with np.load(path, allow_pickle=False) as d:
        xy = np.array(d['xy'], dtype=np.float64)
        tri = np.array(d['triangles'], dtype=np.int64)
        if 'quads' in d and len(d['quads']):
            raise ValueError('Only pure triangular meshes are supported')
        bounds = {k: np.array(d['boundary_' + k], dtype=np.int64) for k in BOUNDARIES}
    if xy.ndim != 2 or xy.shape[1] != 2 or tri.ndim != 2 or tri.shape[1] != 3 or not len(tri):
        raise ValueError('Malformed mesh arrays')
    if not np.isfinite(xy).all() or tri.min() < 0 or tri.max() >= len(xy):
        raise ValueError('Invalid coordinates/connectivity')
    used = np.unique(tri)
    remap = np.full(len(xy), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    for k, ids in bounds.items():
        if not len(ids) or ids.min() < 0 or ids.max() >= len(xy) or (remap[ids] < 0).any():
            raise ValueError(f'Invalid/orphan {k} boundary nodes')
        bounds[k] = np.unique(remap[ids])
    return xy[used], remap[tri], bounds


def mesh_signature(xy, tri, bounds):
    h = hashlib.sha256()
    for a in (xy, tri, *(bounds[k] for k in BOUNDARIES)):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def dual_geometry(xy, triangles, bounds):
    xy = np.asarray(xy, dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.int64).copy()
    p = xy[tri]
    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    if (np.abs(cross) < 1e-20).any():
        raise ValueError('Degenerate triangle')
    tri[cross < 0] = tri[cross < 0][:, [0, 2, 1]]
    p = xy[tri]
    area = np.abs(cross) / 2
    center = p.mean(1)
    volume = np.zeros(len(xy))
    centroid_sum = np.zeros_like(xy)
    src, dst, face, normal = [], [], [], []
    all_edges = []
    for k in range(3):
        i, j = tri[:, k], tri[:, (k + 1) % 3]
        a, b = xy[i], xy[j]
        mid = (a + b) / 2
        tangent = center - mid
        n = np.column_stack((tangent[:, 1], -tangent[:, 0]))
        n *= np.where(np.sum(n * (b - a), axis=1) > 0, 1., -1.)[:, None]
        src.append(i); dst.append(j); face.append((center + mid) / 2); normal.append(n)
        all_edges.append(np.column_stack((i, j)))
        np.add.at(volume, i, area / 3)
        prior = (a + p[:, (k - 1) % 3]) / 2
        subcentroid = ((a + mid + center) + (a + prior + center)) / 6
        np.add.at(centroid_sum, i, (area / 3)[:, None] * subcentroid)
    raw_edges = np.concatenate(all_edges)
    edges, first, counts = np.unique(np.sort(raw_edges, axis=1), axis=0,
                                    return_index=True, return_counts=True)
    if (counts > 2).any() or (volume <= 0).any():
        raise ValueError('Non-manifold edges or nonpositive control volumes')
    boundary_edges = raw_edges[first[counts == 1]]
    memberships = {k: np.isin(boundary_edges, ids).all(1) for k, ids in bounds.items()}
    if not np.all(sum(memberships.values()) == 1):
        raise ValueError('Each exterior edge must belong to exactly one boundary group')
    tags = np.zeros(len(boundary_edges), dtype=np.int64)
    for tag, name in enumerate(BOUNDARIES):
        tags[memberships[name]] = tag
    a, b = xy[boundary_edges[:, 0]], xy[boundary_edges[:, 1]]
    d = b - a
    half_normal = np.column_stack((d[:, 1], -d[:, 0])) / 2
    owners = boundary_edges.T.reshape(-1)
    bf = np.concatenate((.75 * a + .25 * b, .25 * a + .75 * b))
    bn = np.concatenate((half_normal, half_normal))
    src, dst, face, normal = map(np.concatenate, (src, dst, face, normal))
    closure = np.zeros_like(xy)
    np.add.at(closure, src, normal); np.add.at(closure, dst, -normal)
    np.add.at(closure, owners, bn)
    if np.max(np.abs(closure)) > 1e-10 * max(1., np.ptp(xy, axis=0).max()):
        raise ValueError('Dual control-volume face normals do not close')
    return dict(triangles=tri, volume=volume, centroid=centroid_sum / volume[:, None],
                edges=edges, src=src, dst=dst, face=face, normal=normal,
                boundary_owner=owners, boundary_face=bf, boundary_normal=bn,
                boundary_tag=np.tile(tags, 2))


class QuadraticReconstruction(nn.Module):
    def __init__(self, xy, edges, dtype=torch.float32, device='cpu'):
        super().__init__()
        xy = np.asarray(xy, dtype=np.float64)
        neighbors = [set() for _ in xy]
        for i, j in edges:
            neighbors[i].add(j); neighbors[j].add(i)
        sources, targets, coefficients = [], [], []
        for i in range(len(xy)):
            ring = {i}
            found = {i}
            for depth in range(1, 6):
                ring = set().union(*(neighbors[j] for j in ring)) - found
                found |= ring
                ids = np.array(sorted(found - {i}), dtype=np.int64)
                if depth < 2 or len(ids) < 5:
                    continue
                delta = xy[ids] - xy[i]
                scale = np.sqrt(np.mean(np.sum(delta ** 2, axis=1)))
                dx, dy = (delta / scale).T
                a = np.column_stack((dx, dy, .5 * dx ** 2, dx * dy, .5 * dy ** 2))
                w = 1 / np.sqrt(dx ** 2 + dy ** 2)
                aw = a * w[:, None]
                if np.linalg.matrix_rank(aw) == 5 and np.linalg.cond(aw) < 1e8:
                    coef = (np.linalg.pinv(aw, rcond=1e-12) * w).T
                    coef /= np.array([scale, scale, scale ** 2, scale ** 2, scale ** 2])
                    sources.extend([i] * len(ids)); targets.extend(ids); coefficients.extend(coef)
                    break
            else:
                raise ValueError(f'Rank-deficient quadratic stencil at node {i}; refine mesh')
        self.register_buffer('src', torch.tensor(sources, dtype=torch.long, device=device))
        self.register_buffer('dst', torch.tensor(targets, dtype=torch.long, device=device))
        self.register_buffer('coef', torch.tensor(np.array(coefficients), dtype=dtype, device=device))

    def forward(self, f):
        delta = f[:, :, self.dst] - f[:, :, self.src]
        out = f.new_zeros((*f.shape, 5))
        for k in range(5):
            out[..., k].index_add_(2, self.src, delta * self.coef[:, k])
        return out


class FiniteVolume(nn.Module):
    def __init__(self, xy, triangles, bounds, dtype=torch.float32, device='cpu'):
        super().__init__()
        geo = dual_geometry(xy, triangles, bounds)
        self.geometry = geo
        self.register_buffer('xy', torch.as_tensor(xy, dtype=dtype, device=device))
        for key, a in geo.items():
            self.register_buffer(key, torch.as_tensor(a, dtype=torch.long if a.dtype.kind in 'iu' else dtype,
                                                     device=device))
        self.reconstruction = QuadraticReconstruction(xy, geo['edges'], dtype, device)

    def evaluate(self, f, coeff, owners, points):
        dx, dy = (points - self.xy[owners]).unbind(-1)
        a, b, c, d, e = coeff[:, :, owners].unbind(-1)
        value = f[:, :, owners] + a * dx + b * dy + .5 * c * dx ** 2 + d * dx * dy + .5 * e * dy ** 2
        grad = torch.stack((a + c * dx + d * dy, b + d * dx + e * dy), -1)
        return value, grad

    def internal(self, f, coeff):
        a, ga = self.evaluate(f, coeff, self.src, self.face)
        b, gb = self.evaluate(f, coeff, self.dst, self.face)
        return .5 * (a + b), .5 * (ga + gb)

    def integrate(self, interior_flux, boundary_flux):
        result = interior_flux.new_zeros((*interior_flux.shape[:-1], len(self.xy)))
        result.index_add_(2, self.src, interior_flux)
        result.index_add_(2, self.dst, -interior_flux)
        result.index_add_(2, self.boundary_owner, boundary_flux)
        return result / self.volume

    def laplacian(self, f, boundary_gradient=None):
        coeff = self.reconstruction(f)
        _, gi = self.internal(f, coeff)
        _, gb = self.evaluate(f, coeff, self.boundary_owner, self.boundary_face)
        if boundary_gradient is not None:
            gb = boundary_gradient
        return self.integrate((gi * self.normal).sum(-1), (gb * self.boundary_normal).sum(-1))

    def flow_terms(self, velocity, pressure, re, boundary=None):
        """Return convective+pressure-viscous divergence, mass divergence, flux data.

        boundary(points, normals, tags, velocity, pressure, grad_velocity) can
        prescribe boundary face states. With None, polynomial continuation is
        used for manufactured-solution tests, not physical external-flow BCs.
        """
        cv, cp = self.reconstruction(velocity), self.reconstruction(pressure)
        vi, gi = self.internal(velocity, cv)
        pi, _ = self.internal(pressure, cp)
        vb, gb = self.evaluate(velocity, cv, self.boundary_owner, self.boundary_face)
        pb, _ = self.evaluate(pressure, cp, self.boundary_owner, self.boundary_face)
        if boundary is not None:
            vb, pb, gb = boundary(self.boundary_face, self.boundary_normal,
                                  self.boundary_tag, vb, pb, gb)
        nu = 1 / re[:, None, None]

        def flux(v, p, grad, normal):
            mass = (v * normal.T).sum(1, keepdim=True)
            shear_x = 2 * grad[:, 0, :, 0] * normal[:, 0] + (grad[:, 0, :, 1] + grad[:, 1, :, 0]) * normal[:, 1]
            shear_y = (grad[:, 0, :, 1] + grad[:, 1, :, 0]) * normal[:, 0] + 2 * grad[:, 1, :, 1] * normal[:, 1]
            viscous = nu * torch.stack((shear_x, shear_y), 1)
            return v * mass + p * normal.T - viscous, mass, p * normal.T - viscous

        fi, mi, _ = flux(vi, pi, gi, self.normal)
        fb, mb, traction = flux(vb, pb, gb, self.boundary_normal)
        return self.integrate(fi, fb), self.integrate(mi, mb), {
            'boundary_mass': mb, 'wall_force': traction[:, :, self.boundary_tag == 0].sum(-1),
            'boundary_momentum': fb, 'interior_momentum': fi}
