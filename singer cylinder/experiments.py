
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from params import CONFIG, ROOT
from model import write_json


def controlled_signature(c):
    return hashlib.sha256(json.dumps({k: c[k] for k in ('physics', 'loss', 'tolerances', 'training', 'seed')},
                                     sort_keys=True).encode()).hexdigest()


def generate(folder, device=None):
    folder = Path(folder).resolve(); folder.mkdir(parents=True, exist_ok=True)
    base = deepcopy(CONFIG)
    if device:
        base['device'] = device
    manifest = {'groups': {}, 'status': 'configurations_only_not_experimental_results',
                'normalization': 'rho=U_ref=D_or_chord=1',
                'comparability': 'Same dt, physical residual, gates, optimization budget and seed in all ablations.'}
    manifest['groups']['operator_verification'] = {'command': 'python sanity_check.py',
                                                  'checks': ['NUFFT reconstruction/adjoint', 'LS gradient',
                                                             'FV Laplacian', 'manufactured NS', 'mesh convergence']}
    for geometry in ('cylinder', 'NACA0012'):
        entries = []
        for density, cells in (('coarse', 22500), ('medium', 90000), ('fine', 360000)):
            c = deepcopy(base)
            c['geometry'] = ({'name': 'cylinder', 'diameter': 1., 'center': [0., 0.]}
                             if geometry == 'cylinder' else deepcopy(base['geometry']))
            c['cases'] = {'re': [20., 40., 100., 150.], 'amplitude': [.0015]*4}
            if geometry == 'cylinder':
                c['physics']['wake_box'] = [.5, 10., -1.5, 1.5]
            scale = (90000/cells)**.5
            c['grid'].update(target_cells=cells, near_size=base['grid']['near_size']*scale,
                             far_size=base['grid']['far_size']*scale)
            mesh = folder/'meshes'/f'{geometry}_{density}.npz'
            c['grid']['mesh_path'] = str(mesh)
            filename = folder/f'{geometry}_{density}.json'
            write_json(filename, c)
            entries.append({'config': str(filename), 'mesh_command': f'python mesh.py --config "{filename}"',
                            'solve_command': f'python train.py --config "{filename}" --output "{folder / "runs" / filename.stem}"',
                            'mesh_cells': cells, 'Re': c['cases']['re']})
        manifest['groups']['cylinder_reference_cfd' if geometry == 'cylinder' else 'complex_geometry'] = entries
    manifest['groups']['mesh_independence'] = {
        'configurations': [str(folder/f'cylinder_{d}.json') for d in ('coarse', 'medium', 'fine')],
        'required_metrics': ['velocity/pressure error', 'mean Cd', 'Cl rms', 'St', 'updates', 'seconds', 'mass error'],
        'note': 'Also verify time-step and domain-size independence before claiming CFD agreement.'}
    cylinder = json.loads((folder/'cylinder_medium.json').read_text(encoding='utf8'))
    manifest['groups']['ablation'] = []
    for operator in ('nufft_only', 'graph_only', 'graph_nufft'):
        c = deepcopy(cylinder); c['model']['operator'] = operator
        p = folder/f'ablation_{operator}.json'; write_json(p, c)
        manifest['groups']['ablation'].append({'config': str(p), 'controls_hash': controlled_signature(c),
                                              'operator': operator, 'shared_mesh': c['grid']['mesh_path']})
    hashes = {v['controls_hash'] for v in manifest['groups']['ablation']}
    assert len(hashes) == 1
    target = json.loads((folder/'NACA0012_medium.json').read_text(encoding='utf8'))
    target['grid']['algorithms'] = [5]
    target['grid']['mesh_path'] = str(folder/'meshes'/'NACA0012_alternate_topology.npz')
    p = folder/'NACA0012_alternate_topology.json'; write_json(p, target)
    manifest['groups']['cross_mesh'] = {
        'source': str(folder/'NACA0012_medium.json'),
        'targets': [str(folder/f'NACA0012_{d}.json') for d in ('coarse', 'fine')] + [str(p)],
        'initialization': 'One source v2 checkpoint; rebuild target dual volumes/graph/NUFFT quadrature; reset state/optimizer/replay.',
        'command': 'python predict_unseen.py --checkpoint SOURCE/checkpoint.pt --config TARGET.json --re 40 100',
        'compare_with': 'python train.py --config TARGET.json (fresh initialization)'}
    write_json(folder/'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT/'experiments'))
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    generate(args.output, args.device)
    print('Experiment configurations saved:', args.output)
