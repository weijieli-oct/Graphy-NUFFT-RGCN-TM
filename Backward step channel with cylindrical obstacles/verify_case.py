
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from model import FlowPhysics, get_device, write_json
from mesh import channel_vertices, quality_metrics, validate_geometry
from params import CONFIG


@torch.no_grad()
def verify(config, device='cpu'):
    c = deepcopy(config)
    centers,radius = validate_geometry(c)
    physics = FlowPhysics(c,get_device(device))
    fv = physics.fv
    xy = fv.xy.cpu().numpy()
    tri = fv.triangles.cpu().numpy()
    # Perform geometric tolerances against original FP64 mesh, not rounded
    # tensors on the selected compute device.
    with np.load(c['grid']['mesh_path'],allow_pickle=False) as saved:
        xy64,tri64 = saved['xy'],saved['triangles']
    q = quality_metrics(xy64,tri64)
    cfg = c['grid']
    assert abs(len(tri)-cfg['target_cells']) <= cfg['target_cells']*cfg['tolerance']
    assert q['all_positive_area'] and q['min_angle_deg']>=cfg['min_triangle_angle_deg']
    assert q['min_scaled_jacobian']>=cfg['min_scaled_jacobian'] and q['max_edge_ratio']<=cfg['max_aspect']
    unique = fv.edges.cpu().numpy()
    assert len(xy)-len(unique)+len(tri)==-2, 'Expected one connected region and three holes'
    from matplotlib.path import Path as Polygon
    assert Polygon(channel_vertices(c)).contains_points(xy64[tri64].mean(1)).all()
    for center in centers:
        # No centroid lies appreciably inside an obstacle. The curved surface is
        # approximated by chords; the dedicated topology test checks the polygons.
        assert np.linalg.norm(xy64[tri64].mean(1)-center,axis=1).min()>radius*.99
    nfaces = len(fv.boundary_tag)
    amp = fv.xy.new_tensor([.25,1.])
    v = fv.xy.new_zeros((2,2,nfaces))
    p = fv.xy.new_ones((2,1,nfaces))
    g = fv.xy.new_ones((2,2,nfaces,2))
    v,p,g = physics.boundary_flux(amp)(fv.boundary_face,fv.boundary_normal,fv.boundary_tag,v,p,g)
    inlet = fv.boundary_tag==1
    solid = (fv.boundary_tag==0)|(fv.boundary_tag==3)|(fv.boundary_tag==4)
    mass = (v*fv.boundary_normal.T).sum(1)
    expected = -amp*(2*c['geometry']['inlet_half_height'])
    error = float((mass[:,inlet].sum(1)-expected).abs().max())
    assert error<2e-5
    assert not mass[:,solid].count_nonzero()
    state = physics.velocity_bc(fv.xy.new_zeros((2,2,physics.n)),amp)
    assert not state[:,:,physics.solid_nodes].count_nonzero()
    assert float((physics.inlet_amplitude(state)-amp).abs().max())<1e-6
    assert set(physics.body_face_masks)=={'cylinder_1','cylinder_2','cylinder_3'}
    initial = physics.initial(fv.xy.new_tensor([.0015]))
    assert torch.isfinite(initial).all() and float(initial.abs().max())>0
    zero = fv.xy.new_zeros((1,2,physics.n))
    loss,metrics = physics.residual(zero,zero,zero[:,:1],fv.xy.new_tensor([40.]))
    assert float(loss)==0 and max(metrics.values())==0
    return {'passed':True,'device':str(device),'nodes':len(xy),'triangles':len(tri),
            'mesh_signature':physics.signature,'exact_inlet_flux_error':error,
            'wall_mass_leakage':0.0,'zero_state_residuals':metrics,
            'initial_perturbation_max':float(initial.abs().max()),
            'triangle_quality':q,'training_executed':False,'flow_accuracy_validated':False,
            'scope':'Full production mesh reconstruction, boundaries, force partition and zero-field residual. No time stepping.'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',default='verification/case_check.json')
    args=p.parse_args(argv)
    c=json.loads(Path(args.config).read_text(encoding='utf8')) if args.config else CONFIG
    value=verify(c,args.device)
    write_json(args.output,value)
    print(json.dumps(value,indent=2),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
