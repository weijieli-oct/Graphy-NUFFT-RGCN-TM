
from pathlib import Path
import json
import numpy as np


def naca0012(x):
    x=np.asarray(x,dtype=float)
    return .6*(.2969*np.sqrt(np.clip(x,0,1))-.1260*x-.3516*x*x+.2843*x**3-.1036*x**4)


def profile(n=240):
    theta=np.linspace(0,np.pi,n+1)
    x=(1+np.cos(theta))/2
    upper=np.column_stack((x,naca0012(x)))  # TE -> LE
    lower=np.column_stack((x[::-1],-naca0012(x[::-1]))) # LE -> TE
    return upper,lower


def quality_metrics(xy,tri,angle_threshold=17.,scaled_threshold=.25):
    p=xy[tri]
    e=np.roll(p,-1,axis=1)-p
    l=np.linalg.norm(e,axis=-1)
    signed=0.5*np.sum(p[:,:,0]*np.roll(p[:,:,1],-1,axis=1)-np.roll(p[:,:,0],-1,axis=1)*p[:,:,1],axis=1)
    cross=e[:,:,0]*np.roll(e,-1,axis=1)[:,:,1]-e[:,:,1]*np.roll(e,-1,axis=1)[:,:,0]
    sj=np.min(cross/np.maximum(l*np.roll(l,-1,axis=1),1e-30),axis=1)
    cosang=np.sum(-np.roll(e,1,axis=1)*e,axis=-1)/np.maximum(np.roll(l,1,axis=1)*l,1e-30)
    angles=np.rad2deg(np.arccos(np.clip(cosang,-1.,1.)))
    mina=np.min(angles,axis=1)
    aspect=l.max(axis=1)/np.maximum(l.min(axis=1),1e-30)
    worst=int(np.argmin(sj))
    return dict(cells=int(len(tri)),triangles=int(len(tri)),quads=0,
                min_area=float(signed.min()),all_positive_area=bool(np.all(signed>0)),
                min_angle_deg=float(mina.min()),min_scaled_jacobian=float(sj.min()),
                max_edge_ratio=float(aspect.max()),p99_edge_ratio=float(np.percentile(aspect,99)),
                angle_below_threshold=int(np.count_nonzero(mina<angle_threshold)),
                scaled_jacobian_below_threshold=int(np.count_nonzero(sj<scaled_threshold)),
                worst_cell_index=worst,worst_cell_center=p[worst].mean(axis=0).tolist(),
                worst_cell_vertices=p[worst].tolist())


def extract(gmsh, curves):
    tags, coords, _=gmsh.model.mesh.getNodes()
    tags=np.asarray(tags,dtype=np.int64)
    xy=np.asarray(coords,dtype=np.float64).reshape(-1,3)[:,:2]
    if not len(tags): raise RuntimeError('Gmsh generated no nodes')
    # Node tags are not necessarily contiguous or sorted.
    order=np.argsort(tags);sorted_tags=tags[order]
    typ, _, con=gmsh.model.mesh.getElements(dim=2)
    tris=[]
    for element_type, nodes in zip(typ,con):
        if int(element_type)!=2: raise RuntimeError(f'Only first-order triangles allowed, got Gmsh element type {element_type}')
        ids=np.asarray(nodes,dtype=np.int64).reshape(-1,3)
        ix=np.searchsorted(sorted_tags,ids)
        if (ix>=len(sorted_tags)).any() or np.any(sorted_tags[ix]!=ids):raise RuntimeError('Invalid triangle node tags')
        tris.append(order[ix])
    tri=np.vstack(tris).astype(np.int64)
    p=xy[tri]
    signed=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    flipped=signed<0
    tri[flipped]=tri[flipped][:,[0,2,1]]
    boundaries={}
    for name, curve_ids in curves.items():
        node_list=[]
        for curve in curve_ids:
            node_tags,_,_=gmsh.model.mesh.getNodes(1,int(curve),includeBoundary=True)
            node_list.extend(np.asarray(node_tags,dtype=np.int64).tolist())
        ids=np.unique(np.asarray(node_list,dtype=np.int64))
        ix=np.searchsorted(sorted_tags,ids)
        if (ix>=len(sorted_tags)).any() or np.any(sorted_tags[ix]!=ids):raise RuntimeError(f'Invalid {name} boundary tags')
        boundaries[name]=np.unique(order[ix]).astype(np.int64)
        if not len(boundaries[name]):raise RuntimeError(f'Empty boundary {name}')
    # Gmsh may return nodes not referenced by any 2-D element.
    # Compact all node IDs to exactly the nodes used by triangles.  In particular,
    # this prevents zero-area orphan nodes in the nodal finite-volume dual measure.
    used=np.unique(tri.reshape(-1))
    remap=np.full(len(xy),-1,dtype=np.int64)
    remap[used]=np.arange(len(used),dtype=np.int64)
    for name,values in boundaries.items():
        missing=values[remap[values]<0]
        if len(missing):
            raise RuntimeError(f'{name} boundary has {len(missing)} nodes absent from triangles; check geometry/element extraction')
        boundaries[name]=remap[values]
    tri=remap[tri]
    xy=xy[used]
    if len(np.unique(tri)) != len(xy):
        raise RuntimeError('Mesh node compaction failed')
    return xy,tri,boundaries


def build(config,output=None,verbose=True):
    try: import gmsh
    except ImportError as exc:raise RuntimeError('Install gmsh: pip install gmsh') from exc
    cfg=config['grid'];geo=config['geometry']
    out=Path(output or cfg['mesh_path']);out.parent.mkdir(parents=True,exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal',int(verbose))
        gmsh.option.setNumber('Mesh.ElementOrder',1)
        gmsh.option.setNumber('Mesh.RecombineAll',0)
        gmsh.option.setNumber('Mesh.MeshSizeFromPoints',0)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',0)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary',0)
        gmsh.model.add('NACA0012_triangles_only')
        g=gmsh.model.geo
        xmin,xmax,ymin,ymax=[float(cfg[k]) for k in ('xmin','xmax','ymin','ymax')]
        angle=np.deg2rad(float(geo.get('angle_deg',0.0)))
        chord=float(geo.get('chord',1.0))
        def transform(p):
            q=(p-np.array([0.25,0]))*chord
            return [float(.25*chord+q[0]*np.cos(angle)-q[1]*np.sin(angle)),
                    float(q[0]*np.sin(angle)+q[1]*np.cos(angle))]
        if geo['name'].lower() == 'cylinder':
            radius=float(geo.get('diameter',1.))/2
            cx,cy=geo.get('center',[0.,0.])
            center=g.addPoint(cx,cy,0)
            circle=[g.addPoint(cx+radius*np.cos(a),cy+radius*np.sin(a),0)
                    for a in np.arange(4)*np.pi/2]
            wall_curves=[g.addCircleArc(circle[k],center,circle[(k+1)%4]) for k in range(4)]
        elif geo['name'].upper() == 'NACA0012':
            air_upper,air_lower=profile(int(geo.get('surface_samples_per_side',240)))
            # Share the exact LE and TE point tags across the two splines.
            te=g.addPoint(*transform(air_upper[0]),0)
            le=g.addPoint(*transform(air_upper[-1]),0)
            up=[te]+[g.addPoint(*transform(pt),0) for pt in air_upper[1:-1]]+[le]
            lo=[le]+[g.addPoint(*transform(pt),0) for pt in air_lower[1:-1]]+[te]
            wall_curves=[g.addSpline(up),g.addSpline(lo)]
        else:
            raise ValueError('geometry.name must be NACA0012 or cylinder')
        air_loop=g.addCurveLoop(wall_curves)
        corner=[g.addPoint(x,y,0) for x,y in ((xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax))]
        lines=[g.addLine(corner[i],corner[(i+1)%4]) for i in range(4)]
        domain=g.addPlaneSurface([g.addCurveLoop(lines),air_loop])
        g.synchronize()
        edges={'wall':wall_curves, 'inlet':[lines[3]],'outlet':[lines[1]],
               'bottom':[lines[0]],'top':[lines[2]]}
        for tag,(name,entities) in enumerate(edges.items(),1):
            gmsh.model.addPhysicalGroup(1,entities,tag);gmsh.model.setPhysicalName(1,tag,name)
        gmsh.model.addPhysicalGroup(2,[domain],11)
        # Size via distance to actual airfoil wall + wake refinement. Keep exterior sparse.
        field=gmsh.model.mesh.field
        dist=field.add('Distance');field.setNumbers(dist,'CurvesList',wall_curves);field.setNumber(dist,'Sampling',600)
        near=field.add('Threshold');field.setNumber(near,'InField',dist)
        wake=field.add('Box')
        blend=field.add('Min');field.setNumbers(blend,'FieldsList',[near,wake]);field.setAsBackgroundMesh(blend)
        target=int(cfg['target_cells']);tol=float(cfg.get('tolerance',.10))
        history=[];best=None
        # Geometry fixed. Only the size field changes between attempts.
        for alg in cfg.get('algorithms',[6,5]):
            gmsh.option.setNumber('Mesh.Algorithm',alg)
            scale=1.0
            for attempt in range(int(cfg.get('max_attempts_per_algorithm',5))):
                near_size=float(cfg['near_size'])*scale
                far_size=float(cfg['far_size'])*scale
                field.setNumber(near,'SizeMin',near_size)
                field.setNumber(near,'SizeMax',far_size)
                field.setNumber(near,'DistMin',float(cfg.get('near_distance',.07)))
                field.setNumber(near,'DistMax',float(cfg.get('far_distance',2.5)))
                wx0,wx1,wy0,wy1=config['physics']['wake_box']
                for key,val in [('VIn',near_size*1.15),('VOut',far_size),('XMin',wx0),('XMax',wx1),
                                ('YMin',wy0),('YMax',wy1),('Thickness',.4)]: field.setNumber(wake,key,val)
                gmsh.model.mesh.generate(2)
                try:gmsh.model.mesh.optimize('Netgen')
                except Exception:pass
                xy,tri,boundaries=extract(gmsh,edges)
                q=quality_metrics(xy,tri,cfg['min_triangle_angle_deg'],cfg['min_scaled_jacobian'])
                count_pass=abs(q['cells']-target)<=target*tol
                shape_pass=(q['all_positive_area'] and q['min_angle_deg']>=cfg['min_triangle_angle_deg']
                            and q['min_scaled_jacobian']>=cfg['min_scaled_jacobian']
                            and q['max_edge_ratio']<=cfg['max_aspect'])
                passed=bool(count_pass and shape_pass)
                history.append(dict(algorithm=alg,attempt=attempt+1,size_scale=scale,
                                    cells=q['cells'],min_angle_deg=q['min_angle_deg'],
                                    min_scaled_jacobian=q['min_scaled_jacobian'],passed=passed))
                if verbose:print('Mesh trial:',history[-1],flush=True)
                # Preserve the best candidate IN MEMORY. Do not silently leave the last attempt.
                score=abs(q['cells']-target)/target + 8*max(0,cfg['min_triangle_angle_deg']-q['min_angle_deg'])/cfg['min_triangle_angle_deg'] + 8*max(0,cfg['min_scaled_jacobian']-q['min_scaled_jacobian'])/cfg['min_scaled_jacobian']
                if best is None or score<best[0]:best=(score,xy.copy(),tri.copy(),{k:v.copy() for k,v in boundaries.items()},q.copy())
                if passed:break
                scale*=float(np.clip(np.sqrt(max(q['cells'],1)/target),.75,1.35))
                gmsh.model.mesh.clear()
            if passed:break
        _,xy,tri,boundaries,q=best
        q['passed']=bool(q['all_positive_area'] and q['min_angle_deg']>=cfg['min_triangle_angle_deg']
                         and q['min_scaled_jacobian']>=cfg['min_scaled_jacobian']
                         and q['max_edge_ratio']<=cfg['max_aspect']
                         and abs(q['cells']-target)<=target*tol)
        q['target_cells']=target;q['calibration']=history
        from finite_volume import dual_geometry, mesh_signature, DISCRETIZATION
        dual=dual_geometry(xy,tri,boundaries)
        q['mesh_signature']=mesh_signature(xy,tri,boundaries)
        q['geometry']=geo;q['discretization']=DISCRETIZATION
        q['note']='Pure first-order triangles with closed median-dual control volumes. Mesh quality alone does not validate flow accuracy.'
        np.savez_compressed(out,xy=xy,triangles=tri,quads=np.empty((0,4),dtype=np.int64),
                            control_volume=dual['volume'],dual_face=dual['face'],dual_normal=dual['normal'],
                            dual_src=dual['src'],dual_dst=dual['dst'],
                            boundary_face=dual['boundary_face'],boundary_normal=dual['boundary_normal'],
                            boundary_owner=dual['boundary_owner'],boundary_tag=dual['boundary_tag'],
                            **{'boundary_'+k:v for k,v in boundaries.items()})
        out.with_suffix('.json').write_text(json.dumps(q,indent=2),encoding='utf8')
        return q
    finally:gmsh.finalize()


if __name__=='__main__':
    import argparse
    from params import CONFIG
    parser=argparse.ArgumentParser()
    parser.add_argument('--config');parser.add_argument('--output')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text(encoding='utf8')) if args.config else CONFIG
    result=build(config,output=args.output,verbose=True)
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit('Quality gate failed: do NOT train. Inspect worst_cell_center.')
