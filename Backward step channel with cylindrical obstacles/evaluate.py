
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from model import write_json


def statistics(history, start_time, length=1., speed=1., *, rows=None):
    if rows is None:
        with Path(history).open(encoding='utf8') as stream:
            rows = list(csv.DictReader(stream))
    result = []
    for case in sorted({int(r['case']) for r in rows}):
        all_rows = [r for r in rows if int(r['case']) == case]
        selected = [r for r in all_rows if float(r['time']) >= start_time]
        if len(selected) < 4:
            raise ValueError('At least four samples after start_time are required')
        t, cd, cl = (np.array([float(r[k]) for r in selected]) for k in ('time', 'Cd', 'Cl'))
        delta = np.diff(t)
        if not np.all(delta > 0) or not np.allclose(delta, delta[0], rtol=1e-5):
            raise ValueError('History must have strictly increasing, uniform sample times')
        fluctuation = cl-cl.mean()
        rms = np.sqrt(np.mean(fluctuation**2))
        st, note = None, 'No resolved oscillation; St is undefined'
        if rms > 1e-8:
            spectrum = np.abs(np.fft.rfft(fluctuation*np.hanning(len(t))))**2
            spectrum[0] = 0
            k = int(np.argmax(spectrum))
            if k >= 3 and k < len(spectrum)-1:
                st = float(np.fft.rfftfreq(len(t), delta[0])[k]*length/speed)
                note = 'FFT estimate; inspect stationarity and use enough shedding cycles'
            else:
                note = 'Fewer than three resolved cycles or a Nyquist peak; extend sampling'
        result.append({'case': case, 'Re': float(selected[0]['Re']), 'samples': len(t),
                       'window': [float(t[0]), float(t[-1])], 'Cd_mean': float(cd.mean()),
                       'Cl_rms': float(rms), 'St': st, 'St_note': note,
                       'mean_local_updates': float(np.mean([float(r['updates']) for r in selected])),
                       'total_local_updates': int(sum(float(r['updates']) for r in all_rows)),
                       'total_step_seconds': float(sum(float(r['step_seconds']) for r in all_rows)),
                       'max_abs_mass_flux': float(max(abs(float(r['mass_flux'])) for r in selected)),
                       'mean_divergence_rms': float(np.mean([float(r['divergence_rms']) for r in selected]))})
    return result


def body_statistics(history, body_history, start_time, length=1., speed=1.):
    """Separate body force spectra; attach the corresponding shared-step diagnostics."""
    with Path(history).open(encoding='utf8') as stream:
        totals = {(int(r['case']), int(r['step'])): r for r in csv.DictReader(stream)}
    with Path(body_history).open(encoding='utf8') as stream:
        bodies = list(csv.DictReader(stream))
    results = []
    for body in sorted({r['body'] for r in bodies}):
        merged = []
        for row in bodies:
            if row['body'] != body:
                continue
            key = (int(row['case']), int(row['step']))
            if key not in totals:
                raise ValueError('Body history has no matching total history step')
            total = totals[key]
            if float(row['time']) != float(total['time']) or float(row['Re']) != float(total['Re']):
                raise ValueError('Body and total histories disagree in time/Re')
            merged.append({**total, **row})
        for value in statistics(None, start_time, length, speed, rows=merged):
            results.append({'body': body, **value})
    return results


def compare_fields(predicted, reference):
    with np.load(predicted, allow_pickle=False) as d:
        pred = {k: d[k].copy() for k in d.files}
    with np.load(reference, allow_pickle=False) as d:
        ref = {k: d[k].copy() for k in d.files}
    if not np.isclose(pred['time'], ref['time'], rtol=0, atol=1e-9):
        raise ValueError('CFD reference time differs from prediction time')
    if not np.array_equal(pred['re'], ref['re']):
        raise ValueError('Re arrays/case ordering must match exactly')
    same_mesh = pred['xy'].shape == ref['xy'].shape and np.allclose(pred['xy'], ref['xy'], atol=1e-12, rtol=0)
    if same_mesh:
        target_v, target_p = ref['velocity'], ref['pressure']
    else:
        import matplotlib.tri as mtri
        if 'triangles' not in ref:
            raise ValueError('Cross-mesh reference needs triangles to preserve solid holes')
        mesh = mtri.Triangulation(ref['xy'][:, 0], ref['xy'][:, 1], ref['triangles'])
        def interpolate(field):
            output = np.empty((*field.shape[:2], len(pred['xy'])))
            for b in range(field.shape[0]):
                for c in range(field.shape[1]):
                    value = mtri.LinearTriInterpolator(mesh, field[b, c])(*pred['xy'].T)
                    if np.ma.getmaskarray(value).any():
                        raise ValueError('Target nodes outside reference fluid triangulation; no extrapolation allowed')
                    output[b, c] = value
            return output
        target_v, target_p = interpolate(ref['velocity']), interpolate(ref['pressure'])
    w = pred['volume']/pred['volume'].sum()
    result = []
    for i, reynolds in enumerate(pred['re']):
        # Pressure is defined up to a constant; align gauges with the same volume weights.
        dp = pred['pressure'][i]-target_p[i]
        dp -= np.sum(dp*w, axis=-1, keepdims=True)
        dv = pred['velocity'][i]-target_v[i]
        velocity_norm = np.sum(target_v[i]**2*w)
        pressure_centered = target_p[i]-np.sum(target_p[i]*w, axis=-1, keepdims=True)
        pressure_norm = np.sum(pressure_centered**2*w)
        result.append({'Re': float(reynolds), 'velocity_L2_relative': float(np.sqrt(np.sum(dv**2*w)/velocity_norm)) if velocity_norm > 1e-24 else None,
                       'pressure_L2_relative': float(np.sqrt(np.sum(dp**2*w)/pressure_norm)) if pressure_norm > 1e-24 else None,
                       'velocity_rms_absolute': float(np.sqrt(np.sum(dv**2*w)/2)),
                       'pressure_rms_absolute': float(np.sqrt(np.sum(dp**2*w)))})
    return {'reference_path': str(Path(reference).resolve()), 'same_mesh': same_mesh, 'errors': result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history')
    parser.add_argument('--body-history', help='Per-cylinder body_history.csv')
    parser.add_argument('--reference-history')
    parser.add_argument('--start-time', type=float, help='User-selected post-transient statistics window')
    parser.add_argument('--length', type=float, default=1.)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--field'); parser.add_argument('--reference-field')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = {'physical_validation': 'requires review of reference provenance, matching geometry/BCs and convergence'}
    if args.length <= 0 or args.speed <= 0:
        parser.error('length and speed must be positive')
    if args.history:
        if args.start_time is None:
            parser.error('--history requires explicit --start-time')
        result['statistics'] = statistics(args.history, args.start_time, args.length, args.speed)
        if args.body_history:
            result['body_statistics'] = body_statistics(args.history, args.body_history,
                                                        args.start_time, args.length, args.speed)
            result['force_normalization'] = 'Each body and total: F/(0.5*rho*U_ref^2*D), D=1'
            result['timing_note'] = 'Timings/updates/mass/divergence are shared solve diagnostics; do not sum them over bodies or Re.'
        if args.reference_history:
            result['reference_statistics'] = statistics(args.reference_history, args.start_time, args.length, args.speed)
            result['coefficient_comparison'] = []
            if len(result['statistics']) != len(result['reference_statistics']):
                raise ValueError('Reference and prediction must contain the same cases')
            for a, b in zip(result['statistics'], result['reference_statistics']):
                if a['Re'] != b['Re'] or not np.allclose(a['window'], b['window']):
                    raise ValueError('Reference statistics must match Re and time window')
                result['coefficient_comparison'].append({'Re': a['Re'], **{
                    key+'_absolute_error': abs(a[key]-b[key]) if a[key] is not None and b[key] is not None else None
                    for key in ('Cd_mean', 'Cl_rms', 'St')}})
    if args.field or args.reference_field:
        if not (args.field and args.reference_field):
            parser.error('Both --field and --reference-field are required')
        result['field_comparison'] = compare_fields(args.field, args.reference_field)
    if not args.history and not args.field:
        parser.error('Provide a history or field pair')
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
