import csv
import numpy as np
import pytest
from evaluate import statistics, compare_fields
from experiments import generate


def test_statistics_known_shedding_frequency(tmp_path):
    path = tmp_path/'history.csv'
    with path.open('w', newline='') as f:
        fields = ['case', 'Re', 'time', 'Cd', 'Cl', 'updates', 'step_seconds', 'mass_flux', 'divergence_rms']
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for t in np.arange(1000)*.1:
            writer.writerow(dict(case=0, Re=100, time=t, Cd=1.2, Cl=.4*np.sin(2*np.pi*.2*t),
                                 updates=2, step_seconds=.01, mass_flux=1e-5, divergence_rms=1e-4))
    result = statistics(path, 0)[0]
    assert result['Cd_mean'] == pytest.approx(1.2)
    assert result['Cl_rms'] == pytest.approx(.4/np.sqrt(2))
    assert result['St'] == pytest.approx(.2)


def test_reference_field_gauge_and_time(tmp_path):
    data = dict(time=1., re=np.array([40.]), xy=np.array([[0., 0.], [1., 0.], [0., 1.]]),
                volume=np.ones(3), velocity=np.ones((1, 2, 3)), pressure=np.array([[[1., 2., 3.]]]))
    pred, ref = tmp_path/'pred.npz', tmp_path/'ref.npz'
    np.savez(pred, **data)
    data['pressure'] = data['pressure']+5
    np.savez(ref, **data)
    result = compare_fields(pred, ref)['errors'][0]
    assert result['velocity_L2_relative'] == 0 and result['pressure_L2_relative'] == 0
    data['time'] = 2.; np.savez(ref, **data)
    with pytest.raises(ValueError, match='time'):
        compare_fields(pred, ref)


def test_ablation_controls_and_shared_mesh(tmp_path):
    manifest = generate(tmp_path)
    cases = manifest['groups']['ablation']
    assert len({c['controls_hash'] for c in cases}) == 1
    assert len({c['shared_mesh'] for c in cases}) == 1
    assert {c['operator'] for c in cases} == {'nufft_only', 'graph_only', 'graph_nufft'}
