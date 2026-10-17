import json
import numpy as np
import pytest
from sf_training.normalization import normalize, unnormalize, safe_bounds, audit_arrays
from sf_training.validation import validate_metrics


def test_sparse_rotations_survive_collapsed_quantiles_and_roundtrip():
    # Under 1% nonzero rotations: exact q01/q99 collapse despite real motion.
    x = np.zeros((1000, 2), np.float32)
    x[500] = [.675, -.675]
    x[750] = [.45, .675]
    legacy = {"q01": np.quantile(x, .01, axis=0).tolist(),
              "q99": np.quantile(x, .99, axis=0).tolist()}
    with pytest.raises(ValueError, match="Degenerate"):
        normalize(x, legacy)
    safe = safe_bounds(legacy, x.min(0), x.max(0))
    y = normalize(x, safe)
    assert safe["fallback_dimensions"] == [0, 1]
    assert np.max(np.abs(y)) <= 1.00001
    assert y[500, 0] != y[0, 0] and y[500, 1] != y[750, 1]
    np.testing.assert_allclose(unnormalize(y, safe), x, atol=2e-7)
    assert audit_arrays({"actions": x}, {"actions": safe})["actions"]["frames"] == 1000


def test_normal_quantiles_are_preserved_and_outliers_not_clipped():
    stats = {"q01": [-1., 0.], "q99": [1., 2.]}
    safe = safe_bounds(stats, [-2, 0], [3, 4])
    assert safe["q01"] == stats["q01"] and safe["q99"] == stats["q99"]
    assert safe["fallback_dimensions"] == []
    assert normalize(np.array([3., 4.]), safe).min() > 1


def test_constant_coordinate_roundtrip():
    stats = safe_bounds({"q01": [2.], "q99": [2.]}, [2.], [2.])
    assert stats["constant_dimensions"] == [0]
    np.testing.assert_allclose(normalize([2.], stats), [0.], atol=2e-6)
    np.testing.assert_allclose(unnormalize(normalize([2.], stats), stats), [2.])


@pytest.mark.parametrize("x", [[float('nan')], [float('inf')], [1000000.]])
def test_nonfinite_and_extreme_finite_targets_rejected(x):
    with pytest.raises(ValueError, match="extreme"):
        normalize(x, {"q01": [-1.], "q99": [1.]})


@pytest.mark.parametrize("low,high", [([0.], [0.]), ([1.], [0.]), ([float('nan')], [1.])])
def test_invalid_bounds_rejected(low, high):
    with pytest.raises(ValueError):
        normalize([0.], {"q01": low, "q99": high})


def rows():
    return [dict(step=i,loss=.2,flow_mse=.1,alignment=.2,grad_norm=1.,
                 target_abs_max=5.,rare_rotation_samples=int(i==5)) for i in range(1,33)]


def save(path, values):
    path.write_text(''.join(json.dumps(v)+'\n' for v in values))


def test_expanded_smoke_requires_rare_samples_and_complete_metrics(tmp_path):
    path=tmp_path/'metrics.jsonl'
    save(path, rows())
    assert validate_metrics(path)["rare_rotation_samples"] == 1
    save(path, rows()[:3])
    with pytest.raises(ValueError, match="every"):
        validate_metrics(path)
    values=rows()
    for v in values: v['rare_rotation_samples']=0
    save(path, values)
    with pytest.raises(ValueError, match="sparse"):
        validate_metrics(path)


@pytest.mark.parametrize("key,value", [("flow_mse",791030976.),("flow_mse",float('nan')),
                                     ("target_abs_max",1350001.),("grad_norm",float('inf'))])
def test_smoke_rejects_numerical_failure_despite_process_success(tmp_path,key,value):
    path=tmp_path/'metrics.jsonl'; values=rows(); values[4][key]=value
    save(path,values)
    with pytest.raises(ValueError): validate_metrics(path)


def test_old_checkpoint_cannot_be_exported_with_repaired_statistics(tmp_path):
    from sf_training.export import export
    home = tmp_path / '.runtime/training'
    home.mkdir(parents=True)
    (home / 'norm_stats.json').write_text('{"method":"changed"}')
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps(dict(root=str(tmp_path),recipe={},assets={'norm_stats':'old-sha'})))
    # Rejection occurs before any model/checkpoint is loaded.
    with pytest.raises(ValueError, match='Normalization differs'):
        export(plan, 'baseline', tmp_path/'unused.pt', tmp_path/'export')
