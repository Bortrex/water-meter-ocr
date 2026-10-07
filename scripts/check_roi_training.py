"""Focused CPU-friendly checks: python -m scripts.check_roi_training."""
from pathlib import Path
import tempfile
from unittest.mock import patch
import json

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from water_meter_ocr import roi_training as training
from water_meter_ocr.plotting import plot_learning_curves


def tree_equal(a, b):
    return all(np.array_equal(x, y) for x, y in zip(
        jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


def check_numerics():
    y = jnp.array([1, 0, 1, 0], dtype=jnp.float32).reshape(1, 2, 2, 1)
    p = jnp.array([0.8, 0.7, 0.5, 0.1], dtype=jnp.float32).reshape(y.shape)
    expected_dice = 1 - 2 * 1.3 / 4.1
    expected_focal = np.mean([-0.25 * .2**2 * np.log(.8),
                              -.75 * .7**2 * np.log(.3),
                              -.25 * .5**2 * np.log(.5),
                              -.75 * .1**2 * np.log(.9)])
    np.testing.assert_allclose(training.dice_loss(y, p), [expected_dice], rtol=1e-6)
    np.testing.assert_allclose(training.focal_loss(y, p), expected_focal, rtol=1e-6)
    np.testing.assert_allclose(training.roi_loss(y, p), expected_dice + expected_focal, rtol=1e-6)
    np.testing.assert_allclose(training.iou_score(y, p), (1 + 1e-5) / (3 + 1e-5))
    np.testing.assert_allclose(training.fbeta_score(y, p), 0.5)
    z = jnp.zeros_like(y)
    assert float(training.dice_loss(z, z)[0]) == 0
    assert float(training.iou_score(z, z)) == 1
    assert float(training.fbeta_score(z, z)) == 0
    for target, prediction in ((z, z), (y, y), (y, 1-y)):
        assert np.isfinite(training.roi_loss(target, prediction))
        assert np.isfinite(jax.grad(lambda pred: training.roi_loss(target, pred))(prediction)).all()
    aggregate = training.EpochMetrics()
    aggregate.update({'loss': 1., 'counts': np.array([1, 1, 1])}, 1)
    aggregate.update({'loss': 3., 'counts': np.array([9, 0, 0])}, 3)
    result = aggregate.result()
    assert result['loss'] == 2.5
    np.testing.assert_allclose(result['fbeta'], 20 / 22)
    np.testing.assert_allclose(result['iou'], (10 + 1e-5) / (12 + 1e-5))
    samples = [(np.array([i]), np.array([i])) for i in range(1240)]
    order = lambda seed: np.concatenate([x[:, 0] for x, _ in training.batches(
        samples, 16, rng=np.random.default_rng(seed), drop_last=True)])
    assert len(order(1111)) == 1232 and len(set(order(1111))) == 1232
    np.testing.assert_array_equal(order(1111), order(1111))
    assert not np.array_equal(order(1111), order(1112))
    assert len(list(training.batches(samples, 16, drop_last=False))) == 78
    print('Loss reference values, empty masks, epoch aggregation and batching passed.', flush=True)


def main():
    check_numerics()
    print(f'Devices: {jax.devices()}', flush=True)
    state = training.create_train_state()
    x = jax.random.normal(jax.random.PRNGKey(5), (2, 32, 48, 3))
    y = (x[..., :1] > 0).astype(jnp.float32)
    updated, result = training.train_step(state, x, y)
    assert bool(result['gradients_finite']) and np.isfinite(result['loss'])
    assert not tree_equal(state.params, updated.params)
    assert not tree_equal(state.batch_stats, updated.batch_stats)
    assert not tree_equal(state.dropout_key, updated.dropout_key)
    repeated, repeated_result = training.train_step(state, x, y)
    assert tree_equal(updated, repeated)
    changed_rng, _ = training.train_step(state.replace(dropout_key=jax.random.PRNGKey(99)), x, y)
    assert not tree_equal(updated.params, changed_rng.params)
    second, result = training.train_step(updated, x, y)
    assert int(second.step) == 2 and bool(result['gradients_finite'])
    before_stats = jax.device_get(second.batch_stats)
    prediction, metrics = training.eval_step(second, x, y)
    again, _ = training.eval_step(second, x, y)
    np.testing.assert_array_equal(prediction, again)
    assert tree_equal(before_stats, second.batch_stats)
    for metric in (training.iou_score, training.fbeta_score):
        assert 0 <= float(metric(y, prediction)) <= 1
    print('Finite gradients, parameter/BN updates, dropout and deterministic validation passed.', flush=True)
    with tempfile.TemporaryDirectory(prefix='roi-training-check-') as folder:
        root = Path(folder)
        training.save_checkpoint(root / 'roundtrip', second, 1, float(metrics['loss']), 2e-4, False)
        training.save_checkpoint(root / 'roundtrip', second, 2, float(metrics['loss']), 2e-4, False)
        restored = training.restore_checkpoint(root / 'roundtrip', state)
        assert restored['epoch'] == 2
        assert tree_equal(second, restored['state'])
        restored_prediction, _ = training.eval_step(restored['state'], x, y)
        np.testing.assert_array_equal(prediction, restored_prediction)
        next_original, _ = training.train_step(second, x, y)
        next_restored, _ = training.train_step(restored['state'], x, y)
        assert tree_equal(next_original, next_restored)
        # Drive the real epoch loop with controlled validation losses to exercise
        # early stopping and best-state restoration, independent of convergence.
        dataset = [(np.asarray(x[i]), np.asarray(y[i])) for i in range(2)]
        eval_step = training.eval_step
        calls = 0
        def controlled_validation(state, images, masks):
            nonlocal calls
            pred, metric = eval_step(state, images, masks)
            metric = dict(metric, loss=jnp.asarray(1.0 if calls < 2 else 2.0))
            calls += 1
            return pred, metric
        with patch.object(training, 'eval_step', controlled_validation):
            best, history = training.fit(
                state, dataset, dataset, epochs=4, batch_size=2, patience=1,
                checkpoint_dir=root / 'fit', output_dir=root / 'history')
        assert len(history['epoch']) == 2 and int(best.step) == 1
        assert history['val_loss'] == [1., 2.]
        assert json.loads((root / 'history/history.json').read_text()) == history
        assert history['learning_rate'] == [2e-4, 2e-4]
        synthetic = {'epoch': [1, 2, 3], 'train_accuracy': [.3, .5, .7],
                     'val_accuracy': [.4, .6, .5], 'train_edit_distance': [4, 3, 2],
                     'val_edit_distance': [5, 3, 4]}
        fig = plot_learning_curves(synthetic, ('accuracy', 'edit_distance'),
            best_modes={'accuracy': 'max', 'edit_distance': 'min'},
            output_path=root / 'curves.png')
        assert len(fig.axes) == 2 and (root / 'curves.png').stat().st_size > 0
        plt.close(fig)
    print('PASS: checkpoint output/update roundtrip, early stopping, best restore, JSON history and generic plotting.')


if __name__ == '__main__':
    main()
