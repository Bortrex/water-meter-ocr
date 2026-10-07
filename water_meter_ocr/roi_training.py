"""Explicit ROI baseline losses, state, steps, batching and best-model training.

See ROI_TRAINING.md for historical defaults and deliberate numerical differences.
"""
from pathlib import Path
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import train_state

from water_meter_ocr.roi_model import ROIUNet


# Losses and metrics


def dice_loss(y_true, y_pred):
    """Per-image soft Dice loss over HWC; empty/empty has zero loss."""
    intersection = jnp.sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = jnp.sum(y_true + y_pred, axis=(1, 2, 3))
    return jnp.where(denominator > 0,
                     1 - 2 * intersection / jnp.where(denominator > 0, denominator, 1), 0)


def focal_loss(y_true, y_pred):
    """Mean binary focal loss, alpha=.25, gamma=2, Keras epsilon=1e-7."""
    p = jnp.clip(y_pred, 1e-7, 1 - 1e-7)
    return jnp.mean(-0.25 * y_true * (1 - p)**2 * jnp.log(p)
                    - 0.75 * (1 - y_true) * p**2 * jnp.log1p(-p))


def roi_loss(y_true, y_pred):
    return jnp.mean(dice_loss(y_true, y_pred)) + focal_loss(y_true, y_pred)


def pixel_counts(y_true, y_pred):
    prediction = (y_pred > 0.5).astype(jnp.float32)
    return jnp.stack((jnp.sum(prediction * y_true),
                      jnp.sum(prediction * (1 - y_true)),
                      jnp.sum((1 - prediction) * y_true)))


def iou_score(y_true, y_pred):
    tp, fp, fn = pixel_counts(y_true, y_pred)
    return (tp + 1e-5) / (tp + fp + fn + 1e-5)


def fbeta_score(y_true, y_pred):
    """TFA micro FBetaScore default beta=1; zero denominator returns zero."""
    tp, fp, fn = pixel_counts(y_true, y_pred)
    denominator = 2 * tp + fp + fn
    return 2 * tp / jnp.where(denominator > 0, denominator, 1)


# Training state


class ROITrainState(train_state.TrainState):
    # BatchNorm statistics are state, not parameters optimized by gradients.
    # Keep the RNG here so checkpoints also restore the next Dropout update.
    batch_stats: object
    dropout_key: jax.Array


def create_train_state(seed=1111, learning_rate=2e-4, residual=False,
                       input_shape=(1, 32, 32, 3)):
    model = ROIUNet(use_residual_blocks=residual)
    params_key, dropout_key = jax.random.split(jax.random.PRNGKey(seed))
    variables = model.init(params_key, jnp.zeros(input_shape, jnp.float32), train=False)
    # Keras Adam's implicit epsilon was 1e-7 (Optax defaults to 1e-8).
    return ROITrainState.create(
        apply_fn=model.apply, params=variables['params'],
        tx=optax.adam(learning_rate, b1=0.9, b2=0.999, eps=1e-7),
        batch_stats=variables['batch_stats'], dropout_key=dropout_key)


# Train/evaluation steps


# Compile repeated steps; the model/optimizer update remains explicit below.
@jax.jit
def train_step(state, images, masks):
    # JAX randomness is explicit: use one key now and retain the next key.
    next_dropout_key, dropout_rng = jax.random.split(state.dropout_key)

    def objective(params):
        predictions, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats}, images,
            train=True, rngs={'dropout': dropout_rng}, mutable=['batch_stats'])
        loss = roi_loss(masks, predictions)
        return loss, (predictions, updates['batch_stats'])

    # Compute the loss and its parameter gradients together. Auxiliary outputs
    # carry predictions and updated BatchNorm statistics without differentiating them.
    (loss, (predictions, new_batch_stats)), gradients = jax.value_and_grad(
        objective, has_aux=True)(state.params)
    state = state.apply_gradients(grads=gradients)
    state = state.replace(batch_stats=new_batch_stats, dropout_key=next_dropout_key)
    return state, {'loss': loss, 'counts': pixel_counts(masks, predictions)}


@jax.jit
def eval_step(state, images, masks):
    # Evaluation uses stored BatchNorm statistics: no RNG, mutation or gradients.
    predictions = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats},
                                 images, train=False)
    return predictions, {'loss': roi_loss(masks, predictions),
                          'counts': pixel_counts(masks, predictions)}


# Batching


def batches(dataset, batch_size, *, rng=None, drop_last=False):
    """Shuffle indices once, load on demand, and yield only the current NumPy batch."""
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    indices = np.arange(len(dataset))
    if rng is not None:
        rng.shuffle(indices)
    stop = len(indices) - len(indices) % batch_size if drop_last else len(indices)
    for start in range(0, stop, batch_size):
        samples = [dataset[int(i)] for i in indices[start:min(start + batch_size, stop)]]
        yield tuple(np.stack(items) for items in zip(*samples))


# Epoch metrics


class EpochMetrics:
    """Sample-weighted loss; global micro metrics, accumulated in host float64."""
    def __init__(self):
        self.loss_sum = 0.0
        self.samples = 0
        self.counts = np.zeros(3, dtype=np.float64)

    def update(self, result, batch_size):
        result = jax.device_get(result)
        if not np.isfinite(result['loss']) or not np.isfinite(result['counts']).all():
            raise FloatingPointError('Non-finite loss or metric counts')
        self.loss_sum += float(result['loss']) * batch_size
        self.samples += batch_size
        self.counts += result['counts']

    def result(self):
        if not self.samples:
            raise ValueError('No batches: training needs at least one full batch')
        tp, fp, fn = self.counts
        denominator = 2 * tp + fp + fn
        return {'loss': self.loss_sum / self.samples,
                'iou': float((tp + 1e-5) / (tp + fp + fn + 1e-5)),
                'fbeta': float(2 * tp / denominator) if denominator else 0.0}


# Checkpoint helpers


def checkpoint_payload(state, epoch, best_val_loss, learning_rate, residual):
    return {'state': state, 'epoch': int(epoch),
            'best_val_loss': float(best_val_loss),
            'learning_rate': float(learning_rate), 'residual': bool(residual)}


def save_checkpoint(path, state, epoch, best_val_loss, learning_rate, residual):
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(Path(path).resolve(), checkpoint_payload(
            state, epoch, best_val_loss, learning_rate, residual), force=True)
        checkpointer.wait_until_finished()


def restore_checkpoint(path, template_state, learning_rate=2e-4, residual=False):
    """Restore with a matching model/optimizer template; no automatic data-RNG resume."""
    with ocp.StandardCheckpointer() as checkpointer:
        restored = checkpointer.restore(Path(path).resolve(), target=checkpoint_payload(
            template_state, 0, 0.0, learning_rate, residual))
    if bool(restored['residual']) != residual or not np.isclose(
            restored['learning_rate'], learning_rate, rtol=1e-6, atol=0):
        raise ValueError('Checkpoint configuration differs from restore template')
    return restored


# Training loop


def fit(state, train_dataset, valid_dataset, *, epochs=100, batch_size=16,
        learning_rate=2e-4, seed=1111, checkpoint_dir='checkpoints/roi',
        output_dir='outputs/roi', residual=False, patience=15):
    """Fixed-LR baseline. Return best model state and JSON-compatible epoch history."""
    
    if epochs <= 0 or patience <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError('epochs, patience, batch_size and learning_rate must be positive')
    if len(train_dataset) < batch_size or not len(valid_dataset):
        raise ValueError('Need one full training batch and nonempty validation')
    
    best_path = Path(checkpoint_dir).resolve() / 'best'
    output_dir = Path(output_dir)
    history_path = output_dir / 'history.json'
    if best_path.exists() or history_path.exists():
        raise FileExistsError('Choose fresh output/checkpoint directories for this run')
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    history = {key: [] for key in ('epoch', 'train_loss', 'val_loss', 'train_iou',
               'val_iou', 'train_fbeta', 'val_fbeta', 'learning_rate', 'epoch_time')}
    (output_dir / 'config.json').write_text(json.dumps({
        'epochs': epochs, 'batch_size': batch_size, 'validation_batch_size': 1,
        'learning_rate': learning_rate, 'seed': seed, 'residual': residual,
        'patience': patience, 'drop_last': True, 'plateau_reduction': False}, indent=2))
    rng = np.random.default_rng(seed)
    best_loss, waiting = float('inf'), 0
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_metrics, validation_metrics = EpochMetrics(), EpochMetrics()
        for images, masks in batches(train_dataset, batch_size, rng=rng, drop_last=True):
            state, result = train_step(state, jax.device_put(images), jax.device_put(masks))
            train_metrics.update(result, len(images))
        for images, masks in batches(valid_dataset, 1):
            _, result = eval_step(state, jax.device_put(images), jax.device_put(masks))
            validation_metrics.update(result, len(images))
        train_results = train_metrics.result()
        validation_results = validation_metrics.result()
        if validation_results['loss'] < best_loss:
            best_loss, waiting = validation_results['loss'], 0
            save_checkpoint(best_path, state, epoch, best_loss, learning_rate, residual)
        else:
            waiting += 1
        for prefix, values in (('train', train_results), ('val', validation_results)):
            for key, value in values.items():
                history[f'{prefix}_{key}'].append(value)
        history['epoch'].append(epoch)
        history['learning_rate'].append(float(learning_rate))
        history['epoch_time'].append(time.perf_counter() - start)
        temporary = history_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(history, indent=2, allow_nan=False))
        temporary.replace(history_path)
        print(f"Epoch {epoch}/{epochs} loss={train_results['loss']:.5f} val_loss={validation_results['loss']:.5f} "
              f"IoU={train_results['iou']:.4f} val_IoU={validation_results['iou']:.4f} "
              f"lr={learning_rate:g} time={history['epoch_time'][-1]:.1f}s", flush=True)
        if waiting >= patience:
            print(f'Early stopping after {waiting} epochs without improvement.', flush=True)
            break
    restored = restore_checkpoint(best_path, state, learning_rate, residual)
    print(f"Restored best model from epoch {int(restored['epoch'])}.", flush=True)
    return restored['state'], history
