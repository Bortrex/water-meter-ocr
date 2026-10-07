# ROI training baseline

Run from the repository root with the dependencies in `requirements.txt`:

```bash
python -m scripts.check_roi_training
python -m scripts.train_roi --data-dir data --epochs 100 --batch-size 16 \
  --learning-rate 0.0002 --seed 1111 \
  --checkpoint-dir checkpoints/roi --output-dir outputs/roi
```

The default is the standard U-Net. `--residual` selects the experimental variant.
The smoke check uses CPU-friendly synthetic 32x48 inputs with the unchanged
architecture; no full experiment is run by the check. Use `JAX_PLATFORMS=cpu`
when CUDA is unavailable. Full-resolution training and GPU memory capacity have
not been validated by this smoke check.

## Historical definitions

Source: ROI notebook loss/metric/callback cells and internship report sections
3.5–3.7 and 4.1. TensorFlow Addons' historical implementation confirms the
implicit beta=1 default and micro reduction:
https://github.com/tensorflow/addons/blob/v0.12.1/tensorflow_addons/metrics/f_scores.py

For binary target y and sigmoid probability p:

- Dice loss per image: `1 - 2 sum(y*p) / sum(y+p)`, sums over HWC (axes 1,2,3
  on NHWC batches). The combined loss averages these image losses. No additive
  smoothing is applied. A zero denominator gives zero loss instead of the
  notebook's NaN; safe division also keeps gradients finite.
- Focal loss: mean over all batch pixels/channels of
  `-0.25*y*(1-p)^2*log(p) - 0.75*(1-y)*p^2*log(1-p)`.
  Probabilities are clipped to `[1e-7, 1-1e-7]`, the historical implicit Keras
  epsilon. This is algebraically equivalent to the notebook's logit expression.
- ROI loss: mean Dice loss + focal loss, with no extra weighting.
- Predictions for metrics use strict `p > 0.5`. IoU is
  `(TP + 1e-5) / (TP + FP + FN + 1e-5)`. Empty/empty returns 1.
- F-beta is micro F1: `2*TP / (2*TP + FP + FN)`, zero when its denominator
  is zero. The notebook sets `num_classes=2, average='micro', threshold=.5`
  and leaves beta at its default 1. Micro mode uses scalar counts over all
  dimensions, so num_classes=2 does not require inventing a background channel
  for the one-channel model.

Epoch loss is weighted by the number of samples in each batch (all images in
an epoch have the same pixel count). TP/FP/FN accumulate in host float64 for
both epoch metrics. This preserves TFA's micro F1 definition. Global epoch IoU
is **not** the notebook's Keras mean of batch IoU values; the global count-based
aggregation deliberately avoids averaging ratios and is batch-partition invariant.

## Optimization and state

Optax Adam uses learning rate 2e-4, beta1=.9, beta2=.999 and epsilon=1e-7,
matching the implicit Keras hyperparameters. Optax and historical Keras differ
in epsilon placement relative to bias correction, so optimizer trajectories
are not expected to be bit-identical. No weight decay, clipping or schedule.

The report (section 4.1) specifies a fixed learning rate. The executable notebook
also defines ReduceLROnPlateau with factor=.1 and patience=10. This initial
implementation follows the fixed-rate report: plateau reduction is **not
implemented or enabled**. Every history row records the effective fixed rate.

`ROITrainState` retains params, Adam state, step, BatchNorm statistics and a
Dropout key. Each update splits the key and saves updated statistics. Validation
uses running statistics and deterministic Dropout, with no mutable collections.
`fit` expects a state created with matching learning-rate/residual arguments.

The original split and 307 extra examples remain unchanged. Training indices
are shuffled once per epoch with a seeded NumPy generator. Batches are loaded
on demand through ROIDataset, then transferred to JAX. With 1,240 examples and
batch size 16, 77 full batches consume 1,232 examples; 8 shuffled examples are
dropped each epoch. Validation uses all 311 examples in fixed order, batch size
1, at 960x960. The old loader computed shuffled indices but ignored them in
`__getitem__`; applying actual shuffling here is an intentional correction
required by the new pipeline. `--seed` controls initialization, dropout,
augmentation and epoch shuffling; the historical data split stays fixed at 1111.

## Artifacts and restoration

`fit` saves `config.json` and atomically updates `history.json` after each
completed epoch. History contains epoch, train/val loss, IoU and F1, learning
rate and elapsed epoch seconds (including compilation and checkpoint work).
The CLI saves `learning_curves.png` after completion.

Orbax stores the best strict validation-loss improvement in `<checkpoint-dir>/best`.
The checkpoint includes the complete training state, epoch, best validation
loss, learning rate and residual flag. After 15 consecutive epochs without
improvement, training stops; completion always restores and returns the best
checkpoint's state. Existing run artifacts are rejected rather than silently
overwritten by a new run. Checkpoint writes within the current run replace its
previous best checkpoint.

Use `restore_checkpoint(path, create_train_state(...), learning_rate=..., residual=...)`
with the matching architecture/optimizer configuration. The returned dictionary
contains `state` and metadata. Optimizer and dropout state restore, but exact
mid-run data-order/augmentation continuation is not implemented: no resume CLI
or serialization of Albumentations/shuffle RNGs is provided.

Plot saved history without retraining:

```python
import json
import matplotlib.pyplot as plt
from water_meter_ocr.plotting import plot_learning_curves

with open('outputs/roi/history.json') as stream:
    history = json.load(stream)
fig = plot_learning_curves(history, metrics=('loss', 'iou', 'fbeta'),
    best_modes={'loss': 'min', 'iou': 'max', 'fbeta': 'max'},
    output_path='docs/images/roi/learning_curves.png')
plt.show()
```

The plotter accepts arbitrary `train_<metric>`/`val_<metric>` keys and explicit
min/max preferences, including accuracy and edit-distance metrics. The caller
owns the returned Figure and can display or close it.
