"""Generic learning curves from train_<metric>/val_<metric> history lists."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(history, metrics=('loss',), *, best_modes=None,
                         output_path=None):
    """Return an interactive Figure; optionally save and mark validation optima.

    best_modes maps metric names to 'min' or 'max'; omit to skip best markers.
    Callers own figure display/closing. Metrics are not tied to any model.
    """
    if not metrics:
        raise ValueError('Request at least one metric')
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
    for metric, ax in zip(metrics, axes[0]):
        for prefix, label in (('train', 'Training'), ('val', 'Validation')):
            values = np.asarray(history[f'{prefix}_{metric}'], dtype=float)
            epochs = np.asarray(history.get('epoch', np.arange(1, len(values) + 1)))
            if values.ndim != 1 or len(values) != len(epochs):
                raise ValueError(f'Invalid history length for {prefix}_{metric}')
            ax.plot(epochs, values, label=label)
            mode = (best_modes or {}).get(metric)
            if prefix == 'val' and mode is not None:
                if mode not in ('min', 'max'):
                    raise ValueError('Best modes must be min or max')
                if np.isfinite(values).any():
                    index = (np.nanargmin if mode == 'min' else np.nanargmax)(values)
                    ax.scatter(epochs[index], values[index], marker='*', s=100,
                               label=f'Best validation (epoch {epochs[index]})')
        ax.set(xlabel='Epoch', ylabel=metric.replace('_', ' ').title())
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches='tight')
    return fig
