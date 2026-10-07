"""Generic learning curves from train_<metric>/val_<metric> history lists."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from numpy.strings import index

def plot_learning_curves(history, metrics=('loss',), *,
    best_modes=None, output_path=None,):
    """Return an interactive Figure; optionally save and mark validation optima.

    best_modes maps metric names to 'min' or 'max'; omit to skip best markers.
    Callers own figure display/closing. Metrics are not tied to any model.
    """
    if not metrics:
        raise ValueError('Request at least one metric')

    # Three metrics are displayed in a README-friendly layout:
    # one wide plot above two smaller plots.
    if len(metrics) == 3:
        fig = plt.figure(figsize=(10, 8))
        grid = fig.add_gridspec(2, 6, height_ratios=[0.85, 1],
            hspace=0.18, wspace=0.35)

        axes = [
            fig.add_subplot(grid[0, 1:5]),
            fig.add_subplot(grid[1, 0:3]),
            fig.add_subplot(grid[1, 3:6]),
        ]
    else:
        fig, axes = plt.subplots(1, len(metrics), 
                        figsize=(5 * len(metrics), 4), squeeze=False)
        axes = axes[0]

    metric_labels = {
        'loss': 'Loss',
        'iou': 'IoU',
        'fbeta': r'$\mathcal{F}_\beta$',
    }

    for metric, ax in zip(metrics, axes):
        for prefix, label in (('train', 'Training'), ('val', 'Validation')):
            values = np.asarray(history[f'{prefix}_{metric}'],
                dtype=float)
            epochs = np.asarray(history.get(
                    'epoch', np.arange(1, len(values) + 1),
                )
            )

            if values.ndim != 1 or len(values) != len(epochs):
                raise ValueError(
                    f'Invalid history length for {prefix}_{metric}'
                )

            ax.plot(epochs, values, label=label)

            mode = (best_modes or {}).get(metric)

            if prefix == 'val' and mode is not None:
                if mode not in ('min', 'max'):
                    raise ValueError('Best modes must be min or max')

                if np.isfinite(values).any():
                    index = (np.nanargmin if mode == 'min' else np.nanargmax
                    )(values)

                    best_epoch = epochs[index]
                    best_value = values[index]

                    ax.scatter( best_epoch, best_value, marker='*',
                        s=100, color='red', alpha=0.5,
                        label=f'Best validation (epoch {best_epoch})',
                    )

                    metric_label = metric_labels.get( metric,
                        metric.replace('_', ' ').title())

                    ax.annotate(
                        f'Epoch: {int(best_epoch)}\n'
                        f'{metric_label}: {best_value:.4f}',
                        xy=(best_epoch, best_value),
                        xytext=(0, 35 if mode == 'min' else -35),
                        textcoords='offset points',
                        ha='center', fontsize=9,
                    )

        metric_label = metric_labels.get(
            metric,
            metric.replace('_', ' ').title(),
        )

        ax.spines[['top', 'right']].set_visible(False)
        ax.set( xlabel='Epoch', ylabel=metric_label)
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir( parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches='tight')

    return fig

if __name__ == '__main__':
    import matplotlib
    import json
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from water_meter_ocr.plotting import plot_learning_curves
    history = json.load(open('outputs/roi/history.json'))

    fig = plot_learning_curves(history, ('loss', 'iou', 'fbeta'),
                                   best_modes={'loss': 'min', 'iou': 'max', 'fbeta': 'max'},
                                   output_path='docs/images/learning_curves.png')
    plt.close(fig)
