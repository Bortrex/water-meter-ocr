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

def plot_benford_distribution(data_dir, output_path=None):
    """Plot the Benford distribution of leading digits in the dataset."""
    from water_meter_ocr.roi_preprocessing import load_split, load_pair
    from collections import Counter

    def grab_value_from_name(sentence, sep="."):
        '''Function to grab the actual value from image name; due to problems in the CSV file.'''
        start, end = "value_", ".jpg"
        return sentence[sentence.find(start)+len(start) : sentence.rfind(end)].replace("_", sep)
    
    def first_significant_digit(value):
        """Return the first non-zero digit of a meter reading."""
        value = str(value).strip().replace(",", ".")

        for char in value:
            if char in "123456789":
                return int(char)
        return None
    
    split = load_split(data_dir)
    
    readings = [grab_value_from_name(filename) for filename in split.train + split.validation]
    first_digits = [
        first_significant_digit(value)
        for value in readings
    ]
    first_digits = [
        digit for digit in first_digits
        if digit is not None
    ]

    counts = Counter(first_digits)
    digits = np.arange(1, 10)

    observed_counts = np.array(
        [counts[digit] for digit in digits]
    )
    observed = observed_counts / observed_counts.sum()

    # Benford's law:
    # P(d) = log10(1 + 1/d)
    expected = np.log10(1 + 1 / digits)

    # mean absolute deviation- measure of the difference.
    mad = np.mean(np.abs(observed - expected))

    colors = plt.cm.nipy_spectral(
        np.linspace(0.05, 0.90, len(digits))
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars = ax.bar(digits,observed * 100, color=colors,
        alpha=0.85, label="Observed",
    )

    ax.plot(digits, expected * 100, marker="o", linestyle="--",
        linewidth=2, label="Benford's law",
    )

    for bar, percentage in zip(bars, observed * 100):
        ax.annotate(
            f"{percentage:.1f}%",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
            ),
            xytext=(4, 4), textcoords="offset points",
            ha="left", va="bottom", fontsize=9,
        )

    ax.text(0.9, 0.9, f"MAD: {mad:.4f}",transform=ax.transAxes,
        ha="right", va="top", fontsize=10)

    ax.set_xlabel("First significant digit")
    ax.set_ylabel("Frequency (%)")
    ax.set_xticks(digits)
    ax.set_title("First significant digit distribution")

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    return fig


if __name__ == '__main__':
    import matplotlib
    import json
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from water_meter_ocr.plotting import plot_learning_curves
    history = json.load(open('outputs/roi/history.json'))

    # fig = plot_learning_curves(history, ('loss', 'iou', 'fbeta'),
    #                                best_modes={'loss': 'min', 'iou': 'max', 'fbeta': 'max'},
    #                                output_path='docs/images/learning_curves.png')

    fig = plot_benford_distribution('data', output_path='docs/images/benford_distribution.png')
    plt.close(fig)
