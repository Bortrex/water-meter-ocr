"""Train the ROI baseline: python -m scripts.train_roi --help."""
import argparse
from pathlib import Path

from water_meter_ocr.roi_preprocessing import load_split, ROIDataset
from water_meter_ocr.roi_training import create_train_state, fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints/roi'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/roi'))
    parser.add_argument('--residual', action='store_true')
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error('epochs, batch-size and learning-rate must be positive')
    split = load_split(args.data_dir)
    train = ROIDataset(split.train, args.data_dir, extra_filenames=split.extra, seed=args.seed)
    valid = ROIDataset(split.validation, args.data_dir, validation=True, seed=args.seed)
    state = create_train_state(args.seed, args.learning_rate, args.residual)
    _, history = fit(state, train, valid, epochs=args.epochs, batch_size=args.batch_size,
                     learning_rate=args.learning_rate, seed=args.seed,
                     checkpoint_dir=args.checkpoint_dir, output_dir=args.output_dir,
                     residual=args.residual)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from water_meter_ocr.plotting import plot_learning_curves
    fig = plot_learning_curves(history, ('loss', 'iou', 'fbeta'),
                               best_modes={'loss': 'min', 'iou': 'max', 'fbeta': 'max'},
                               output_path=args.output_dir / 'learning_curves.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
