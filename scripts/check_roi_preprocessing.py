"""Run local ROI checks without writing datasets, from the repository root:

    python -m scripts.check_roi_preprocessing --data-dir data
"""

import argparse
from copy import deepcopy
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from water_meter_ocr.roi_preprocessing import (
    ROIDataset, extra_augmentation, imagenet_preprocess, load_pair, load_split,
    training_augmentation, validation_augmentation,
)


def check_geometry():
    # An asymmetric synthetic foreground makes incorrect paired geometry visible.
    mask = np.zeros((960, 720, 1), dtype=np.float32)
    mask[100:650, 90:300] = 1
    mask[550:850, 280:620] = 1
    image = np.repeat(mask, 3, axis=2)
    geometry = deepcopy(training_augmentation().transforms[:3])
    geometry[0].p = 1  # Exercise affine geometry rather than a skipped transform.
    extra = extra_augmentation()
    extra.transforms[1].p = 1
    for label, transform in (
        ("main", A.Compose(geometry, seed=1111)),
        ("extra", extra),
        ("validation", validation_augmentation()),
    ):
        for _ in range(5):
            sample = transform(image=image, mask=mask)
            foreground = sample["image"][..., 0] >= 0.5
            target = sample["mask"][..., 0] == 1
            iou = np.count_nonzero(foreground & target) / np.count_nonzero(foreground | target)
            # Main images use bilinear sampling; masks use nearest-neighbor.
            assert iou > 0.995, (label, iou)
            if label != "main":
                np.testing.assert_array_equal(foreground, target)
    print("Paired geometric alignment: passed (5 forced cases per policy)")


def main(data_dir):
    assert A.__version__ == "2.0.8", A.__version__
    split = load_split(data_dir)
    names = pd.read_csv(data_dir / "data.csv")["photo_name"].to_numpy()
    assert len(names) == len(set(names)) == 1244
    for folder in ("images", "masks"):
        assert {p.name for p in (data_dir / folder).glob("*.jpg")} == set(names)
    assert (len(split.train), len(split.validation), len(split.extra)) == (933, 311, 307)
    assert not set(split.train) & set(split.validation)
    assert len(set(split.extra)) == 307
    assert set(split.extra) <= set(split.train)
    expected_train, expected_valid = train_test_split(names, test_size=0.25, random_state=1111)
    np.testing.assert_array_equal(split.train, expected_train)
    np.testing.assert_array_equal(split.validation, expected_valid)
    np.testing.assert_array_equal(split.extra, np.random.RandomState(1111).choice(
        expected_train, size=int(len(expected_train) * 0.33), replace=False))
    for name in names:
        image, mask = load_pair(data_dir, name)
        assert image.shape == (960, 720, 3) and mask.shape == (960, 720, 1)
        assert image.dtype == np.uint8 and mask.dtype == np.float32
        assert np.isin(mask, [0, 1]).all()
    print("All 1244 CSV/image/mask records matched and decoded")

    train = ROIDataset(split.train, data_dir, extra_filenames=split.extra)
    valid = ROIDataset(split.validation, data_dir, validation=True)
    assert len(train) == 1240 and len(valid) == 311
    print("Split: 933 train / 311 validation; 307 unique extras; 1240 logical training examples")
    for label, dataset, index, size in (
        ("main", train, 0, 720), ("extra", train, 933, 720),
        ("validation", valid, 0, 960),
    ):
        image, mask = dataset[index]
        assert image.shape == (size, size, 3)
        assert mask.shape == (size, size, 1)
        assert image.dtype == mask.dtype == np.float32
        assert np.isfinite(image).all() and np.isin(mask, [0, 1]).all()
        print(f"{label}: image {image.shape}, mask {mask.shape}, float32, binary mask")

    # Validation padding must match OpenCV reflection exactly and be deterministic.
    image, mask = load_pair(data_dir, split.validation[0])
    expected_image = cv2.copyMakeBorder(image, 0, 0, 120, 120, cv2.BORDER_REFLECT_101)
    expected_mask = cv2.copyMakeBorder(mask[..., 0], 0, 0, 120, 120, cv2.BORDER_REFLECT_101)
    for _ in range(2):
        actual_image, actual_mask = valid[0]
        np.testing.assert_array_equal(actual_image, imagenet_preprocess(expected_image))
        np.testing.assert_array_equal(actual_mask[..., 0], expected_mask)

    pixel = np.array([[[10, 20, 30]]], dtype=np.uint8)
    np.testing.assert_allclose(imagenet_preprocess(pixel),
                               [[[-73.939, -96.779, -113.68]]], atol=1e-5)
    np.testing.assert_array_equal(pixel, [[[10, 20, 30]]])
    # Independent instances with the same seed reproduce both RNG streams.
    first = ROIDataset(split.train, data_dir, extra_filenames=split.extra)
    second = ROIDataset(split.train, data_dir, extra_filenames=split.extra)
    for index in (933, 0, 933, 0):
        for actual, expected in zip(first[index], second[index]):
            np.testing.assert_array_equal(actual, expected)
    check_geometry()
    print("Reflection, caffe preprocessing and seeded repeatability: passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    main(parser.parse_args().data_dir)
