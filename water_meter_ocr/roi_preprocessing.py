"""On-demand ROI preprocessing from the 2021 notebook (Albumentations 2.0.8).

Usage::

    split = load_split("data")
    train = ROIDataset(split.train, "data", extra_filenames=split.extra)
    valid = ROIDataset(split.validation, "data", validation=True)
    image, mask = train[0]

Outputs are HWC float32 arrays. A seeded dataset reproduces the same sequence
of calls; its augmentation RNG advances on each access. No samples are cached
or written to disk. Create separate instances for independent consumers.
"""

from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 1111


@dataclass(frozen=True)
class ROISplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    extra: tuple[str, ...]


def load_split(data_dir = "data"):
    """Split photo_name in CSV order and select the notebook's extra copies."""
    root = Path(data_dir)
    names = pd.read_csv(root / "data.csv")["photo_name"]
    if names.isna().any() or names.duplicated().any():
        raise ValueError("photo_name must contain unique, non-null filenames")
    for name in names:
        for folder in ("images", "masks"):
            path = root / folder / name
            if not path.is_file():
                raise FileNotFoundError(path)
    train, validation = train_test_split(
        names.to_numpy(), test_size=0.25, random_state=SEED
    )
    extra = np.random.RandomState(SEED).choice(
        train, size=int(len(train) * 0.33), replace=False
    )
    return ROISplit(tuple(train), tuple(validation), tuple(extra))


def load_pair(data_dir, filename):
    """Load RGB image and thresholded single-channel mask at 960h x 720w."""
    root = Path(data_dir)
    image = cv2.imread(str(root / "images" / filename))
    mask = cv2.imread(str(root / "masks" / filename), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise ValueError(f"Cannot decode image/mask pair: {filename}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (720, 960), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (720, 960), interpolation=cv2.INTER_NEAREST)
    return image, (mask >= 200).astype(np.float32)[..., None]


def training_augmentation(seed = SEED):
    return A.Compose([
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.1, rotate_limit=15,
            interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.5,
        ),
        A.PadIfNeeded(
            min_height=720, min_width=720, border_mode=cv2.BORDER_CONSTANT,
            fill=0, fill_mask=0, p=1,
        ),
        A.RandomCrop(height=720, width=720, p=1),
        # Old IAA noise std was 0.01–0.05 times 255, shared across channels.
        A.GaussNoise(std_range=(0.01, 0.05), mean_range=(0, 0),
                     per_channel=False, p=0.2),
        A.OneOf([
            # RandomContrast: disable brightness in the combined replacement.
            A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2,
                                       brightness_by_max=True, p=1),
            A.HueSaturationValue(p=1),
        ], p=0.77),
        A.OneOf([
            A.CLAHE(p=1),
            # RandomBrightness: disable contrast in the combined replacement.
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0,
                                       brightness_by_max=True, p=1),
            A.RandomGamma(p=1),
        ], p=0.77),
        A.OneOf([
            # Same IAASharpen ranges; supported implementation replaces imgaug.
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ], p=0.77),
    ], seed=seed)


def extra_augmentation(seed = SEED):
    return A.Compose([
        A.Resize(height=720, width=720, interpolation=cv2.INTER_NEAREST,
                 mask_interpolation=cv2.INTER_NEAREST, p=1),
        A.Rotate(limit=70, p=0.45, border_mode=cv2.BORDER_CONSTANT,
                 interpolation=cv2.INTER_NEAREST,
                 mask_interpolation=cv2.INTER_NEAREST, fill=0, fill_mask=0),
    ], seed=seed)


def validation_augmentation(seed = SEED):
    # 0.5.1 padded by reflection; 2.0.8 defaults to constant padding instead.
    return A.Compose([
        A.PadIfNeeded(min_height=768, min_width=960,
                      border_mode=cv2.BORDER_REFLECT_101, p=1),
    ], seed=seed)


def imagenet_preprocess(image):
    """Keras default caffe preprocessing: RGB -> BGR, subtract means, no scaling."""
    image = image[..., ::-1].astype(np.float32, copy=True)
    image -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
    return np.ascontiguousarray(image)


def sanitize_mask(mask):
    mask = np.round(mask).clip(0, 1).astype(np.float32)
    return mask[..., None] if mask.ndim == 2 else mask


class ROIDataset:
    """Indexable, unbatched ROI samples; extra copies follow normal examples.

    Policies are attached to indices rather than filename access history, so
    shuffling/repeated reads cannot accidentally switch an example's policy.
    """

    def __init__(self, filenames, data_dir = "data",
                 *, extra_filenames = (),
                 validation = False, seed = SEED):
        normal, extra = tuple(filenames), tuple(extra_filenames)
        if validation and extra:
            raise ValueError("Validation cannot contain extra training copies")
        if len(set(extra)) != len(extra) or not set(extra).issubset(normal):
            raise ValueError("Extra filenames must be a unique subset of training")
        self.filenames = normal + extra
        self.normal_count = len(normal)
        self.data_dir = Path(data_dir)
        self.augmentation = (validation_augmentation(seed) if validation
                             else training_augmentation(seed))
        self.extra_augmentation = extra_augmentation(seed) if extra else None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        image, mask = load_pair(self.data_dir, self.filenames[index])
        transform = (self.extra_augmentation if index >= self.normal_count
                     else self.augmentation)
        sample = transform(image=image, mask=mask)
        return imagenet_preprocess(sample["image"]), sanitize_mask(sample["mask"])
