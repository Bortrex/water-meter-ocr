"""Flax ROI U-Net based on cell 41 of the 2021 ROI detection notebook.

Inputs are float NHWC batches with three channels and spatial sizes divisible
by 16 (including 720 and 960). Outputs are NHWC one-channel probabilities.
Pass train=False for evaluation, using the initialized params and batch_stats.
Training-mode application needs mutable=['batch_stats'] and a 'dropout' RNG;
the returned batch_stats must be retained by the eventual training code.
"""

import jax.numpy as jnp
from flax import linen as nn


class ConvAct(nn.Module):
    """He-normal spatial convolution, optional BatchNorm, optional Swish."""

    features: int
    batch_norm: bool = False
    strides: tuple[int, int] = (1, 1)
    activate: bool = True

    @nn.compact
    def __call__(self, x, *, train):
        x = nn.Conv(
            self.features, (3, 3), strides=self.strides, padding="SAME",
            use_bias=not self.batch_norm, kernel_init=nn.initializers.he_normal(),
            name="conv",
        )(x)
        if self.batch_norm:
            # Match Keras defaults, especially epsilon (Flax defaults to 1e-5).
            x = nn.BatchNorm(use_running_average=not train, momentum=0.99,
                             epsilon=1e-3, name="batch_norm")(x)
        return nn.swish(x) if self.activate else x


class ConvBlock(nn.Module):
    """Two spatial convolutions; residual mode adds a shortcut before Swish."""

    features: int
    batch_norm: bool = False
    dropout_rate: float = 0.0
    use_residual_blocks: bool = False

    @nn.compact
    def __call__(self, x, *, train):
        shortcut = x
        x = ConvAct(self.features, self.batch_norm, name="first")(x, train=train)
        if self.dropout_rate:
            x = nn.Dropout(rate=self.dropout_rate, name="dropout")(
                x, deterministic=not train
            )
        x = ConvAct(self.features, self.batch_norm,
                    activate=not self.use_residual_blocks, name="second")(x, train=train)
        if self.use_residual_blocks:
            if shortcut.shape[-1] != self.features:
                shortcut = nn.Conv(
                    self.features, (1, 1), padding="SAME", use_bias=False,
                    kernel_init=nn.initializers.he_normal(), name="projection",
                )(shortcut)
            x = nn.swish(x + shortcut)
        return x


class EncoderBlock(nn.Module):
    features: int
    batch_norm: bool = False
    use_residual_blocks: bool = False

    @nn.compact
    def __call__(self, x, *, train):
        skip = ConvBlock(self.features, batch_norm=self.batch_norm,
                         use_residual_blocks=self.use_residual_blocks,
                         name="block")(x, train=train)
        # The notebook's learned downsampler has bias and Swish, but no BN.
        x = ConvAct(self.features, strides=(2, 2), name="downsample")(
            skip, train=train
        )
        return x, skip


class DecoderBlock(nn.Module):
    features: int
    dropout_rate: float
    use_residual_blocks: bool = False

    @nn.compact
    def __call__(self, x, skip, *, train):
        x = nn.ConvTranspose(
            self.features, (2, 2), strides=(2, 2), padding="SAME",
            # Keras stores transposed-convolution kernels as HWOI.
            transpose_kernel=True, kernel_init=nn.initializers.he_normal(),
            name="upsample",
        )(x)
        if x.shape[1:3] != skip.shape[1:3]:
            raise ValueError(f"Upsampled and skip shapes differ: {x.shape}, {skip.shape}")
        x = jnp.concatenate((x, skip), axis=-1)
        return ConvBlock(self.features, dropout_rate=self.dropout_rate,
                         use_residual_blocks=self.use_residual_blocks,
                         name="block")(x, train=train)


class ROIUNet(nn.Module):
    """Four-level U-Net; optional residual blocks preserve the outer topology.

    Baseline: encoder widths 16/32/64/128, bottleneck 256, decoder 128/64/32/16.
    BatchNorm is confined to the 64/128 encoder blocks and the bottleneck.
    Decoder dropout sits between convolutions, with rates 0.2/0.2/0.1/0.1.
    """

    use_residual_blocks: bool = False

    @nn.compact
    def __call__(self, x, *, train = False):
        if x.ndim != 4 or x.shape[-1] != 3:
            raise ValueError(f"Expected NHWC input with 3 channels, got {x.shape}")
        if any(size <= 0 or size % 16 for size in x.shape[1:3]):
            raise ValueError("Input height and width must be positive multiples of 16")
        if not jnp.issubdtype(x.dtype, jnp.floating):
            raise ValueError("Expected floating-point preprocessed images")
        skips = []
        for i, features in enumerate((16, 32, 64, 128)):
            x, skip = EncoderBlock(
                features, batch_norm=i >= 2,
                use_residual_blocks=self.use_residual_blocks, name=f"encoder_{i}",
            )(x, train=train)
            skips.append(skip)
        x = ConvBlock(256, batch_norm=True,
                      use_residual_blocks=self.use_residual_blocks,
                      name="bottleneck")(x, train=train)
        for i, (features, dropout_rate, skip) in enumerate(zip(
            (128, 64, 32, 16), (0.2, 0.2, 0.1, 0.1), reversed(skips)
        )):
            x = DecoderBlock(
                features, dropout_rate, use_residual_blocks=self.use_residual_blocks,
                name=f"decoder_{i}",
            )(x, skip, train=train)
        # The original output layer used Keras's default Glorot, not He-normal.
        x = nn.Conv(1, (1, 1), kernel_init=nn.initializers.glorot_uniform(),
                    name="segmentation")(x)
        return nn.sigmoid(x)
