"""Architecture checks only. Run from the repository root:

    python -m scripts.check_roi_model

Set JAX_PLATFORMS=cpu to explicitly run without a GPU. No dataset is required.
"""

import jax
import jax.numpy as jnp
import numpy as np

from water_meter_ocr.roi_model import ROIUNet


def check_output(output, shape):
    output = np.asarray(output)
    assert output.shape == shape, output.shape
    assert output.dtype == np.float32, output.dtype
    assert np.isfinite(output).all()
    assert ((output >= 0) & (output <= 1)).all()


def check_variant(residual):
    model = ROIUNet(use_residual_blocks=residual)
    label = "residual" if residual else "standard"
    print(f"{label}: initializing on (1, 720, 720, 3)", flush=True)
    inputs = jnp.ones((1, 720, 720, 3), dtype=jnp.float32)
    # Evaluation initialization creates both collections without needing dropout RNG.
    variables = jax.jit(lambda key, x: model.init(key, x, train=False))(
        jax.random.key(1111), inputs
    )
    jax.block_until_ready(variables)
    assert set(variables) == {"params", "batch_stats"}
    count = sum(leaf.size for leaf in jax.tree_util.tree_leaves(variables["params"]))
    state_count = sum(leaf.size for leaf in jax.tree_util.tree_leaves(variables["batch_stats"]))
    print(f"{label}: {count:,} trainable parameters; {state_count:,} BatchNorm state values", flush=True)
    evaluate = jax.jit(lambda state, x: model.apply(state, x, train=False))
    shapes = []
    for size in (720, 960):
        output = evaluate(variables, jnp.ones((1, size, size, 3), jnp.float32))
        check_output(output, (1, size, size, 1))
        shapes.append(output.shape)
        print(f"{label}: (1, {size}, {size}, 3) -> {output.shape}; finite probabilities", flush=True)

    # Exercise state updates and dropout on a cheap, rectangular input. This is
    # a forward-mode check only: no gradients, optimizer, loss or training loop.
    small = jax.random.normal(jax.random.key(7), (2, 32, 48, 3))
    train_forward = jax.jit(lambda state, x, key: model.apply(
        state, x, train=True, rngs={"dropout": key}, mutable=["batch_stats"]
    ))
    first, updates = train_forward(variables, small, jax.random.key(8))
    repeated, _ = train_forward(variables, small, jax.random.key(8))
    different, _ = train_forward(variables, small, jax.random.key(9))
    check_output(first, (2, 32, 48, 1))
    check_output(different, (2, 32, 48, 1))
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, different), "Dropout RNG had no effect"
    assert any(not np.array_equal(old, new) for old, new in zip(
        jax.tree_util.tree_leaves(variables["batch_stats"]),
        jax.tree_util.tree_leaves(updates["batch_stats"]),
    )), "BatchNorm state did not update"
    assert all(np.isfinite(np.asarray(x)).all()
               for x in jax.tree_util.tree_leaves(updates["batch_stats"]))
    updated = {"params": variables["params"], "batch_stats": updates["batch_stats"]}
    evaluated = evaluate(updated, small)
    check_output(evaluated, (2, 32, 48, 1))
    np.testing.assert_array_equal(evaluated, evaluate(updated, small))
    print(f"{label}: BatchNorm updates, seeded dropout and RNG-free deterministic evaluation passed", flush=True)
    return count, shapes


def main():
    print(f"JAX {jax.__version__}; devices: {jax.devices()}", flush=True)
    standard_count, standard_shapes = check_variant(False)
    jax.clear_caches()
    residual_count, residual_shapes = check_variant(True)
    assert standard_shapes == residual_shapes
    assert residual_count > standard_count
    print(f"PASS: identical geometry; standard={standard_count:,}, residual={residual_count:,}")


if __name__ == "__main__":
    main()
