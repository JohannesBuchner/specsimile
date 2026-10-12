import os
import tempfile
import numpy as np
import pytest
import torch

from specsimile.decoders import CutoffPLDecoder, SBPLDecoder, DenseDecoder
from specsimile.emulator import Emulator
from specsimile.store import DatasetWriter, DatasetReader
import specsimile.jxrun as jxrun

import jax.numpy as jnp


def _make_cutoffpl_dataset(path: str, seed: int = 42) -> None:
    """Create a small CutoffPL training dataset with 20 samples."""
    rng = np.random.default_rng(seed)
    x = np.logspace(0, 2, 15)  # 15-point x grid
    N = 20
    paramnames = ["logA", "gamma", "logxcut"]

    with DatasetWriter(path, N=N, x=x, paramnames=paramnames) as writer:
        for _ in range(N):
            logA = rng.uniform(0.0, 2.0)
            gamma = rng.uniform(1.0, 3.0)
            logxcut = rng.uniform(1.0, 2.0)
            xcut = 10.0 ** logxcut
            y = (10.0 ** logA) * x ** (-gamma) * np.exp(-x / xcut)
            writer.append(x, y, np.array([logA, gamma, logxcut]))


def _build_and_train_emulator(model_path: str, dataset_path: str) -> Emulator:
    """Build a 1-layer 2-node CutoffPL emulator and train for 1 epoch."""
    decoder = CutoffPLDecoder(gamma_lo=0.5, gamma_hi=6.0, xcut_lo=1.0, xcut_hi=1e3)
    emu = Emulator(
        model_path=model_path,
        decoder=decoder,
        shape=[2],
        param_bounds_policy='warn',
    )
    dataset = DatasetReader(dataset_path)
    emu.fit(
        dataset,
        epochs=1,
        lr=1e-3,
        batch_size=10,
        # train_split=0.8,
        # patience=100,
        seed=7,
    )
    dataset.close()
    return emu


def _make_sbpl_dataset(path: str, seed: int = 42) -> None:
    """Create a small SBPL training dataset with 20 samples."""
    rng = np.random.default_rng(seed)
    x = np.logspace(1, 3, 15)  # 15-point x grid
    N = 20
    paramnames = ["lognorm", "s1", "s2", "xbrk", "width"]

    with DatasetWriter(path, N=N, x=x, paramnames=paramnames) as writer:
        for _ in range(N):
            lognorm = rng.uniform(0.0, 2.0)
            s1 = rng.uniform(-3.0, -1.0)
            s2 = rng.uniform(-5.0, -3.0)
            xbrk = rng.uniform(50.0, 500.0)
            width = rng.uniform(0.5, 5.0)

            # Evaluate SBPL model directly
            q = np.log(x / xbrk) / width
            q = np.clip(q, -50.0, 50.0)
            num = np.exp(q) + np.exp(-q)
            den = 2.0
            xrel = x / xbrk
            expo = ((s2 - s1) / 2.0) * width
            expo = np.clip(expo, -10.0, 10.0)
            norm_val = 10.0 ** lognorm
            y = norm_val * xrel ** ((s1 + s2 + 2.0) / 2.0) * (num / den) ** expo / xrel
            y = np.clip(y, 1e-30, None)

            writer.append(x, y, np.array([lognorm, s1, s2, xbrk, width]))


def _build_and_train_sbpl_emulator(model_path: str, dataset_path: str) -> Emulator:
    """Build a 1-layer 4-node SBPL emulator and train for 1 epoch."""
    decoder = SBPLDecoder(
        logx_lo=1.0,
        logx_hi=3.0,
        slope_scale=2.5,
        logwidth_lo=-1.0,
        logwidth_hi=2.0,
    )
    emu = Emulator(
        model_path=model_path,
        decoder=decoder,
        shape=[4],
        acti="LeakyReLU",
        device="cpu",
    )
    dataset = DatasetReader(dataset_path)
    emu.fit(
        dataset,
        epochs=1,
        lr=1e-3,
        batch_size=10,
        train_split=0.8,
        patience=100,
        seed=7,
    )
    dataset.close()
    return emu


def _make_dense_dataset(path: str, seed: int = 42) -> None:
    """Create a small Dense (pass-through) training dataset with 20 samples."""
    rng = np.random.default_rng(seed)
    x = np.linspace(1.0, 10.0, 8)  # 8-point x grid
    N = 20
    paramnames = ["p0", "p1", "p2"]

    with DatasetWriter(path, N=N, x=x, paramnames=paramnames) as writer:
        for _ in range(N):
            p = rng.uniform(0.0, 1.0, size=3)
            # y is just a simple linear function of params, same dim as x
            y = np.array([
                p[0] * xi + p[1] * xi ** 2 + p[2]
                for xi in x
            ])
            writer.append(x, y, p)


def _build_and_train_dense_emulator(model_path: str, dataset_path: str) -> Emulator:
    """Build a 1-layer 4-node Dense emulator and train for 1 epoch."""
    decoder = DenseDecoder(y_dim=8)
    emu = Emulator(
        model_path=model_path,
        decoder=decoder,
        shape=[4],
        acti="LeakyReLU",
        device="cpu",
    )
    dataset = DatasetReader(dataset_path)
    emu.fit(
        dataset,
        epochs=1,
        lr=1e-3,
        batch_size=10,
        train_split=0.8,
        patience=100,
        seed=7,
    )
    dataset.close()
    return emu


def test_cutoffpl_emulator_vs_jxrun():
    """
    Train a tiny CutoffPL emulator, run it with fiducial inputs via the
    Emulator.transform() interface, then load the checkpoint through
    jxrun.load_decoder_bundle / jxrun.jax_encoder_forward and verify that
    both paths produce identical outputs (within float64 tolerance).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = os.path.join(tmpdir, "dataset.h5")
        model_path = os.path.join(tmpdir, "emu.pt")

        _make_cutoffpl_dataset(dataset_path, seed=42)
        emu = _build_and_train_emulator(model_path, dataset_path)

        # Fiducial input parameters [logA, gamma, logxcut]
        fiducial_params = np.array([[13.0, 2.0, 1.5]])

        # --- Path 1: specsimile Emulator ---
        # Use the x grid from the training dataset
        dataset = DatasetReader(dataset_path)
        x_grid = dataset.x.copy()
        dataset.close()

        emu.decoder.x = x_grid
        y_specsimile = emu.transform(
            fiducial_params,
            x=x_grid,
            return_latent_params=False,
        )  # shape (1, len(x_grid))

        # --- Path 2: jxrun ---
        bundle = jxrun.load_decoder_bundle(model_path, prefix="test", verbose=False)

        # Inputs are normalised by jax_encoder_forward using the norm
        # embedded in the bundle.
        x_jax = jnp.array(fiducial_params, dtype=jnp.float64)  # (1, 3)
        z_phys_jax = jxrun.jax_encoder_forward(x_jax, bundle, prefix="test")
        # z_phys_jax has shape (1, latent_dim=3): [logA_raw, gamma_raw, logxcut_raw]

        # Apply cutoffpl decoder in JAX to get y
        decoder_config = bundle.get("test_decoder_config", {})
        gamma_lo = decoder_config.get("gamma_lo", 0.5)
        gamma_hi = decoder_config.get("gamma_hi", 6.0)
        xcut_lo_val = decoder_config.get("xcut_lo", 1.0)
        xcut_hi_val = decoder_config.get("xcut_hi", 1e3)
        logxcut_lo = float(np.log10(xcut_lo_val))
        logxcut_hi = float(np.log10(xcut_hi_val))

        # latent_to_params
        params_jax = jxrun.jax_cutoffpl_latent_to_params(
            z_phys_jax,
            gamma_lo=gamma_lo,
            gamma_hi=gamma_hi,
            logxcut_lo=logxcut_lo,
            logxcut_hi=logxcut_hi,
        )
        # params_jax: (1, 3) -> [logA, gamma, xcut]
        logA_j = params_jax[:, 0]
        gamma_j = params_jax[:, 1]
        xcut_j = params_jax[:, 2]

        ln10 = np.log(10.0)
        x_jgrid = jnp.array(x_grid, dtype=jnp.float64)[None, :]  # (1, D)
        lnA = logA_j[:, None] * ln10
        lnx = jnp.log(jnp.clip(x_jgrid, 1e-300, None))
        lny = lnA - gamma_j[:, None] * lnx - (x_jgrid / xcut_j[:, None])
        logy_jax = lny / ln10  # log10(y)
        y_jax = 10.0 ** logy_jax  # linear y, shape (1, D)

        y_specsimile_np = np.asarray(y_specsimile)
        y_jax_np = np.asarray(y_jax)

        assert y_specsimile_np.shape == y_jax_np.shape, (
            f"Shape mismatch: specsimile {y_specsimile_np.shape} vs "
            f"jxrun {y_jax_np.shape}"
        )

        np.testing.assert_allclose(
            y_specsimile_np,
            y_jax_np,
            rtol=1e-5,
            atol=1e-10,
            err_msg="CutoffPL emulator outputs differ between specsimile and jxrun paths",
        )


def test_sbpl_emulator_vs_jxrun():
    """
    Train a tiny SBPL emulator, run it with fiducial inputs via the
    Emulator.transform() interface, then load the checkpoint through
    jxrun.load_decoder_bundle / jxrun.jax_encoder_forward and verify that
    both paths produce identical outputs (within float64 tolerance).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = os.path.join(tmpdir, "sbpl_dataset.h5")
        model_path = os.path.join(tmpdir, "sbpl_emu.pt")

        _make_sbpl_dataset(dataset_path, seed=42)
        emu = _build_and_train_sbpl_emulator(model_path, dataset_path)

        # Fiducial input parameters [lognorm, s1, s2, xbrk, width]
        fiducial_params = np.array([[1.0, -2.0, -4.0, 100.0, 2.0]])

        # --- Path 1: specsimile Emulator ---
        dataset = DatasetReader(dataset_path)
        x_grid = dataset.x.copy()
        dataset.close()

        emu.decoder.x = x_grid
        y_specsimile = emu.transform(
            fiducial_params,
            x=x_grid,
            return_latent_params=False,
        )  # shape (1, len(x_grid))

        # --- Path 2: jxrun ---
        bundle = jxrun.load_decoder_bundle(model_path, prefix="sbpl", verbose=False)

        x_jax = jnp.array(fiducial_params, dtype=jnp.float64)  # (1, 5)
        z_phys_jax = jxrun.jax_encoder_forward(x_jax, bundle, prefix="sbpl")
        # z_phys_jax has shape (1, latent_dim=5)

        # Read decoder config
        decoder_config = bundle.get("sbpl_decoder_config", {})
        slope_scale = decoder_config.get("slope_scale", 2.5)
        logx_lo = decoder_config.get("logx_lo", 1.0)
        logx_hi = decoder_config.get("logx_hi", 3.0)
        logw_lo = decoder_config.get("logwidth_lo", -1.0)
        logw_hi = decoder_config.get("logwidth_hi", 2.0)

        # latent_to_params via jxrun
        params_jax = jxrun._jax_sbpl_latent_to_params(
            z_phys_jax,
            slope_scale=slope_scale,
            logx_lo=logx_lo,
            logx_hi=logx_hi,
            logw_lo=logw_lo,
            logw_hi=logw_hi,
        )
        # params_jax: (1, 5) -> [lognorm, s1, s2, xbrk, width]
        lognorm_j = params_jax[:, 0]
        s1_j = params_jax[:, 1]
        s2_j = params_jax[:, 2]
        xbrk_j = params_jax[:, 3]
        width_j = params_jax[:, 4]

        # Evaluate SBPL forward in JAX (mirrors SBPLDecoder.forward without x0)
        x_jgrid = jnp.array(x_grid, dtype=jnp.float64)[None, :]  # (1, D)
        x_jgrid = jnp.clip(x_jgrid, 1e-300, None)

        q = jnp.log(x_jgrid / xbrk_j[:, None]) / width_j[:, None]
        q = jnp.clip(q, -50.0, 50.0)
        den = 2.0  # no x0 pivot
        xrel = x_jgrid / xbrk_j[:, None]
        num = jnp.exp(q) + jnp.exp(-q)
        ratio = num / (den + 1e-30)
        expo = ((s2_j - s1_j) / 2.0)[:, None] * width_j[:, None]
        expo = jnp.clip(expo, -10.0, 10.0)
        norm_val = 10.0 ** lognorm_j
        y_lin = norm_val[:, None] * xrel ** (((s1_j + s2_j + 2.0) / 2.0)[:, None]) * (ratio ** expo) / xrel
        y_lin = jnp.clip(y_lin, 1e-30, None)
        # log10 then back to linear (matches decoder output path)
        logy_jax = jnp.log10(y_lin)
        y_jax = 10.0 ** logy_jax

        y_specsimile_np = np.asarray(y_specsimile)
        y_jax_np = np.asarray(y_jax)

        assert y_specsimile_np.shape == y_jax_np.shape, (
            f"Shape mismatch: specsimile {y_specsimile_np.shape} vs "
            f"jxrun {y_jax_np.shape}"
        )

        np.testing.assert_allclose(
            y_specsimile_np,
            y_jax_np,
            rtol=1e-5,
            atol=1e-10,
            err_msg="SBPL emulator outputs differ between specsimile and jxrun paths",
        )


def test_dense_emulator_vs_jxrun():
    """
    Train a tiny Dense (pass-through) emulator, run it with fiducial inputs
    via the Emulator.transform() interface, then load the checkpoint through
    jxrun.load_decoder_bundle / jxrun.jax_encoder_forward and verify that
    both paths produce identical outputs (within float64 tolerance).

    For DenseDecoder the forward pass is identity: decoder(z) = z, so the
    emulator output IS the latent vector (after denormalisation).  The jxrun
    path just needs to run the encoder forward pass and denormalise.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = os.path.join(tmpdir, "dense_dataset.h5")
        model_path = os.path.join(tmpdir, "dense_emu.pt")

        _make_dense_dataset(dataset_path, seed=42)
        emu = _build_and_train_dense_emulator(model_path, dataset_path)

        # Fiducial input parameters [p0, p1, p2]
        fiducial_params = np.array([[0.3, 0.5, 0.7]])

        # --- Path 1: specsimile Emulator ---
        dataset = DatasetReader(dataset_path)
        x_grid = dataset.x.copy()
        dataset.close()

        emu.decoder.x = x_grid
        y_specsimile = emu.transform(
            fiducial_params,
            x=x_grid,
            return_latent_params=False,
        )  # shape (1, len(x_grid)=8)

        # --- Path 2: jxrun ---
        bundle = jxrun.load_decoder_bundle(model_path, prefix="dense", verbose=False)

        x_jax = jnp.array(fiducial_params, dtype=jnp.float64)  # (1, 3)
        # jax_encoder_forward applies input normalisation, runs MLP layers,
        # and applies latent denormalisation -> returns z_phys (= y for DenseDecoder)
        z_phys_jax = jxrun.jax_encoder_forward(x_jax, bundle, prefix="dense")
        # For DenseDecoder the decoder is identity, so z_phys IS the output.
        # The emulator stored norm["y"]["log"]=False so no log transform is needed.
        y_jax = z_phys_jax  # shape (1, 8)

        y_specsimile_np = np.asarray(y_specsimile)
        y_jax_np = np.asarray(y_jax)

        assert y_specsimile_np.shape == y_jax_np.shape, (
            f"Shape mismatch: specsimile {y_specsimile_np.shape} vs "
            f"jxrun {y_jax_np.shape}"
        )

        np.testing.assert_allclose(
            y_specsimile_np,
            y_jax_np,
            rtol=1e-5,
            atol=1e-10,
            err_msg="Dense emulator outputs differ between specsimile and jxrun paths",
        )


def test_inside_param_bounds_consistent_with_verify_bundle_param_bounds():
    """
    Train a tiny CutoffPL emulator, then verify that Emulator.inside_param_bounds
    and jxrun.verify_bundle_param_bounds give consistent answers for a range of
    input parameter samples.

    The bounds stored in the emulator are derived from the min/max across the
    training samples.  Both functions should agree on whether each row is
    inside or outside those bounds.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = os.path.join(tmpdir, "dataset_bounds.h5")
        model_path = os.path.join(tmpdir, "emu_bounds.pt")

        # Build a dataset with known parameter ranges so we can craft both
        # inside and outside samples deterministically.
        rng = np.random.default_rng(0)
        x = np.logspace(0, 2, 10)
        N = 30
        paramnames = ["logA", "gamma", "logxcut"]

        # Training ranges: logA in [0,2], gamma in [1,3], logxcut in [1,2]
        with DatasetWriter(dataset_path, N=N, x=x, paramnames=paramnames) as writer:
            for _ in range(N):
                logA = rng.uniform(0.0, 2.0)
                gamma = rng.uniform(1.0, 3.0)
                logxcut = rng.uniform(1.0, 2.0)
                xcut = 10.0 ** logxcut
                y = (10.0 ** logA) * x ** (-gamma) * np.exp(-x / xcut)
                writer.append(x, y, np.array([logA, gamma, logxcut]))

        # Train emulator (1 epoch is enough; we only need the bounds)
        decoder = CutoffPLDecoder(gamma_lo=0.5, gamma_hi=6.0, xcut_lo=1.0, xcut_hi=1e3)
        emu = Emulator(
            model_path=model_path,
            decoder=decoder,
            shape=[2],
            acti="LeakyReLU",
            device="cpu",
        )
        dataset = DatasetReader(dataset_path)
        emu.fit(
            dataset,
            epochs=1,
            lr=1e-3,
            batch_size=10,
            train_split=0.8,
            patience=100,
            seed=3,
        )
        dataset.close()

        # Load the bundle via jxrun
        bundle = jxrun.load_decoder_bundle(model_path, prefix="bnd", verbose=False)

        # Retrieve the actual min/max bounds that were stored during training
        assert emu.train_param_bounds is not None, "train_param_bounds should be set after fit()"
        pmin = np.array(emu.train_param_bounds["min"])
        pmax = np.array(emu.train_param_bounds["max"])

        # Construct test parameter rows:
        #  - some clearly inside the training range
        #  - some clearly outside (below min or above max for each dimension)
        n_inside = 5

        inside_params = np.column_stack([
            rng.uniform(pmin[0], pmax[0], size=n_inside),
            rng.uniform(pmin[1], pmax[1], size=n_inside),
            rng.uniform(pmin[2], pmax[2], size=n_inside),
        ])

        # Outside samples: push one parameter well beyond each bound
        margin = 0.5  # ensure they are clearly outside (> tolerance)
        outside_params = np.array([
            [pmin[0] - margin, (pmin[1] + pmax[1]) / 2, (pmin[2] + pmax[2]) / 2],  # logA below min
            [pmax[0] + margin, (pmin[1] + pmax[1]) / 2, (pmin[2] + pmax[2]) / 2],  # logA above max
            [(pmin[0] + pmax[0]) / 2, pmin[1] - margin, (pmin[2] + pmax[2]) / 2],  # gamma below min
            [(pmin[0] + pmax[0]) / 2, pmax[1] + margin, (pmin[2] + pmax[2]) / 2],  # gamma above max
            [(pmin[0] + pmax[0]) / 2, (pmin[1] + pmax[1]) / 2, pmin[2] - margin],  # logxcut below min
            [(pmin[0] + pmax[0]) / 2, (pmin[1] + pmax[1]) / 2, pmax[2] + margin],  # logxcut above max
        ])

        all_params = np.vstack([inside_params, outside_params])

        # --- Emulator.inside_param_bounds ---
        emu_mask = emu.inside_param_bounds(all_params)

        # --- jxrun.verify_bundle_param_bounds (policy='ignore' so no exception) ---
        bundle_mask = jxrun.verify_bundle_param_bounds(
            bundle,
            prefix="bnd",
            params=all_params,
            param_bounds_tol=0.0,
            param_bounds_policy="ignore",
        )

        assert emu_mask.shape == bundle_mask.shape, (
            f"Mask shape mismatch: emu={emu_mask.shape}, bundle={bundle_mask.shape}"
        )

        # Both masks must agree on every row
        np.testing.assert_array_equal(
            emu_mask,
            bundle_mask,
            err_msg=(
                "inside_param_bounds and verify_bundle_param_bounds disagree.\n"
                f"emu_mask   = {emu_mask}\n"
                f"bundle_mask= {bundle_mask}\n"
                f"params=\n{all_params}\n"
                f"pmin={pmin}\npmax={pmax}"
            ),
        )

        # Sanity checks: the inside_params rows should all be True,
        # the outside_params rows should all be False.
        assert np.all(emu_mask[:n_inside]), (
            f"Expected all inside_params to be in-bounds, got {emu_mask[:n_inside]}"
        )
        assert not np.any(emu_mask[n_inside:]), (
            f"Expected all outside_params to be out-of-bounds, got {emu_mask[n_inside:]}"
        )

        # Also verify the single-sample (1-D) path for verify_bundle_param_bounds
        single_inside = inside_params[0]
        single_outside = outside_params[0]

        mask_single_in = jxrun.verify_bundle_param_bounds(
            bundle, prefix="bnd", params=single_inside,
            param_bounds_tol=0.0, param_bounds_policy="ignore",
        )
        mask_single_out = jxrun.verify_bundle_param_bounds(
            bundle, prefix="bnd", params=single_outside,
            param_bounds_tol=0.0, param_bounds_policy="ignore",
        )

        assert mask_single_in.shape == (1,), f"Expected shape (1,), got {mask_single_in.shape}"
        assert mask_single_out.shape == (1,), f"Expected shape (1,), got {mask_single_out.shape}"
        assert bool(mask_single_in[0]), "Single inside sample should be in-bounds"
        assert not bool(mask_single_out[0]), "Single outside sample should be out-of-bounds"

        # Verify that the 'error' policy raises for out-of-bounds inputs
        with pytest.raises(ValueError):
            jxrun.verify_bundle_param_bounds(
                bundle, prefix="bnd", params=outside_params,
                param_bounds_tol=0.0, param_bounds_policy="error",
            )

        # Verify that a non-negative tolerance widens the accepted region:
        # a sample just outside should become inside when tol >= margin
        tol = margin + 0.1
        mask_with_tol = jxrun.verify_bundle_param_bounds(
            bundle, prefix="bnd", params=outside_params,
            param_bounds_tol=tol, param_bounds_policy="ignore",
        )
        assert np.all(mask_with_tol), (
            f"With tol={tol} all outside_params should become in-bounds, "
            f"got {mask_with_tol}"
        )

        # Negative tolerance disables bounds checking -> all True
        mask_neg_tol = jxrun.verify_bundle_param_bounds(
            bundle, prefix="bnd", params=all_params,
            param_bounds_tol=-1.0, param_bounds_policy="ignore",
        )
        assert np.all(mask_neg_tol), (
            "Negative tolerance should disable bounds checking (all True)"
        )
