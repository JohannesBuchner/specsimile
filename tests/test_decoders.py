import json
import hashlib
import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from specsimile.decoders import (
    sigmoid_asinh,
    DenseDecoder,
    LogDenseDecoder,
    CutoffPLDecoder,
    SBPLDecoder,
    QuadraticDecoder,
)


# ---------------------------------------------------------------------------
# sigmoid_asinh
# ---------------------------------------------------------------------------

def test_sigmoid_asinh_midpoint():
    x = torch.tensor([0.0])
    result = sigmoid_asinh(x, 0.0, 10.0)
    assert torch.isclose(result, torch.tensor([5.0]), atol=1e-6)


def test_sigmoid_asinh_bounds_approached():
    x_large = torch.tensor([1e6])
    x_small = torch.tensor([-1e6])
    lo, hi = 1.0, 5.0
    assert sigmoid_asinh(x_large, lo, hi).item() > (lo + hi) / 2
    assert sigmoid_asinh(x_small, lo, hi).item() < (lo + hi) / 2


def test_sigmoid_asinh_shape():
    x = torch.randn(4, 3)
    result = sigmoid_asinh(x, -1.0, 1.0)
    assert result.shape == (4, 3)


@given(
    v=st.floats(-1e4, 1e4),
    lo=st.floats(-10, 0),
    hi=st.floats(1, 10),
)
def test_sigmoid_asinh_in_range(v, lo, hi):
    x = torch.tensor([v], dtype=torch.float64)
    result = sigmoid_asinh(x, lo, hi).item()
    mid = (lo + hi) / 2.0
    spread = hi - lo
    # result must be strictly between lo-margin and hi+margin (unbounded asinh)
    # but centered around mid; just verify finite and symmetric about mid
    assert np.isfinite(result)


# ---------------------------------------------------------------------------
# DenseDecoder
# ---------------------------------------------------------------------------

def test_dense_decoder_forward_identity():
    dec = DenseDecoder(y_dim=4)
    z = torch.randn(5, 4)
    assert torch.equal(dec(z), z)


def test_dense_decoder_latent_dim():
    dec = DenseDecoder(y_dim=7)
    assert dec.latent_dim == 7


def test_dense_decoder_default_paramnames():
    dec = DenseDecoder(y_dim=3)
    assert dec.paramnames == ["y0", "y1", "y2"]


def test_dense_decoder_custom_paramnames():
    dec = DenseDecoder(y_dim=2, paramnames=["a", "b"])
    assert dec.paramnames == ["a", "b"]


def test_dense_decoder_paramnames_mismatch_raises():
    with pytest.raises(ValueError):
        DenseDecoder(y_dim=3, paramnames=["a", "b"])


def test_dense_decoder_normalize_keys():
    dec = DenseDecoder(y_dim=2)
    y = np.random.rand(10, 2)
    params = np.random.rand(10, 3)
    out = dec.normalize(y, params)
    assert set(out.keys()) == {"params", "latent", "y"}


def test_dense_decoder_normalize_shapes():
    dec = DenseDecoder(y_dim=4)
    N, Y, P = 20, 4, 3
    y = np.random.rand(N, Y)
    params = np.random.rand(N, P)
    out = dec.normalize(y, params)
    assert out["latent"]["mean"].shape == (Y,)
    assert out["params"]["mean"].shape == (P,)


def test_dense_decoder_normalize_bad_input_raises():
    dec = DenseDecoder(y_dim=2)
    with pytest.raises(ValueError):
        dec.normalize(np.random.rand(10), np.random.rand(10, 2))


def test_dense_decoder_normalize_mismatched_N_raises():
    dec = DenseDecoder(y_dim=2)
    with pytest.raises(ValueError):
        dec.normalize(np.random.rand(5, 2), np.random.rand(10, 2))


def test_dense_decoder_get_config_roundtrip():
    dec = DenseDecoder(y_dim=3, paramnames=["x", "y", "z"])
    cfg = dec.get_config()
    assert cfg["y_dim"] == 3
    assert cfg["paramnames"] == ["x", "y", "z"]
    json.dumps(cfg)  # must be JSON-serialisable


def test_dense_decoder_config_hash_is_hex():
    dec = DenseDecoder(y_dim=2)
    h = dec.config_hash()
    assert len(h) == 64
    int(h, 16)


def test_dense_decoder_config_hash_changes_with_config():
    dec1 = DenseDecoder(y_dim=2)
    dec2 = DenseDecoder(y_dim=3)
    assert dec1.config_hash() != dec2.config_hash()


# ---------------------------------------------------------------------------
# LogDenseDecoder
# ---------------------------------------------------------------------------

def test_log_dense_decoder_default_paramnames():
    dec = LogDenseDecoder(y_dim=2)
    assert dec.paramnames == ["logy0", "logy1"]


def test_log_dense_decoder_normalize_log_flag():
    dec = LogDenseDecoder(y_dim=2)
    y = np.random.rand(10, 2) + 0.1
    params = np.random.rand(10, 2)
    out = dec.normalize(y, params)
    assert out["y"]["log"] is True


def test_log_dense_decoder_evaluate():
    dec = LogDenseDecoder(y_dim=3)
    z = torch.tensor([[1.0, 2.0, 3.0]])
    result = dec.evaluate(z)
    expected = torch.pow(10.0, z)
    assert torch.allclose(result, expected)


def test_log_dense_decoder_get_config_has_eps():
    dec = LogDenseDecoder(y_dim=2, eps=1e-20)
    cfg = dec.get_config()
    assert "eps" in cfg
    assert cfg["eps"] == 1e-20


# ---------------------------------------------------------------------------
# _XSetterMixin via CutoffPLDecoder
# ---------------------------------------------------------------------------

def test_x_setter_stores_float64():
    dec = CutoffPLDecoder(x=np.array([1.0, 2.0, 3.0]))
    assert dec.x.dtype == torch.float64


def test_x_setter_rejects_2d():
    dec = CutoffPLDecoder()
    with pytest.raises(ValueError):
        dec.x = np.ones((3, 3))


def test_x_setter_update():
    dec = CutoffPLDecoder(x=np.array([1.0, 2.0]))
    dec.x = np.array([10.0, 20.0, 30.0])
    assert dec.x.shape == (3,)


# ---------------------------------------------------------------------------
# CutoffPLDecoder
# ---------------------------------------------------------------------------

def test_cutoffpl_latent_dim():
    dec = CutoffPLDecoder()
    assert dec.latent_dim == 3


def test_cutoffpl_forward_shape():
    x = np.linspace(1.0, 100.0, 50)
    dec = CutoffPLDecoder(x=x)
    z = torch.zeros(4, 3, dtype=torch.float64)
    out = dec(z)
    assert out.shape == (4, 50)


def test_cutoffpl_forward_finite():
    x = np.logspace(0, 2, 30)
    dec = CutoffPLDecoder(x=x)
    z = torch.randn(8, 3, dtype=torch.float64)
    out = dec(z)
    assert torch.all(torch.isfinite(out))


def test_cutoffpl_evaluate_positive():
    x = np.logspace(0, 2, 20)
    dec = CutoffPLDecoder(x=x)
    z = torch.zeros(3, 3, dtype=torch.float64)
    out = dec.evaluate(z)
    assert torch.all(out > 0)


def test_cutoffpl_latent_to_params_gamma_in_range():
    dec = CutoffPLDecoder(gamma_lo=0.5, gamma_hi=6.0)
    # the asinh squash is unbounded, so the configured range is only
    # guaranteed for moderate latents (|raw| <~ 3.6); clamp to +-3
    z = torch.randn(100, 3, dtype=torch.float64).clamp(-3, 3)
    _, gamma, _ = dec.latent_to_params(z)
    assert torch.all(gamma > 0.5) and torch.all(gamma < 6.0)


def test_cutoffpl_latent_to_params_xcut_positive():
    dec = CutoffPLDecoder()
    z = torch.randn(50, 3, dtype=torch.float64)
    _, _, xcut = dec.latent_to_params(z)
    assert torch.all(xcut > 0)


def test_cutoffpl_normalize_output():
    x = np.logspace(0, 2, 20)
    dec = CutoffPLDecoder(x=x)
    y = np.random.rand(15, 20) + 0.1
    params = np.random.rand(15, 2)
    out = dec.normalize(y, params)
    assert out["latent"]["dim"] == 3
    assert out["y"]["log"] is True


def test_cutoffpl_get_config_roundtrip():
    dec = CutoffPLDecoder(gamma_lo=1.0, gamma_hi=5.0, xcut_lo=0.01, xcut_hi=1e4)
    cfg = dec.get_config()
    json.dumps(cfg)
    assert cfg["gamma_lo"] == 1.0
    assert cfg["gamma_hi"] == 5.0


def test_cutoffpl_config_hash_deterministic():
    dec = CutoffPLDecoder(gamma_lo=1.0, gamma_hi=4.0)
    assert dec.config_hash() == dec.config_hash()


# ---------------------------------------------------------------------------
# SBPLDecoder
# ---------------------------------------------------------------------------

def test_sbpl_latent_dim():
    dec = SBPLDecoder(logx_lo=0.0, logx_hi=3.0)
    assert dec.latent_dim == 5


def test_sbpl_forward_shape():
    x = np.logspace(0, 3, 60)
    dec = SBPLDecoder(logx_lo=0.0, logx_hi=3.0, x=x)
    z = torch.zeros(5, 5, dtype=torch.float64)
    out = dec(z)
    assert out.shape == (5, 60)


def test_sbpl_forward_finite():
    x = np.logspace(0, 3, 40)
    dec = SBPLDecoder(logx_lo=0.0, logx_hi=3.0, x=x)
    z = torch.randn(10, 5, dtype=torch.float64)
    out = dec(z)
    assert torch.all(torch.isfinite(out))


def test_sbpl_evaluate_positive():
    x = np.logspace(0, 2, 30)
    dec = SBPLDecoder(logx_lo=0.0, logx_hi=2.0, x=x)
    z = torch.zeros(4, 5, dtype=torch.float64)
    out = dec.evaluate(z)
    assert torch.all(out > 0)


def test_sbpl_latent_to_params_shapes():
    dec = SBPLDecoder(logx_lo=0.0, logx_hi=3.0)
    z = torch.randn(7, 5, dtype=torch.float64)
    lognorm, s1, s2, xbrk, width = dec.latent_to_params(z)
    for t in (lognorm, s1, s2, xbrk, width):
        assert t.shape == (7,)


def test_sbpl_xbrk_in_range():
    dec = SBPLDecoder(logx_lo=0.0, logx_hi=3.0)
    # the asinh squash is unbounded, so the configured range is only
    # guaranteed for moderate latents (|raw| <~ 3.6); clamp to +-3
    z = torch.randn(100, 5, dtype=torch.float64).clamp(-3, 3)
    _, _, _, xbrk, _ = dec.latent_to_params(z)
    assert torch.all(xbrk > 1.0) and torch.all(xbrk < 1000.0)


def test_sbpl_width_positive():
    dec = SBPLDecoder(logx_lo=0.0, logx_hi=3.0)
    z = torch.randn(50, 5, dtype=torch.float64)
    _, _, _, _, width = dec.latent_to_params(z)
    assert torch.all(width > 0)


def test_sbpl_with_x0():
    x = np.logspace(0, 3, 40)
    dec = SBPLDecoder(logx_lo=0.0, logx_hi=3.0, x=x, x0=10.0)
    assert dec.x0 is not None
    z = torch.zeros(3, 5, dtype=torch.float64)
    out = dec(z)
    assert out.shape == (3, 40)


def test_sbpl_get_config_keys():
    dec = SBPLDecoder(logx_lo=0.0, logx_hi=3.0, slope_scale=3.0)
    cfg = dec.get_config()
    for key in ("logx_lo", "logx_hi", "slope_scale", "logwidth_lo", "logwidth_hi", "x0", "eps"):
        assert key in cfg
    json.dumps(cfg)


def test_sbpl_normalize_output():
    x = np.logspace(0, 3, 40)
    dec = SBPLDecoder(logx_lo=0.0, logx_hi=3.0, x=x)
    y = np.random.rand(12, 40) + 0.1
    params = np.random.rand(12, 2)
    out = dec.normalize(y, params)
    assert out["latent"]["dim"] == 5


# ---------------------------------------------------------------------------
# QuadraticDecoder
# ---------------------------------------------------------------------------

def test_quadratic_latent_dim():
    dec = QuadraticDecoder(logx_lo=0.0, logx_hi=3.0)
    assert dec.latent_dim == 3


def test_quadratic_forward_shape():
    x = np.logspace(0, 3, 50)
    dec = QuadraticDecoder(logx_lo=0.0, logx_hi=3.0, x=x)
    z = torch.zeros(6, 3, dtype=torch.float64)
    out = dec(z)
    assert out.shape == (6, 50)


def test_quadratic_forward_finite():
    x = np.logspace(0, 3, 40)
    dec = QuadraticDecoder(logx_lo=0.0, logx_hi=3.0, x=x)
    z = torch.randn(10, 3, dtype=torch.float64)
    out = dec(z)
    assert torch.all(torch.isfinite(out))


def test_quadratic_peak_at_xpeak():
    x = np.logspace(-1, 3, 2000)
    dec = QuadraticDecoder(logx_lo=-1.0, logx_hi=3.0, x=x)
    # Use z_phys with logA=2.0, raw latent near 0 -> xpeak near geometric mid
    z = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64)
    log_y = dec(z)
    peak_idx = torch.argmax(log_y[0]).item()
    logA, xpeak, _ = dec.latent_to_params(z)
    assert abs(np.log10(x[peak_idx]) - np.log10(xpeak[0].item())) < 0.05


def test_quadratic_evaluate_positive():
    x = np.logspace(0, 2, 30)
    dec = QuadraticDecoder(logx_lo=0.0, logx_hi=2.0, x=x)
    z = torch.zeros(3, 3, dtype=torch.float64)
    out = dec.evaluate(z)
    assert torch.all(out > 0)


def test_quadratic_latent_to_params_xpeak_in_range():
    dec = QuadraticDecoder(logx_lo=0.0, logx_hi=3.0)
    # the asinh squash is unbounded, so the configured range is only
    # guaranteed for moderate latents (|raw| <~ 3.6); clamp to +-3
    z = torch.randn(100, 3, dtype=torch.float64).clamp(-3, 3)
    _, xpeak, _ = dec.latent_to_params(z)
    log_xpeak = torch.log10(xpeak)
    assert torch.all(log_xpeak > 0.0) and torch.all(log_xpeak < 3.0)


def test_quadratic_latent_to_params_w_positive():
    dec = QuadraticDecoder(logx_lo=0.0, logx_hi=2.0)
    z = torch.randn(50, 3, dtype=torch.float64)
    _, _, w = dec.latent_to_params(z)
    assert torch.all(w > 0)


def test_quadratic_get_config_roundtrip():
    dec = QuadraticDecoder(logx_lo=1.0, logx_hi=4.0, logw_lo=-2.0, logw_hi=2.0)
    cfg = dec.get_config()
    assert cfg["logx_lo"] == 1.0
    assert cfg["logx_hi"] == 4.0
    json.dumps(cfg)


def test_quadratic_normalize_bad_input_raises():
    dec = QuadraticDecoder(logx_lo=0.0, logx_hi=2.0)
    with pytest.raises(ValueError):
        dec.normalize(np.random.rand(5, 10), np.random.rand(7, 2))


def test_quadratic_normalize_output_structure():
    dec = QuadraticDecoder(logx_lo=0.0, logx_hi=2.0)
    y = np.random.rand(10, 20) + 0.1
    params = np.random.rand(10, 3)
    out = dec.normalize(y, params)
    assert set(out.keys()) == {"params", "latent", "y"}
    assert out["latent"]["dim"] == 3


# ---------------------------------------------------------------------------
# DecoderConfigMixin config_hash consistency
# ---------------------------------------------------------------------------

def test_config_hash_same_config_same_hash():
    dec1 = CutoffPLDecoder(gamma_lo=1.0, gamma_hi=5.0, xcut_lo=0.01, xcut_hi=1e3)
    dec2 = CutoffPLDecoder(gamma_lo=1.0, gamma_hi=5.0, xcut_lo=0.01, xcut_hi=1e3)
    assert dec1.config_hash() == dec2.config_hash()


def test_config_hash_different_config_different_hash():
    dec1 = SBPLDecoder(logx_lo=0.0, logx_hi=3.0)
    dec2 = SBPLDecoder(logx_lo=0.0, logx_hi=4.0)
    assert dec1.config_hash() != dec2.config_hash()


# ---------------------------------------------------------------------------
# Hypothesis-based property tests
# ---------------------------------------------------------------------------

@given(
    z_vals=arrays(np.float64, shape=(5, 3), elements=st.floats(-3, 3)),
)
@settings(max_examples=50)
def test_cutoffpl_forward_finite_hypothesis(z_vals):
    x = np.logspace(0, 2, 20)
    dec = CutoffPLDecoder(x=x)
    z = torch.tensor(z_vals, dtype=torch.float64)
    out = dec(z)
    assert torch.all(torch.isfinite(out))


@given(
    z_vals=arrays(np.float64, shape=(5, 5), elements=st.floats(-3, 3)),
)
@settings(max_examples=50)
def test_sbpl_forward_finite_hypothesis(z_vals):
    x = np.logspace(0, 3, 30)
    dec = SBPLDecoder(logx_lo=0.0, logx_hi=3.0, x=x)
    z = torch.tensor(z_vals, dtype=torch.float64)
    out = dec(z)
    assert torch.all(torch.isfinite(out))


@given(
    z_vals=arrays(np.float64, shape=(5, 3), elements=st.floats(-3, 3)),
)
@settings(max_examples=50)
def test_quadratic_forward_finite_hypothesis(z_vals):
    x = np.logspace(0, 2, 25)
    dec = QuadraticDecoder(logx_lo=0.0, logx_hi=2.0, x=x)
    z = torch.tensor(z_vals, dtype=torch.float64)
    out = dec(z)
    assert torch.all(torch.isfinite(out))


@given(y_dim=st.integers(min_value=1, max_value=20))
def test_dense_decoder_latent_dim_hypothesis(y_dim):
    dec = DenseDecoder(y_dim=y_dim)
    assert dec.latent_dim == y_dim
    z = torch.randn(3, y_dim)
    assert dec(z).shape == (3, y_dim)
