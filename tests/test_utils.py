import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from specsimile.utils import (
    apply_log_mask,
    as_2d,
    denormalize,
    ensure_finite,
    normalize,
    _p95_l2,
)


# ---------------------------------------------------------------------------
# as_2d
# ---------------------------------------------------------------------------

def test_as_2d_1d_returns_2d_and_flag():
    a = np.array([1.0, 2.0, 3.0])
    out, was_1d = as_2d(a)
    assert was_1d is True
    assert out.ndim == 2
    assert out.shape == (1, 3)


def test_as_2d_2d_returns_same_shape_and_flag():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    out, was_1d = as_2d(a)
    assert was_1d is False
    assert out.shape == (2, 2)


def test_as_2d_dtype_is_float64():
    a = np.array([1, 2, 3], dtype=int)
    out, _ = as_2d(a)
    assert out.dtype == np.float64


def test_as_2d_3d_raises():
    a = np.ones((2, 3, 4))
    with pytest.raises(ValueError, match="expected 1D or 2D array"):
        as_2d(a)


@given(arrays(np.float64, st.integers(1, 50), elements=st.floats(-1e6, 1e6)))
def test_as_2d_1d_hypothesis(a):
    out, was_1d = as_2d(a)
    assert was_1d is True
    assert out.shape == (1, len(a))


# ---------------------------------------------------------------------------
# ensure_finite
# ---------------------------------------------------------------------------

def test_ensure_finite_passes_for_finite_array():
    ensure_finite("x", np.array([1.0, 2.0, 3.0]))


def test_ensure_finite_raises_for_nan():
    with pytest.raises(ValueError, match="non-finite"):
        ensure_finite("x", np.array([1.0, np.nan]))


def test_ensure_finite_raises_for_inf():
    with pytest.raises(ValueError, match="non-finite"):
        ensure_finite("x", np.array([np.inf, 1.0]))


def test_ensure_finite_raises_for_neg_inf():
    with pytest.raises(ValueError, match="non-finite"):
        ensure_finite("x", np.array([-np.inf]))


def test_ensure_finite_error_includes_name():
    with pytest.raises(ValueError, match="myarray"):
        ensure_finite("myarray", np.array([np.nan]))


# ---------------------------------------------------------------------------
# apply_log_mask
# ---------------------------------------------------------------------------

def test_apply_log_mask_no_columns():
    X = np.array([[1.0, 10.0], [2.0, 100.0]])
    mask = np.array([False, False])
    out = apply_log_mask(X, mask, eps=1e-8)
    np.testing.assert_array_equal(out, X)


def test_apply_log_mask_all_columns():
    X = np.array([[10.0, 100.0]])
    mask = np.array([True, True])
    out = apply_log_mask(X, mask, eps=1e-8)
    np.testing.assert_allclose(out, [[1.0, 2.0]])


def test_apply_log_mask_single_column():
    X = np.array([[10.0, 5.0], [100.0, 5.0]])
    mask = np.array([True, False])
    out = apply_log_mask(X, mask, eps=1e-8)
    np.testing.assert_allclose(out[:, 0], [1.0, 2.0])
    np.testing.assert_array_equal(out[:, 1], X[:, 1])


def test_apply_log_mask_clips_non_positive():
    X = np.array([[0.0, -1.0]])
    mask = np.array([True, True])
    eps = 1e-8
    out = apply_log_mask(X, mask, eps=eps)
    expected = np.log10(eps)
    np.testing.assert_allclose(out, [[expected, expected]])


def test_apply_log_mask_does_not_modify_input():
    X = np.array([[10.0, 100.0]])
    X_orig = X.copy()
    mask = np.array([True, True])
    apply_log_mask(X, mask, eps=1e-8)
    np.testing.assert_array_equal(X, X_orig)


def test_apply_log_mask_wrong_mask_shape_raises():
    X = np.array([[1.0, 2.0, 3.0]])
    mask = np.array([True, False])
    with pytest.raises(ValueError, match="log_mask shape"):
        apply_log_mask(X, mask, eps=1e-8)


# ---------------------------------------------------------------------------
# normalize / denormalize
# ---------------------------------------------------------------------------

def test_normalize_zero_mean_unit_std():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    mean = np.array([0.0, 0.0])
    std = np.array([1.0, 1.0])
    np.testing.assert_array_equal(normalize(X, mean, std), X)


def test_normalize_shifts_and_scales():
    X = np.array([[10.0, 20.0]])
    mean = np.array([5.0, 10.0])
    std = np.array([5.0, 10.0])
    out = normalize(X, mean, std)
    np.testing.assert_allclose(out, [[1.0, 1.0]])


def test_denormalize_reverses_normalize():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    mean = np.array([1.5, 2.5])
    std = np.array([0.5, 1.5])
    Xn = normalize(X, mean, std)
    X_recovered = denormalize(Xn, mean, std)
    np.testing.assert_allclose(X_recovered, X)


@given(
    arrays(np.float64, (5, 3), elements=st.floats(-100, 100)),
    arrays(np.float64, 3, elements=st.floats(-10, 10)),
    arrays(np.float64, 3, elements=st.floats(0.1, 10)),
)
@settings(max_examples=200)
def test_normalize_denormalize_roundtrip(X, mean, std):
    Xn = normalize(X, mean, std)
    X_recovered = denormalize(Xn, mean, std)
    np.testing.assert_allclose(X_recovered, X, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# _p95_l2
# ---------------------------------------------------------------------------

def test_p95_l2_identical_arrays_is_zero():
    y = np.ones((10, 3))
    assert _p95_l2(y, y) == 0.0


def test_p95_l2_known_value():
    yhat = np.array([[3.0, 0.0], [0.0, 0.0]])
    y = np.zeros((2, 2))
    result = _p95_l2(yhat, y, percentile=50)
    np.testing.assert_allclose(result, np.percentile([3.0, 0.0], 50))


def test_p95_l2_returns_float():
    yhat = np.random.default_rng(0).random((20, 4))
    y = np.random.default_rng(1).random((20, 4))
    result = _p95_l2(yhat, y)
    assert isinstance(result, float)


def test_p95_l2_custom_percentile():
    rng = np.random.default_rng(42)
    yhat = rng.random((100, 2))
    y = np.zeros((100, 2))
    errors = np.linalg.norm(yhat - y, axis=1)
    expected_50 = float(np.percentile(errors, 50))
    assert _p95_l2(yhat, y, percentile=50) == pytest.approx(expected_50)
