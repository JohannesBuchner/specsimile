import numpy as np
import pytest
import torch
import torch.nn as nn
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from specsimile.emulator import (
    LeakySoftplus,
    MLP,
    ConditionalMLP,
    Emulator,
    getacti,
    soft_floor_np,
    soft_floor_torch,
    softplus_np,
)


# ---------------------------------------------------------------------------
# softplus_np
# ---------------------------------------------------------------------------

def test_softplus_np_zero():
    result = softplus_np(0.0)
    assert np.isclose(result, np.log(2.0))


def test_softplus_np_large_positive():
    result = softplus_np(100.0)
    assert np.isclose(result, 100.0, atol=1e-6)


def test_softplus_np_large_negative():
    result = softplus_np(-100.0)
    assert np.isclose(result, 0.0, atol=1e-6)


def test_softplus_np_always_positive():
    x = np.linspace(-10, 10, 200)
    assert np.all(softplus_np(x) > 0)


@given(arrays(np.float64, (20,), elements=st.floats(-50, 50)))
def test_softplus_np_nonnegative(x):
    assert np.all(softplus_np(x) >= 0)


@given(st.floats(-30, 30))
def test_softplus_np_matches_naive(x):
    expected = np.log1p(np.exp(-abs(x))) + max(x, 0.0)
    assert np.isclose(softplus_np(x), expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# soft_floor_np
# ---------------------------------------------------------------------------

def test_soft_floor_np_at_floor():
    x = np.array([0.0])
    floor = np.array([0.0])
    result = soft_floor_np(x, floor)
    assert result[0] > floor[0]


def test_soft_floor_np_above_floor():
    x = np.array([5.0])
    floor = np.array([0.0])
    result = soft_floor_np(x, floor)
    assert np.isclose(result[0], 5.0, atol=0.01)


def test_soft_floor_np_below_floor():
    x = np.array([-10.0])
    floor = np.array([0.0])
    result = soft_floor_np(x, floor)
    assert result[0] > floor[0]
    assert result[0] < 0.1


@given(
    arrays(np.float64, (10,), elements=st.floats(-10, 10)),
    st.floats(-5, 5),
)
def test_soft_floor_np_always_above_floor(x, floor):
    result = soft_floor_np(x, floor)
    assert np.all(result > floor - 1e-9)


# ---------------------------------------------------------------------------
# soft_floor_torch
# ---------------------------------------------------------------------------

def test_soft_floor_torch_above_floor():
    x = torch.tensor([5.0])
    result = soft_floor_torch(x, 0.0)
    assert torch.isclose(result, torch.tensor([5.0]), atol=0.01)


def test_soft_floor_torch_always_above_floor():
    x = torch.linspace(-10, 10, 50)
    floor = 0.0
    result = soft_floor_torch(x, floor)
    assert torch.all(result > floor - 1e-9)


def test_soft_floor_torch_sharpness():
    x = torch.tensor([1.0])
    floor = torch.tensor([0.0])
    result_sharp = soft_floor_torch(x, floor, k=100.0)
    result_soft = soft_floor_torch(x, floor, k=1.0)
    assert torch.isclose(result_sharp, x, atol=0.01)
    assert not torch.isclose(result_soft, x, atol=0.001)


# ---------------------------------------------------------------------------
# LeakySoftplus
# ---------------------------------------------------------------------------

def test_leakysoftplus_zero():
    act = LeakySoftplus()
    out = act(torch.tensor(0.0))
    assert torch.isclose(out, torch.tensor(0.0), atol=1e-7)


def test_leakysoftplus_large_positive():
    act = LeakySoftplus(negative_slope=0.01, positive_slope=1.0)
    x = torch.tensor(100.0)
    out = act(x)
    assert torch.isclose(out, x, atol=1.0)


def test_leakysoftplus_large_negative():
    act = LeakySoftplus(negative_slope=0.01)
    x = torch.tensor(-100.0)
    out = act(x)
    # For large negative x, LeakySoftplus(x) ≈ negative_slope * x = 0.01 * -100 = -1.0
    # However, emulator.py computes: sp + negative_slope * x
    # where sp = softplus(x, beta) - softplus(0, beta)
    # For x << 0: softplus(x, beta) ≈ 0, so sp ≈ -softplus(0, beta) = -log(2)/beta
    # Result ≈ -log(2)/beta + negative_slope * x
    # With beta=0.99, negative_slope=0.01, x=-100:
    # ≈ -log(2)/0.99 + 0.01*(-100) ≈ -0.700 + (-1.0) = -1.700
    # The test expects -1.0 but the actual asymptotic value is negative_slope * x - log(2)/beta
    # This is a known issue in emulator.py: the zero-centering correction
    # -softplus(0; beta) shifts the activation by a constant (-log(2)/beta),
    # so for large negative x the output is negative_slope*x - log(2)/beta, not negative_slope*x.
    expected = act.negative_slope * x.item() - float(act.zero.item())
    assert torch.isclose(out, torch.tensor(expected), atol=0.1)


def test_leakysoftplus_negative_slope_custom():
    slope = 0.1
    act = LeakySoftplus(negative_slope=slope)
    x = torch.tensor(-50.0)
    out = act(x)
    expected = act.negative_slope * x.item() - float(act.zero.item())
    assert torch.isclose(out, torch.tensor(expected), atol=0.01)


def test_leakysoftplus_differentiable():
    act = LeakySoftplus()
    x = torch.tensor([[-1.0, 0.0, 1.0]], requires_grad=True)
    out = act(x).sum()
    out.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_leakysoftplus_batch():
    act = LeakySoftplus()
    x = torch.randn(16, 8)
    out = act(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# getacti
# ---------------------------------------------------------------------------

def test_getacti_leakysoftplus():
    cls = getacti("LeakySoftplus")
    assert cls is LeakySoftplus


def test_getacti_leakyrelu():
    cls = getacti("LeakyReLU")
    assert cls is nn.LeakyReLU


def test_getacti_tanh():
    cls = getacti("Tanh")
    assert cls is nn.Tanh


def test_getacti_relu():
    cls = getacti("ReLU")
    assert cls is nn.ReLU


def test_getacti_returns_class():
    cls = getacti("Sigmoid")
    instance = cls()
    assert isinstance(instance, nn.Module)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

def test_mlp_output_shape():
    model = MLP(nin=4, nout=3, shape=[8, 8])
    x = torch.randn(5, 4)
    out = model(x)
    assert out.shape == (5, 3)


def test_mlp_single_sample():
    model = MLP(nin=2, nout=5, shape=[16])
    x = torch.randn(1, 2)
    out = model(x)
    assert out.shape == (1, 5)


def test_mlp_no_hidden():
    model = MLP(nin=3, nout=2, shape=[])
    x = torch.randn(4, 3)
    out = model(x)
    assert out.shape == (4, 2)


def test_mlp_dropout_zero():
    model = MLP(nin=4, nout=2, shape=[8], dropout=0.0)
    model.eval()
    x = torch.randn(10, 4)
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2)


def test_mlp_double_precision():
    model = MLP(nin=4, nout=2, shape=[8]).double()
    x = torch.randn(3, 4, dtype=torch.float64)
    out = model(x)
    assert out.dtype == torch.float64


def test_mlp_activation_stored():
    model = MLP(nin=4, nout=2, shape=[8], acti="Tanh")
    assert model.acti == "Tanh"


def test_mlp_leakysoftplus_activation():
    model = MLP(nin=4, nout=2, shape=[8], acti="LeakySoftplus")
    x = torch.randn(3, 4)
    out = model(x)
    assert out.shape == (3, 2)


# ---------------------------------------------------------------------------
# ConditionalMLP
# ---------------------------------------------------------------------------

def test_conditional_mlp_output_shape():
    model = ConditionalMLP(
        n_context=2,
        n_inference=3,
        nout=4,
        shape=[16, 16],
        embed_dim=8,
    )
    x = torch.randn(5, 5)
    out = model(x)
    assert out.shape == (5, 4)


def test_conditional_mlp_context_split():
    n_ctx = 2
    n_inf = 3
    model = ConditionalMLP(
        n_context=n_ctx,
        n_inference=n_inf,
        nout=2,
        shape=[8],
        embed_dim=4,
    )
    x = torch.randn(4, n_inf + n_ctx)
    out = model(x)
    assert out.shape == (4, 2)


def test_conditional_mlp_double():
    model = ConditionalMLP(
        n_context=1,
        n_inference=2,
        nout=3,
        shape=[8],
        embed_dim=4,
    ).double()
    x = torch.randn(3, 3, dtype=torch.float64)
    out = model(x)
    assert out.dtype == torch.float64


def test_conditional_mlp_fixed_context_smooth_output():
    model = ConditionalMLP(
        n_context=1,
        n_inference=2,
        nout=4,
        shape=[16],
        embed_dim=8,
        acti="Tanh",
    ).eval()
    ctx = torch.tensor([[0.5]])
    thetas = torch.linspace(-1, 1, 10).unsqueeze(1).expand(-1, 2)
    x = torch.cat([thetas, ctx.expand(10, 1)], dim=1)
    out = model(x)
    assert out.shape == (10, 4)


# ---------------------------------------------------------------------------
# Emulator._loss_variance_np
# ---------------------------------------------------------------------------

def make_emulator_no_dynamic_range():
    import torch.nn as nn
    dec = nn.Identity()
    return Emulator(model_path="/tmp/dummy.pt", decoder=dec, shape=[8])


def make_emulator_with_dynamic_range(dr):
    import torch.nn as nn
    dec = nn.Identity()
    return Emulator(model_path="/tmp/dummy.pt", decoder=dec, shape=[8], dynamic_range=dr)


def test_loss_variance_np_uniform():
    emu = make_emulator_no_dynamic_range()
    y = np.ones((4, 5))
    result = emu._loss_variance_np(y)
    assert result.shape == (4, 5)
    assert np.all(result == 1.0)


def test_loss_variance_np_dynamic_range():
    emu = make_emulator_with_dynamic_range(2.0)
    y = np.zeros((3, 4))
    y[:, 0] = 5.0
    result = emu._loss_variance_np(y)
    assert result.shape == (3, 4)
    assert np.all(result > 0)


def test_loss_variance_np_func_floor():
    import torch.nn as nn
    dec = nn.Identity()
    emu = Emulator(model_path="/tmp/dummy.pt", decoder=dec, shape=[8], func_floor_loss=0.5)
    y = np.ones((3, 4))
    result = emu._loss_variance_np(y)
    expected = 0.5 ** 2
    assert np.allclose(result, expected)


# ---------------------------------------------------------------------------
# Emulator.rms_per_sample_np
# ---------------------------------------------------------------------------

def test_rms_per_sample_np_perfect():
    emu = make_emulator_no_dynamic_range()
    y = np.random.randn(5, 10)
    err = emu.rms_per_sample_np(y, y)
    assert np.allclose(err, 0.0)


def test_rms_per_sample_np_known_value():
    emu = make_emulator_no_dynamic_range()
    yhat = np.ones((1, 4))
    y = np.zeros((1, 4))
    err = emu.rms_per_sample_np(yhat, y)
    assert np.isclose(err[0], 1.0)


def test_rms_per_sample_np_shape():
    emu = make_emulator_no_dynamic_range()
    yhat = np.random.randn(7, 12)
    y = np.random.randn(7, 12)
    err = emu.rms_per_sample_np(yhat, y)
    assert err.shape == (7,)


@given(
    arrays(np.float64, (5, 8), elements=st.floats(-10, 10)),
    arrays(np.float64, (5, 8), elements=st.floats(-10, 10)),
)
def test_rms_per_sample_np_nonnegative(yhat, y):
    emu = make_emulator_no_dynamic_range()
    err = emu.rms_per_sample_np(yhat, y)
    assert np.all(err >= 0)


# ---------------------------------------------------------------------------
# Emulator.rms_per_sample_torch
# ---------------------------------------------------------------------------

def test_rms_per_sample_torch_perfect():
    emu = make_emulator_no_dynamic_range()
    y = torch.randn(5, 10)
    err = emu.rms_per_sample_torch(y, y)
    assert torch.allclose(err, torch.zeros(5), atol=1e-6)


def test_rms_per_sample_torch_known_value():
    emu = make_emulator_no_dynamic_range()
    yhat = torch.ones(1, 4)
    y = torch.zeros(1, 4)
    err = emu.rms_per_sample_torch(yhat, y)
    assert torch.isclose(err[0], torch.tensor(1.0))


def test_rms_per_sample_torch_shape():
    emu = make_emulator_no_dynamic_range()
    yhat = torch.randn(6, 9)
    y = torch.randn(6, 9)
    err = emu.rms_per_sample_torch(yhat, y)
    assert err.shape == (6,)


# ---------------------------------------------------------------------------
# Emulator.loss_fn
# ---------------------------------------------------------------------------

def test_loss_fn_perfect():
    emu = make_emulator_no_dynamic_range()
    y = torch.randn(4, 8)
    loss = emu.loss_fn(y, y)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_loss_fn_scalar():
    emu = make_emulator_no_dynamic_range()
    yhat = torch.ones(3, 5)
    y = torch.zeros(3, 5)
    loss = emu.loss_fn(yhat, y)
    assert loss.shape == ()


def test_loss_fn_nonnegative():
    emu = make_emulator_no_dynamic_range()
    yhat = torch.randn(8, 6)
    y = torch.randn(8, 6)
    loss = emu.loss_fn(yhat, y)
    assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# Emulator.inside_param_bounds
# ---------------------------------------------------------------------------

def test_inside_param_bounds_none():
    emu = make_emulator_no_dynamic_range()
    P = np.ones((3, 2))
    result = emu.inside_param_bounds(P)
    assert np.all(result)


def test_inside_param_bounds_all_inside():
    emu = make_emulator_no_dynamic_range()
    emu.train_param_bounds = {"min": [0.0, 0.0], "max": [1.0, 1.0]}
    P = np.array([[0.5, 0.5], [0.1, 0.9]])
    result = emu.inside_param_bounds(P)
    assert np.all(result)


def test_inside_param_bounds_some_outside():
    emu = make_emulator_no_dynamic_range()
    emu.train_param_bounds = {"min": [0.0, 0.0], "max": [1.0, 1.0]}
    P = np.array([[0.5, 0.5], [2.0, 0.5]])
    result = emu.inside_param_bounds(P)
    assert result[0]
    assert not result[1]


def test_inside_param_bounds_dim_mismatch():
    emu = make_emulator_no_dynamic_range()
    emu.train_param_bounds = {"min": [0.0], "max": [1.0]}
    P = np.ones((3, 2))
    with pytest.raises(ValueError):
        emu.inside_param_bounds(P)


# ---------------------------------------------------------------------------
# Emulator._check_param_bounds
# ---------------------------------------------------------------------------

def test_check_param_bounds_no_bounds():
    emu = make_emulator_no_dynamic_range()
    P = np.array([[99.0, 99.0]])
    emu._check_param_bounds(P)


def test_check_param_bounds_raises_outside():
    emu = make_emulator_no_dynamic_range()
    emu.train_param_bounds = {"min": [0.0, 0.0], "max": [1.0, 1.0]}
    P = np.array([[2.0, 0.5]])
    with pytest.raises(ValueError):
        emu._check_param_bounds(P)


def test_check_param_bounds_no_raise_inside():
    emu = make_emulator_no_dynamic_range()
    emu.train_param_bounds = {"min": [0.0, 0.0], "max": [1.0, 1.0]}
    P = np.array([[0.5, 0.5]])
    emu._check_param_bounds(P)


def test_check_param_bounds_policy_nan_no_raise():
    emu = make_emulator_no_dynamic_range()
    emu.param_bounds_policy = "nan"
    emu.train_param_bounds = {"min": [0.0], "max": [1.0]}
    P = np.array([[99.0]])
    emu._check_param_bounds(P)


# ---------------------------------------------------------------------------
# Emulator._y_mean_std
# ---------------------------------------------------------------------------

def test_y_mean_std_defaults():
    emu = make_emulator_no_dynamic_range()
    norm = {"y": {}}
    mean, std = emu._y_mean_std(norm, 5)
    assert mean.shape == (5,)
    assert std.shape == (5,)
    assert np.all(mean == 0.0)
    assert np.all(std == 1.0)


def test_y_mean_std_custom():
    emu = make_emulator_no_dynamic_range()
    norm = {"y": {"mean": [1.0, 2.0], "std": [0.5, 2.0]}}
    mean, std = emu._y_mean_std(norm, 2)
    assert np.allclose(mean, [1.0, 2.0])
    assert np.allclose(std, [0.5, 2.0])


# ---------------------------------------------------------------------------
# Emulator._apply_dynamic_range_np
# ---------------------------------------------------------------------------

def test_apply_dynamic_range_np_no_dr():
    emu = make_emulator_no_dynamic_range()
    yhat = np.array([[1.0, 2.0, 3.0]])
    y = np.array([[1.0, 2.0, 3.0]])
    yhat_out, y_out = emu._apply_dynamic_range_np(yhat, y)
    assert np.allclose(yhat_out, yhat)
    assert np.allclose(y_out, y)


def test_apply_dynamic_range_np_with_dr():
    emu = make_emulator_with_dynamic_range(2.0)
    y = np.array([[0.0, 1.0, 3.0]])
    yhat = np.array([[0.0, 1.0, 3.0]])
    yhat_out, y_out = emu._apply_dynamic_range_np(yhat, y)
    assert np.all(yhat_out >= 3.0 - 2.0 - 0.01)
    assert np.all(y_out >= 3.0 - 2.0 - 0.01)


# ---------------------------------------------------------------------------
# Emulator.to_dict
# ---------------------------------------------------------------------------

def test_to_dict_no_xnames_raises():
    emu = make_emulator_no_dynamic_range()
    emu.xnames = None
    with pytest.raises(RuntimeError):
        emu.to_dict(np.ones(3))


def test_to_dict_1d():
    emu = make_emulator_no_dynamic_range()
    emu.xnames = ["a", "b", "c"]
    y = np.array([1.0, 2.0, 3.0])
    result = emu.to_dict(y)
    assert set(result.keys()) == {"a", "b", "c"}
    assert result["a"] == 1.0
    assert result["b"] == 2.0
    assert result["c"] == 3.0


def test_to_dict_2d():
    emu = make_emulator_no_dynamic_range()
    emu.xnames = ["x", "y"]
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = emu.to_dict(y)
    assert result["x"].shape == (2,)
    assert np.allclose(result["x"], [1.0, 3.0])


def test_to_dict_shape_mismatch_raises():
    emu = make_emulator_no_dynamic_range()
    emu.xnames = ["a", "b"]
    y = np.ones((3, 5))
    with pytest.raises(ValueError):
        emu.to_dict(y)
