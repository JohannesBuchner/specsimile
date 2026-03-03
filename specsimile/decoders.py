# specsimile/decoders.py
#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn

ln10 = np.log(10.0)


def sigmoid_asinh(x, lo, hi):
    """Smooth squashing R -> (lo, hi) using asinh."""
    spread = hi - lo
    mid = (lo + hi) / 2.0
    return mid + (spread / 4.0) * torch.asinh(x)


class _XSetterMixin:
    """
    Allow setting decoder.x later via: decoder.x = new_grid
    Stores a float64 torch buffer called _x.
    """
    def _set_x(self, x):
        x = np.asarray(x, float)
        if x.ndim != 1:
            raise ValueError(f"x must be 1D, got {x.shape}")
        t = torch.tensor(x, dtype=torch.float64)
        if hasattr(self, "_x"):
            # overwrite buffer
            self._buffers["_x"] = t
        else:
            self.register_buffer("_x", t)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._set_x(value)


class DenseDecoder(nn.Module):
    """latent_dim = y_dim ; forward(z)=z."""
    paramnames = None

    def __init__(self, y_dim: int):
        super().__init__()
        self.y_dim = int(y_dim)
        self.latent_dim = self.y_dim
        self.paramnames = [f"y{i}" for i in range(self.y_dim)]

    def normalize(self, y: np.ndarray, params: np.ndarray) -> dict:
        y = np.asarray(y, dtype=np.float64)
        params = np.asarray(params, dtype=np.float64)
        if y.ndim != 2 or params.ndim != 2 or y.shape[0] != params.shape[0]:
            raise ValueError("y must be (N,Y), params must be (N,P) with same N")

        return {
            "y": {"log": False, "eps": 1e-30},
            "params": {"log_mask": np.zeros(params.shape[1], bool), "mean": params.mean(axis=0), "std": params.std(axis=0) + 1e-12, "eps": 1e-30},
            "latent_dim": self.latent_dim,
            "latent": {"mean": y.mean(axis=0), "std": y.std(axis=0) + 1e-12, "log_mask": np.zeros(self.latent_dim, bool)},
        }

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def latent_to_params(self, z: torch.Tensor) -> torch.Tensor:
        return z


class LogDenseDecoder(nn.Module):
    """latent_dim=y_dim ; forward(z)=z interpreted as log10(y)."""
    paramnames = None

    def __init__(self, y_dim: int, eps: float = 1e-30):
        super().__init__()
        self.y_dim = int(y_dim)
        self.latent_dim = self.y_dim
        self.eps = float(eps)
        self.paramnames = [f"logy{i}" for i in range(self.y_dim)]

    def normalize(self, y: np.ndarray, params: np.ndarray) -> dict:
        y = np.asarray(y, dtype=np.float64)
        params = np.asarray(params, dtype=np.float64)
        if y.ndim != 2 or params.ndim != 2 or y.shape[0] != params.shape[0]:
            raise ValueError("y must be (N,Y), params must be (N,P) with same N")

        ylog = np.log10(np.clip(y, self.eps, None))
        return {
            "y": {"log": True, "eps": self.eps},
            "params": {"log_mask": np.zeros(params.shape[1], bool), "mean": params.mean(axis=0), "std": params.std(axis=0) + 1e-12, "eps": 1e-30},
            "latent_dim": self.latent_dim,
            "latent": {"mean": ylog.mean(axis=0), "std": ylog.std(axis=0) + 1e-12, "log_mask": np.zeros(self.latent_dim, bool)},
        }

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def evaluate(self, z: torch.Tensor) -> torch.Tensor:
        return torch.pow(10.0, z)

    def latent_to_params(self, z: torch.Tensor) -> torch.Tensor:
        return z


class CutoffPLDecoder(nn.Module, _XSetterMixin):
    """
    Generic cutoff power-law on x-grid:
      y(x) = A * x^{-gamma} * exp(-x/xcut)
    Decoder returns log10(y).

    latent (physical) z_phys = [logA, gamma_raw, logxcut_raw]
      gamma = squash(gamma_raw, gamma_lo, gamma_hi)
      log10(xcut) = squash(logxcut_raw, logxcut_lo, logxcut_hi)
    """
    paramnames = ["logA", "gamma", "xcut"]

    def __init__(
        self,
        *,
        x=None,
        gamma_lo=0.5,
        gamma_hi=6.0,
        xcut_lo=1e-3,
        xcut_hi=1e6,
        eps=1e-30,
    ):
        super().__init__()
        if x is not None:
            self._set_x(x)
        self.eps = float(eps)
        self.latent_dim = 3

        self.gamma_lo = float(gamma_lo)
        self.gamma_hi = float(gamma_hi)
        self.logxcut_lo = float(np.log10(xcut_lo))
        self.logxcut_hi = float(np.log10(xcut_hi))

    def normalize(self, y: np.ndarray, params: np.ndarray) -> dict:
        y = np.asarray(y, np.float64)
        params = np.asarray(params, np.float64)
        if y.ndim != 2 or params.ndim != 2 or y.shape[0] != params.shape[0]:
            raise ValueError("y must be (N,D), params must be (N,P) with same N")

        ylog = np.log10(np.clip(y, self.eps, None))
        amp = np.max(ylog, axis=1)
        amp = amp[np.isfinite(amp)]
        if len(amp) == 0:
            raise ValueError("non-finite y in normalize")

        latent_mean = np.zeros(self.latent_dim)
        latent_std = np.ones(self.latent_dim)
        latent_mean[0] = float(np.median(amp))
        latent_std[0] = float(np.std(amp) + 1e-6)

        return {
            "y": {"log": True, "eps": self.eps},
            "params": {"log_mask": np.zeros(params.shape[1], bool), "mean": params.mean(axis=0), "std": params.std(axis=0) + 1e-12, "eps": 1e-30},
            "latent_dim": self.latent_dim,
            "latent": {"mean": latent_mean, "std": latent_std, "log_mask": np.zeros(self.latent_dim, bool)},
        }

    def latent_to_params(self, z_phys: torch.Tensor):
        z_phys = z_phys.to(torch.float64)
        logA = z_phys[:, 0]
        gamma = sigmoid_asinh(z_phys[:, 1], self.gamma_lo, self.gamma_hi)
        logxcut = sigmoid_asinh(z_phys[:, 2], self.logxcut_lo, self.logxcut_hi)
        xcut = torch.pow(10.0, logxcut)
        return logA, gamma, xcut

    def forward(self, z_phys: torch.Tensor) -> torch.Tensor:
        logA, gamma, xcut = self.latent_to_params(z_phys)
        x = torch.clamp(self.x[None, :], min=1e-300)
        lnA = logA[:, None] * ln10
        lnx = torch.log(x)
        lny = lnA - gamma[:, None] * lnx - (x / xcut[:, None])
        return lny / ln10

    def evaluate(self, z_phys: torch.Tensor) -> torch.Tensor:
        return torch.pow(10.0, self.forward(z_phys))


class SBPLDecoder(nn.Module, _XSetterMixin):
    """
    Generic smooth broken power law (SBPL) on x-grid, returning log10(y).

    Physical latent parameters:
      z_phys = [lognorm, s1_raw, s2_raw, logxbrk_raw, logwidth_raw]

    Mapping:
      s1 = slope_scale * asinh(s1_raw)
      s2 = slope_scale * asinh(s2_raw)
      xbrk = 10**squash(logxbrk_raw, logx_lo-pad, logx_hi+pad)
      width = 10**squash(logwidth_raw, logw_lo, logw_hi)
    Normalized such that y(x0)=10**lognorm.
    """
    paramnames = ["lognorm", "slope1", "slope2", "xbreak", "width"]

    def __init__(
        self,
        *,
        x=None,
        x0=None,
        slope_scale=2.5,
        logx_pad=1.0,
        logwidth_lo=-1.0,
        logwidth_hi=2.0,
        eps=1e-30,
    ):
        super().__init__()
        if x is not None:
            self._set_x(x)
        if x0 is not None:
            self.register_buffer("_x0", torch.tensor(float(x0), dtype=torch.float64))
        self.slope_scale = float(slope_scale)
        self.logx_pad = float(logx_pad)
        self.logwidth_lo = float(logwidth_lo)
        self.logwidth_hi = float(logwidth_hi)
        self.eps = float(eps)
        self.latent_dim = 5

    @property
    def x0(self):
        return getattr(self, '_x0', None)

    def normalize(self, y: np.ndarray, params: np.ndarray) -> dict:
        y = np.asarray(y, np.float64)
        params = np.asarray(params, np.float64)
        if y.ndim != 2 or params.ndim != 2 or y.shape[0] != params.shape[0]:
            raise ValueError("y must be (N,D), params must be (N,P) with same N")

        ylog = np.log10(np.clip(y, self.eps, None))
        amp = np.max(ylog, axis=1)
        amp = amp[np.isfinite(amp)]
        if len(amp) == 0:
            raise ValueError("non-finite y in normalize")

        latent_mean = np.zeros(self.latent_dim)
        latent_std = np.ones(self.latent_dim)
        latent_mean[0] = float(np.median(amp))
        latent_std[0] = float(np.std(amp) + 1e-6)

        return {
            "y": {"log": True, "eps": self.eps},
            "params": {"log_mask": np.zeros(params.shape[1], bool), "mean": params.mean(axis=0), "std": params.std(axis=0) + 1e-12, "eps": 1e-30},
            "latent_dim": self.latent_dim,
            "latent": {"mean": latent_mean, "std": latent_std, "log_mask": np.zeros(self.latent_dim, bool)},
        }

    def latent_to_params(self, z_phys: torch.Tensor):
        z_phys = z_phys.to(torch.float64)
        lognorm = z_phys[:, 0]
        s1 = self.slope_scale * torch.asinh(z_phys[:, 1])
        s2 = self.slope_scale * torch.asinh(z_phys[:, 2])

        x = self.x
        logx_lo = float(np.log10(x.min().item()))
        logx_hi = float(np.log10(x.max().item()))
        logxbrk = sigmoid_asinh(z_phys[:, 3], logx_lo - self.logx_pad, logx_hi + self.logx_pad)
        xbrk = torch.pow(10.0, logxbrk)

        logw = sigmoid_asinh(z_phys[:, 4], self.logwidth_lo, self.logwidth_hi)
        width = torch.pow(10.0, logw)

        return lognorm, s1, s2, xbrk, width

    def forward(self, z_phys: torch.Tensor) -> torch.Tensor:
        lognorm, s1, s2, xbrk, width = self.latent_to_params(z_phys)

        x = torch.clamp(self.x[None, :], min=1e-300)

        # SBPL-like smooth transition in log space
        q = torch.log(x / xbrk[:, None]) / width[:, None]
        q = torch.clamp(q, -50.0, 50.0)
        if self.x0 is not None:
            x0 = torch.clamp(self.x0, min=1e-300)
            qpiv = torch.log(x0 / xbrk) / width
            qpiv = torch.clamp(qpiv, -50.0, 50.0)
            den = torch.exp(qpiv[:, None]) + torch.exp(-qpiv[:, None])
            xrel = x / x0
        else:
            den = 1 + 1
            x0 = xbrk[:, None]
            xrel = x / x0

        num = torch.exp(q) + torch.exp(-q)
        ratio = num / (den + 1e-30)

        expo = ((s2 - s1) / 2.0)[:, None] * width[:, None]
        expo = torch.clamp(expo, -10.0, 10.0)

        norm = torch.pow(10.0, lognorm)
        y = norm[:, None] * xrel ** (((s1 + s2 + 2.0) / 2.0)[:, None]) * (ratio ** expo) * xrel
        y = torch.clamp(y, min=self.eps)
        return torch.log10(y)

    def evaluate(self, z_phys: torch.Tensor) -> torch.Tensor:
        return torch.pow(10.0, self.forward(z_phys))


class QuadraticDecoder(nn.Module, _XSetterMixin):
    """
    Single quadratic (in log10 x) peak shape, returning log10(y).

    Model:
      t = log10(x / x_peak)
      log10(y) = logA - (t / w)**2

    Physical latent:
      z_phys = [logA, logxpeak_raw, logw_raw]

    logxpeak is squashed to [log10(xmin), log10(xmax)]
    logw squashed to [logw_lo, logw_hi]
    Normalization is at the peak by construction: y(x_peak)=10**logA.

    This is the generic replacement for your torus log-quadratic (single component).
    """
    paramnames = ["logA", "x_peak", "sigma"]

    def __init__(self, x=None, *, logw_lo=-1.0, logw_hi=1.0, logx_pad=1, eps=1e-30):
        super().__init__()
        if x is not None:
            self._set_x(x)
        self.logw_lo = float(logw_lo)
        self.logw_hi = float(logw_hi)
        self.eps = float(eps)
        self.latent_dim = 3
        self.logx_pad = float(logx_pad)

    def normalize(self, y: np.ndarray, params: np.ndarray) -> dict:
        y = np.asarray(y, np.float64)
        params = np.asarray(params, np.float64)
        if y.ndim != 2 or params.ndim != 2 or y.shape[0] != params.shape[0]:
            raise ValueError("y must be (N,D), params must be (N,P) with same N")

        ylog = np.log10(np.clip(y, self.eps, None))
        amp = np.max(ylog, axis=1)
        amp = amp[np.isfinite(amp)]
        if len(amp) == 0:
            raise ValueError("non-finite y in normalize")

        latent_mean = np.zeros(self.latent_dim)
        latent_std = np.ones(self.latent_dim)
        latent_mean[0] = float(np.median(amp))
        latent_std[0] = float(np.std(amp) + 1e-6)

        return {
            "y": {"log": True, "eps": self.eps},
            "params": {"log_mask": np.zeros(params.shape[1], bool), "mean": params.mean(axis=0), "std": params.std(axis=0) + 1e-12, "eps": 1e-30},
            "latent_dim": self.latent_dim,
            "latent": {"mean": latent_mean, "std": latent_std, "log_mask": np.zeros(self.latent_dim, bool)},
        }

    def latent_to_params(self, z_phys: torch.Tensor):
        z_phys = z_phys.to(torch.float64)
        logA = z_phys[:, 0]

        x = self.x
        logx_lo = float(np.log10(x.min().item())) - self.logx_pad
        logx_hi = float(np.log10(x.max().item())) + self.logx_pad
        logxpeak = sigmoid_asinh(z_phys[:, 1], logx_lo, logx_hi)
        xpeak = torch.pow(10.0, logxpeak)

        logw = sigmoid_asinh(z_phys[:, 2], self.logw_lo, self.logw_hi)
        w = torch.pow(10.0, logw)

        return logA, xpeak, w

    def forward(self, z_phys: torch.Tensor) -> torch.Tensor:
        logA, xpeak, w = self.latent_to_params(z_phys)
        x = torch.clamp(self.x[None, :], min=1e-300)
        t = torch.log10(x / xpeak[:, None])
        logy = logA[:, None] - (t / w[:, None]) ** 2
        return logy

    def evaluate(self, z_phys: torch.Tensor) -> torch.Tensor:
        return torch.pow(10.0, self.forward(z_phys))
