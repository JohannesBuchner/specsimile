# specsimile/decoders.py
#!/usr/bin/env python3
import json
import hashlib
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union

ln10 = np.log(10.0)


def sigmoid_asinh(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Smooth squashing R -> (lo, hi) using asinh.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of arbitrary shape.
    lo : float
        Lower bound of the output range.
    hi : float
        Upper bound of the output range.

    Returns
    -------
    torch.Tensor
        Squashed tensor with values in (lo, hi), same shape as ``x``.
    """
    spread = hi - lo
    mid = (lo + hi) / 2.0
    return mid + (spread / 4.0) * torch.asinh(x)


class _XSetterMixin:
    """
    Allow setting decoder.x later via: decoder.x = new_grid
    Stores a float64 torch buffer called _x.
    """
    def _set_x(self, x: Union[np.ndarray, list]) -> None:
        """Register or overwrite the x-grid buffer.

        Parameters
        ----------
        x : array-like
            1-D array of x-grid values. Will be cast to float64.

        Raises
        ------
        ValueError
            If ``x`` is not 1-D.
        """
        x = np.asarray(x, float)
        if x.ndim != 1:
            raise ValueError(f"x must be 1D, got {x.shape}")
        t = torch.tensor(x, dtype=torch.float64)
        if hasattr(self, "_x"):
            self._buffers["_x"] = t
        else:
            self.register_buffer("_x", t)

    @property
    def x(self) -> torch.Tensor:
        """torch.Tensor: The registered x-grid buffer (float64, 1-D)."""
        return self._x

    @x.setter
    def x(self, value: Union[np.ndarray, list]) -> None:
        self._set_x(value)


class DecoderConfigMixin:
    """
    Provide a JSON-serializable decoder config for checkpointing.
    Only include init args that affect normalize() or latent_to_params().
    Do NOT include x (grid) because x is dataset/user provided at inference.
    """
    def get_config(self) -> dict:
        """Return a JSON-serializable configuration dictionary.

        Returns
        -------
        dict
            Dictionary of constructor arguments that affect
            ``normalize()`` or ``latent_to_params()``.
        """
        return {}

    def config_hash(self) -> str:
        """Return a SHA-256 hex digest of the serialized configuration.

        Returns
        -------
        str
            64-character hexadecimal string uniquely identifying the config.
        """
        s = json.dumps(self.get_config(), sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _normalize_params_stats(params: np.ndarray) -> dict:
    """Compute normalisation statistics for physical parameters.

    Parameters
    ----------
    params : np.ndarray
        Physical input parameters, shape ``(N, P)``.

    Returns
    -------
    dict
        Dictionary with keys ``"log_mask"``, ``"mean"``, ``"std"``,
        and ``"eps"``.
    """
    return {
        "log_mask": np.zeros(params.shape[1], bool),
        "mean": params.mean(axis=0),
        "std": params.std(axis=0) + 1e-12,
        "eps": 1e-30,
    }


def _normalize_amplitude_latent(latent_dim: int, ylog: np.ndarray) -> dict:
    """Compute normalisation statistics for a latent space where the first
    dimension encodes the log10-amplitude (peak value).

    Parameters
    ----------
    latent_dim : int
        Total number of latent dimensions.
    ylog : np.ndarray
        Log10-transformed spectra, shape ``(N, D)``.

    Returns
    -------
    dict
        Dictionary with keys ``"dim"``, ``"mean"``, ``"std"``, and
        ``"log_mask"``.

    Raises
    ------
    ValueError
        If all per-sample peak amplitudes are non-finite.
    """
    amp = np.max(ylog, axis=1)
    amp = amp[np.isfinite(amp)]
    if len(amp) == 0:
        raise ValueError("non-finite y in normalize")
    latent_mean = np.zeros(latent_dim)
    latent_std = np.ones(latent_dim)
    latent_mean[0] = float(np.median(amp))
    latent_std[0] = float(np.std(amp) + 1e-6)
    return {
        "dim": latent_dim,
        "mean": latent_mean,
        "std": latent_std,
        "log_mask": np.zeros(latent_dim, bool),
    }


def _check_normalize_inputs(y: np.ndarray, params: np.ndarray) -> None:
    """Validate shapes of ``y`` and ``params`` for ``normalize()``.

    Parameters
    ----------
    y : np.ndarray
        Observed spectra / outputs, shape ``(N, Y)``.
    params : np.ndarray
        Physical input parameters, shape ``(N, P)``.

    Raises
    ------
    ValueError
        If ``y`` or ``params`` are not 2-D, or their leading dimensions
        differ.
    """
    if y.ndim != 2 or params.ndim != 2 or y.shape[0] != params.shape[0]:
        raise ValueError("y must be (N,D), params must be (N,P) with same N")


def _infer_logx_bounds_from_x(
    x: np.ndarray,
    logx_lo: Optional[float],
    logx_hi: Optional[float],
    current_lo: Optional[float],
    current_hi: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Infer missing log10 x-bounds from an x-grid array.

    Parameters
    ----------
    x : np.ndarray
        1-D x-grid array.
    logx_lo : float or None
        Explicitly supplied lower bound (takes priority).
    logx_hi : float or None
        Explicitly supplied upper bound (takes priority).
    current_lo : float or None
        Currently stored lower bound on the object.
    current_hi : float or None
        Currently stored upper bound on the object.

    Returns
    -------
    tuple of (float or None, float or None)
        Updated ``(logx_lo, logx_hi)`` with any inferrable values filled in.
    """
    x = np.asarray(x, float)
    if logx_lo is None and current_lo is None:
        logx_lo = float(np.log10(x.min()))
    if logx_hi is None and current_hi is None:
        logx_hi = float(np.log10(x.max()))
    return logx_lo, logx_hi


class DenseDecoder(nn.Module, DecoderConfigMixin):
    """latent_dim = y_dim ; forward(z)=z."""

    def __init__(self, y_dim: Optional[int] = None, paramnames: Optional[List[str]] = None) -> None:
        """Initialise a pass-through dense decoder.

        Parameters
        ----------
        y_dim : int, optional
            Dimensionality of the output (and latent) space.
        paramnames : list of str, optional
            Names for each output parameter. Defaults to
            ``["y0", "y1", ...]`` when not provided.

        Raises
        ------
        ValueError
            If ``paramnames`` is provided but its length differs from
            ``y_dim``.
        """
        super().__init__()
        if y_dim is not None:
            self.configure(y_dim=y_dim, paramnames=paramnames)

    def configure(self, y_dim: int, paramnames: List[str] = None, **kwargs) -> None:
        """Configure the dense decoder after construction.

        Parameters
        ----------
        y_dim : int
            Dimensionality of the output (and latent) space.
        paramnames : list of str, optional
            Names for each output parameter. Defaults to
            ``["y0", "y1", ...]`` when not provided.
        **kwargs
            Additional keyword arguments are accepted and ignored.

        Raises
        ------
        ValueError
            If ``paramnames`` is provided but its length differs from
            ``y_dim``.
        """
        self.y_dim = int(y_dim)
        self.latent_dim = self.y_dim
        if paramnames is None:
            self.paramnames = [f"y{i}" for i in range(self.y_dim)]
        else:
            self.paramnames = paramnames
        if len(self.paramnames) != self.y_dim:
            raise ValueError(f"paramnames={paramnames} must be same length as y_dim={y_dim}")

    def normalize(self, y: np.ndarray, params: np.ndarray) -> dict:
        """Compute normalisation statistics from training data.

        Parameters
        ----------
        y : np.ndarray
            Observed spectra / outputs, shape ``(N, Y)``.
        params : np.ndarray
            Physical input parameters, shape ``(N, P)``.

        Returns
        -------
        dict
            Nested dictionary with keys ``"params"``, ``"latent"``, and
            ``"y"``, each containing the statistics required to standardise
            that quantity at training and inference time.

        Raises
        ------
        ValueError
            If ``y`` or ``params`` are not 2-D, or their leading dimensions
            differ.
        """
        y = np.asarray(y, dtype=np.float64)
        params = np.asarray(params, dtype=np.float64)
        _check_normalize_inputs(y, params)

        if not hasattr(self, 'y_dim') or self.y_dim is None:
            self.configure(y_dim=y.shape[1])

        return {
            "params": _normalize_params_stats(params),
            "latent": {
                "dim": self.latent_dim,
                "mean": y.mean(axis=0),
                "std": y.std(axis=0) + 1e-12,
                "log_mask": np.zeros(self.latent_dim, bool),
            },
            "y": {"log": False, "eps": 1e-30},
        }

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Pass latent vector through unchanged.

        Parameters
        ----------
        z : torch.Tensor
            Latent tensor of shape ``(N, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Same tensor ``z``, shape ``(N, latent_dim)``.
        """
        return z

    def latent_to_params(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent codes to physical parameters (identity here).

        Parameters
        ----------
        z : torch.Tensor
            Latent tensor of shape ``(N, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Physical parameter tensor, shape ``(N, latent_dim)``.
        """
        return z

    def get_config(self) -> dict:
        """Return serialisable constructor arguments.

        Returns
        -------
        dict
            Dictionary with keys ``"y_dim"`` and ``"paramnames"``.
        """
        return dict(
            y_dim=int(self.y_dim),
            paramnames=list(self.paramnames),
        )

class LogDenseDecoder(DenseDecoder):
    """latent_dim=y_dim ; forward(z)=z interpreted as log10(y)."""

    def __init__(self, y_dim: Optional[int] = None, eps: float = 1e-30, paramnames: Optional[List[str]] = None) -> None:
        """Initialise a log-space dense decoder.

        Parameters
        ----------
        y_dim : int, optional
            Dimensionality of the output (and latent) space.
        eps : float, optional
            Small floor applied before taking ``log10`` to avoid
            ``log(0)``. Default is ``1e-30``.
        paramnames : list of str, optional
            Names for each output parameter. Defaults to
            ``["logy0", "logy1", ...]`` when not provided.
        """
        self.eps = float(eps)
        if y_dim is not None:
            super().__init__(
                y_dim=y_dim,
                paramnames=paramnames or [f"logy{i}" for i in range(y_dim)])
        else:
            super().__init__()

    def configure(self, y_dim: int, paramnames: List[str] = None, **kwargs) -> None:
        """Configure the log-space dense decoder after construction.

        Parameters
        ----------
        y_dim : int
            Dimensionality of the output (and latent) space.
        paramnames : list of str, optional
            Names for each output parameter. Defaults to
            ``["logy0", "logy1", ...]`` when not provided.
        **kwargs
            Additional keyword arguments are accepted and ignored.
        """
        super().configure(
            y_dim=y_dim,
            paramnames=paramnames or [f"logy{i}" for i in range(y_dim)])

    def normalize(self, y: np.ndarray, params: np.ndarray) -> dict:
        """Compute normalisation statistics in log10 space.

        Parameters
        ----------
        y : np.ndarray
            Observed spectra / outputs (linear scale), shape ``(N, Y)``.
        params : np.ndarray
            Physical input parameters, shape ``(N, P)``.

        Returns
        -------
        dict
            Nested dictionary with keys ``"params"``, ``"latent"``, and
            ``"y"``, where ``"y"["log"]`` is ``True``.

        Raises
        ------
        ValueError
            If ``y`` or ``params`` are not 2-D, or their leading dimensions
            differ.
        """
        y = np.asarray(y, dtype=np.float64)
        params = np.asarray(params, dtype=np.float64)
        _check_normalize_inputs(y, params)

        if not hasattr(self, 'y_dim') or self.y_dim is None:
            self.configure(y_dim=y.shape[1])

        ylog = np.log10(np.clip(y, self.eps, None))
        return {
            "params": _normalize_params_stats(params),
            "latent": {
                "dim": self.latent_dim,
                "mean": ylog.mean(axis=0),
                "std": ylog.std(axis=0) + 1e-12,
                "log_mask": np.zeros(self.latent_dim, bool),
            },
            "y": {"log": True, "eps": self.eps},
        }

    def evaluate(self, z: torch.Tensor) -> torch.Tensor:
        """Convert log10-space latent codes to linear-scale outputs.

        Parameters
        ----------
        z : torch.Tensor
            Latent tensor interpreted as ``log10(y)``, shape
            ``(N, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Linear-scale output tensor, shape ``(N, latent_dim)``.
        """
        return torch.pow(10.0, z)

    def get_config(self) -> dict:
        """Return serialisable constructor arguments.

        Returns
        -------
        dict
            Dictionary extending the parent config with key ``"eps"``.
        """
        cfg = super().get_config()
        cfg.update(dict(eps=float(self.eps)))
        return cfg


class CutoffPLDecoder(nn.Module, _XSetterMixin, DecoderConfigMixin):
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
        x: Optional[np.ndarray] = None,
        gamma_lo: float = 0.5,
        gamma_hi: float = 6.0,
        xcut_lo: float = 1e-3,
        xcut_hi: float = 1e6,
        eps: float = 1e-30,
    ) -> None:
        """Initialise a cutoff power-law decoder.

        Parameters
        ----------
        x : array-like, optional
            1-D x-grid on which the spectrum is evaluated. Can be set
            later via ``decoder.x = new_grid``.
        gamma_lo : float, optional
            Lower bound for the squashed spectral index ``gamma``.
            Default is ``0.5``.
        gamma_hi : float, optional
            Upper bound for the squashed spectral index ``gamma``.
            Default is ``6.0``.
        xcut_lo : float, optional
            Lower physical bound for the cutoff ``xcut`` (linear scale).
            Default is ``1e-3``.
        xcut_hi : float, optional
            Upper physical bound for the cutoff ``xcut`` (linear scale).
            Default is ``1e6``.
        eps : float, optional
            Floor applied before ``log10`` to avoid ``log(0)``.
            Default is ``1e-30``.
        """
        super().__init__()
        self.eps = float(eps)
        self.latent_dim = 3

        self.gamma_lo = float(gamma_lo)
        self.gamma_hi = float(gamma_hi)
        self.logxcut_lo = float(np.log10(xcut_lo))
        self.logxcut_hi = float(np.log10(xcut_hi))
        if x is not None:
            self._set_x(x)

    def configure(
        self,
        *,
        x: Optional[np.ndarray] = None,
        gamma_lo: Optional[float] = None,
        gamma_hi: Optional[float] = None,
        xcut_lo: Optional[float] = None,
        xcut_hi: Optional[float] = None,
        eps: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Configure the cutoff power-law decoder after construction.

        Parameters
        ----------
        x : array-like, optional
            1-D x-grid on which the spectrum is evaluated.
        gamma_lo : float, optional
            Lower bound for the squashed spectral index ``gamma``.
        gamma_hi : float, optional
            Upper bound for the squashed spectral index ``gamma``.
        xcut_lo : float, optional
            Lower physical bound for the cutoff ``xcut`` (linear scale).
        xcut_hi : float, optional
            Upper physical bound for the cutoff ``xcut`` (linear scale).
        eps : float, optional
            Floor applied before ``log10`` to avoid ``log(0)``.
        **kwargs
            Additional keyword arguments are accepted and ignored.
        """
        if x is not None:
            self._set_x(x)
        if eps is not None:
            self.eps = float(eps)
        if gamma_lo is not None:
            self.gamma_lo = float(gamma_lo)
        if gamma_hi is not None:
            self.gamma_hi = float(gamma_hi)
        if xcut_lo is not None:
            self.logxcut_lo = float(np.log10(xcut_lo))
        if xcut_hi is not None:
            self.logxcut_hi = float(np.log10(xcut_hi))

    def normalize(self, y: np.ndarray, params: np.ndarray) -> dict:
        """Compute normalisation statistics for the cutoff power-law model.

        Parameters
        ----------
        y : np.ndarray
            Observed spectra (linear scale), shape ``(N, D)``.
        params : np.ndarray
            Physical input parameters, shape ``(N, P)``.

        Returns
        -------
        dict
            Nested dictionary with keys ``"params"``, ``"latent"``, and
            ``"y"``.

        Raises
        ------
        ValueError
            If ``y`` or ``params`` are not 2-D, their leading dimensions
            differ, or all amplitudes are non-finite.
        """
        y = np.asarray(y, np.float64)
        params = np.asarray(params, np.float64)
        _check_normalize_inputs(y, params)
        ylog = np.log10(np.clip(y, self.eps, None))
        return {
            "params": _normalize_params_stats(params),
            "latent": _normalize_amplitude_latent(self.latent_dim, ylog),
            "y": {"log": True, "eps": self.eps},
        }

    def latent_to_params(self, z_phys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map raw latent codes to physical parameters.

        Parameters
        ----------
        z_phys : torch.Tensor
            Raw latent tensor of shape ``(N, 3)`` with columns
            ``[logA, gamma_raw, logxcut_raw]``.

        Returns
        -------
        logA : torch.Tensor
            Log10-amplitude, shape ``(N,)``.
        gamma : torch.Tensor
            Squashed spectral index in ``(gamma_lo, gamma_hi)``,
            shape ``(N,)``.
        xcut : torch.Tensor
            Cutoff value in linear scale, shape ``(N,)``.
        """
        z_phys = z_phys.to(torch.float64)
        logA = z_phys[:, 0]
        gamma = sigmoid_asinh(z_phys[:, 1], self.gamma_lo, self.gamma_hi)
        logxcut = sigmoid_asinh(z_phys[:, 2], self.logxcut_lo, self.logxcut_hi)
        xcut = torch.pow(10.0, logxcut)
        return logA, gamma, xcut

    def forward(self, z_phys: torch.Tensor) -> torch.Tensor:
        """Evaluate the cutoff power-law model and return ``log10(y)``.

        Parameters
        ----------
        z_phys : torch.Tensor
            Raw latent tensor of shape ``(N, 3)``.

        Returns
        -------
        torch.Tensor
            Log10-spectrum evaluated on the x-grid, shape ``(N, len(x))``.
        """
        logA, gamma, xcut = self.latent_to_params(z_phys)
        x = torch.clamp(self.x[None, :], min=1e-300)
        lnA = logA[:, None] * ln10
        lnx = torch.log(x)
        lny = lnA - gamma[:, None] * lnx - (x / xcut[:, None])
        return lny / ln10

    def evaluate(self, z_phys: torch.Tensor) -> torch.Tensor:
        """Evaluate the cutoff power-law model and return linear-scale ``y``.

        Parameters
        ----------
        z_phys : torch.Tensor
            Raw latent tensor of shape ``(N, 3)``.

        Returns
        -------
        torch.Tensor
            Linear-scale spectrum evaluated on the x-grid,
            shape ``(N, len(x))``.
        """
        return torch.pow(10.0, self.forward(z_phys))

    def get_config(self) -> dict:
        """Return serialisable constructor arguments.

        Returns
        -------
        dict
            Dictionary with keys ``"gamma_lo"``, ``"gamma_hi"``,
            ``"xcut_lo"``, ``"xcut_hi"``, and ``"eps"``.
        """
        return dict(
            gamma_lo=float(self.gamma_lo),
            gamma_hi=float(self.gamma_hi),
            xcut_lo=float(10.0 ** self.logxcut_lo),
            xcut_hi=float(10.0 ** self.logxcut_hi),
            eps=float(self.eps),
        )


class SBPLDecoder(nn.Module, _XSetterMixin, DecoderConfigMixin):
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
        logx_lo: Optional[float] = None,
        logx_hi: Optional[float] = None,
        *,
        x: Optional[np.ndarray] = None,
        x0: Optional[float] = None,
        slope_scale: float = 2.5,
        logwidth_lo: float = -1.0,
        logwidth_hi: float = 2.0,
        eps: float = 1e-30,
    ) -> None:
        """Initialise a smooth broken power-law decoder.

        Parameters
        ----------
        logx_lo : float, optional
            Lower bound of the log10-break-position range. Inferred from
            ``x`` if not provided.
        logx_hi : float, optional
            Upper bound of the log10-break-position range. Inferred from
            ``x`` if not provided.
        x : array-like, optional
            1-D x-grid on which the spectrum is evaluated. Can be set
            later via ``decoder.x = new_grid``.
        x0 : float, optional
            Pivot point at which ``y(x0) = 10**lognorm``. Defaults to
            ``xbrk`` when not provided.
        slope_scale : float, optional
            Scaling factor applied to ``asinh`` of the raw slope
            parameters. Default is ``2.5``.
        logwidth_lo : float, optional
            Lower bound for the log10-width of the transition region.
            Default is ``-1.0``.
        logwidth_hi : float, optional
            Upper bound for the log10-width of the transition region.
            Default is ``2.0``.
        eps : float, optional
            Floor applied before ``log10``. Default is ``1e-30``.
        """
        super().__init__()
        self.slope_scale = float(slope_scale)
        self.logwidth_lo = float(logwidth_lo)
        self.logwidth_hi = float(logwidth_hi)
        self.eps = float(eps)
        self.latent_dim = 5
        self.logx_lo = None
        self.logx_hi = None
        if x is not None:
            self._set_x(x)
            logx_lo, logx_hi = _infer_logx_bounds_from_x(x, logx_lo, logx_hi, self.logx_lo, self.logx_hi)
        if logx_lo is not None:
            self.logx_lo = float(logx_lo)
        if logx_hi is not None:
            self.logx_hi = float(logx_hi)
        if x0 is not None:
            self.register_buffer("_x0", torch.tensor(float(x0), dtype=torch.float64))

    def configure(
        self,
        *,
        x: Optional[np.ndarray] = None,
        logx_lo: Optional[float] = None,
        logx_hi: Optional[float] = None,
        x0: Optional[float] = None,
        slope_scale: Optional[float] = None,
        logwidth_lo: Optional[float] = None,
        logwidth_hi: Optional[float] = None,
        eps: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Configure the SBPL decoder after construction.

        Parameters
        ----------
        x : array-like, optional
            1-D x-grid on which the spectrum is evaluated.
        logx_lo : float, optional
            Lower bound of the log10-break-position range. Inferred from
            ``x`` if not provided.
        logx_hi : float, optional
            Upper bound of the log10-break-position range. Inferred from
            ``x`` if not provided.
        x0 : float, optional
            Pivot point at which ``y(x0) = 10**lognorm``.
        slope_scale : float, optional
            Scaling factor applied to ``asinh`` of the raw slope parameters.
        logwidth_lo : float, optional
            Lower bound for the log10-width of the transition region.
        logwidth_hi : float, optional
            Upper bound for the log10-width of the transition region.
        eps : float, optional
            Floor applied before ``log10``.
        **kwargs
            Additional keyword arguments are accepted and ignored.
        """
        if x is not None:
            self._set_x(x)
            logx_lo, logx_hi = _infer_logx_bounds_from_x(x, logx_lo, logx_hi, self.logx_lo, self.logx_hi)
        if logx_lo is not None:
            self.logx_lo = float(logx_lo)
        if logx_hi is not None:
            self.logx_hi = float(logx_hi)
        if x0 is not None:
            self.register_buffer("_x0", torch.tensor(float(x0), dtype=torch.float64))
        if slope_scale is not None:
            self.slope_scale = float(slope_scale)
        if logwidth_lo is not None:
            self.logwidth_lo = float(logwidth_lo)
        if logwidth_hi is not None:
            self.logwidth_hi = float(logwidth_hi)
        if eps is not None:
            self.eps = float(eps)

    @property
    def x0(self) -> Optional[torch.Tensor]:
        """torch.Tensor or None: Registered pivot point buffer (scalar, float64)."""
        return getattr(self, '_x0', None)

    def _ensure_logx_bounds(self) -> None:
        """Infer logx bounds from the registered x-grid if not yet set.

        Raises
        ------
        ValueError
            If ``logx_lo`` or ``logx_hi`` are missing and no x-grid is
            available to infer them from.
        """
        if self.logx_lo is None or self.logx_hi is None:
            if hasattr(self, '_x') and self._x is not None:
                xarr = self._x.detach().cpu().numpy()
                if self.logx_lo is None:
                    self.logx_lo = float(np.log10(xarr.min()))
                if self.logx_hi is None:
                    self.logx_hi = float(np.log10(xarr.max()))
            else:
                raise ValueError("logx_lo and logx_hi must be set before normalize()")

    def normalize(self, y: np.ndarray, params: np.ndarray) -> dict:
        """Compute normalisation statistics for the SBPL model.

        Parameters
        ----------
        y : np.ndarray
            Observed spectra (linear scale), shape ``(N, D)``.
        params : np.ndarray
            Physical input parameters, shape ``(N, P)``.

        Returns
        -------
        dict
            Nested dictionary with keys ``"params"``, ``"latent"``, and
            ``"y"``.

        Raises
        ------
        ValueError
            If ``y`` or ``params`` are not 2-D, their leading dimensions
            differ, or all amplitudes are non-finite.
        """
        y = np.asarray(y, np.float64)
        params = np.asarray(params, np.float64)
        _check_normalize_inputs(y, params)
        self._ensure_logx_bounds()
        ylog = np.log10(np.clip(y, self.eps, None))
        return {
            "params": _normalize_params_stats(params),
            "latent": _normalize_amplitude_latent(self.latent_dim, ylog),
            "y": {"log": True, "eps": self.eps},
        }

    def latent_to_params(
        self, z_phys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map raw latent codes to physical SBPL parameters.

        Parameters
        ----------
        z_phys : torch.Tensor
            Raw latent tensor of shape ``(N, 5)`` with columns
            ``[lognorm, s1_raw, s2_raw, logxbrk_raw, logwidth_raw]``.

        Returns
        -------
        lognorm : torch.Tensor
            Log10-normalisation, shape ``(N,)``.
        s1 : torch.Tensor
            Slope below the break, shape ``(N,)``.
        s2 : torch.Tensor
            Slope above the break, shape ``(N,)``.
        xbrk : torch.Tensor
            Break position in linear scale, shape ``(N,)``.
        width : torch.Tensor
            Transition width in linear scale, shape ``(N,)``.
        """
        z_phys = z_phys.to(torch.float64)
        lognorm = z_phys[:, 0]
        s1 = self.slope_scale * torch.asinh(z_phys[:, 1])
        s2 = self.slope_scale * torch.asinh(z_phys[:, 2])

        logxbrk = sigmoid_asinh(z_phys[:, 3], self.logx_lo, self.logx_hi)
        xbrk = torch.pow(10.0, logxbrk)

        logw = sigmoid_asinh(z_phys[:, 4], self.logwidth_lo, self.logwidth_hi)
        width = torch.pow(10.0, logw)

        return lognorm, s1, s2, xbrk, width

    def forward(self, z_phys: torch.Tensor) -> torch.Tensor:
        """Evaluate the SBPL model and return ``log10(y)``.

        Parameters
        ----------
        z_phys : torch.Tensor
            Raw latent tensor of shape ``(N, 5)``.

        Returns
        -------
        torch.Tensor
            Log10-spectrum evaluated on the x-grid, shape ``(N, len(x))``.
        """
        lognorm, s1, s2, xbrk, width = self.latent_to_params(z_phys)

        x = torch.clamp(self.x[None, :], min=1e-300)

        q = torch.log(x / xbrk[:, None]) / width[:, None]
        q = torch.clamp(q, -50.0, 50.0)
        if self.x0 is not None:
            x0 = torch.clamp(self.x0, min=1e-300)
            qpiv = torch.log(x0 / xbrk) / width
            qpiv = torch.clamp(qpiv, -50.0, 50.0)
            den = torch.exp(qpiv[:, None]) + torch.exp(-qpiv[:, None])
        else:
            den = 1 + 1
            x0 = xbrk[:, None]
        xrel = x / x0

        num = torch.exp(q) + torch.exp(-q)
        ratio = num / (den + 1e-30)

        expo = ((s2 - s1) / 2.0)[:, None] * width[:, None]
        expo = torch.clamp(expo, -10.0, 10.0)

        norm = torch.pow(10.0, lognorm)
        y = norm[:, None] * xrel ** (((s1 + s2 + 2.0) / 2.0)[:, None]) * (ratio ** expo) / xrel
        y = torch.clamp(y, min=self.eps)
        return torch.log10(y)

    def evaluate(self, z_phys: torch.Tensor) -> torch.Tensor:
        """Evaluate the SBPL model and return linear-scale ``y``.

        Parameters
        ----------
        z_phys : torch.Tensor
            Raw latent tensor of shape ``(N, 5)``.

        Returns
        -------
        torch.Tensor
            Linear-scale spectrum evaluated on the x-grid,
            shape ``(N, len(x))``.
        """
        return torch.pow(10.0, self.forward(z_phys))

    def get_config(self) -> dict:
        """Return serialisable constructor arguments.

        Returns
        -------
        dict
            Dictionary with keys ``"logx_lo"``, ``"logx_hi"``,
            ``"slope_scale"``, ``"logwidth_lo"``, ``"logwidth_hi"``,
            ``"x0"``, and ``"eps"``.
        """
        return dict(
            logx_lo=float(self.logx_lo),
            logx_hi=float(self.logx_hi),
            slope_scale=float(self.slope_scale),
            logwidth_lo=float(self.logwidth_lo),
            logwidth_hi=float(self.logwidth_hi),
            x0=None if self.x0 is None else float(self.x0.detach().cpu().item()),
            eps=float(self.eps),
        )

class QuadraticDecoder(nn.Module, _XSetterMixin, DecoderConfigMixin):
    """
    Single quadratic (in log10 x) peak shape, returning log10(y).

    Model:
      t = log10(x / x_peak)
      ln(y) = logA - (t / w)**2

    Physical latent:
      z_phys = [logA, logxpeak_raw, logw_raw]

    logxpeak is squashed to [log10(xmin), log10(xmax)]
    logw squashed to [logw_lo, logw_hi]
    Normalization is at the peak by construction: y(x_peak)=exp(logA).

    This is the generic replacement for the GRAHSP torus ln-quadratic (single component).
    """
    paramnames = ["logA", "x_peak", "sigma"]

    def __init__(
        self,
        logx_lo: Optional[float] = None,
        logx_hi: Optional[float] = None,
        x: Optional[np.ndarray] = None,
        *,
        logw_lo: float = -1.0,
        logw_hi: float = 1.0,
        eps: float = 1e-30,
    ) -> None:
        """Initialise a log-quadratic peak decoder.

        Parameters
        ----------
        logx_lo : float, optional
            Lower bound of the log10-peak-position range. Inferred from
            ``x`` if not provided.
        logx_hi : float, optional
            Upper bound of the log10-peak-position range. Inferred from
            ``x`` if not provided.
        x : array-like, optional
            1-D x-grid on which the spectrum is evaluated. Can be set
            later via ``decoder.x = new_grid``.
        logw_lo : float, optional
            Lower bound for the log10-width parameter. Default is ``-1.0``.
        logw_hi : float, optional
            Upper bound for the log10-width parameter. Default is ``1.0``.
        eps : float, optional
            Floor applied before ``log10``. Default is ``1e-30``.
        """
        super().__init__()
        self.logw_lo = float(logw_lo)
        self.logw_hi = float(logw_hi)
        self.eps = float(eps)
        self.latent_dim = 3
        self.logx_lo = None
        self.logx_hi = None
        if x is not None:
            self._set_x(x)
            logx_lo, logx_hi = _infer_logx_bounds_from_x(x, logx_lo, logx_hi, self.logx_lo, self.logx_hi)
        if logx_lo is not None:
            self.logx_lo = float(logx_lo)
        if logx_hi is not None:
            self.logx_hi = float(logx_hi)

    def configure(
        self,
        *,
        x: Optional[np.ndarray] = None,
        logx_lo: Optional[float] = None,
        logx_hi: Optional[float] = None,
        logw_lo: Optional[float] = None,
        logw_hi: Optional[float] = None,
        eps: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Configure the quadratic peak decoder after construction.

        Parameters
        ----------
        x : array-like, optional
            1-D x-grid on which the spectrum is evaluated.
        logx_lo : float, optional
            Lower bound of the log10-peak-position range. Inferred from
            ``x`` if not provided.
        logx_hi : float, optional
            Upper bound of the log10-peak-position range. Inferred from
            ``x`` if not provided.
        logw_lo : float, optional
            Lower bound for the log10-width parameter.
        logw_hi : float, optional
            Upper bound for the log10-width parameter.
        eps : float, optional
            Floor applied before ``log10``.
        **kwargs
            Additional keyword arguments are accepted and ignored.
        """
        if x is not None:
            self._set_x(x)
            logx_lo, logx_hi = _infer_logx_bounds_from_x(x, logx_lo, logx_hi, self.logx_lo, self.logx_hi)
        if logx_lo is not None:
            self.logx_lo = float(logx_lo)
        if logx_hi is not None:
            self.logx_hi = float(logx_hi)
        if logw_lo is not None:
            self.logw_lo = float(logw_lo)
        if logw_hi is not None:
            self.logw_hi = float(logw_hi)
        if eps is not None:
            self.eps = float(eps)

    def _ensure_logx_bounds(self) -> None:
        """Infer logx bounds from the registered x-grid if not yet set.

        Raises
        ------
        ValueError
            If ``logx_lo`` or ``logx_hi`` are missing and no x-grid is
            available to infer them from.
        """
        if self.logx_lo is None or self.logx_hi is None:
            if hasattr(self, '_x') and self._x is not None:
                xarr = self._x.detach().cpu().numpy()
                if self.logx_lo is None:
                    self.logx_lo = float(np.log10(xarr.min()))
                if self.logx_hi is None:
                    self.logx_hi = float(np.log10(xarr.max()))
            else:
                raise ValueError("logx_lo and logx_hi must be set before normalize()")

    def normalize(self, y: np.ndarray, params: np.ndarray) -> dict:
        """Compute normalisation statistics for the quadratic peak model.

        Parameters
        ----------
        y : np.ndarray
            Observed spectra (linear scale), shape ``(N, D)``.
        params : np.ndarray
            Physical input parameters, shape ``(N, P)``.

        Returns
        -------
        dict
            Nested dictionary with keys ``"params"``, ``"latent"``, and
            ``"y"``.

        Raises
        ------
        ValueError
            If ``y`` or ``params`` are not 2-D, their leading dimensions
            differ, or all amplitudes are non-finite.
        """
        y = np.asarray(y, np.float64)
        params = np.asarray(params, np.float64)
        _check_normalize_inputs(y, params)
        self._ensure_logx_bounds()
        ylog = np.log10(np.clip(y, self.eps, None))
        return {
            "params": _normalize_params_stats(params),
            "latent": _normalize_amplitude_latent(self.latent_dim, ylog),
            "y": {"log": True, "eps": self.eps},
        }

    def latent_to_params(
        self, z_phys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map raw latent codes to physical quadratic-peak parameters.

        Parameters
        ----------
        z_phys : torch.Tensor
            Raw latent tensor of shape ``(N, 3)`` with columns
            ``[logA, logxpeak_raw, logw_raw]``.

        Returns
        -------
        logA : torch.Tensor
            Log10-amplitude at the peak, shape ``(N,)``.
        xpeak : torch.Tensor
            Peak position in linear scale, shape ``(N,)``.
        w : torch.Tensor
            Width parameter in linear scale, shape ``(N,)``.
        """
        z_phys = z_phys.to(torch.float64)
        logA = z_phys[:, 0]

        logxpeak = sigmoid_asinh(z_phys[:, 1], self.logx_lo, self.logx_hi)
        xpeak = torch.pow(10.0, logxpeak)

        logw = sigmoid_asinh(z_phys[:, 2], self.logw_lo, self.logw_hi)
        w = torch.pow(10.0, logw)

        return logA, xpeak, w

    def forward(self, z_phys: torch.Tensor) -> torch.Tensor:
        """Evaluate the quadratic peak model and return ``log10(y)``.

        Parameters
        ----------
        z_phys : torch.Tensor
            Raw latent tensor of shape ``(N, 3)``.

        Returns
        -------
        torch.Tensor
            Log10-spectrum evaluated on the x-grid, shape ``(N, len(x))``.
        """
        logA, xpeak, w = self.latent_to_params(z_phys)
        x = torch.clamp(self.x[None, :], min=1e-300)
        t = torch.log10(x / xpeak[:, None])
        logy = logA[:, None] - (t / w[:, None]) ** 2
        return logy

    def evaluate(self, z_phys: torch.Tensor) -> torch.Tensor:
        """Evaluate the quadratic peak model and return linear-scale ``y``.

        Parameters
        ----------
        z_phys : torch.Tensor
            Raw latent tensor of shape ``(N, 3)``.

        Returns
        -------
        torch.Tensor
            Linear-scale spectrum evaluated on the x-grid,
            shape ``(N, len(x))``.
        """
        return torch.pow(10, self.forward(z_phys))

    def get_config(self) -> dict:
        """Return serialisable constructor arguments.

        Returns
        -------
        dict
            Dictionary with keys ``"logx_lo"``, ``"logx_hi"``,
            ``"logw_lo"``, ``"logw_hi"``, and ``"eps"``.
        """
        return dict(
            logx_lo=float(self.logx_lo),
            logx_hi=float(self.logx_hi),
            logw_lo=float(self.logw_lo),
            logw_hi=float(self.logw_hi),
            eps=float(self.eps),
        )
