#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn

ln10 = np.log(10.0)

def sigmoid(x, lo, hi):
    spread = hi - lo
    # sigmoid returns 0..1, map to lo .. hi:
    # return lo + (hi - lo) * torch.sigmoid(x)
    # asinh/2 returns ~ -1..+1
    mid = (lo + hi) / 2.0
    return mid + (spread / 4.0) * torch.asinh(x)

class DenseDecoder(nn.Module):
    """
    Dense emulator decoder: y_pre = z.

    Intended for cases where the function already returns the quantity
    in the space you want to emulate (linear or log), i.e. "user returns log if desired".

    latent_dim = y_dim
    """
    def __init__(self, y_dim: int):
        super().__init__()
        self.y_dim = int(y_dim)

    def normalize(self, y: np.ndarray, params: np.ndarray) -> dict:
        y = np.asarray(y, dtype=np.float64)
        params = np.asarray(params, dtype=np.float64)
        if y.ndim != 2:
            raise ValueError(f"y must be 2D (N,Y), got {y.shape}")
        if params.ndim != 2:
            raise ValueError(f"params must be 2D (N,P), got {params.shape}")
        if y.shape[0] != params.shape[0]:
            raise ValueError("y and params must have same N")

        y_mean = y.mean(axis=0)
        y_std = y.std(axis=0) + 1e-12

        p_mean = params.mean(axis=0)
        p_std = params.std(axis=0) + 1e-12
        p_log_mask = np.zeros(params.shape[1], dtype=bool)

        return {
            "y": {"log": False, "mean": y_mean, "std": y_std, "eps": 1e-30},
            "params": {"log_mask": p_log_mask, "mean": p_mean, "std": p_std, "eps": 1e-30},
            "latent_dim": self.y_dim,
        }

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B,Y)
        return z

    def latent_to_params(self, z: torch.Tensor) -> torch.Tensor:
        # for dense, "decoder parameters" are just the predicted y_pre
        return z


class LogDenseDecoder(nn.Module):
    """
    Dense emulator decoder for strictly-positive quantities spanning many decades.

    The encoder predicts z = log10(y) in a stabilized / normalized way.
    The decoder maps z -> y via y = 10**z.

    Important: This violates the "user returns log if desired" principle *unless*
    you use it intentionally for cases where your function returns *linear* y but
    you want the emulator to operate in log-space.

    Behavior:
      - decoder.forward(z) returns log10(y)  (pre-normalized y-space)
      - decoder.evaluate(z) returns y (linear space)
      - normalize(y, params) sets norm['y']['log']=True and uses stats of log10(y)

    That matches the convention used in your parametric decoders (often outputting log10(y)).
    """
    def __init__(self, y_dim: int, eps: float = 1e-30):
        super().__init__()
        self.y_dim = int(y_dim)
        self.eps = float(eps)

    def normalize(self, y: np.ndarray, params: np.ndarray) -> dict:
        y = np.asarray(y, dtype=np.float64)
        params = np.asarray(params, dtype=np.float64)
        if y.ndim != 2:
            raise ValueError(f"y must be 2D (N,Y), got {y.shape}")
        if params.ndim != 2:
            raise ValueError(f"params must be 2D (N,P), got {params.shape}")
        if y.shape[0] != params.shape[0]:
            raise ValueError("y and params must have same N")

        # work in log10-space for y normalization and tolerance checks
        ylog = np.log10(np.clip(y, self.eps, None))
        y_mean = ylog.mean(axis=0)
        y_std = ylog.std(axis=0) + 1e-12

        p_mean = params.mean(axis=0)
        p_std = params.std(axis=0) + 1e-12
        p_log_mask = np.zeros(params.shape[1], dtype=bool)

        return {
            "y": {"log": True, "mean": y_mean, "std": y_std, "eps": self.eps},
            "params": {"log_mask": p_log_mask, "mean": p_mean, "std": p_std, "eps": 1e-30},
            "latent_dim": self.y_dim,
        }

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Return log10(y) in pre-normalized y-space.

        This keeps training consistent with the parametric decoders which often return log10(y).
        """
        return z

    def evaluate(self, z: torch.Tensor) -> torch.Tensor:
        """
        Return y in linear space (positive).
        """
        return torch.pow(10.0, z)

    def latent_to_params(self, z: torch.Tensor) -> torch.Tensor:
        # The natural "decoder parameters" here are log10(y)
        return z


class CutoffPLDecoder(nn.Module):
    """
    Cutoff power-law decoder producing log10(y) on an energy grid E (keV).

      y(E) = A * E^{-Gamma} * exp(-E/Ecut)

    Updated design (matching SBPLDecoder latent normalization)
    ----------------------------------------------------------
    - Remove lognorm_scale/lognorm_shift parameters from __init__.
    - Decoder.forward(z_phys) returns log10(y) (pre-normalized y-space).
    - normalize(y, params) declares:
        * norm['y']['log']=True and identity mean/std (no per-bin normalization)
        * norm['latent'] with:
            - mean/std applied to latent[0] (logA) using per-sample amplitude statistics
            - mean=0,std=1 for other latent dims by default
        * norm['params'] standardization (all linear unless you want log masks)

    Latent convention (physical latent z_phys)
    ------------------------------------------
      z_phys = [logA, gamma_raw, logecut_raw]

    where:
      Gamma = gamma_lo + (gamma_hi-gamma_lo)*sigmoid(gamma_raw)
      logEcut = log10(ecut_lo) + (log10(ecut_hi)-log10(ecut_lo))*sigmoid(logecut_raw)
      Ecut = 10**logEcut
    """

    def __init__(self, E_keV, kind="hot", eps=1e-30):
        super().__init__()
        self.register_buffer("E", torch.tensor(np.asarray(E_keV, float), dtype=torch.float64))
        self.kind = str(kind)
        self.eps = float(eps)
        self.latent_dim = 3

        # soft bounds via sigmoid transforms
        self.gamma_lo = 0.5
        self.gamma_hi = 6.0
        self.ecut_lo = 0.01
        self.ecut_hi = 10000.0
        self.log_ecut_lo = np.log10(self.ecut_lo)
        self.log_ecut_hi = np.log10(self.ecut_hi)

    def normalize(self, y: np.ndarray, params: np.ndarray) -> dict:
        """
        y: (N, D) in linear space (>=0), e.g. photon flux per keV.
        params: (N, P)

        We operate in log10-space for y and return norm['y']['log']=True
        with identity mean/std (no per-bin normalization).

        Latent normalization:
          - latent_mean[0], latent_std[0] derived from per-sample max(log10(y))
            as a proxy for amplitude logA.
          - other latent dims: mean=0, std=1.
        """
        y = np.asarray(y, dtype=np.float64)
        params = np.asarray(params, dtype=np.float64)
        if y.ndim != 2:
            raise ValueError(f"y must be 2D (N,D), got {y.shape}")
        if params.ndim != 2:
            raise ValueError(f"params must be 2D (N,P), got {params.shape}")
        if y.shape[0] != params.shape[0]:
            raise ValueError("y and params must have same number of rows")

        ylog = np.log10(np.clip(y, self.eps, None))
        a = np.max(ylog, axis=1)
        a = a[np.isfinite(a)]
        if len(a) == 0:
            raise ValueError("could not compute amplitude statistics (non-finite y?)")

        latent_mean = np.zeros(self.latent_dim, dtype=np.float64)
        latent_std = np.ones(self.latent_dim, dtype=np.float64)
        latent_mean[0] = float(np.median(a))
        latent_std[0] = float(np.std(a) + 1e-6)
        #latent_mean[0] = 50
        #latent_std[0] = 3

        # params normalization (default all linear)
        p_mean = params.mean(axis=0)
        p_std = params.std(axis=0) + 1e-12

        #y_mean = ylog.mean()
        #y_std = ylog.std(axis=0) + 1e-6

        return {
            "y": {"log": True, "eps": self.eps}, #, "mean": y_mean, "std": y_std},
            "params": {"mean": p_mean, "std": p_std},
            "latent_dim": self.latent_dim,
            "latent": {"mean": latent_mean, "std": latent_std, "log_mask": np.zeros(self.latent_dim, dtype=bool)},
        }

    def latent_to_params(self, z_phys: torch.Tensor):
        """
        z_phys: (B,3) physical latent (after Emulator denormalizes with norm['latent']).
        Returns (logA, Gamma, Ecut_keV).
        """
        z_phys = z_phys.to(dtype=torch.float64)
        logA = z_phys[:, 0]
        gamma_raw = z_phys[:, 1]
        logecut_raw = z_phys[:, 2]

        Gamma = sigmoid(gamma_raw, self.gamma_lo, self.gamma_hi)

        logEcut = sigmoid(logecut_raw, self.log_ecut_lo, self.log_ecut_hi)
        Ecut = 10.0 ** logEcut
        return logA, Gamma, Ecut

    def forward(self, z_phys: torch.Tensor) -> torch.Tensor:
        """
        Return log10(y(E)) on self.E grid.
        """
        logA, Gamma, Ecut = self.latent_to_params(z_phys)
        E = self.E[None, :]

        lnA = (logA[:, None] * ln10)
        lnE = torch.log(torch.clamp(E, min=1e-300))
        lny = lnA - Gamma[:, None] * lnE - (E / Ecut[:, None])
        log10y = lny / ln10
        return log10y

    def evaluate(self, z_phys: torch.Tensor) -> torch.Tensor:
        """
        Return y(E) in linear space.
        """
        return torch.pow(10.0, self.forward(z_phys))


class SBPLDecoder(nn.Module):
    """
    SBPL decoder producing log10(y) on a wavelength grid x (nm).

    Latent convention (updated per your decision)
    ---------------------------------------------
    - Decoder.forward(z) returns log10(y).
    - We DO NOT apply any y-mean/y-std normalization across wavelength bins.
      So norm['y']['mean']=0, norm['y']['std']=1 (identity), but norm['y']['log']=True.
    - Instead, we normalize only the *amplitude latent coordinate* (z[:,0]) via:
        z_phys0 = z_norm0 * latent_std0 + latent_mean0
      where latent_mean0/std0 are computed from training data.
    - All other latent coordinates keep mean=0, std=1.

    Therefore the Emulator must support norm['latent'] and apply it:
      z_phys = z_norm * latent_std + latent_mean
      yhat_log = decoder.forward(z_phys)

    latent z meaning (physical):
      z_phys = [lognorm, lam1_raw, lam2_raw, logxbrk_raw, logLambda_raw]
    and the decoder constrains lam1/lam2/logxbrk/logLambda via sigmoids as before.

    normalize(y, params) returns:
      - norm['y']['log']=True and identity mean/std for y
      - norm['params'] (default: all linear; mean/std from data)
      - norm['latent'] with mean/std only on index 0

    """

    def __init__(self, lam_nm, x0_nm=510.0, eps=1e-30):
        super().__init__()
        self.register_buffer("x", torch.tensor(np.asarray(lam_nm, float), dtype=torch.float64))
        self.register_buffer("x0", torch.tensor(float(x0_nm), dtype=torch.float64))
        self.eps = float(eps)
        self.latent_dim = 5
        self.log_lam_nm_lo = np.log10(lam_nm.min())
        self.log_lam_nm_hi = np.log10(lam_nm.max())

    def normalize(self, y: np.ndarray, params: np.ndarray) -> dict:
        y = np.asarray(y, dtype=np.float64)
        params = np.asarray(params, dtype=np.float64)
        if y.ndim != 2:
            raise ValueError(f"y must be 2D (N,Y), got {y.shape}")
        if params.ndim != 2:
            raise ValueError(f"params must be 2D (N,P), got {params.shape}")
        if y.shape[0] != params.shape[0]:
            raise ValueError("y and params must have same number of rows")

        # amplitude proxy in log space: use per-sample maximum of log10(y)
        ylog = np.log10(np.clip(y, self.eps, None))
        a = np.max(ylog, axis=1)
        a = a[np.isfinite(a)]
        if len(a) == 0:
            raise ValueError("could not compute amplitude statistics for SBPL (non-finite y?)")

        latent_mean = np.zeros(self.latent_dim, dtype=np.float64)
        latent_std = np.ones(self.latent_dim, dtype=np.float64)

        latent_mean[0] = float(np.median(a))
        latent_std[0] = float(np.std(a) + 1e-6)

        # params normalization: default all linear, standardize columns
        p_log_mask = np.zeros(params.shape[1], dtype=bool)
        p_mean = params.mean(axis=0)
        p_std = params.std(axis=0) + 1e-12

        # y normalization: identity in log space (decoder already returns log10(y))

        return {
            "y": {"log": True, "eps": self.eps},
            "params": {"log_mask": p_log_mask, "mean": p_mean, "std": p_std, "eps": 1e-30},
            "latent_dim": self.latent_dim,
            "latent": {"mean": latent_mean, "std": latent_std, "log_mask": np.zeros(self.latent_dim, dtype=bool)},
        }

    def latent_to_params(self, z_phys: torch.Tensor):
        """
        Map physical latent vector -> SBPL parameters.

        z_phys[:,0] is lognorm directly (already denormalized by Emulator).
        """
        z_phys = z_phys.to(dtype=torch.float64)

        lognorm = z_phys[:, 0]
        lam1 = 2.5 * torch.asinh(z_phys[:, 1])
        lam2 = 2.5 * torch.asinh(z_phys[:, 2])

        # break location
        logxbrk = sigmoid(z_phys[:, 3], self.log_lam_nm_lo - 1, self.log_lam_nm_hi + 1)
        xbrk = 10.0 ** logxbrk

        # Lambda width in dex: 0.1 to 100
        logLambda = sigmoid(z_phys[:, 4], -1, 2)
        Lambda = 10.0 ** logLambda

        return lognorm, lam1, lam2, xbrk, Lambda

    def forward(self, z_phys: torch.Tensor) -> torch.Tensor:
        """
        Return log10(y) on self.x grid.

        Note: input is *physical latent* z_phys (not normalized). The Emulator applies
        latent normalization inversion before calling this.
        """
        lognorm, lam1, lam2, xbrk, Lambda = self.latent_to_params(z_phys)
        norm = 10.0 ** lognorm
        x = self.x[None, :]
        x0 = self.x0

        q = torch.log(x / xbrk[:, None]) / Lambda[:, None]
        qpiv = torch.log(x0 / xbrk) / Lambda
        q = torch.clamp(q, -50.0, 50.0)
        qpiv = torch.clamp(qpiv, -50.0, 50.0)

        num = torch.exp(q) + torch.exp(-q)
        den = torch.exp(qpiv[:, None]) + torch.exp(-qpiv[:, None])
        ratio = num / (den + 1e-30)

        expo = ((lam2 - lam1) / 2.0)[:, None] * Lambda[:, None]
        expo = torch.clamp(expo, -10.0, 10.0)

        y = norm[:, None] * (x / x0) ** (((lam1 + lam2 + 2.0) / 2.0)[:, None]) * (ratio ** expo) * (x0 / x)
        y = torch.clamp(y, min=self.eps)
        return torch.log10(y)

    def evaluate(self, z_phys: torch.Tensor) -> torch.Tensor:
        """
        Return y in linear space.
        """
        return torch.pow(10.0, self.forward(z_phys))



class TorusDoubleLogQuadraticDecoder(nn.Module):
    """
    Two-component log-quadratic torus shape model on a wavelength grid (um).

    Latent definition (physical z_phys, dim=6)
    ------------------------------------------
    z_phys = [u0, u1, u2, u3, logL12_phys, u5]
      u0,u1,u2,u3,u5 are unconstrained reals that are squashed with sigmoid into ranges
      logL12_phys is the physical log10 of y at lam_ref (up to the model definition)

    The mapping is:
      log_lam_hot  in [0.0, 0.8]   => lam_hot  ~ [1, 6.3] um
      log_lam_cool in [0.9, 1.9]   => lam_cool ~ [8, 79] um
      log_w_hot    in [-1, 2]      => w_hot    ~ [0.1, 100]
      log_w_cool   in [-1, 2]
      log_peak_ratio in [-4, 1]    => peak_ratio ~ [1e-4, 10]
      and normalization enforces y(lam_ref)=10**logL12_phys.

    Notes
    -----
    - This decoder returns log10(y) to match your tolerance-in-log-space workflow.
    - If your stored y can be <= 0, you must fix that upstream; log-space requires positivity.
    """

    def __init__(self, lam_um, lam_ref_um=12.0, eps=1e-30):
        super().__init__()
        self.register_buffer("lam", torch.tensor(np.asarray(lam_um, float), dtype=torch.float64))
        self.lam_ref = float(lam_ref_um)
        self.eps = float(eps)
        self.latent_dim = 6

        # In practice this is just a constant shift between log10(nuLnu) and log10(Lnu).
        # If you store Lnu as y, and want the parameter to correspond to L12 in nuLnu, you can use this.
        # If you store nuLnu as y, set lam_logfactor=0 effectively by not using it.
        # We'll keep it, but it only affects interpretation of the 5th parameter.
        self.lam_logfactor = np.log10(2.99792458e14 / self.lam_ref)  # log10(nu[Hz]) with lam in um

    def normalize(self, y: np.ndarray, params: np.ndarray) -> dict:
        y = np.asarray(y, dtype=np.float64)
        params = np.asarray(params, dtype=np.float64)
        if y.ndim != 2:
            raise ValueError(f"y must be 2D (N,D), got {y.shape}")
        if params.ndim != 2:
            raise ValueError(f"params must be 2D (N,P), got {params.shape}")
        if y.shape[0] != params.shape[0]:
            raise ValueError("y and params must have same number of rows")

        # amplitude proxy: per-sample max(log10(y))
        ylog = np.log10(np.clip(y, self.eps, None))
        a = np.max(ylog, axis=1)
        a = a[np.isfinite(a)]
        if len(a) == 0:
            raise ValueError("could not compute amplitude stats for torus decoder")

        latent_mean = np.zeros(self.latent_dim, dtype=np.float64)
        latent_std = np.ones(self.latent_dim, dtype=np.float64)

        # normalize only the amplitude coordinate (index 4)
        latent_mean[4] = float(np.median(a))
        latent_std[4] = float(np.std(a) + 1e-6)

        # params normalization: default all linear
        p_log_mask = np.zeros(params.shape[1], dtype=bool)
        p_mean = params.mean(axis=0)
        p_std = params.std(axis=0) + 1e-12

        # y normalization: identity in log space
        return {
            "y": {"log": True, "eps": self.eps},
            "params": {"log_mask": p_log_mask, "mean": p_mean, "std": p_std, "eps": 1e-30},
            "latent_dim": self.latent_dim,
            "latent": {
                "mean": latent_mean,
                "std": latent_std,
                "log_mask": np.zeros(self.latent_dim, dtype=bool),
            },
        }

    def latent_to_params(self, z_phys: torch.Tensor):
        """
        Return interpretable parameters:
          (logL12, log_peak_ratio, lam_hot_um, lam_cool_um, w_hot, w_cool)

        Where logL12 is the physical latent amplitude coordinate (already denormalized by Emulator).
        """
        z_phys = z_phys.to(dtype=torch.float64)

        log_lam_hot = 0.0 + 0.8 * sigmoid(z_phys[:, 0])
        log_lam_cool = 0.9 + 1.0 * torch.sigmoid(z_phys[:, 1])
        lam_hot = 10.0 ** log_lam_hot
        lam_cool = 10.0 ** log_lam_cool

        log_w_hot = -1.0 + 3.0 * torch.sigmoid(z_phys[:, 2])
        log_w_cool = -1.0 + 3.0 * torch.sigmoid(z_phys[:, 3])
        w_hot = 10.0 ** log_w_hot
        w_cool = 10.0 ** log_w_cool

        logL12 = z_phys[:, 4]  # already in log10 space
        log_peak_ratio = -4.0 + 8.0 * torch.sigmoid(z_phys[:, 5])

        return logL12, log_peak_ratio, lam_hot, lam_cool, w_hot, w_cool

    def forward(self, z_phys: torch.Tensor) -> torch.Tensor:
        """
        Return log10(y(lam)) on self.lam grid.

        This produces log10(y) directly (pre-space).
        """
        logL12, log_peak_ratio, lam_hot, lam_cool, w_hot, w_cool = self.latent_to_params(z_phys)

        lam = self.lam[None, :]  # (1, D)

        # log10 distance from peaks
        xh = torch.log10(lam / lam_hot[:, None])   # (B, D)
        xc = torch.log10(lam / lam_cool[:, None])  # (B, D)

        # Shapes (positive): exp(-(x/w)^2) in ln-space
        ln_hot_shape = - (xh / w_hot[:, None])**2
        ln_cool_shape = - (xc / w_cool[:, None])**2

        # amplitudes: hot relative to cool
        ln_A_hot = (log_peak_ratio[:, None]) * ln10
        ln_A_cool = torch.zeros_like(ln_A_hot)

        ln_term_hot = ln_A_hot + ln_hot_shape
        ln_term_cool = ln_A_cool + ln_cool_shape
        ln_y_unnorm = torch.logaddexp(ln_term_hot, ln_term_cool)

        # reference at lam_ref
        lam_ref = torch.as_tensor(self.lam_ref, dtype=lam.dtype, device=lam.device)[None, None]
        xh_ref = torch.log10(lam_ref / lam_hot[:, None])
        xc_ref = torch.log10(lam_ref / lam_cool[:, None])

        ln_hot_ref = ln_A_hot[:, :1] + (-(xh_ref / w_hot[:, None])**2)
        ln_cool_ref = ln_A_cool[:, :1] + (-(xc_ref / w_cool[:, None])**2)
        ln_y_ref = torch.logaddexp(ln_hot_ref, ln_cool_ref)

        # enforce y(lam_ref) = 10**logL12
        ln_L12 = (logL12[:, None]) * ln10
        ln_y = ln_y_unnorm + ln_L12 - ln_y_ref

        return ln_y / ln10

    def evaluate(self, z_phys: torch.Tensor) -> torch.Tensor:
        """Return y in linear space."""
        return torch.pow(10.0, self.forward(z_phys))
