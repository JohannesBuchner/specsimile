from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
#import torch.nn.functional as F
#import math

from . import utils
from .store import DatasetReader

LOGTHRESH = float(np.log10(3.0))

class MLP(nn.Module):
    def __init__(self, nin: int, nout: int, shape: list[int]):
        super().__init__()
        layers = []
        prev = nin
        for width in shape:
            layers += [nn.Linear(prev, width), nn.LeakyReLU()]
            prev = width
        layers += [nn.Linear(prev, nout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def softplus_np(z):
    z = np.asarray(z, dtype=np.float64)
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)

def soft_floor_np(x, floor, k=10.0):
    z = k * (x - floor)
    return floor + softplus_np(z) / k

def soft_floor_torch(x, floor, k=10.0):
    # smooth approximation to max(x, floor)
    return floor + torch.nn.functional.softplus(k * (x - floor))/k


class Emulator:
    def __init__(self, model_path: str, decoder, shape: list[int], *, device=None, dynamic_range=None, func_floor_loss=None, loss_y_weights=None, param_bounds_policy='error'):
        self.model_path = str(model_path)
        self.decoder = decoder
        self.shape = list(shape)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self._loaded = False
        self._encoder: MLP | None = None
        self._norm: dict | None = None
        self._meta: dict = {}

        self.fit_mean: float | None = None
        self.fit_median: float | None = None
        self.fit_p95: float | None = None
        self.dynamic_range = dynamic_range
        self.func_floor_loss = func_floor_loss
        self.train_param_bounds: dict | None = None
        self.param_bounds_policy = param_bounds_policy
        self.param_bounds_tol: float = 0.0
        if loss_y_weights is None:
            self.loss_y_weights = 1.
        else:
            self.loss_y_weights = torch.tensor(
                loss_y_weights.reshape((1, -1)),
                dtype=torch.float64, device=self.device)

    # ---------- helpers ----------
    def _y_mean_std(self, norm: dict, y_dim: int):
        yinfo = norm.get("y", {})
        mean = yinfo.get("mean", None)
        std = yinfo.get("std", None)
        mean = np.zeros(y_dim, np.float64) if mean is None else np.asarray(mean, np.float64)
        std = np.ones(y_dim, np.float64) if std is None else np.asarray(std, np.float64)
        return mean, std

    def inside_param_bounds(self, P: np.ndarray):
        b = self.train_param_bounds
        if b is None or self.param_bounds_tol < 0:
            return
        pmin = np.asarray(b["min"], dtype=np.float64).reshape(1, -1)
        pmax = np.asarray(b["max"], dtype=np.float64).reshape(1, -1)
        if P.shape[1] != pmin.shape[1]:
            raise ValueError("Parameter dimension mismatch vs stored bounds")

        mask_inside = np.all(np.logical_and(
            P > pmin - self.param_bounds_tol,
            P < pmax + self.param_bounds_tol
        ), axis=1)
        return mask_inside
            

    def _check_param_bounds(self, P: np.ndarray):
        b = self.train_param_bounds
        if b is None or self.param_bounds_tol < 0 or self.param_bounds_policy != 'error':
            return
        bad = ~self.inside_param_bounds(P)
        if np.any(bad):
            i = int(np.where(bad)[0][0])
            pmin = np.asarray(b["min"], dtype=np.float64).reshape(1, -1)
            pmax = np.asarray(b["max"], dtype=np.float64).reshape(1, -1)
            if self.paramnames is not None:
                s = ""
                for j, (pname, plo, phi) in enumerate(zip(self.paramnames, b["min"], b["max"])):
                    if not (P[i, j] > plo - self.param_bounds_tol and P[i, j] < phi + self.param_bounds_tol):
                        s += f"Input parameter {j+1} {pname}={P[i,j]} outside training bounds [{plo}..{phi}] (first offending row {i}).\n"
            raise ValueError(
                s + f"Input params outside training bounds (first offending row {i}).\n"
                f"params={P[i]}\n"
                f"min={pmin[0]}\n"
                f"max={pmax[0]}\n"
                f"paramnames={self.paramnames}"
            )

    def _apply_norm_params(self, P: np.ndarray, norm: dict) -> np.ndarray:
        pn = norm["params"]
        if "log_mask" in pn:
            P = utils.apply_log_mask(P, pn["log_mask"], pn.get("eps", 1e-30))
        return utils.normalize(P, pn["mean"], pn["std"])

    def _apply_latent_norm_inv(self, z_norm: np.ndarray, norm: dict) -> np.ndarray:
        lat = norm["latent"]
        if 'std' in lat:
            std = np.asarray(lat["std"], np.float64).reshape(1, -1)
        else:
            std = 1.0
        if 'mean' in lat:
            mean = np.asarray(lat["mean"], np.float64).reshape(1, -1)
        else:
            mean = 0.0
        return z_norm * std + mean

    def _target_y_pre(self, Y: np.ndarray, norm: dict) -> np.ndarray:
        if bool(norm["y"].get("log", False)):
            eps = float(norm["y"].get("eps", 1e-30))
            Ypre = np.log10(np.clip(Y, eps, None))
        else:
            Ypre = Y
        mean, std = self._y_mean_std(norm, Ypre.shape[1])
        return (Ypre - mean[None, :]) / std[None, :]

    # ---------- checkpoint IO ----------
    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(self.model_path)

        ckpt = torch.load(self.model_path, map_location="cpu")

        self._norm = ckpt["norm"]
        self._meta = ckpt.get("meta", {})
        self.paramnames = self._meta.get("paramnames", None)
        print(f"loading emulator from {self.model_path}. Meta info: {self._meta}")

        self.train_param_bounds = self._meta.get("train_param_bounds", None)
        # validate decoder identity/config if present
        dmeta = self._meta.get("decoder", None)
        if dmeta is not None:
            want_cls = dmeta.get("class_name")
            got_cls = self.decoder.__class__.__name__
            if want_cls is not None and want_cls != got_cls:
                raise ValueError(f"Decoder class mismatch: ckpt has {want_cls}, but you provided {got_cls}")

            want_cfg = dmeta.get("config", None)
            if want_cfg is not None and hasattr(self.decoder, "get_config"):
                got_cfg = self.decoder.get_config()
                if got_cfg != want_cfg:
                    raise ValueError(
                        "Decoder config mismatch between checkpoint and provided decoder.\n"
                        f"ckpt: {want_cfg}\n"
                        f"got:  {got_cfg}"
                    )
        self.fit_mean = ckpt.get("fit_mean", None)
        self.fit_median = ckpt.get("fit_median", None)
        self.fit_p95 = ckpt.get("fit_p95", None)

        nin = int(ckpt["nin"])
        nout = int(ckpt["nout"])
        self._encoder = MLP(nin=nin, nout=nout, shape=ckpt["shape"])
        self._encoder.load_state_dict(ckpt["encoder_state"])
        self._encoder.to(self.device).double().eval()
        self._loaded = True
        return self._loaded

    def _save(self, *, norm: dict, encoder: nn.Module, nin: int, nout: int, meta: dict, fit_stats: dict):
        ckpt = dict(
            shape=self.shape,
            norm=norm,
            meta=meta,
            nin=int(nin),
            nout=int(nout),
            encoder_state=encoder.state_dict(),
            **fit_stats,
        )
        torch.save(ckpt, self.model_path)

    def _arch_string(self, xgrid: np.ndarray, P: np.ndarray, Y: np.ndarray) -> str:
        #xdim = int(np.asarray(xgrid).shape[0])   # optional, just for display
        in_dim = int(P.shape[1])
        ydim = int(Y.shape[1])

        enc_str = "-".join(str(w) for w in self.shape)

        dec = self.decoder
        dec_name = dec.__class__.__name__.replace('Decoder', '')

        latent_dim = getattr(dec, "latent_dim", None)
        if latent_dim is None:
            latent_dim = "?"

        return f"[P:{in_dim}]-[MLP:{enc_str}]->[latent:{latent_dim}]-{dec_name}-[y:{ydim}]"


    # ---------- loss weighting (uncertainty floors) ----------
    def _loss_variance_torch(self, y: torch.Tensor) -> torch.Tensor:
        """
        Return per-sample, per-point sigma in y_pre space, shape (B, D).

        dynamic_range: sigma_dyn = max(y) - dynamic_range   (broadcast to (B,D))
        func_floor_loss: sigma_func = constant
        If both given: sigma^2 = sigma_dyn^2 + sigma_func^2  (quadrature)
        """
        B, D = y.shape
        sig2 = None

        if self.dynamic_range is not None:
            dr = float(self.dynamic_range)
            ymax = torch.max(y, dim=1, keepdim=True).values  # (B,1)
            sigma_dyn = ymax - dr                            # (B,1)
            # ensure strictly positive to avoid div-by-zero / negative variance
            sigma_dyn = torch.clamp(sigma_dyn, min=1e-12)
            sig2 = sigma_dyn**2

        if self.func_floor_loss is not None:
            sigma_func = float(self.func_floor_loss)
            sigma_func = max(sigma_func, 1e-12)
            sig2_func = y.new_full((B, 1), sigma_func**2)
            sig2 = sig2_func if sig2 is None else (sig2 + sig2_func)

        if sig2 is None:
            # no weighting -> sigma=1
            return y.new_ones((B, 1)).expand(B, D)

        return sig2.expand(B, D)           # (B,D)

    def _loss_variance_np(self, y: np.ndarray) -> np.ndarray:
        """
        Numpy version of _loss_variance_torch. Returns (N,D).
        """
        y = np.asarray(y, dtype=np.float64)
        N, D = y.shape
        sig2 = None

        if self.dynamic_range is not None:
            dr = float(self.dynamic_range)
            ymax = np.max(y, axis=1, keepdims=True)  # (N,1)
            sigma_dyn = ymax - dr
            sigma_dyn = np.clip(sigma_dyn, 1e-12, None)
            sig2 = sigma_dyn**2

        if self.func_floor_loss is not None:
            sigma_func = max(float(self.func_floor_loss), 1e-12)
            sig2_func = np.full((N, 1), sigma_func**2, dtype=np.float64)
            sig2 = sig2_func if sig2 is None else (sig2 + sig2_func)

        if sig2 is None:
            return np.ones((N, D), dtype=np.float64)

        return np.repeat(sig2, D, axis=1)  # (N,D)

    def _apply_dynamic_range_torch(self, yhat: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply dynamic-range clamping in the *same space* as yhat,y (y_pre space).
        """
        if self.dynamic_range is None:
            return yhat, y

        dr = float(self.dynamic_range)
        ymax = torch.max(y, dim=1, keepdim=True).values  # (B,1)

        # global minimum allowed ymax (in y_pre / standardized-target space)
        floor = ymax - dr
        if self.func_floor_loss is not None:
            floor = torch.maximum(floor, floor.new_tensor(float(self.func_floor_loss)))

        return soft_floor_torch(yhat, floor=floor), soft_floor_torch(y, floor=floor)

    def _apply_dynamic_range_np(self, yhat: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply dynamic-range clamping in y_pre space (numpy).
        """
        if self.dynamic_range is None:
            return yhat, y
        dr = float(self.dynamic_range)
        ymax = np.max(y, axis=1, keepdims=True)

        floor = ymax - dr
        if self.func_floor_loss is not None:
            floor = np.maximum(floor, self.func_floor_loss)

        return soft_floor_np(yhat, floor=floor), soft_floor_np(y, floor=floor)

    def rms_per_sample_torch(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Returns (B,) RMS in y_pre space, with dynamic-range clamp applied if configured.
        """
        yhat2, y2 = self._apply_dynamic_range_torch(yhat, y)
        return torch.sqrt(torch.mean((yhat2 - y2) ** 2 * self.loss_y_weights, dim=1))
        ivar = 1. / self._loss_variance_torch(y)         # (B,D)
        r2 = (yhat - y) ** 2
        w = ivar * self.loss_y_weights
        return torch.sqrt(torch.mean(r2 * w, dim=1))

    def rms_per_sample_np(self, yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Returns (N,) RMS in y_pre space, with dynamic-range clamp applied if configured.
        """
        yhat2, y2 = self._apply_dynamic_range_np(np.asarray(yhat), np.asarray(y))
        return np.sqrt(np.mean((yhat2 - y2) ** 2 * self.loss_y_weights, axis=1))
        ivar = 1. / self._loss_variance_np(y)            # (N,D)
        r2 = (yhat - y) ** 2
        return np.sqrt(np.mean(r2 * ivar * self.loss_y_weights, axis=1))

    def loss_fn(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        yhat, y: (B, D) in the *training target space* (usually log10(y) or y_pre).
        dynamic_range:
          - if None: plain MSE
          - else: clamp both yhat and y to at least (per-sample max - dynamic_range)
        """
        return self.rms_per_sample_torch(yhat, y).mean()

    # ---------- training ----------
    def fit(
        self,
        dataset: DatasetReader,
        *,
        epochs: int,
        lr: float,
        batch_size: int,
        train_split: float = 0.8,
        patience: int = 15,
        seed: int = 123,
        overwrite: bool = True,
    ) -> str:
        Y = dataset.y.astype(np.float64)
        P = dataset.params.astype(np.float64)
        Pmin = P.min(axis=0)
        Pmax = P.max(axis=0)
        x = dataset.x.astype(np.float64)
        utils.ensure_finite("P", P)
        utils.ensure_finite("Y", Y)

        # set x on decoder (required by new contract)
        self.decoder.x = x
        dec = self.decoder.to(self.device).double().eval()
        for p in dec.parameters():
            p.requires_grad_(False)

        norm_info = dec.normalize(Y, P)
        for k, v in norm_info.items():
            print("  Decoder normalisation:", k, v)
        latent_dim = int(norm_info.get("latent_dim", 0))
        if latent_dim <= 0:
            raise ValueError("decoder.normalize must return positive latent_dim")
        if "latent" not in norm_info:
            raise ValueError("decoder.normalize must return norm['latent']")

        Pn = self._apply_norm_params(P, norm_info)
        T = self._target_y_pre(Y, norm_info)
        print("emulator architecture:", self._arch_string(x, P, Y))

        n = len(dataset)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        ntr = max(1, int(round(train_split * n)))
        tr_idx = idx[:ntr]
        va_idx = idx[ntr:] if ntr < n else idx[:1]

        Xtr = torch.tensor(Pn[tr_idx], dtype=torch.float64, device=self.device)
        Xva = torch.tensor(Pn[va_idx], dtype=torch.float64, device=self.device)
        Ttr = torch.tensor(T[tr_idx], dtype=torch.float64, device=self.device)
        Tva = torch.tensor(T[va_idx], dtype=torch.float64, device=self.device)

        nin = P.shape[1]
        nout = latent_dim
        enc = MLP(nin=nin, nout=nout, shape=self.shape).to(self.device).double()
        opt = torch.optim.Adam(enc.parameters(), lr=lr)

        lat = norm_info["latent"]
        if "mean" in lat:
            lat_mean = torch.tensor(np.asarray(lat["mean"]).reshape(1, -1), dtype=torch.float64, device=self.device)
        else:
            lat_mean = 0.0
        if "std" in lat:
            lat_std = torch.tensor(np.asarray(lat["std"]).reshape(1, -1), dtype=torch.float64, device=self.device)
        else:
            lat_std = 1.0

        y_mean_np, y_std_np = self._y_mean_std(norm_info, Y.shape[1])
        y_mean = torch.tensor(y_mean_np.reshape(1, -1), dtype=torch.float64, device=self.device)
        y_std = torch.tensor(y_std_np.reshape(1, -1), dtype=torch.float64, device=self.device)

        best = np.inf
        best_state = None
        bad = 0

        for ep in range(int(epochs)):
            enc.train()
            perm = rng.permutation(len(Xtr))
            for i0 in range(0, len(Xtr), batch_size):
                sl = perm[i0:i0 + batch_size]
                xb = Xtr[sl]
                tb = Ttr[sl]

                z_norm = torch.clamp(enc(xb), -25, 25)
                z_phys = z_norm * lat_std + lat_mean
                y_pre = dec(z_phys)
                y_hat = (y_pre - y_mean) / y_std

                loss = self.loss_fn(y_hat, tb)**2
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                opt.step()

            enc.eval()
            with torch.no_grad():
                z_norm = torch.clamp(enc(Xva), -25, 25)
                z_phys = z_norm * lat_std + lat_mean
                y_pre = dec(z_phys)
                y_hat = (y_pre - y_mean) / y_std
                vloss = self.loss_fn(y_hat, Tva).item()

            if vloss < best - 1e-4:
                best = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
                bad = 0
                print(f"Epoch {ep+1}/{epochs}: RMS loss={vloss:.3f}")
            else:
                bad += 1
                print(f"Epoch {ep+1}/{epochs}: RMS loss={vloss:.3f} [{bad}/{patience}]")
                if bad >= patience:
                    break

        if best_state is not None:
            enc.load_state_dict(best_state)
        enc.eval()

        # ---- fit statistics on validation set in tolerance space (y_pre, not standardized) ----
        with torch.no_grad():
            z_norm = torch.clamp(enc(Xva), -25, 25).detach().cpu().numpy()
        z_phys = self._apply_latent_norm_inv(z_norm, norm_info)

        zt = torch.tensor(z_phys, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            y_pre_hat = dec(zt).detach().cpu().numpy()

        y_pre_true = (
            np.log10(np.clip(Y[va_idx], float(norm_info["y"].get("eps", 1e-30)), None))
            if bool(norm_info["y"].get("log", False)) else Y[va_idx]
        )
        err = self.rms_per_sample_np(y_pre_hat, y_pre_true)

        self.fit_mean = float(np.mean(err))
        self.fit_median = float(np.percentile(err, 50))
        self.fit_p95 = float(np.percentile(err, 95))

        meta = dict(
            paramnames=dataset.paramnames,
            xlabel=dataset.xlabel,
            ylabel=dataset.ylabel,
            xunit=dataset.xunit,
            yunit=dataset.yunit,
            decoder_paramnames=getattr(self.decoder, "paramnames", None),
            decoder=dict(
                class_name=self.decoder.__class__.__name__,
                module=self.decoder.__class__.__module__,
                config=self.decoder.get_config() if hasattr(self.decoder, "get_config") else None,
            ),
            train_param_bounds=dict(min=Pmin.tolist(), max=Pmax.tolist())
        )
        fit_stats = dict(fit_mean=self.fit_mean, fit_median=self.fit_median, fit_p95=self.fit_p95)

        self._save(norm=norm_info, encoder=enc, nin=nin, nout=nout, meta=meta, fit_stats=fit_stats)

        # keep in memory
        self._norm = norm_info
        self._encoder = enc.to(self.device).double().eval()
        self._meta = meta
        self._loaded = True
        self.paramnames = dataset.paramnames

        return self.model_path

    # ---------- inference ----------
    def transform(self, params, *, x=None, dataset: DatasetReader | None = None, return_latent_params=True):
        # set decoder x from dataset unless user overrides x
        if return_latent_params:
            if x is not None:
                raise ValueError("no x should be set when return_latent_params=True")
            if dataset is not None:
                raise ValueError("no dataset should be set when return_latent_params=True")
        if x is not None:
            self.decoder.x = np.asarray(x, dtype=np.float64)
        elif dataset is not None:
            self.decoder.x = dataset.x
        dec = self.decoder.to(self.device).double().eval()
        for p in dec.parameters():
            p.requires_grad_(False)

        P, squeeze = utils.as_2d(params)
        utils.ensure_finite("params", P)

        self._check_param_bounds(P)
        norm = self._norm
        Pn = self._apply_norm_params(P, norm)

        X = torch.tensor(Pn, dtype=torch.float64, device=self.device)

        with torch.no_grad():
            z_norm = torch.clamp(self._encoder(X), -25, 25).detach().cpu().numpy()
        z_phys = self._apply_latent_norm_inv(z_norm, norm)

        if x is None:
            if not return_latent_params:
                return z_phys[0] if squeeze else z_phys
            if hasattr(dec, "latent_to_params"):
                zt = torch.tensor(z_phys, dtype=torch.float64, device=self.device)
                with torch.no_grad():
                    pars = dec.latent_to_params(zt)
                if isinstance(pars, tuple):
                    pars = torch.stack([p for p in pars], dim=1)
                out = pars.detach().cpu().numpy()
                return out[0] if squeeze else out
            return z_phys[0] if squeeze else z_phys

        zt = torch.tensor(z_phys, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            y_pre = dec(zt).detach().cpu().numpy()

        if bool(norm["y"].get("log", False)):
            y = 10.0 ** y_pre
        else:
            y = y_pre
        if self.param_bounds_policy == 'nan':
            bad = ~self.inside_param_bounds(P)
            y[bad,:] = np.nan
        return y[0] if squeeze else y

    def evaluate(self, dataset: DatasetReader, *, nmax=None, seed=123, return_decoded_params=True):
        P = dataset.params.astype(np.float64)
        Y = dataset.y.astype(np.float64)
        xgrid = dataset.x.astype(np.float64)
        norm = self._norm

        n = len(dataset)
        if nmax is not None and n > int(nmax):
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, size=int(nmax), replace=False)
            P = P[idx]
            Y = Y[idx]
        else:
            idx = np.arange(n)

        self.decoder.x = xgrid
        dec = self.decoder.to(self.device).double().eval()
        for p in dec.parameters():
            p.requires_grad_(False)

        Pn = self._apply_norm_params(P, norm)
        X = torch.tensor(Pn, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            z_norm = torch.clamp(self._encoder(X), -25, 25).detach().cpu().numpy()
        z_phys = self._apply_latent_norm_inv(z_norm, norm)

        zt = torch.tensor(z_phys, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            y_pre_hat = dec(zt).detach().cpu().numpy()

        y_pre_true = (
            np.log10(np.clip(Y, float(norm["y"].get("eps", 1e-30)), None))
            if bool(norm["y"].get("log", False)) else Y
        )
        err = self.rms_per_sample_np(y_pre_hat, y_pre_true)
        yhat = (10.0 ** y_pre_hat) if bool(norm["y"].get("log", False)) else y_pre_hat

        dec_params = None
        if return_decoded_params and hasattr(dec, "latent_to_params"):
            with torch.no_grad():
                pars = dec.latent_to_params(zt)
            if isinstance(pars, tuple):
                pars = torch.stack([p for p in pars], dim=1)
            dec_params = pars.detach().cpu().numpy()

        return dict(
            idx=idx,
            x=xgrid,
            params=P,
            y_true=Y,
            yhat=yhat,
            y_pre_true=y_pre_true,
            y_pre_hat=y_pre_hat,
            err=err,
            z_norm=z_norm,
            z_phys=z_phys,
            dec_params=dec_params,
            xlabel=dataset.xlabel,
            ylabel=dataset.ylabel,
            xunit=dataset.xunit,
            yunit=dataset.yunit,
        )
