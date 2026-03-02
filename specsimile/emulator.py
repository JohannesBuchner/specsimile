from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn

from . import utils
from .store import DatasetReader


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


class Emulator:
    def __init__(self, model_path: str, decoder, shape: list[int], *, device=None):
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

    # ---------- helpers ----------
    def _y_mean_std(self, norm: dict, y_dim: int):
        yinfo = norm.get("y", {})
        mean = yinfo.get("mean", None)
        std = yinfo.get("std", None)
        mean = np.zeros(y_dim, np.float64) if mean is None else np.asarray(mean, np.float64)
        std = np.ones(y_dim, np.float64) if std is None else np.asarray(std, np.float64)
        return mean, std

    def _apply_norm_params(self, P: np.ndarray, norm: dict) -> np.ndarray:
        pn = norm["params"]
        if "log_mask" in pn:
            P = utils.apply_log_mask(P, pn["log_mask"], pn.get("eps", 1e-30))
        return utils.normalize(P, pn["mean"], pn["std"])

    def _apply_latent_norm_inv(self, z_norm: np.ndarray, norm: dict) -> np.ndarray:
        lat = norm["latent"]
        mean = np.asarray(lat["mean"], np.float64).reshape(1, -1)
        std = np.asarray(lat["std"], np.float64).reshape(1, -1)
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
        x = dataset.x.astype(np.float64)
        utils.ensure_finite("P", P)
        utils.ensure_finite("Y", Y)

        # set x on decoder (required by new contract)
        self.decoder.x = x
        dec = self.decoder.to(self.device).double().eval()
        for p in dec.parameters():
            p.requires_grad_(False)

        norm_info = dec.normalize(Y, P)
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
        lat_mean = torch.tensor(np.asarray(lat["mean"]).reshape(1, -1), dtype=torch.float64, device=self.device)
        lat_std = torch.tensor(np.asarray(lat["std"]).reshape(1, -1), dtype=torch.float64, device=self.device)

        y_mean_np, y_std_np = self._y_mean_std(norm_info, Y.shape[1])
        y_mean = torch.tensor(y_mean_np.reshape(1, -1), dtype=torch.float64, device=self.device)
        y_std = torch.tensor(y_std_np.reshape(1, -1), dtype=torch.float64, device=self.device)

        def rms_loss(yhat, ytrue):
            return torch.sqrt(torch.mean((yhat - ytrue) ** 2, dim=1)).mean()

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

                z_norm = torch.clamp(enc(xb), -10, 10)
                z_phys = z_norm * lat_std + lat_mean
                y_pre = dec(z_phys)
                y_hat = (y_pre - y_mean) / y_std

                loss = rms_loss(y_hat, tb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                opt.step()

            enc.eval()
            with torch.no_grad():
                z_norm = torch.clamp(enc(Xva), -10, 10)
                z_phys = z_norm * lat_std + lat_mean
                y_pre = dec(z_phys)
                y_hat = (y_pre - y_mean) / y_std
                vloss = rms_loss(y_hat, Tva).item()

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
            z_norm = torch.clamp(enc(Xva), -10, 10).detach().cpu().numpy()
        z_phys = self._apply_latent_norm_inv(z_norm, norm_info)

        zt = torch.tensor(z_phys, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            y_pre_hat = dec(zt).detach().cpu().numpy()

        y_pre_true = (
            np.log10(np.clip(Y[va_idx], float(norm_info["y"].get("eps", 1e-30)), None))
            if bool(norm_info["y"].get("log", False)) else Y[va_idx]
        )
        err = np.sqrt(np.mean((y_pre_hat - y_pre_true) ** 2, axis=1))

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
        )
        fit_stats = dict(fit_mean=self.fit_mean, fit_median=self.fit_median, fit_p95=self.fit_p95)

        self._save(norm=norm_info, encoder=enc, nin=nin, nout=nout, meta=meta, fit_stats=fit_stats)

        # keep in memory
        self._norm = norm_info
        self._encoder = enc.to(self.device).double().eval()
        self._meta = meta
        self._loaded = True

        return self.model_path

    # ---------- inference ----------
    def transform(self, params, *, x=None, dataset: DatasetReader | None = None, return_latent_params=True):
        if dataset is None:
            raise ValueError("dataset must be provided (needed for x-grid and consistent decoder usage)")

        # set decoder x from dataset unless user overrides x
        grid = dataset.x if x is None else np.asarray(x, dtype=np.float64)
        self.decoder.x = grid
        dec = self.decoder.to(self.device).double().eval()
        for p in dec.parameters():
            p.requires_grad_(False)

        P, squeeze = utils.as_2d(params)
        utils.ensure_finite("params", P)

        norm = self._norm
        Pn = self._apply_norm_params(P, norm)
        X = torch.tensor(Pn, dtype=torch.float64, device=self.device)

        with torch.no_grad():
            z_norm = torch.clamp(self._encoder(X), -10, 10).detach().cpu().numpy()
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
            z_norm = torch.clamp(self._encoder(X), -10, 10).detach().cpu().numpy()
        z_phys = self._apply_latent_norm_inv(z_norm, norm)

        zt = torch.tensor(z_phys, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            y_pre_hat = dec(zt).detach().cpu().numpy()

        y_pre_true = (
            np.log10(np.clip(Y, float(norm["y"].get("eps", 1e-30)), None))
            if bool(norm["y"].get("log", False)) else Y
        )
        err = np.sqrt(np.mean((y_pre_hat - y_pre_true) ** 2, axis=1))
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
        )
