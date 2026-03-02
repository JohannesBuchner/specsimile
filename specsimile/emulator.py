# specsimile/emulator.py
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
    """
    Stateless emulator model:
      - no training-data I/O
      - train/evaluate/transform take a DatasetReader when needed

    Decoder contract (unchanged)
    ----------------------------
    decoder = decoder_cls(x_grid) must be nn.Module with:
      - forward(z_phys) -> y_pre
      - normalize(y, params) -> norm dict with:
            norm['y']['log'] : bool
            norm['params'] : {mean, std, optional log_mask, eps}
            norm['latent'] : {mean, std, log_mask}
            norm['latent_dim'] : int
      - latent_to_params(z_phys) optional
    """

    def __init__(self, filename: str, decoder_cls, shape: list[int], *, device=None):
        self.filename = str(filename)
        self.decoder_cls = decoder_cls
        self.shape = list(shape)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        base = os.path.splitext(self.filename)[0]
        self.model_path = base + "_model.pt"

        self._loaded = False
        self._encoder: MLP | None = None
        self._norm: dict | None = None
        self._meta: dict = {}  # store paramnames/labels/units from dataset at train-time

    # ---------- norm helpers ----------

    def _y_mean_std(self, norm: dict, y_dim: int) -> tuple[np.ndarray, np.ndarray]:
        yinfo = norm.get("y", {})
        mean = yinfo.get("mean", None)
        std = yinfo.get("std", None)

        mean = np.zeros(y_dim, dtype=np.float64) if mean is None else np.asarray(mean, dtype=np.float64)
        std = np.ones(y_dim, dtype=np.float64) if std is None else np.asarray(std, dtype=np.float64)

        if mean.shape != (y_dim,) or std.shape != (y_dim,):
            raise ValueError(f"y mean/std must have shape ({y_dim},), got {mean.shape}, {std.shape}")
        if not np.all(std > 0):
            raise ValueError("y std must be > 0")
        return mean, std

    def _apply_norm_params(self, P: np.ndarray, norm: dict) -> np.ndarray:
        pn = norm["params"]
        if "log_mask" in pn:
            P2 = utils.apply_log_mask(P, pn["log_mask"], pn.get("eps", 1e-30))
        else:
            P2 = P
        return utils.normalize(P2, pn["mean"], pn["std"])

    def _apply_latent_norm_inv(self, z_norm: np.ndarray, norm: dict) -> np.ndarray:
        lat = norm["latent"]
        mean = np.asarray(lat["mean"], dtype=np.float64).reshape(1, -1)
        std = np.asarray(lat["std"], dtype=np.float64).reshape(1, -1)
        if not np.all(std > 0):
            raise ValueError("latent std must be > 0")
        return z_norm * std + mean

    def _apply_y_pre_norm(self, y_pre: np.ndarray, norm: dict) -> np.ndarray:
        y_pre = np.asarray(y_pre, dtype=np.float64)
        mean, std = self._y_mean_std(norm, y_pre.shape[1])
        return (y_pre - mean[None, :]) / std[None, :]

    def _target_y_pre(self, Y: np.ndarray, norm: dict) -> np.ndarray:
        yinfo = norm["y"]
        if bool(yinfo.get("log", False)):
            eps = float(yinfo.get("eps", 1e-30))
            Ypre = np.log10(np.clip(Y, eps, None))
        else:
            Ypre = Y
        return self._apply_y_pre_norm(Ypre, norm)

    # ---------- model IO ----------

    def load(self):
        if self._loaded:
            return
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Missing trained model: {self.model_path}")

        ckpt = torch.load(self.model_path, map_location="cpu")
        self._norm = ckpt["norm"]
        self._meta = ckpt.get("meta", {})

        nin = int(ckpt["nin"])
        nout = int(ckpt["nout"])
        self._encoder = MLP(nin=nin, nout=nout, shape=ckpt["shape"])
        self._encoder.load_state_dict(ckpt["encoder_state"])
        self._encoder.to(self.device).double().eval()
        self._loaded = True

    def _save(self, *, norm: dict, encoder: nn.Module, nin: int, nout: int, meta: dict):
        ckpt = dict(
            shape=self.shape,
            norm=norm,
            meta=meta,
            nin=int(nin),
            nout=int(nout),
            encoder_state=encoder.state_dict(),
        )
        torch.save(ckpt, self.model_path)

    # ---------- training ----------

    def train(
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
        """
        Train encoder to predict z_norm.
        Always saves the trained model at the end (unless overwrite=False and file exists).

        Returns: model_path
        """
        if (not overwrite) and os.path.exists(self.model_path):
            return self.model_path

        Y = dataset.y.astype(np.float64)
        P = dataset.params.astype(np.float64)
        x = dataset.x.astype(np.float64)

        utils.ensure_finite("P", P)
        utils.ensure_finite("Y", Y)

        # decoder-defined normalization
        dec = self.decoder_cls(x).to(self.device).double().eval()
        for p in dec.parameters():
            p.requires_grad_(False)

        norm_info = dec.normalize(Y, P)
        if "latent" not in norm_info:
            raise ValueError("decoder.normalize must return norm['latent']")
        latent_dim = int(norm_info.get("latent_dim", 0))
        if latent_dim <= 0:
            raise ValueError("norm['latent_dim'] must be a positive int")

        # normalize inputs + build targets
        Pn = self._apply_norm_params(P, norm_info)
        T = self._target_y_pre(Y, norm_info)

        # split
        n = len(Pn)
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

        # cached norm tensors
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
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            enc.load_state_dict(best_state)
        enc.eval()

        meta = dict(
            filename=self.filename,
            paramnames=dataset.paramnames,
            xlabel=dataset.xlabel,
            ylabel=dataset.ylabel,
            xunit=dataset.xunit,
            yunit=dataset.yunit,
        )
        self._save(norm=norm_info, encoder=enc, nin=nin, nout=nout, meta=meta)

        # keep loaded instance in memory too
        self._norm = norm_info
        self._encoder = enc
        self._encoder.to(self.device).double().eval()
        self._meta = meta
        self._loaded = True

        return self.model_path

    # ---------- inference ----------

    def transform(self, params, *, x=None, dataset: DatasetReader | None = None, return_latent_params=True):
        """
        If x is not None: return y in original output space on x.
        If x is None: return decoded decoder parameters (latent_to_params) if available.
        Needs dataset (or trained meta + a provided x) to normalize params / construct decoder.
        """
        self.load()

        if dataset is None:
            # We still need param normalization stats; those are in self._norm, so dataset is optional.
            # But if x is None and you want latent_to_params, we need an x-grid to instantiate decoder.
            pass

        P, squeeze = utils.as_2d(params)
        utils.ensure_finite("params", P)

        norm = self._norm
        Pn = self._apply_norm_params(P, norm)
        X = torch.tensor(Pn, dtype=torch.float64, device=self.device)

        with torch.no_grad():
            z_norm = torch.clamp(self._encoder(X), -10, 10).detach().cpu().numpy()
        z_phys = self._apply_latent_norm_inv(z_norm, norm)

        # No x requested: return decoder parameters
        if x is None:
            if not return_latent_params:
                return z_phys[0] if squeeze else z_phys

            # need an x-grid to instantiate decoder for latent_to_params
            if dataset is None:
                raise ValueError("dataset must be provided when x=None (need dataset.x to build decoder)")
            dec = self.decoder_cls(dataset.x).to(self.device).double().eval()
            for p in dec.parameters():
                p.requires_grad_(False)

            if hasattr(dec, "latent_to_params"):
                zt = torch.tensor(z_phys, dtype=torch.float64, device=self.device)
                with torch.no_grad():
                    pars = dec.latent_to_params(zt)
                if isinstance(pars, tuple):
                    pars = torch.stack([p for p in pars], dim=1)
                pars = pars.detach().cpu().numpy()
                return pars[0] if squeeze else pars
            return z_phys[0] if squeeze else z_phys

        # Evaluate on provided x
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError(f"x must be 1D, got {x.shape}")

        dec = self.decoder_cls(x).to(self.device).double().eval()
        for p in dec.parameters():
            p.requires_grad_(False)

        zt = torch.tensor(z_phys, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            y_pre = dec(zt).detach().cpu().numpy()

        if bool(norm["y"].get("log", False)):
            y = 10.0 ** y_pre
        else:
            y = y_pre

        return y[0] if squeeze else y

    def evaluate(self, dataset: DatasetReader, *, nmax=None, seed=123, return_decoded_params=True):
        """
        Evaluate emulator on a dataset.
        Returns same dict as before (minus tolerance-related items).
        """
        self.load()

        P = dataset.params.astype(np.float64)
        Y = dataset.y.astype(np.float64)
        xgrid = dataset.x.astype(np.float64)
        norm = self._norm

        n = len(Y)
        if nmax is not None and n > int(nmax):
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, size=int(nmax), replace=False)
            P = P[idx]
            Y = Y[idx]
        else:
            idx = np.arange(n)

        Pn = self._apply_norm_params(P, norm)
        X = torch.tensor(Pn, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            z_norm = torch.clamp(self._encoder(X), -10, 10).detach().cpu().numpy()
        z_phys = self._apply_latent_norm_inv(z_norm, norm)

        dec = self.decoder_cls(xgrid).to(self.device).double().eval()
        for p in dec.parameters():
            p.requires_grad_(False)

        zt = torch.tensor(z_phys, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            y_pre_hat = dec(zt).detach().cpu().numpy()

        y_pre_true = (
            np.log10(np.clip(Y, float(norm["y"].get("eps", 1e-30)), None))
            if bool(norm["y"].get("log", False)) else Y
        )

        # RMS per sample in tolerance space (but no "tolerance logic")
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
