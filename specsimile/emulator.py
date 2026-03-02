import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import emulator_utils as utils
import matplotlib.pyplot as plt
import matplotlib as mpl


class MLP(nn.Module):
    def __init__(self, nin: int, nout: int, shape: list):
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
    Updated Emulator with latent normalization and optional y mean/std.

    Decoder contract
    ---------------
    - decoder = decoder_cls(x_grid) must be an nn.Module with:
        - forward(z_phys) -> y_pre
          where y_pre is either y (if norm['y']['log']=False) or log10(y) (if True)
        - latent_to_params(z_phys) (optional but used for returning params)
        - normalize(y, params) -> norm dict with at least:
            norm['y']['log'] : bool
            norm['params'] : {log_mask, mean, std, eps}
            norm['latent'] : {mean, std, log_mask}   (mean/std shape (latent_dim,))
            norm['latent_dim'] : int

      norm['y']['mean'] and norm['y']['std'] may be missing => treated as 0 and 1.
    """

    def __init__(self, filename, decoder_cls, shape, tolerance=0.01, dynamic_range=None, device=None, paramnames=None):
        self.filename = filename
        self.decoder_cls = decoder_cls
        self.shape = shape
        self.tolerance = float(tolerance)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.train_filename = filename
        self.train_file = None
        self.num_data_training = 0
        self.N = 0
        self.paramnames = paramnames

        base = os.path.splitext(self.train_filename)[0]
        self.model_path = base + "_model.pt"

        self._loaded = False
        self._encoder = None
        self._norm = None
        self.dynamic_range = dynamic_range
        self._loss_fn = nn.MSELoss()
        self._y_peak_floor = None

    def loss_fn(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        yhat, y: (B, D) in the *training target space* (usually log10(y) or y_pre).
        dynamic_range:
          - if None: plain MSE
          - else: clamp both yhat and y to at least (per-sample max - dynamic_range)
        """
        return self.rms_per_sample_torch(yhat, y).mean()

    def close(self):
        if self.train_file is not None:
            try:
                self.train_file.close()
                self.train_file = None
            except Exception:
                pass

    # ---------- recording ----------

    def start_recording(self, N, x, y, params):
        self.train_file = h5py.File(self.train_filename, mode="w")
        N = int(N)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        params = np.asarray(params, dtype=np.float64)

        if x.ndim != 1:
            raise ValueError(f"x must be 1D, got {x.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got {y.shape}")
        if params.ndim != 1:
            raise ValueError(f"params must be 1D, got {params.shape}")

        for key in ("x", "y", "params"):
            if key in self.train_file:
                raise RuntimeError(f"Dataset '{key}' already exists in {self.filename}")

        self.num_data_training = 0
        self.N = N

        self.train_file.create_dataset("x", data=x)
        self.train_file.create_dataset("y", dtype=np.float64, shape=(N, len(y)))
        self.train_file.create_dataset("params", dtype=np.float64, shape=(N, len(params)))
        self.train_file.create_dataset("paramnames", data=self.paramnames)

        self.train_file.attrs["num_data_training"] = self.num_data_training
        self.train_file.attrs["N"] = self.N
        self.train_file.flush()

    def append(self, x, y, params):
        if not all(k in self.train_file for k in ("x", "y", "params")):
            raise RuntimeError("Call start_recording(...) first (datasets missing).")

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        params = np.asarray(params, dtype=np.float64)

        if x.ndim != 1 or y.ndim != 1 or params.ndim != 1:
            raise ValueError("x,y,params must be 1D arrays")

        if len(y) != self.train_file["y"].shape[1]:
            raise ValueError("y length mismatch")
        if len(params) != self.train_file["params"].shape[1]:
            raise ValueError("params length mismatch")

        x0 = self.train_file["x"][()]
        if x.shape != x0.shape or not np.allclose(x, x0, rtol=0, atol=0):
            raise ValueError("x does not match stored x grid")

        i = self.num_data_training
        if i >= self.N:
            raise RuntimeError(f"Training file full (N={self.N}).")

        utils.ensure_finite("params", params)
        utils.ensure_finite("y", y)

        self.train_file["y"][i, :] = y
        self.train_file["params"][i, :] = params
        self.num_data_training += 1
        self.train_file.attrs["num_data_training"] = self.num_data_training
        self.train_file.flush()
        # done training, can close file for writing
        if self.num_data_training == len(self.train_file["y"]):
            self.train_file.close()
            self.train_file = None
        return i

    # ---------- norm helpers (NEW) ----------

    def _y_mean_std(self, norm: dict, y_dim: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Allow norm['y']['mean'] and/or norm['y']['std'] to be missing.
        Defaults: mean=0, std=1.
        """
        yinfo = norm.get("y", {})
        mean = yinfo.get("mean", None)
        std = yinfo.get("std", None)

        if mean is None:
            mean = np.zeros(y_dim, dtype=np.float64)
        else:
            mean = np.asarray(mean, dtype=np.float64)
        if std is None:
            std = np.ones(y_dim, dtype=np.float64)
        else:
            std = np.asarray(std, dtype=np.float64)

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
        """
        Convert z_norm -> z_phys using norm['latent'].
        """
        lat = norm["latent"]
        mean = np.asarray(lat["mean"], dtype=np.float64).reshape(1, -1)
        std = np.asarray(lat["std"], dtype=np.float64).reshape(1, -1)
        if not np.all(std > 0):
            raise ValueError("latent std must be > 0")
        return z_norm * std + mean

    def _apply_y_pre_norm(self, y_pre: np.ndarray, norm: dict) -> np.ndarray:
        """
        Apply optional y mean/std normalization in pre-space (either y or log10(y)).
        """
        y_pre = np.asarray(y_pre, dtype=np.float64)
        y_dim = y_pre.shape[1]
        mean, std = self._y_mean_std(norm, y_dim)
        return (y_pre - mean[None, :]) / std[None, :]

    def _target_y_pre(self, Y: np.ndarray, norm: dict) -> np.ndarray:
        """
        Convert stored Y (function output) -> y_pre target space:
          if norm['y']['log']: log10(Y)
          else: Y
        then apply optional mean/std.
        """
        yinfo = norm["y"]
        if bool(yinfo.get("log", False)):
            eps = float(yinfo.get("eps", 1e-30))
            Ypre = np.log10(np.clip(Y, eps, None))
        else:
            Ypre = Y
        return self._apply_y_pre_norm(Ypre, norm)


    def _apply_dynamic_range_torch(self, yhat: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply dynamic-range clamping in the *same space* as yhat,y (y_pre space).
        """
        if self.dynamic_range is None:
            return yhat, y

        dr = float(self.dynamic_range)
        ymax = torch.max(y, dim=1, keepdim=True).values  # (B,1)

        # global minimum allowed ymax (in y_pre / standardized-target space)
        if hasattr(self, "_y_peak_floor") and (self._y_peak_floor is not None):
            min_ymax = float(self._y_peak_floor) - dr
            #print(f"replacing ymax={ymax.min()} with {min_ymax}")
            ymax = torch.maximum(ymax, torch.tensor(min_ymax, dtype=y.dtype, device=y.device))

        floor = ymax - dr
        return torch.clamp(yhat, min=floor), torch.clamp(y, min=floor)

    def _apply_dynamic_range_np(self, yhat: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply dynamic-range clamping in y_pre space (numpy).
        """
        if self.dynamic_range is None:
            return yhat, y
        dr = float(self.dynamic_range)
        ymax = np.max(y, axis=1, keepdims=True)

        if hasattr(self, "_y_peak_floor") and (self._y_peak_floor is not None):
            min_ymax = float(self._y_peak_floor) - dr
            #print(f"replacing ymax={ymax.min()} with {min_ymax}")
            ymax = np.maximum(ymax, min_ymax)

        floor = ymax - dr
        return np.maximum(yhat, floor), np.maximum(y, floor)

    def rms_per_sample_torch(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Returns (B,) RMS in y_pre space, with dynamic-range clamp applied if configured.
        """
        yhat2, y2 = self._apply_dynamic_range_torch(yhat, y)
        return torch.sqrt(torch.mean((yhat2 - y2) ** 2, dim=1))

    def rms_per_sample_np(self, yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Returns (N,) RMS in y_pre space, with dynamic-range clamp applied if configured.
        """
        yhat2, y2 = self._apply_dynamic_range_np(np.asarray(yhat), np.asarray(y))
        return np.sqrt(np.mean((yhat2 - y2) ** 2, axis=1))

    # ---------- model IO ----------

    def _load_model_if_available(self):
        if self._loaded:
            return
        if not os.path.exists(self.model_path):
            return

        ckpt = torch.load(self.model_path, map_location="cpu")
        self._norm = ckpt["norm"]
        self._y_peak_floor = ckpt.get("y_peak_floor", None)

        nin = int(ckpt["nin"])
        nout = int(ckpt["nout"])
        self._encoder = MLP(nin=nin, nout=nout, shape=ckpt["shape"])
        self._encoder.load_state_dict(ckpt["encoder_state"])
        self._encoder.to(self.device).double().eval()
        self._loaded = True
        if "paramnames" in ckpt:
            self.paramnames = [p.decode('utf-8') for p in ckpt["paramnames"]]
        else:
            self.paramnames = None
        print('load', self.paramnames)

    def _save_model(self, norm: dict, encoder: nn.Module, nin: int, nout: int):
        ckpt = dict(
            shape=self.shape,
            decoder_name=self.decoder_cls.__name__,
            norm=norm,
            paramnames = self.paramnames,
            nin=int(nin),
            nout=int(nout),
            y_peak_floor=self._y_peak_floor,
            encoder_state=encoder.state_dict(),
        )
        torch.save(ckpt, self.model_path)

    def _arch_string(self, xgrid: np.ndarray, P: np.ndarray, Y: np.ndarray) -> str:
        #xdim = int(np.asarray(xgrid).shape[0])   # optional, just for display
        in_dim = int(P.shape[1])
        ydim = int(Y.shape[1])

        enc_str = "-".join(str(w) for w in self.shape)

        dec = self.decoder_cls(xgrid)
        dec_name = dec.__class__.__name__.replace('Decoder', '')

        latent_dim = getattr(dec, "latent_dim", None)
        if latent_dim is None:
            latent_dim = "?"

        return f"[P:{in_dim}]-[MLP:{enc_str}]->[latent:{latent_dim}]-{dec_name}-[y:{ydim}]"
    
    def train_if_needed(
        self,
        epochs,
        lr,
        batch_size,
        min_points=200,
        train_split=0.8,
        patience=15,
        seed=123,
    ):
        """
        Train encoder to predict z_norm; decoder maps z_phys->y_pre.
        Target is y_pre (log10(y) if norm['y']['log'] else y), optionally mean/std normalized.
        Accept if p95 L2 in y_pre-space (log if requested) < tolerance.
        """
        if os.path.exists(self.model_path):
            return False

        if self.train_file is None:
            self.train_file = h5py.File(self.train_filename, mode="r")
        Y = self.train_file["y"][()].astype(np.float64)
        P = self.train_file["params"][()].astype(np.float64)
        utils.ensure_finite("P", P)
        utils.ensure_finite("Y", Y)
        if self.paramnames is None:
            self.paramnames = [p for p in self.train_file["paramnames"][()]]
        print("training data:", Y.shape, P.shape)

        self.num_data_training = len(Y)
        n = self.num_data_training
        if n < int(min_points):
            return False

        # compute norm via decoder
        x = self.train_file["x"][()]
        print("emulator architecture:", self._arch_string(x, P, Y))
        dec = self.decoder_cls(x).to(self.device).double().eval()
        for p in dec.parameters():
            p.requires_grad_(False)

        norm_info = dec.normalize(Y, P)
        if "latent" not in norm_info:
            raise ValueError("decoder.normalize must return norm['latent'] for latent normalization")
        for k, v in norm_info.items():
            print(f'Standardization for {k}: {v}')

        # normalize inputs
        Pn = self._apply_norm_params(P, norm_info)

        # build targets in y_pre-space (+ optional mean/std)
        T = self._target_y_pre(Y, norm_info)  # (N, Ydim)

        # split
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        ntr = max(1, int(round(train_split * n)))
        tr_idx = idx[:ntr]
        va_idx = idx[ntr:] if ntr < n else idx[:1]

        Xtr = torch.tensor(Pn[tr_idx], dtype=torch.float64, device=self.device)
        Xva = torch.tensor(Pn[va_idx], dtype=torch.float64, device=self.device)
        Ttr = torch.tensor(T[tr_idx], dtype=torch.float64, device=self.device)
        Tva = torch.tensor(T[va_idx], dtype=torch.float64, device=self.device)

        latent_dim = int(norm_info.get("latent_dim", 0))
        if latent_dim <= 0:
            raise ValueError("norm['latent_dim'] must be a positive int")

        # build raw targets in y_pre-space
        yinfo = norm_info["y"]
        if bool(yinfo.get("log", False)):
            # eps = float(yinfo.get("eps", 1e-30))
            y_pre_all = np.log10(Y)
        else:
            y_pre_all = Y

        y_pre_tr = y_pre_all[tr_idx]  # (Ntr, Ydim)

        with torch.no_grad():
            #t = torch.tensor(y_pre_tr, dtype=torch.float64, device=self.device)
            t = np.nanmax(y_pre_tr, 0)
            y_mean = np.nanmean(t)
            y_std = np.nanstd(t) + 1e-12
            self._y_peak_floor = (y_mean - 3.0 * y_std).item()
            print(f"setting floor to {self._y_peak_floor} based on mean={y_mean} and std={y_std}")
        assert np.isfinite(self._y_peak_floor), self._y_peak_floor
        del y_pre_tr, y_pre_all, t

        nin = P.shape[1]
        nout = latent_dim
        enc = MLP(nin=nin, nout=nout, shape=self.shape).to(self.device).double()

        opt = torch.optim.Adam(enc.parameters(), lr=lr)

        # cache latent mean/std on device
        lat = norm_info["latent"]
        lat_mean = torch.tensor(np.asarray(lat["mean"]).reshape(1, -1), dtype=torch.float64, device=self.device)
        lat_std = torch.tensor(np.asarray(lat["std"]).reshape(1, -1), dtype=torch.float64, device=self.device)

        # cache y mean/std on device (allow missing)
        y_dim = Y.shape[1]
        y_mean_np, y_std_np = self._y_mean_std(norm_info, y_dim)
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

                z_norm = torch.clamp(enc(xb), -10, 10)
                z_phys = z_norm * lat_std + lat_mean
                y_pre = dec(z_phys)  # (B, Ydim) in pre-space (log if requested)
                y_hat = (y_pre - y_mean) / y_std

                loss = self.loss_fn(y_hat, tb)
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

        # evaluate tolerance in y_pre-space (log if norm['y']['log'] is True)
        with torch.no_grad():
            z_norm = torch.clamp(enc(Xva), -10, 10).detach().cpu().numpy()
        z_phys = self._apply_latent_norm_inv(z_norm, norm_info)

        # decode
        zt = torch.tensor(z_phys, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            y_pre_hat = dec(zt).detach().cpu().numpy()

        # Compare in y_pre-space:
        # - if log=True: y_pre = log10(y), so compare to log10(Y)
        # - else: y_pre = y, so compare to Y
        # Apply p95 L2 on raw y_pre (NOT the standardized version); your tolerance is in that space.
        y_pre_true = (np.log10(np.clip(Y[va_idx], float(norm_info["y"].get("eps", 1e-30)), None))
                      if bool(norm_info["y"].get("log", False)) else Y[va_idx])
        rms = self.rms_per_sample_np(y_pre_hat, y_pre_true)
        err_med = float(np.percentile(rms, 50))
        err_p95 = float(np.percentile(rms, 95))
        print(f"Final performance: median loss={err_med:.3f}")
        print(f"Final performance: 95% quantile loss={err_p95:.3f} [desired tolerance: {self.tolerance}]")

        if err_p95 < self.tolerance:
            self._save_model(norm=norm_info, encoder=enc, nin=nin, nout=nout)
            return True
        print("Need more training data or a better model.")
        return False


    def transform(self, params, x=None, return_latent_params=True):
        """
        params: (P,) or (B,P)

        If x is None:
          returns decoder parameters via decoder.latent_to_params(z_phys) if available
          (else returns z_phys).
        If x is not None:
          evaluates decoder on provided x (fresh decoder instance) and returns:
            - if norm['y']['log'] True: returns 10**log10(y) (linear y)
            - else: returns y
        """
        self._load_model_if_available()
        if not self._loaded:
            self._load_model_if_available()
        if not self._loaded:
            raise RuntimeError(f"Model not trained yet (missing {self.model_path}).")

        P, squeeze = utils.as_2d(params)
        utils.ensure_finite("params", P)

        norm = self._norm

        # normalize params
        Pn = self._apply_norm_params(P, norm)
        X = torch.tensor(Pn, dtype=torch.float64, device=self.device)

        with torch.no_grad():
            z_norm = torch.clamp(self._encoder(X), -10, 10).detach().cpu().numpy()

        z_phys = self._apply_latent_norm_inv(z_norm, norm)

        # If no x requested: return decoder params
        if x is None:
            if not return_latent_params:
                return z_phys[0] if squeeze else z_phys

            # instantiate decoder on stored x grid (or any, doesn't matter for latent_to_params)
            x0 = self.train_file["x"][()]
            dec = self.decoder_cls(x0).to(self.device).double().eval()
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

        # Evaluate on provided x grid
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


    def evaluate(self, nmax=None, seed=123, return_decoded_params=True):
        """
        Evaluate emulator on stored training points.

        Returns a dict with:
          x               : (X,)
          params          : (N,P)
          y_true          : (N,Y)  original function output space
          yhat            : (N,Y)  original function output space
          y_pre_true      : (N,Y)  tolerance space: log10(y) if norm['y']['log'] else y
          y_pre_hat       : (N,Y)  tolerance space
          err             : (N,)   L2 per sample in tolerance space
          z_norm          : (N,L)
          z_phys          : (N,L)
          dec_params      : (N,K) or None  (decoder.latent_to_params if available)
        """
        self._load_model_if_available()
        if not self._loaded:
            self._load_model_if_available()
        if not self._loaded:
            raise RuntimeError(f"Model not trained yet (missing {self.model_path}).")

        if self.train_file is None:
            self.train_file = h5py.File(self.train_filename, mode="r")
        P = self.train_file["params"][()].astype(np.float64)
        Y = self.train_file["y"][()].astype(np.float64)
        xgrid = self.train_file["x"][()].astype(np.float64)
        norm = self._norm
        if self.paramnames is None:
            self.paramnames = [p for p in self.train_file["paramnames"][()]]

        self.num_data_training = len(Y)
        n = self.num_data_training
        if n <= 0:
            raise RuntimeError("No training data recorded.")

        # optional downsample
        if nmax is not None and n > int(nmax):
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, size=int(nmax), replace=False)
            P = P[idx]
            Y = Y[idx]
        else:
            idx = np.arange(n)

        # predict z_norm
        Pn = self._apply_norm_params(P, norm)
        X = torch.tensor(Pn, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            z_norm = torch.clamp(self._encoder(X), -10, 10).detach().cpu().numpy()

        # denormalize latent
        z_phys = self._apply_latent_norm_inv(z_norm, norm)

        # decode in pre-space
        dec = self.decoder_cls(xgrid).to(self.device).double().eval()
        for p in dec.parameters():
            p.requires_grad_(False)

        zt = torch.tensor(z_phys, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            y_pre_hat = dec(zt).detach().cpu().numpy()

        # true in pre-space
        y_pre_true = (np.log10(np.clip(Y, float(norm["y"].get("eps", 1e-30)), None))
                      if bool(norm["y"].get("log", False)) else Y)

        # error in tolerance space
        err = self.rms_per_sample_np(y_pre_hat, y_pre_true)

        # convert both to original output space for convenience
        if bool(norm["y"].get("log", False)):
            yhat = 10.0 ** y_pre_hat
        else:
            yhat = y_pre_hat

        # decoded physical decoder parameters (optional)
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

    # ----------------------------
    # Plot helpers
    # ----------------------------

    def plot_error_hist(self, eval_out, bins=50, logx=True):
        """
        Histogram of per-sample L2 errors in tolerance space.
        Returns (fig, ax).
        """
        err = np.asarray(eval_out["err"], float)
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        e = err[np.isfinite(err)]
        if len(e) == 0:
            ax.text(0.5, 0.5, "no finite errors", ha="center", va="center")
            return fig, ax

        ax.hist(e, bins=bins, color="0.6", edgecolor="0.2")
        p95 = np.percentile(e, 95)
        mean_RMS = np.mean(e)
        ax.axvline(self.tolerance, color="C2", lw=2, ls="--", label=f"tolerance={self.tolerance:.3g}")
        ax.axvline(mean_RMS, color="C4", lw=2, ls="--", label=f"mean={mean_RMS:.3g}")
        ax.axvline(p95, color="C3", lw=2, label=f"p95={p95:.3g}")
        ax.set_xlabel("RMS error (tolerance space)")
        ax.set_ylabel("count")
        #if logx and np.nanmax(e) / max(np.nanmin(e[e > 0]), 1e-30) > 100:
        #    ax.set_xscale("log")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig, ax

    def plot_fit_examples(self, eval_out, n_first=12, n_worst=12, yscale="log"):
        """
        Overlay y_true vs yhat on x grid for a few samples.
        Uses original output space (so if tolerance is log-space, we still show linear y).

        Returns (fig, axes) with shape (2, ncol).
        """
        x = eval_out["x"]
        y = eval_out["y_true"]
        yhat = eval_out["yhat"]
        err = eval_out["err"]

        n = len(err)
        n_first = min(int(n_first), n)
        n_worst = min(int(n_worst), n)

        idx_first = np.arange(n_first)
        idx_worst = np.argsort(err)[-n_worst:][::-1]

        show = np.concatenate([idx_first, idx_worst])
        nshow = len(show)
        ncol = max(1, int(np.ceil(nshow / 2)))
        ncol2 = max(1, ncol // 2)

        fig, axes = plt.subplots(
            4, ncol2, figsize=(3.2 * ncol2, 6 * 2),
            squeeze=False, sharex=True,
        )

        def _plot(ax, i):
            yt = np.asarray(y[i], float)
            yp = np.asarray(yhat[i], float)
            if yscale == "log":
                ax.loglog(x, np.clip(yt, 1e-300, None), "k-", lw=1.5, label="true")
                ax.loglog(x, np.clip(yp, 1e-300, None), "r--", lw=1.2, label="pred")
            else:
                ax.plot(x, yt, "k-", lw=1.5, label="true")
                ax.plot(x, yp, "r--", lw=1.2, label="pred")

            ax.set_title(f"i={i}  err={err[i]:.3g}", fontsize=9)
            ax.grid(True, which="both", alpha=0.2)

        for j, i in enumerate(idx_first[::2]):
            _plot(axes[0, j], i)
        for j, i in enumerate(idx_first[1::2]):
            _plot(axes[1, j], i)
        for j, i in enumerate(idx_worst[::2]):
            _plot(axes[2, j], i)
        for j, i in enumerate(idx_worst[1::2]):
            _plot(axes[3, j], i)


        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        return fig, axes

    def plot_param_corner_scatter(self, eval_out, param_names=None, max_points=5000):
        """
        Pairwise scatter of input params colored by log10(err).
        Returns (fig, axes).
        """
        P = np.asarray(eval_out["params"], float)
        err = np.asarray(eval_out["err"], float)
        N, D = P.shape
        if N == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "no points", ha="center", va="center")
            return fig, np.array([[ax]])

        if N > max_points:
            idx = np.random.choice(N, size=max_points, replace=False)
            P = P[idx]
            err = err[idx]
            N = max_points

        c = np.log10(np.clip(err, 1e-30, None))
        vmin, vmax = np.percentile(c, [5, 95])
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("viridis_r")

        fig, axes = plt.subplots(D, D, figsize=(2.1 * D, 2.1 * D), squeeze=False)

        if param_names is None:
            param_names = self.paramnames
        if param_names is None or len(param_names) != D:
            param_names = [f"p{i}" for i in range(D)]
        labels = param_names

        for i in range(D):
            for j in range(D):
                ax = axes[i, j]
                if i == j:
                    ax.hist(P[:, j], bins=30, color="0.7")
                    ax.set_yticks([])
                elif i > j:
                    ax.scatter(P[:, j], P[:, i], c=c, s=6, cmap=cmap, norm=norm, alpha=0.6, linewidths=0)
                else:
                    ax.axis("off")

                if i == D - 1 and j <= i:
                    ax.set_xlabel(labels[j], fontsize=8, rotation=45)
                else:
                    ax.set_xticks([])
                if j == 0 and i > 0:
                    ax.set_ylabel(labels[i], fontsize=8)
                else:
                    ax.set_yticks([])

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
        cbar.set_label("log10(L2 error)")

        fig.tight_layout()
        return fig, axes

    def plot_decoder_param_corner_scatter(self, eval_out, param_names=None, max_points=5000):
        """
        Pairwise scatter of decoded decoder parameters (from latent_to_params), colored by log10(err).
        Returns (fig, axes). If decoder params not available, raises.
        """
        Q = eval_out.get("dec_params", None)
        if Q is None:
            raise RuntimeError("dec_params not available (decoder may not implement latent_to_params)")

        Q = np.asarray(Q, float)
        err = np.asarray(eval_out["err"], float)
        N, D = Q.shape

        if N > max_points:
            idx = np.random.choice(N, size=max_points, replace=False)
            Q = Q[idx]
            err = err[idx]
            N = max_points

        c = np.log10(np.clip(err, 1e-30, None))
        vmin, vmax = np.percentile(c, [5, 95])
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("viridis_r")

        fig, axes = plt.subplots(D, D, figsize=(2.1 * D, 2.1 * D), squeeze=False)
        labels = param_names if (param_names is not None and len(param_names) == D) else [f"q{i}" for i in range(D)]

        for i in range(D):
            for j in range(D):
                ax = axes[i, j]
                if i == j:
                    ax.hist(Q[:, j], bins=30, color="0.7")
                    ax.set_yticks([])
                elif i > j:
                    ax.scatter(Q[:, j], Q[:, i], c=c, s=6, cmap=cmap, norm=norm, alpha=0.6, linewidths=0)
                else:
                    ax.axis("off")

                if i == D - 1 and j <= i:
                    ax.set_xlabel(labels[j], fontsize=8, rotation=45)
                else:
                    ax.set_xticks([])
                if j == 0 and i > 0:
                    ax.set_ylabel(labels[i], fontsize=8)
                else:
                    ax.set_yticks([])

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
        cbar.set_label("log10(L2 error)")

        fig.tight_layout()
        return fig, axes
