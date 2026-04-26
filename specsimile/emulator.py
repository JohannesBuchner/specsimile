from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils
from .store import DatasetReader


class LeakySoftplus(nn.Module):
    """Smooth, zero-centered activation with a configurable negative slope.

    Combines a bias-corrected softplus (for the positive region) with a
    global linear term, giving a kink-free replacement for LeakyReLU that
    is infinitely differentiable everywhere.

    Parameters
    ----------
    negative_slope : float, optional
        Slope of the activation for large negative inputs (x << 0).
        Equivalent to the ``negative_slope`` parameter of ``LeakyReLU``.
        Default is 0.01.
    positive_slope : float, optional
        Target asymptotic slope for large positive inputs. Default is 1.0.

    Notes
    -----
    The forward pass computes::

        LeakySoftplus(x) = softplus(x; beta) - softplus(0; beta)
                           + negative_slope * x

    where ``beta = positive_slope - negative_slope``.

    The derivative is::

        d/dx LeakySoftplus(x) = beta * sigmoid(beta * x)
                                + negative_slope

    Limiting behaviour:

    * x → +∞ : derivative → beta + negative_slope = positive_slope (≈ 1)
    * x = 0  : derivative = beta / 2 + negative_slope
               = (positive_slope - negative_slope) / 2 + negative_slope
               = positive_slope / 2 + negative_slope / 2
    * x → -∞ : derivative → negative_slope

    The zero-centering correction ``- softplus(0; beta)`` ensures
    ``LeakySoftplus(0) = 0`` exactly, which improves gradient flow at
    initialisation when pre-activations are near zero.

    Examples
    --------
    >>> act = LeakySoftplus(negative_slope=0.01)
    >>> act(torch.tensor(0.0))   # exactly zero
    tensor(0.)
    >>> act(torch.tensor(10.0))  # ≈ 10 * 1.0 for large positive x
    tensor(9.9...)
    >>> act(torch.tensor(-10.0)) # ≈ -0.1  (negative_slope * x)
    tensor(-0.0999...)
    """
    def __init__(self, negative_slope: float = 0.01, positive_slope: float = 1.0):
        super().__init__()
        self.negative_slope = negative_slope
        self.beta = positive_slope - negative_slope   # beta_adj: corrected softplus sharpness
        self.register_buffer(
            'zero',
            F.softplus(torch.zeros(1), beta=self.beta)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the LeakySoftplus activation elementwise.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of any shape and dtype.

        Returns
        -------
        torch.Tensor
            Activated tensor, same shape and dtype as ``x``.
            Satisfies ``forward(0) == 0`` exactly.
        """
        sp = F.softplus(x, beta=self.beta) - self.zero
        return sp + self.negative_slope * x

def getacti(actiname):
    """Return the activation function class for a string name."""
    if actiname == 'LeakySoftplus':
        return LeakySoftplus
    else:
        return getattr(nn, actiname)


class MLP(nn.Module):
    def __init__(self, nin: int, nout: int, shape: list[int],
                 acti='LeakyReLU', dropout: float = 0.0):
        super().__init__()
        self.acti = acti
        layers = []
        prev = nin
        for width in shape:
            layers += [nn.Linear(prev, width), getacti(acti)()]
            if dropout > 0.0:
                layers += [nn.Dropout(p=dropout)]
            prev = width
        layers += [nn.Linear(prev, nout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConditionalMLP(nn.Module):
    """
    Separates 'context' parameters (e.g. redshift, fixed during inference)
    from 'inference' parameters (must produce smooth gradients).
    
    Architecture:
      context_net(z) -> embedding of size embed_dim
      main_net([theta, embedding]) -> output
      
    This means: for fixed z, output is a smooth function of theta only,
    because the embedding is a fixed vector and the main_net is smooth.
    """
    def __init__(
        self,
        n_context: int,
        n_inference: int,
        nout: int,
        shape: list[int],
        embed_dim: int = 16,
        acti: str = 'Tanh',
        context_acti: str = 'LeakyReLU',
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_context = n_context
        self.n_inference = n_inference
        self.acti_name = acti
        self.dropout = dropout
        acti_cls = getacti(acti)
        context_acti_cls = getacti(context_acti)
        
        # Context encoder: z -> embedding
        ctx_layers = []
        prev = n_context
        for width in shape[:1]:  # single hidden layer for context
            ctx_layers += [nn.Linear(prev, width), context_acti_cls()]
            prev = width
        ctx_layers += [nn.Linear(prev, embed_dim)]
        self.context_net = nn.Sequential(*ctx_layers)
        
        # Main network: [theta, embedding] -> latent
        main_layers = []
        prev = n_inference + embed_dim
        for width in shape:
            main_layers += [nn.Linear(prev, width), acti_cls()]
            if dropout > 0.0:
                main_layers += [nn.Dropout(p=dropout)]
            prev = width
        main_layers += [nn.Linear(prev, nout)]
        self.main_net = nn.Sequential(*main_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_inference, n_context)
        ctx = x[:, -self.n_context:]
        theta = x[:, :-self.n_context]
        embedding = self.context_net(ctx)   # (B, embed_dim)
        combined = torch.cat([theta, embedding], dim=1)
        return self.main_net(combined)


def softplus_np(z):
    z = np.asarray(z, dtype=np.float64)
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def soft_floor_np(x, floor, k=10.0):
    z = k * (x - floor)
    return floor + softplus_np(z) / k


def soft_floor_torch(x, floor, k=10.0):
    return floor + torch.nn.functional.softplus(k * (x - floor)) / k


class Emulator:
    def __init__(
        self,
        model_path: str,
        decoder,
        shape: list[int],
        acti: str = "LeakyReLU",
        n_context_params: int = 0,
        embed_dim: int = 16,
        context_acti: str = "ReLU",
        *,
        device=None,
        dynamic_range=None,
        func_floor_loss=None,
        loss_y_weights=None,
        param_bounds_policy: str = 'error',
        smooth_weight: float = 0.0,
        smooth_frac: float = 1e-4,
        dropout: float = 0,
    ):
        """
        Parameters
        ----------
        smooth_weight : float
            Weight λ of the finite-difference smoothness penalty relative to
            the main RMS loss.  0.0 disables the penalty entirely.
        smooth_frac : float
            Step size as a fraction of the per-parameter training range
            (hi - lo).  Default 1e-4 means a step of 0.01 % of the range.
        """
        self.model_path = str(model_path)
        self.decoder = decoder
        self.shape = list(shape)
        self.acti = acti
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

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
        self.xnames: list[str] | None = None

        self.paramnames: list | None = None
        self.n_context_params = int(n_context_params)
        self.embed_dim = int(embed_dim)
        self.context_acti = context_acti
        self.dropout = dropout

        self.smooth_weight = float(smooth_weight)
        self.smooth_frac = float(smooth_frac)
        # set during fit() once the parameter ranges are known
        self._smooth_step: torch.Tensor | None = None

        if loss_y_weights is None:
            self.loss_y_weights = 1.0
        else:
            self.loss_y_weights = torch.tensor(
                loss_y_weights.reshape((1, -1)),
                dtype=torch.float64,
                device=self.device,
            )

    # ---------- helpers ----------

    def _y_mean_std(self, norm: dict, y_dim: int):
        yinfo = norm.get("y", {})
        mean = yinfo.get("mean", None)
        std = yinfo.get("std", None)
        mean = (
            np.zeros(y_dim, np.float64) if mean is None
            else np.asarray(mean, np.float64)
        )
        std = (
            np.ones(y_dim, np.float64) if std is None
            else np.asarray(std, np.float64)
        )
        return mean, std

    def inside_param_bounds(self, P: np.ndarray):
        b = self.train_param_bounds
        if b is None or self.param_bounds_tol < 0:
            return
        pmin = np.asarray(b["min"], dtype=np.float64).reshape(1, -1)
        pmax = np.asarray(b["max"], dtype=np.float64).reshape(1, -1)
        if P.shape[1] != pmin.shape[1]:
            raise ValueError("Parameter dimension mismatch vs stored bounds")
        return np.all(
            (P > pmin - self.param_bounds_tol) & (P < pmax + self.param_bounds_tol),
            axis=1,
        )

    def _check_param_bounds(self, P: np.ndarray):
        b = self.train_param_bounds
        if b is None or self.param_bounds_tol < 0 or self.param_bounds_policy != 'error':
            return
        bad = ~self.inside_param_bounds(P)
        if np.any(bad):
            i = int(np.where(bad)[0][0])
            s = ""
            if self.paramnames is not None:
                for j, (pname, plo, phi) in enumerate(
                    zip(self.paramnames, b["min"], b["max"])
                ):
                    if not (
                        P[i, j] > plo - self.param_bounds_tol
                        and P[i, j] < phi + self.param_bounds_tol
                    ):
                        s += (
                            f"Input parameter {j+1} {pname}={P[i,j]} "
                            f"outside training bounds [{plo}..{phi}] "
                            f"(first offending row {i}).\n"
                        )
            pmin = np.asarray(b["min"], dtype=np.float64)
            pmax = np.asarray(b["max"], dtype=np.float64)
            raise ValueError(
                s
                + f"Input params outside training bounds (first offending row {i}).\n"
                f"params={P[i]}\nmin={pmin}\nmax={pmax}\n"
                f"paramnames={self.paramnames}"
            )

    def _apply_norm_params(self, P: np.ndarray, norm: dict) -> np.ndarray:
        pn = norm["params"]
        if "log_mask" in pn:
            P = utils.apply_log_mask(P, pn["log_mask"], pn.get("eps", 1e-30))
        return utils.normalize(P, pn["mean"], pn["std"])

    def _apply_latent_norm_inv(self, z_norm: np.ndarray, norm: dict) -> np.ndarray:
        lat = norm["latent"]
        std = np.asarray(lat["std"], np.float64).reshape(1, -1) if "std" in lat else 1.0
        mean = np.asarray(lat["mean"], np.float64).reshape(1, -1) if "mean" in lat else 0.0
        return z_norm * std + mean

    def _target_y_pre(self, Y: np.ndarray, norm: dict) -> np.ndarray:
        if bool(norm["y"].get("log", False)):
            eps = float(norm["y"].get("eps", 1e-30))
            Ypre = np.log10(np.clip(Y, eps, None))
        else:
            Ypre = Y
        mean, std = self._y_mean_std(norm, Ypre.shape[1])
        return (Ypre - mean[None, :]) / std[None, :]

    def _build_smooth_step(self, Pmin_inf, Pmax_inf, norm):
        """
        Build per-parameter step sizes in *normalised* input space.

        Parameters
        ----------
        Pmin_inf, Pmax_inf : (n_inf,) arrays
            Physical min/max of the inference parameters only
            (context params excluded).
        norm : dict
            Normalisation info from decoder.normalize().

        Returns
        -------
        torch.Tensor, shape (1, n_inf)
            Step size for each inference parameter in normalised space.
        """
        pn = norm["params"]
        std_norm = np.asarray(pn["std"], dtype=np.float64)
        n_inf = len(Pmin_inf)
        std_norm_inf = std_norm[:n_inf]                        # inference params only

        delta_phys = self.smooth_frac * (Pmax_inf - Pmin_inf) # physical step
        delta_norm = delta_phys / np.clip(std_norm_inf, 1e-30, None)  # normalised step

        return torch.tensor(
            delta_norm.reshape(1, -1), dtype=torch.float64, device=self.device
        )
    def _taylor_consistency_penalty(
        self, xb, enc, dec, lat_mean, lat_std, y_mean, y_std,
        direction, *, create_graph: bool = True,
    ):
        """
        Penalise gradients inconsistent with nearby function values.

            residual = f(x + delta) - f(x) - delta · grad_f(x)

        Parameters
        ----------
        create_graph : bool
            True during training (penalty must be differentiable w.r.t. weights).
            False during validation (only need the scalar value).
        """
        B = xb.shape[0]
        n_ctx = self.n_context_params

        assert self._smooth_step is not None
        # _smooth_step: (1, n_inf), already in normalised-input units,
        # scaled by the physical parameter range fraction smooth_frac.
        delta_inf = direction * self._smooth_step          # (B, n_inf)

        # Pad context dimensions with zero (context does not move)
        if n_ctx > 0:
            delta_full = torch.cat([
                delta_inf,
                xb.new_zeros(B, n_ctx),
            ], dim=1)
        else:
            delta_full = delta_inf

        def forward_standardized(x_in):
            z_norm = torch.clamp(enc(x_in), -25, 25)
            z_phys = z_norm * lat_std + lat_mean
            y_pre  = dec(z_phys)
            return (y_pre - y_mean) / y_std

        # --- f(x + delta): detached, no gradient needed ---
        with torch.no_grad():
            fx_perturbed = forward_standardized(xb + delta_full)   # (B, Y)

        # --- f(x): keep in graph so encoder weights get gradients ---
        # Do NOT detach xb. The gradient of the penalty w.r.t. encoder
        # weights flows through fx -> enc(xb) -> encoder parameters.
        xb_req = xb.requires_grad_(True)  # in-place, no detach
        fx = forward_standardized(xb_req)

        # --- grad_f(x) w.r.t. xb: needs create_graph=True so that
        #     the gradient of (residual^2) w.r.t. encoder weights
        #     can be computed via the chain:
        #     residual -> linear_prediction -> grad_fx -> fx -> enc weights
        # ---
        grad_fx = torch.autograd.grad(
            outputs=fx.sum(),
            inputs=xb_req,
            create_graph=create_graph,
            retain_graph=True,         # fx is used again below for residual
        )[0]                                                       # (B, n_input)

        # --- Taylor residual ---
        linear_prediction = (grad_fx * delta_full).sum(dim=1, keepdim=True)  # (B, 1)

        # fx_perturbed is detached (computed under no_grad).
        # fx is in the graph — this is what lets the penalty train the encoder.
        residual = (fx_perturbed - fx) - linear_prediction                    # (B, Y)

        return (residual ** 2).mean()

    def _gradient_magnitude_penalty(
        self, xb, enc, dec, lat_mean, lat_std, y_mean, y_std,
        *, create_graph: bool = True,
    ):
        """
        Penalise large input-gradients at training points.

        Computes d f(x) / d x_inf (inference dims only) and penalises
        its squared magnitude. This directly discourages spikes in the
        learned function at training locations.

        For a piecewise-linear network (LeakyReLU), this penalises the
        slope of each active linear piece, pushing the network toward
        shallower, wider pieces — exactly what reduces gradient noise
        seen by HMC.
        """
        B = xb.shape[0]
        n_ctx = self.n_context_params
        n_inf = xb.shape[1] - n_ctx

        def forward_standardized(x_in):
            z_norm = torch.clamp(enc(x_in), -25, 25)
            z_phys = z_norm * lat_std + lat_mean
            y_pre  = dec(z_phys)
            return (y_pre - y_mean) / y_std        # (B, Y)

        xb_req = xb.detach().requires_grad_(True)
        fx = forward_standardized(xb_req)          # (B, Y)

        # Sum over outputs and batch to get a scalar, then differentiate.
        # create_graph=True so that loss.backward() can propagate through
        # grad_fx back to the encoder weights.
        grad_fx = torch.autograd.grad(
            outputs=fx.sum(),
            inputs=xb_req,
            create_graph=create_graph,
        )[0]                                       # (B, n_input)

        # Penalise only inference dims (context is fixed during HMC anyway)
        grad_inf = grad_fx[:, :n_inf]              # (B, n_inf)

        return (grad_inf ** 2).mean()


    def _gradient_penalty(self, xb, enc, dec, lat_mean, lat_std, y_mean, y_std):
        if self.smooth_weight == 0.0:
            return xb.new_tensor(0.0)

        n_ctx = self.n_context_params
        B = xb.shape[0]
        n_inf = xb.shape[1] - n_ctx if n_ctx > 0 else xb.shape[1]

        x_inf = xb[:, :-n_ctx] if n_ctx > 0 else xb
        x_ctx = xb[:, -n_ctx:] if n_ctx > 0 else None

        # Random unit direction in inference-param space, one per sample
        v = torch.randn(B, n_inf, dtype=xb.dtype, device=xb.device)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-30)   # (B, n_inf)

        def forward_fn(x_inf_single, x_ctx_single):
            if n_ctx > 0:
                x_full = torch.cat([
                    x_inf_single.unsqueeze(0),
                    x_ctx_single.unsqueeze(0)
                ], dim=1)
            else:
                x_full = x_inf_single.unsqueeze(0)
            z_norm = torch.clamp(enc(x_full), -25, 25)
            z_phys = z_norm * lat_std + lat_mean
            y_pre = dec(z_phys).squeeze(0)
            return (y_pre - y_mean.squeeze(0)) / y_std.squeeze(0)  # standardized

        if n_ctx > 0:
            jvp_fn = torch.func.vmap(
                lambda xi, xc, vi: torch.func.jvp(
                    lambda t: forward_fn(t, xc), (xi,), (vi,)
                )[1]  # take only the tangent, not the primal
            )
            directional_deriv = jvp_fn(x_inf, x_ctx, v)  # (B, Y)
        else:
            jvp_fn = torch.func.vmap(
                lambda xi, vi: torch.func.jvp(
                    forward_fn, (xi,), (vi,)
                )[1]
            )
            directional_deriv = jvp_fn(x_inf, v)

        return (directional_deriv ** 2).mean()

    # ---------- checkpoint IO ----------

    def load(self, verbose: bool = True):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(self.model_path)

        ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)

        self._norm = ckpt["norm"]
        self._meta = ckpt.get("meta", {})
        self.paramnames = self._meta.get("paramnames", None)
        self.xnames = self._meta.get("xnames", None)

        if verbose:
            print(f"loading emulator from {self.model_path}:")
            for k, v in self._meta.items():
                print(f" {k:10s}: {v}")

        self.train_param_bounds = self._meta.get("train_param_bounds", None)

        dmeta = self._meta.get("decoder", None)
        if dmeta is not None:
            want_cls = dmeta.get("class_name")
            got_cls = self.decoder.__class__.__name__
            if want_cls is not None and want_cls != got_cls:
                raise ValueError(
                    f"Decoder class mismatch: ckpt has {want_cls}, "
                    f"but you provided {got_cls}"
                )
            want_cfg = dmeta.get("config", None)
            if want_cfg is not None and hasattr(self.decoder, "get_config"):
                got_cfg = self.decoder.get_config()
                if got_cfg != want_cfg:
                    raise ValueError(
                        "Decoder config mismatch between checkpoint and "
                        f"provided decoder.\nckpt: {want_cfg}\ngot:  {got_cfg}"
                    )

        self.fit_mean = ckpt.get("fit_mean", None)
        self.fit_median = ckpt.get("fit_median", None)
        self.fit_p95 = ckpt.get("fit_p95", None)
        self.shape = ckpt["shape"]
        self.acti = ckpt.get("acti", "LeakyReLU")
        self.n_context_params = int(ckpt.get("n_context_params", 0))
        self.embed_dim = int(ckpt.get("embed_dim", 16))
        self.context_acti = ckpt.get("context_acti", "ReLU")
        self.dropout = float(ckpt.get('dropout', 0.0))

        nin = int(ckpt["nin"])
        nout = int(ckpt["nout"])
        n_inference = nin - self.n_context_params
        if self.n_context_params > 0:
            self._encoder = ConditionalMLP(
                n_context=self.n_context_params,
                n_inference=n_inference,
                nout=nout,
                shape=self.shape,
                embed_dim=self.embed_dim,
                acti=self.acti,
                context_acti=self.context_acti,
                dropout=self.dropout,
            ).to(self.device).double()
        else:
            self._encoder = MLP(
                nin=nin, nout=nout, shape=self.shape, acti=self.acti, dropout=self.dropout,
            ).to(self.device).double()
        self._encoder.load_state_dict(ckpt["encoder_state"])
        self._encoder.to(self.device).double().eval()
        self._loaded = True
        return self._loaded

    def _save(
        self,
        *,
        norm: dict,
        encoder: nn.Module,
        nin: int,
        nout: int,
        meta: dict,
        fit_stats: dict,
    ):
        ckpt = dict(
            shape=self.shape,
            acti=self.acti,
            n_context_params=self.n_context_params,
            embed_dim=self.embed_dim,
            context_acti=self.context_acti,
            dropout=self.dropout,
            norm=norm,
            meta=meta,
            nin=int(nin),
            nout=int(nout),
            encoder_state=encoder.state_dict(),
            **fit_stats,
        )
        torch.save(ckpt, self.model_path)

    def _arch_string(self, xgrid: np.ndarray, P: np.ndarray, Y: np.ndarray) -> str:
        in_dim = int(P.shape[1])
        ydim = int(Y.shape[1])
        enc_str = "-".join(str(w) for w in self.shape)
        dec_name = self.decoder.__class__.__name__.replace("Decoder", "")
        latent_dim = getattr(self.decoder, "latent_dim", "?")
        enc_str += f'^{self.acti}'
        if self.n_context_params:
            enc_str += f'|{self.n_context_params:d}'
            enc_str += f'{self.context_acti}'
        if self.dropout > 0:
            enc_str += f'|Drop{self.dropout}'
        return (
            f"[P:{in_dim}]-[MLP:{enc_str}]->[latent:{latent_dim}]"
            f"-{dec_name}-[y:{ydim}]"
        )

    # ---------- loss weighting ----------

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
        return sig2.expand(B, D)

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
        ymax = torch.max(y, dim=1, keepdim=True).values
        floor = ymax - dr
        if self.func_floor_loss is not None:
            floor = torch.maximum(floor, floor.new_tensor(float(self.func_floor_loss)))

        return soft_floor_torch(yhat, floor=floor), soft_floor_torch(y, floor=floor)

    def _apply_dynamic_range_np(
        self, yhat: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply dynamic-range clamping in y_pre space (numpy).
        """
        if self.dynamic_range is None:
            return yhat, y
        dr = float(self.dynamic_range)
        ymax = np.max(y, axis=1, keepdims=True)

        floor = ymax - dr
        if self.func_floor_loss is not None:
            floor = np.maximum(floor, float(self.func_floor_loss))

        return soft_floor_np(yhat, floor=floor), soft_floor_np(y, floor=floor)

    def rms_per_sample_torch(
        self, yhat: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns (B,) RMS in y_pre space, with dynamic-range clamp applied if configured.
        """
        yhat2, y2 = self._apply_dynamic_range_torch(yhat, y)
        return torch.sqrt(
            torch.mean((yhat2 - y2) ** 2 * self.loss_y_weights, dim=1)
        )

    def rms_per_sample_np(self, yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Returns (N,) RMS in y_pre space, with dynamic-range clamp applied if configured.
        """
        yhat2, y2 = self._apply_dynamic_range_np(
            np.asarray(yhat), np.asarray(y)
        )
        return np.sqrt(
            np.mean((yhat2 - y2) ** 2 * self.loss_y_weights, axis=1)
        )

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

        if self.smooth_weight > 0.0:
            print(
                f"Smoothness penalty (gradient/JVP): weight={self.smooth_weight}"
            )

        n = len(dataset)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        ntr = max(1, int(round(train_split * n)))
        tr_idx = idx[:ntr]
        va_idx = idx[ntr:] if ntr < n else idx[:1]

        dtype = torch.float64
        Xtr = torch.tensor(Pn[tr_idx], dtype=dtype, device=self.device)
        Xva = torch.tensor(Pn[va_idx], dtype=dtype, device=self.device)
        Ttr = torch.tensor(T[tr_idx], dtype=dtype, device=self.device)
        Tva = torch.tensor(T[va_idx], dtype=dtype, device=self.device)

        nin = P.shape[1]
        nout = latent_dim
        if self.n_context_params > 0:
            enc = ConditionalMLP(
                n_context=self.n_context_params,
                n_inference=nin - self.n_context_params,
                nout=nout,
                shape=self.shape,
                embed_dim=self.embed_dim,
                acti=self.acti,
                context_acti=self.context_acti,
                dropout=self.dropout,
            ).to(self.device).double()
        else:
            enc = MLP(nin=nin, nout=nout, shape=self.shape, acti=self.acti).to(self.device).double()
        opt = torch.optim.Adam(enc.parameters(), lr=lr)

        lat = norm_info["latent"]
        lat_mean = (
            torch.tensor(
                np.asarray(lat["mean"]).reshape(1, -1),
                dtype=dtype, device=self.device,
            )
            if "mean" in lat else 0.0
        )
        lat_std = (
            torch.tensor(
                np.asarray(lat["std"]).reshape(1, -1),
                dtype=dtype, device=self.device,
            )
            if "std" in lat else 1.0
        )

        y_mean_np, y_std_np = self._y_mean_std(norm_info, Y.shape[1])
        y_mean = torch.tensor(
            y_mean_np.reshape(1, -1), dtype=dtype, device=self.device
        )
        y_std = torch.tensor(
            y_std_np.reshape(1, -1), dtype=dtype, device=self.device
        )
        if self.smooth_weight > 0.0:
            n_inf = P.shape[1] - self.n_context_params
            # prepare random directions in R^P, uniform on the unit ball,
            # then scale each component by the per-parameter step size
            # --- Build perturbation in inference-param space only ---
            self._smooth_step = self._build_smooth_step(
                Pmin[:n_inf], Pmax[:n_inf], norm_info
            )

        best = np.inf
        best_state = None
        bad = 0

        for ep in range(int(epochs)):
            enc.train()
            perm = rng.permutation(len(Xtr))
            if self.smooth_weight > 0.0:
                rand_normal = torch.randn(len(Xtr), n_inf, dtype=Xtr.dtype, device=Xtr.device)
                rand_normal = rand_normal / (rand_normal.norm(dim=1, keepdim=True) + 1e-30)
                rand_normal_va = torch.randn(len(Xva), n_inf, dtype=Xva.dtype, device=Xva.device)
                rand_normal_va = rand_normal_va / (rand_normal_va.norm(dim=1, keepdim=True) + 1e-30)
            for i0 in range(0, len(Xtr), batch_size):
                sl = perm[i0: i0 + batch_size]
                xb = Xtr[sl]
                tb = Ttr[sl]

                z_norm = torch.clamp(enc(xb), -25, 25)
                z_phys = z_norm * lat_std + lat_mean
                y_pre = dec(z_phys)
                y_hat = (y_pre - y_mean) / y_std

                main_loss = self.loss_fn(y_hat, tb) ** 2

                if self.smooth_weight > 0.0:
                    if True:
                        smooth_loss = self._taylor_consistency_penalty(
                            xb, enc, dec, lat_mean, lat_std, y_mean, y_std,
                            # instead of sl, we intentionally use i0, to have linearly increasing memory access
                            # the numbers are random anyway.
                            direction=rand_normal[i0: i0 + batch_size],
                            create_graph=True,  # differentiable
                        )
                    else:
                        smooth_loss = self._gradient_magnitude_penalty(
                            xb, enc, dec, lat_mean, lat_std, y_mean, y_std,
                            create_graph=True,
                        )
                    loss = main_loss + self.smooth_weight * smooth_loss
                else:
                    loss = main_loss

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                opt.step()

            # validation uses only the main loss so that early stopping
            # reflects predictive accuracy, not smoothness
            enc.eval()
            with torch.no_grad():
                z_norm = torch.clamp(enc(Xva), -25, 25)
                z_phys = z_norm * lat_std + lat_mean
                y_pre = dec(z_phys)
                y_hat = (y_pre - y_mean) / y_std
                vloss = self.loss_fn(y_hat, Tva).item() ** 2

            if self.smooth_weight > 0.0:
                # No torch.no_grad() here: _taylor_consistency_penalty needs
                # autograd.grad for f(x), even with create_graph=False.
                # But enc should remain in eval() mode.
                
                # Sample a subset to keep validation fast
                if True:
                    vsmooth = self._taylor_consistency_penalty(
                        Xva, enc, dec, lat_mean, lat_std, y_mean, y_std,
                        direction=rand_normal_va,
                        create_graph=False,  # not differentiable
                    ).item()
                else:
                    vsmooth = self._gradient_magnitude_penalty(
                        Xva, enc, dec, lat_mean, lat_std, y_mean, y_std,
                        create_graph=False,
                    ).item()
                vtotal = vloss + self.smooth_weight * vsmooth
                loss_str = (
                    f"MSE={vloss:.3f}  "
                    f"+smooth={vsmooth:.3f} * {self.smooth_weight} -> "
                    f"{vtotal:.3f}"
                )
            else:
                vtotal = vloss
                loss_str = f"MSE={vloss:.3f}"

            if vtotal < best - 1e-4:
                best = vtotal
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in enc.state_dict().items()
                }
                bad = 0
                print(f"Epoch {ep+1}/{epochs}: {loss_str}")
            else:
                bad += 1
                print(f"Epoch {ep+1}/{epochs}: {loss_str} [{bad}/{patience}]")
                if bad >= patience:
                    break

        if best_state is not None:
            enc.load_state_dict(best_state)
        enc.eval()

        with torch.no_grad():
            z_norm = torch.clamp(enc(Xva), -25, 25).detach().cpu().numpy()
        z_phys = self._apply_latent_norm_inv(z_norm, norm_info)

        zt = torch.tensor(z_phys, dtype=dtype, device=self.device)
        with torch.no_grad():
            y_pre_hat = dec(zt).detach().cpu().numpy()

        y_pre_true = (
            np.log10(
                np.clip(Y[va_idx], float(norm_info["y"].get("eps", 1e-30)), None)
            )
            if bool(norm_info["y"].get("log", False))
            else Y[va_idx]
        )
        err = self.rms_per_sample_np(y_pre_hat, y_pre_true)

        self.fit_mean = float(np.mean(err))
        self.fit_median = float(np.percentile(err, 50))
        self.fit_p95 = float(np.percentile(err, 95))
        self.xnames = dataset.xnames

        meta = dict(
            paramnames=dataset.paramnames,
            xnames=dataset.xnames,
            xlabel=dataset.xlabel,
            ylabel=dataset.ylabel,
            xunit=dataset.xunit,
            yunit=dataset.yunit,
            decoder_paramnames=getattr(self.decoder, "paramnames", None),
            decoder=dict(
                class_name=self.decoder.__class__.__name__,
                module=self.decoder.__class__.__module__,
                config=(
                    self.decoder.get_config()
                    if hasattr(self.decoder, "get_config") else None
                ),
            ),
            train_param_bounds=dict(min=Pmin.tolist(), max=Pmax.tolist()),
        )
        fit_stats = dict(
            fit_mean=self.fit_mean,
            fit_median=self.fit_median,
            fit_p95=self.fit_p95,
        )

        self._save(
            norm=norm_info, encoder=enc, nin=nin, nout=nout,
            meta=meta, fit_stats=fit_stats,
        )

        # keep in memory
        self._norm = norm_info
        self._encoder = enc.to(self.device).double().eval()
        self._meta = meta
        self._loaded = True
        self.paramnames = dataset.paramnames

        return self.model_path

    # ---------- inference ----------

    def transform(
        self,
        params,
        *,
        x=None,
        dataset: DatasetReader | None = None,
        return_latent_params: bool = True,
    ):
        # set decoder x from dataset unless user overrides x
        if return_latent_params:
            if x is not None:
                raise ValueError("no x should be set when return_latent_params=True")
            if dataset is not None:
                raise ValueError(
                    "no dataset should be set when return_latent_params=True"
                )
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
                    pars = torch.stack(list(pars), dim=1)
                out = pars.detach().cpu().numpy()
                return out[0] if squeeze else out
            return z_phys[0] if squeeze else z_phys

        zt = torch.tensor(z_phys, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            y_pre = dec(zt).detach().cpu().numpy()

        y = 10.0 ** y_pre if bool(norm["y"].get("log", False)) else y_pre
        if self.param_bounds_policy == 'nan':
            y[~self.inside_param_bounds(P), :] = np.nan
        return y[0] if squeeze else y

    def to_dict(self, y: np.ndarray) -> dict[str, np.ndarray]:
        """
        Convert a y array (N, Y) or (Y,) into a dict keyed by xnames.

        Requires that xnames were stored during training (i.e. the dataset
        had a xnames column).  Raises RuntimeError if xnames are unavailable.

        Parameters
        ----------
        y : (Y,) or (N, Y) float array as returned by transform() or evaluate()

        Returns
        -------
        dict mapping each output name to its column, shape (N,) or scalar.
        """
        if self.xnames is None:
            raise RuntimeError(
                "xnames are not available for this emulator.  "
                "Re-train with a dataset that has xnames set, or set "
                "emulator.xnames manually."
            )
        y = np.asarray(y)
        squeeze = y.ndim == 1
        if squeeze:
            y = y[None, :]
        if y.shape[1] != len(self.xnames):
            raise ValueError(
                f"y has {y.shape[1]} columns but xnames has "
                f"{len(self.xnames)} entries"
            )
        result = {name: y[:, i] for i, name in enumerate(self.xnames)}
        if squeeze:
            result = {k: v[0] for k, v in result.items()}
        return result

    def evaluate(
        self,
        dataset: DatasetReader,
        *,
        nmax=None,
        seed: int = 123,
        return_decoded_params: bool = True,
    ):
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

        dtype = torch.float64
        Pn = self._apply_norm_params(P, norm)
        X = torch.tensor(Pn, dtype=dtype, device=self.device)
        with torch.no_grad():
            z_norm = torch.clamp(self._encoder(X), -25, 25).detach().cpu().numpy()
        z_phys = self._apply_latent_norm_inv(z_norm, norm)

        zt = torch.tensor(z_phys, dtype=dtype, device=self.device)
        with torch.no_grad():
            y_pre_hat = dec(zt).detach().cpu().numpy()

        y_pre_true = (
            np.log10(np.clip(Y, float(norm["y"].get("eps", 1e-30)), None))
            if bool(norm["y"].get("log", False))
            else Y
        )
        err = self.rms_per_sample_np(y_pre_hat, y_pre_true)
        yhat = 10.0 ** y_pre_hat if bool(norm["y"].get("log", False)) else y_pre_hat

        dec_params = None
        if return_decoded_params and hasattr(dec, "latent_to_params"):
            with torch.no_grad():
                pars = dec.latent_to_params(zt)
            if isinstance(pars, tuple):
                pars = torch.stack(list(pars), dim=1)
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
            xnames=self.xnames,
            xlabel=dataset.xlabel,
            ylabel=dataset.ylabel,
            xunit=dataset.xunit,
            yunit=dataset.yunit,
        )
