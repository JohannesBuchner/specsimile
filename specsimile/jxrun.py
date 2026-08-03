"""Functions for using the emulators from jax """
from typing import Dict, List, Union
import jax
import jax.numpy as jnp
import numpy as np
import torch




# ============================================================================
# Part 2: Extract weights from PyTorch emulator models (specsimile)
# ============================================================================

def extract_pytorch_emulator_weights(model_path: str, verbose: bool = True) -> tuple:
    """
    Extract weights from a specsimile Emulator saved as a PyTorch checkpoint.

    Parameters
    ----------
    model_path : str
        Path to the PyTorch .pt checkpoint file.
    verbose : bool, optional
        If True, print diagnostic information about the loaded checkpoint.
        Default is True.

    Returns
    -------
    result : dict of str to np.ndarray
        Dictionary mapping parameter names to numpy arrays, plus metadata
        entries (e.g. '_norm', 'decoder_type', 'paramnames', etc.).
    checkpoint : dict
        The raw checkpoint dictionary as loaded by torch.load.
    """
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    result = {}
    
    if verbose:
        print(f"  Loaded {model_path}:")
        print(f"    top-level keys: { {k: type(v).__name__ for k, v in checkpoint.items()} }")
    
    # Find state dict
    state = None
    for key in ['model_state_dict', 'state_dict', 'encoder_state']:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            has_tensors = any(isinstance(v, torch.Tensor) for v in checkpoint[key].values())
            if has_tensors:
                state = checkpoint[key]
                if verbose:
                    print(f"    Using state dict from '{key}'")
                break
    
    if state is None:
        state = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
    
    for key, val in state.items():
        if isinstance(val, torch.Tensor):
            result[key] = val.detach().numpy()
        elif isinstance(val, np.ndarray):
            result[key] = val
    
    # Extract normalization
    for norm_key in ('norm', '_norm'):
        if norm_key in checkpoint and isinstance(checkpoint[norm_key], dict):
            result['_norm'] = checkpoint[norm_key]
            break
    
    # Extract other metadata
    for meta_key in ('decoder_type', 'decoder_config', 'shape', 'meta',
                     'x', 'paramnames', 'dynamic_range', 'func_floor_loss',
                     'fit_mean', 'fit_median', 'fit_p95', 'nin', 'nout'):
        if meta_key in checkpoint:
            result[meta_key] = checkpoint[meta_key]
    
    tensor_keys = sorted([k for k in result if isinstance(result[k], np.ndarray)])
    if verbose:
        print(f"    tensor keys ({len(tensor_keys)}): {tensor_keys}")
    
    if '_norm' in result:
        norm = result['_norm']
        if verbose:
            print(f"    norm keys: {sorted(norm.keys())}")
        for nk, nv in sorted(norm.items()):
            if isinstance(nv, dict):
                for k2, v2 in sorted(nv.items()):
                    shape = v2.shape if hasattr(v2, 'shape') else ''
                    if verbose:
                        print(f"      norm[{nk}][{k2}] = {type(v2).__name__} {shape} {v2 if not hasattr(v2, 'shape') else v2.flatten()[:5]}")
            else:
                if verbose:
                    print(f"      norm[{nk}] = {type(nv).__name__} {nv}")
    
    return result, checkpoint

def jax_encoder_forward(x: jnp.ndarray, bundle: dict, prefix: str) -> jnp.ndarray:
    """
    Forward pass through a specsimile encoder stored in *bundle* under *prefix*.

    Handles both plain MLP and ConditionalMLP (when n_context_params > 0).

    Parameters
    ----------
    x : jnp.ndarray
        Input array of shape (N, n_inputs).
    bundle : dict
        Parameter bundle as returned by :func:`load_decoder_bundle`.
    prefix : str
        Key prefix used when the bundle was built.

    Returns
    -------
    h : jnp.ndarray
        Denormalised latent / decoder parameters of shape (N, latent_dim).
    """
    p = f'{prefix}_' if prefix else ''
    h = x

    # --- Input normalisation ------------------------------------------------
    log_mask_key = f'{p}norm_x_log_mask'
    if log_mask_key in bundle:
        log_mask = bundle[log_mask_key]
        eps = bundle.get(f'{p}norm_x_eps', 1e-30)
        h_log = jnp.log10(jnp.clip(h, eps, None))
        h = jnp.where(jnp.array(log_mask)[None, :], h_log, h)

    x_mean_key = f'{p}norm_x_mean'
    x_std_key  = f'{p}norm_x_std'
    if x_mean_key in bundle:
        h = (h - bundle[x_mean_key]) / jnp.clip(bundle[x_std_key], 1e-10, None)

    n_context = bundle.get(f'{p}n_context_params', 0)

    acti_name = bundle[f'{p}acti']
    ctx_acti_name = bundle.get(f'{p}context_acti', 'ReLU')
    if n_context > 0:
        # ConditionalMLP: split context and inference parts
        ctx   = h[:, -n_context:]      # (N, n_context)
        theta = h[:, :-n_context]      # (N, n_inference)

        # Context embedding
        ctx_out = ctx
        ctx_n_layers = bundle.get(f'{p}ctx_n_layers', 0)
        for i in range(ctx_n_layers):
            ctx_out = ctx_out @ bundle[f'{p}ctx_W{i}'] + bundle[f'{p}ctx_b{i}']
            if i < ctx_n_layers - 1:
                ctx_out = _jax_apply_activation(ctx_out, ctx_acti_name)

        # Concatenate theta + embedding
        h = jnp.concatenate([theta, ctx_out], axis=1)
    
    # --- Main MLP -----------------------------------------------------------
    n_layers = bundle[f'{p}n_layers']
    for i in range(n_layers):
        W = bundle[f'{p}W{i}']
        b = bundle[f'{p}b{i}']
        h = h @ W + b
        if i < n_layers - 1:
            h = _jax_apply_activation(h, acti_name)

    # --- Output denormalisation ---------------------------------------------
    y_mean_key = f'{p}norm_y_mean'
    y_std_key  = f'{p}norm_y_std'
    if y_mean_key in bundle:
        h = h * bundle[y_std_key] + bundle[y_mean_key]

    return h

def _jax_sigmoid_asinh(x: jnp.ndarray, lo: float, hi: float) -> jnp.ndarray:
    """
    JAX version of specsimile sigmoid_asinh: R -> (lo, hi).

    Parameters
    ----------
    x : jnp.ndarray
        Input array of arbitrary shape.
    lo : float
        Lower bound of the output range.
    hi : float
        Upper bound of the output range.

    Returns
    -------
    jnp.ndarray
        Transformed array of the same shape as *x*, with values in (lo, hi).
    """
    spread = hi - lo
    mid = (lo + hi) / 2.0
    return mid + (spread / 4.0) * jnp.arcsinh(x)


def _jax_cutoffpl_latent_to_params(
    z_phys: jnp.ndarray,
    gamma_lo: float = 0.5,
    gamma_hi: float = 6.0,
    logxcut_lo: float = -1.0,
    logxcut_hi: float = 6.0,
) -> jnp.ndarray:
    """
    JAX version of CutoffPLDecoder.latent_to_params.

    Parameters
    ----------
    z_phys : jnp.ndarray
        Input latent array of shape (N, 3), with columns
        [logA, gamma_raw, logxcut_raw].
    gamma_lo : float, optional
        Lower bound for the photon index gamma. Default is 0.5.
    gamma_hi : float, optional
        Upper bound for the photon index gamma. Default is 6.0.
    logxcut_lo : float, optional
        Lower bound for log10(xcut). Default is -1.0.
    logxcut_hi : float, optional
        Upper bound for log10(xcut). Default is 6.0.

    Returns
    -------
    jnp.ndarray
        Physical parameter array of shape (N, 3) with columns
        [logA, gamma, xcut].
    """
    logA = z_phys[:, 0]
    gamma = _jax_sigmoid_asinh(z_phys[:, 1], gamma_lo, gamma_hi)
    logxcut = _jax_sigmoid_asinh(z_phys[:, 2], logxcut_lo, logxcut_hi)
    xcut = 10.0 ** logxcut
    return jnp.column_stack([logA, gamma, xcut])


def _jax_cutoffpl_latent_to_params_warm(
    z_phys: jnp.ndarray,
    gamma_lo: float = 0.5,
    gamma_hi: float = 6.0,
    logxcut_lo: float = -1.0,
    logxcut_hi: float = 1.0,
) -> jnp.ndarray:
    """
    JAX version of CutoffPLDecoder.latent_to_params for warm corona.
    xcut_lo=1e-1, xcut_hi=10 => logxcut_lo=-1, logxcut_hi=1.

    Parameters
    ----------
    z_phys : jnp.ndarray
        Input latent array of shape (N, 3), with columns
        [logA, gamma_raw, logxcut_raw].
    gamma_lo : float, optional
        Lower bound for the photon index gamma. Default is 0.5.
    gamma_hi : float, optional
        Upper bound for the photon index gamma. Default is 6.0.
    logxcut_lo : float, optional
        Lower bound for log10(xcut). Default is -1.0.
    logxcut_hi : float, optional
        Upper bound for log10(xcut). Default is 1.0.

    Returns
    -------
    jnp.ndarray
        Physical parameter array of shape (N, 3) with columns
        [logA, gamma, xcut].
    """
    logA = z_phys[:, 0]
    gamma = _jax_sigmoid_asinh(z_phys[:, 1], gamma_lo, gamma_hi)
    logxcut = _jax_sigmoid_asinh(z_phys[:, 2], logxcut_lo, logxcut_hi)
    xcut = 10.0 ** logxcut
    return jnp.column_stack([logA, gamma, xcut])


def _jax_sbpl_latent_to_params(
    z_phys: jnp.ndarray,
    slope_scale: float = 2.5,
    logx_lo: float = 1.0,
    logx_hi: float = 3.0,
    logw_lo: float = -1.0,
    logw_hi: float = 2.0,
) -> jnp.ndarray:
    """
    JAX version of SBPLDecoder.latent_to_params.

    Parameters
    ----------
    z_phys : jnp.ndarray
        Input latent array of shape (N, 5), with columns
        [lognorm, s1_raw, s2_raw, logxbrk_raw, logw_raw].
    slope_scale : float, optional
        Scaling factor applied to arcsinh-transformed slopes. Default is 2.5.
    logx_lo : float, optional
        Lower bound for log10(x_break). Default is 1.0.
    logx_hi : float, optional
        Upper bound for log10(x_break). Default is 3.0.
    logw_lo : float, optional
        Lower bound for log10(width). Default is -1.0.
    logw_hi : float, optional
        Upper bound for log10(width). Default is 2.0.

    Returns
    -------
    jnp.ndarray
        Physical parameter array of shape (N, 5) with columns
        [lognorm, uvslope, plslope, plbendloc, plbendwidth].

    Notes
    -----
    duallike.py names these as:
      lognorm_510, uvslope, plslope, plbendloc, plbendwidth = dec5.transpose()
    where dec5 comes from emu.predict_sbpl_params which calls
    transform(..., return_latent_params=True) -> latent_to_params.
    """
    lognorm = z_phys[:, 0]
    s1 = slope_scale * jnp.arcsinh(z_phys[:, 1])
    s2 = slope_scale * jnp.arcsinh(z_phys[:, 2])
    logxbrk = _jax_sigmoid_asinh(z_phys[:, 3], logx_lo, logx_hi)
    xbrk = 10.0 ** logxbrk
    logw = _jax_sigmoid_asinh(z_phys[:, 4], logw_lo, logw_hi)
    width = 10.0 ** logw
    return jnp.column_stack([lognorm, s1, s2, xbrk, width])


def _jax_quadratic_latent_to_params(
    z_phys: jnp.ndarray,
    logx_lo: float = -1.0,
    logx_hi: float = 2.0,
    logw_lo: float = -1.0,
    logw_hi: float = 1.0,
) -> jnp.ndarray:
    """
    JAX version of QuadraticDecoder.latent_to_params.

    Parameters
    ----------
    z_phys : jnp.ndarray
        Input latent array of shape (N, 3), with columns
        [logA, logxpeak_raw, logw_raw].
    logx_lo : float, optional
        Lower bound for log10(x_peak). Default is -1.0.
    logx_hi : float, optional
        Upper bound for log10(x_peak). Default is 2.0.
    logw_lo : float, optional
        Lower bound for log10(width). Default is -1.0.
    logw_hi : float, optional
        Upper bound for log10(width). Default is 1.0.

    Returns
    -------
    jnp.ndarray
        Physical parameter array of shape (N, 3) with columns
        [logA, xpeak, w].

    Notes
    -----
    emuloader.py then converts:
      out[:, 0] *= ln10    (logA -> lnA)
      out[:, 2] /= sqrt(ln10)  (sigma log10 -> sigma ln)
    """
    logA = z_phys[:, 0]
    logxpeak = _jax_sigmoid_asinh(z_phys[:, 1], logx_lo, logx_hi)
    xpeak = 10.0 ** logxpeak
    logw = _jax_sigmoid_asinh(z_phys[:, 2], logw_lo, logw_hi)
    w = 10.0 ** logw
    return jnp.column_stack([logA, xpeak, w])


def load_emulator(path: str, verbose: bool = False) -> dict:
    """
    Load a specsimile emulator into a flat bundle **without** a prefix.

    This is the legacy interface used by ``difflike.load_jax_emulator_bundle``
    which adds its own prefix when merging into the combined bundle::

        for k, v in specsimile.jxrun.load_emulator(path).items():
            bundle[f'{prefix}_{k}'] = v

    Parameters
    ----------
    path : str
        Path to the PyTorch .pt checkpoint file.
    verbose : bool, optional
        If True, print diagnostic information. Default is False.

    Returns
    -------
    dict
        Flat parameter bundle with keys: W0, b0, …, n_layers,
        norm_x_mean, norm_x_std, … (no prefix).
    """
    return load_decoder_bundle(path, prefix='', verbose=verbose)

def _jax_apply_activation(h: jnp.ndarray, acti_name: str) -> jnp.ndarray:
    """
    Apply a named activation function to an array.

    Parameters
    ----------
    h : jnp.ndarray
        Input array of arbitrary shape.
    acti_name : str
        Name of the activation function. Supported values are:
        'LeakyReLU', 'ReLU', 'Tanh', 'Softplus', 'GELU', 'SELU',
        'LeakySoftplus'.

    Returns
    -------
    jnp.ndarray
        Activated array of the same shape as *h*.

    Raises
    ------
    ValueError
        If *acti_name* is not one of the supported activation names.
    """
    if acti_name == 'LeakyReLU':
        return jax.nn.leaky_relu(h, negative_slope=0.01)
    elif acti_name == 'ReLU':
        return jax.nn.relu(h)
    elif acti_name == 'Tanh':
        return jnp.tanh(h)
    elif acti_name == 'Softplus':
        return jax.nn.softplus(h)
    elif acti_name == 'GELU':
        return jax.nn.gelu(h)
    elif acti_name == 'SELU':
        return jax.nn.selu(h)
    elif acti_name == 'LeakySoftplus':
        beta = 0.99 # positive_slope - negative_slope = 1.0 - 0.01
        negative_slope = 0.01
        # softplus(0, beta) = log(2) / beta  (the zero-centering offset)
        zero = jnp.log(2.0) / beta
        sp = jnp.log1p(jnp.exp(beta * h)) / beta - zero
        return sp + negative_slope * h
    else:
        raise ValueError(f"Unknown activation: {acti_name!r}")

def load_decoder_bundle(path: str, prefix: str = '', verbose: bool = False) -> dict:
    """
    Load a specsimile emulator checkpoint into a flat JAX-compatible parameter
    bundle.  All keys are prefixed with *prefix* (plus underscore if non-empty).

    The returned dict contains:
      {prefix_}W0, {prefix_}b0, ...  – MLP weight/bias matrices
      {prefix_}n_layers              – number of layers
      {prefix_}norm_x_mean/std       – input normalisation
      {prefix_}norm_x_log_mask       – which input columns are log-transformed
      {prefix_}norm_x_eps            – epsilon for log transform
      {prefix_}norm_y_mean/std       – latent (output) denormalisation
      {prefix_}norm_y_log            – whether decoder interprets output as log10
      {prefix_}norm_y_eps
      {prefix_}decoder_class         – string name of decoder class
      {prefix_}n_context_params      – number of context params (for ConditionalMLP)

    Parameters
    ----------
    path : str
        Path to the PyTorch .pt checkpoint.
    prefix : str, optional
        Key prefix added to every entry.  Use e.g. ``'hot'`` so that the
        resulting keys are ``hot_W0``, ``hot_b0``, … matching the convention
        in *difflike.py*. Default is ''.
    verbose : bool, optional
        If True, print diagnostic information about layers and normalisation.
        Default is False.

    Returns
    -------
    dict
        Flat parameter bundle where all values are JAX arrays or plain Python
        scalars/booleans.
    """
    p = f'{prefix}_' if prefix else ''
    emu_weights, ckpt_raw = extract_pytorch_emulator_weights(path, verbose=verbose)
    bundle = {}

    # --- MLP weights --------------------------------------------------------
    # Handle both plain MLP (net.0.weight, net.2.weight, …) and
    # ConditionalMLP (main_net.0.weight, context_net.0.weight, …).
    # We only export the *main* (inference) network layers here; the context
    # network is handled separately when n_context_params > 0.

    # Detect ConditionalMLP vs plain MLP by key names
    has_conditional = any(k.startswith('main_net.') for k in emu_weights if isinstance(emu_weights[k], np.ndarray))

    # Read n_context_params from checkpoint if present
    acti_name = ckpt_raw.get('acti', 'LeakyReLU')
    context_acti_name = ckpt_raw.get('context_acti', 'ReLU')
    bundle[f'{p}acti'] = acti_name
    bundle[f'{p}context_acti'] = context_acti_name

    n_context_params = int(ckpt_raw.get('n_context_params', 0))
    embed_dim = int(ckpt_raw.get('embed_dim', 16))
    bundle[f'{p}n_context_params'] = n_context_params
    bundle[f'{p}embed_dim'] = embed_dim

    # Decoder class name (for downstream consumers)
    meta = ckpt_raw.get('meta', {})
    dec_meta = meta.get('decoder', {})
    bundle[f'{p}decoder_class'] = dec_meta.get('class_name', 'Unknown')
    bundle[f'{p}decoder_config'] = dec_meta.get('config', {})

    pnames = meta.get('paramnames', None)
    if pnames is not None:
        bundle[f'{p}paramnames'] = list(pnames)

    ynames = meta.get('ynames', None)
    if ynames is not None:
        bundle[f'{p}ynames'] = list(ynames)

    if has_conditional:
        # --- Main network layers (main_net.0.weight, main_net.0.bias, …) ---
        main_w_keys = sorted(
            [k for k in emu_weights
             if isinstance(emu_weights[k], np.ndarray)
             and k.startswith('main_net.')
             and k.endswith('.weight')
             and emu_weights[k].ndim == 2],
            key=lambda k: int(k.split('.')[1])
        )
        jax_idx = 0
        for w_key in main_w_keys:
            b_key = w_key.replace('.weight', '.bias')
            if b_key not in emu_weights:
                continue
            W = emu_weights[w_key]
            b = emu_weights[b_key]
            bundle[f'{p}W{jax_idx}'] = jnp.array(W.T.astype(np.float64))
            bundle[f'{p}b{jax_idx}'] = jnp.array(b.astype(np.float64))
            jax_idx += 1
        bundle[f'{p}n_layers'] = jax_idx

        # --- Context network layers (context_net.0.weight, …) --------------
        ctx_w_keys = sorted(
            [k for k in emu_weights
             if isinstance(emu_weights[k], np.ndarray)
             and k.startswith('context_net.')
             and k.endswith('.weight')
             and emu_weights[k].ndim == 2],
            key=lambda k: int(k.split('.')[1])
        )
        ctx_idx = 0
        for w_key in ctx_w_keys:
            b_key = w_key.replace('.weight', '.bias')
            if b_key not in emu_weights:
                continue
            W = emu_weights[w_key]
            b = emu_weights[b_key]
            bundle[f'{p}ctx_W{ctx_idx}'] = jnp.array(W.T.astype(np.float64))
            bundle[f'{p}ctx_b{ctx_idx}'] = jnp.array(b.astype(np.float64))
            ctx_idx += 1
        bundle[f'{p}ctx_n_layers'] = ctx_idx

    else:
        # Plain MLP: net.0.weight, net.2.weight, …
        weight_keys = sorted(
            [k for k in emu_weights
             if isinstance(emu_weights[k], np.ndarray)
             and k.endswith('.weight')
             and emu_weights[k].ndim == 2],
            key=lambda k: k
        )
        jax_idx = 0
        for w_key in weight_keys:
            b_key = w_key.replace('.weight', '.bias')
            if b_key not in emu_weights:
                continue
            W = emu_weights[w_key]
            b = emu_weights[b_key]
            bundle[f'{p}W{jax_idx}'] = jnp.array(W.T.astype(np.float64))
            bundle[f'{p}b{jax_idx}'] = jnp.array(b.astype(np.float64))
            jax_idx += 1
        bundle[f'{p}n_layers'] = jax_idx

    if verbose and bundle[f'{p}n_layers'] > 0:
        shapes = " -> ".join(
            str(tuple(bundle[f'{p}W{j}'].shape))
            for j in range(bundle[f'{p}n_layers'])
        )
        print(f"  {prefix} {bundle[f'{p}n_layers']} layers: {shapes}")

    # --- Normalisation ------------------------------------------------------
    if '_norm' in emu_weights:
        norm = emu_weights['_norm']

        if 'params' in norm and isinstance(norm['params'], dict):
            p_norm = norm['params']
            if 'mean' in p_norm:
                bundle[f'{p}norm_x_mean'] = jnp.array(
                    np.asarray(p_norm['mean'], dtype=np.float64))
            if 'std' in p_norm:
                bundle[f'{p}norm_x_std'] = jnp.array(
                    np.asarray(p_norm['std'], dtype=np.float64))
            if 'log_mask' in p_norm:
                bundle[f'{p}norm_x_log_mask'] = np.asarray(
                    p_norm['log_mask'], dtype=bool)
            if 'eps' in p_norm:
                bundle[f'{p}norm_x_eps'] = float(p_norm['eps'])
        elif 'x' in norm and isinstance(norm['x'], dict):
            x_norm = norm['x']
            if 'mean' in x_norm:
                bundle[f'{p}norm_x_mean'] = jnp.array(
                    np.asarray(x_norm['mean'], dtype=np.float64))
            if 'std' in x_norm:
                bundle[f'{p}norm_x_std'] = jnp.array(
                    np.asarray(x_norm['std'], dtype=np.float64))

        if 'latent' in norm and isinstance(norm['latent'], dict):
            l_norm = norm['latent']
            if 'mean' in l_norm:
                bundle[f'{p}norm_y_mean'] = jnp.array(
                    np.asarray(l_norm['mean'], dtype=np.float64))
            if 'std' in l_norm:
                bundle[f'{p}norm_y_std'] = jnp.array(
                    np.asarray(l_norm['std'], dtype=np.float64))
            if 'log_mask' in l_norm:
                bundle[f'{p}norm_y_log_mask'] = np.asarray(
                    l_norm['log_mask'], dtype=bool)
        elif 'y' in norm and isinstance(norm['y'], dict):
            y_norm_dict = norm['y']
            if 'mean' in y_norm_dict:
                bundle[f'{p}norm_y_mean'] = jnp.array(
                    np.asarray(y_norm_dict['mean'], dtype=np.float64))
            if 'std' in y_norm_dict:
                bundle[f'{p}norm_y_std'] = jnp.array(
                    np.asarray(y_norm_dict['std'], dtype=np.float64))

        if 'y' in norm and isinstance(norm['y'], dict):
            y_norm_dict = norm['y']
            if 'log' in y_norm_dict:
                bundle[f'{p}norm_y_log'] = bool(y_norm_dict['log'])
            if 'eps' in y_norm_dict:
                bundle[f'{p}norm_y_eps'] = float(y_norm_dict['eps'])

        if verbose:
            print(f"  {prefix} norm: "
                  f"x_mean={'yes' if f'{p}norm_x_mean' in bundle else 'no'} "
                  f"x_std={'yes' if f'{p}norm_x_std' in bundle else 'no'} "
                  f"y_mean={'yes' if f'{p}norm_y_mean' in bundle else 'no'} "
                  f"y_std={'yes' if f'{p}norm_y_std' in bundle else 'no'} "
                  f"y_log={bundle.get(f'{p}norm_y_log', False)}")

    return bundle



def get_bundle_paramnames(bundle: dict, prefix: str) -> List[str]:
    """
    Return the input parameter names for the emulator stored under *prefix*.

    Parameters
    ----------
    bundle : dict
        Parameter bundle as returned by :func:`load_decoder_bundle`.
    prefix : str
        Key prefix under which the emulator was stored, e.g. ``'hot'``.

    Returns
    -------
    list of str
        Ordered list of input parameter names as stored in the checkpoint.

    Raises
    ------
    KeyError
        If the bundle does not contain paramnames for this prefix (e.g. the
        checkpoint was saved before paramnames were recorded in meta).
    """
    p = f'{prefix}_' if prefix else ''
    key = f'{p}paramnames'
    return list(bundle[key])


def get_bundle_ynames(bundle: dict, prefix: str) -> List[str]:
    """
    Return the output (y) column names for the emulator stored under *prefix*.

    Parameters
    ----------
    bundle : dict
        Parameter bundle as returned by :func:`load_decoder_bundle`.
    prefix : str
        Key prefix under which the emulator was stored, e.g. ``'hot'``.

    Returns
    -------
    list of str
        Ordered list of output column names as stored in the checkpoint.

    Raises
    ------
    KeyError
        If the bundle does not contain ynames for this prefix.
    """
    p = f'{prefix}_' if prefix else ''
    key = f'{p}ynames'
    if key not in bundle:
        raise KeyError(
            f"Bundle does not contain '{key}'. "
            f"Re-save the emulator so that meta['ynames'] is stored."
        )
    return list(bundle[key])

def jax_encoder_forward_kw(
    bundle: dict,
    prefix: str,
    kwargs: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """
    Keyword-argument wrapper around :func:`jax_encoder_forward`.

    Instead of passing a pre-stacked ``(N, n_params)`` array you pass each
    input feature by name.  The function looks up the stored ``paramnames``
    for this emulator, assembles the columns in the correct order, and calls
    the underlying forward pass.

    Parameters
    ----------
    bundle : dict
        Parameter bundle as returned by :func:`load_decoder_bundle` or
        :func:`load_emulator`.  Must contain ``{prefix}_paramnames``.
    prefix : str
        Key prefix under which the emulator was stored, e.g. ``'hot'``.
    kwargs : dict of str to jnp.ndarray
        One entry per input feature, named exactly as stored in
        ``paramnames``.  Each array must have shape (N,) and all must share
        the same leading dimension N.

    Returns
    -------
    jnp.ndarray
        Raw encoder output of shape (N, n_outputs), denormalised but before
        any decoder transform.

    Raises
    ------
    KeyError
        If a required input feature is missing from *kwargs*, or if the
        bundle does not contain paramnames.

    Examples
    --------
    >>> out = jax_encoder_forward_kw(bundle, 'L',
    ...     {'M_BH': log10_M_BH, 'log_mdot': log_mdot, 'cos_inc': cos_inc})
    """
    paramnames = get_bundle_paramnames(bundle, prefix)

    # --- Assemble input array in the order the emulator was trained on ------
    # Collect columns
    x = jnp.column_stack([kwargs[name] for name in paramnames])

    # --- Forward pass -------------------------------------------------------
    h = jax_encoder_forward(x, bundle, prefix)   # (N, n_outputs)

    return h

def jax_encoder_forward_kw_dict(
    bundle: dict,
    prefix: str,
    kwargs: Dict[str, jnp.ndarray],
) -> Dict[str, jnp.ndarray]:
    """
    Keyword-argument wrapper around :func:`jax_encoder_forward` returning a
    dict keyed by output name.

    Instead of passing a pre-stacked ``(N, n_params)`` array you pass each
    input feature by name.  The function looks up the stored ``paramnames``
    for this emulator, assembles the columns in the correct order, and calls
    the underlying forward pass.

    Parameters
    ----------
    bundle : dict
        Parameter bundle as returned by :func:`load_decoder_bundle` or
        :func:`load_emulator`.  Must contain ``{prefix}_paramnames`` and
        ``{prefix}_ynames``.
    prefix : str
        Key prefix under which the emulator was stored, e.g. ``'hot'``.
    kwargs : dict of str to jnp.ndarray
        One entry per input feature, named exactly as stored in
        ``paramnames``.  Each array must have shape (N,) and all must share
        the same leading dimension N.

    Returns
    -------
    dict of str to jnp.ndarray
        Keys are the ynames of the emulator; values are arrays of shape (N,).

    Raises
    ------
    KeyError
        If a required input feature is missing from *kwargs*, or if the
        bundle does not contain paramnames / ynames.

    Examples
    --------
    >>> out = jax_encoder_forward_kw_dict(bundle, 'L',
    ...     {'M_BH': log10_M_BH, 'log_mdot': log_mdot, 'cos_inc': cos_inc})
    >>> logLUV = out['LUV']      # (N,)
    >>> logL2keV = out['L2keV']  # (N,)
    """
    h = jax_encoder_forward_kw(bundle, prefix, kwargs)

    # --- Wrap output in a dict keyed by ynames ------------------------------
    ynames = get_bundle_ynames(bundle, prefix)
    return {name: h[:, i] for i, name in enumerate(ynames)}
