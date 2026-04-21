"""Functions for using the emulators from jax """
from typing import Dict #, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
import torch




# ============================================================================
# Part 2: Extract weights from PyTorch emulator models (specsimile)
# ============================================================================

def extract_pytorch_emulator_weights(model_path: str, verbose: bool=True) -> Dict[str, np.ndarray]:
    """
    Extract weights from a specsimile Emulator saved as a PyTorch checkpoint.
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
    
    # Extract normalization - specsimile uses 'norm' not '_norm'
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
    
    return result

def _jax_encoder_forward(x, bundle, prefix):
    """
    Forward pass through a specsimile encoder.
    
    specsimile pipeline:
    1. Input: optionally log10 some params (per log_mask), then standardize
    2. MLP forward with ReLU activations
    3. Output: denormalize with latent mean/std
    4. If y_log: the denormalized output represents log10 of the physical quantity
       (but for decoders like CutoffPL, SBPL, Quadratic, the decoder params 
        are NOT exponentiated — they're used directly as decoder parameters.
        Only LogDenseDecoder/DenseDecoder would exponentiate.)
    
    IMPORTANT: For CutoffPLDecoder, SBPLDecoder, QuadraticDecoder,
    y_log=True means the TRAINING target was log10(y), so the network
    outputs are already the decoder parameters in whatever space they
    need to be. The denormalization gives back the decoder parameters.
    We should NOT exponentiate for these decoders.
    
    For LogDenseDecoder (used by L_info), y_log=True means the output
    IS log10 of the physical luminosities, so we DO exponentiate.
    
    But actually, looking at emuloader.py more carefully, the Emulator.predict()
    method returns the DECODER PARAMETERS, not the decoded spectrum.
    The decoder then interprets those parameters.
    
    For CutoffPLDecoder: params are [log_norm, gamma, Ecut_or_log_Ecut]
    For SBPLDecoder: params are [log_norm, uvslope, plslope, bendloc, log_bendwidth]
    For QuadraticDecoder: params are [logA, xpeak, sigma]
    For LogDenseDecoder: params are log10(y_values), so output = 10^params
    
    The y_log flag and latent normalization handle this correctly:
    - Network predicts in standardized space
    - Denormalize: output = network_out * latent_std + latent_mean
    - This gives the decoder parameters directly
    - y_log is about how training data was preprocessed, not about the final step
    """
    h = x
    
    # Input: apply log_mask if present, then standardize
    log_mask_key = f'{prefix}_norm_x_log_mask'
    if log_mask_key in bundle:
        log_mask = bundle[log_mask_key]
        if np.any(log_mask):
            eps = bundle.get(f'{prefix}_norm_x_eps', 1e-30)
            # Only log10 the columns where log_mask is True
            h_log = jnp.log10(jnp.clip(h, eps, None))
            h = jnp.where(jnp.array(log_mask)[None, :], h_log, h)
    
    x_mean_key = f'{prefix}_norm_x_mean'
    x_std_key = f'{prefix}_norm_x_std'
    if x_mean_key in bundle:
        h = (h - bundle[x_mean_key]) / jnp.clip(bundle[x_std_key], 1e-10, None)
    
    # MLP forward
    n_layers = bundle[f'{prefix}_n_layers']
    for i in range(n_layers):
        W = bundle[f'{prefix}_W{i}']
        b = bundle[f'{prefix}_b{i}']
        h = h @ W + b
        if i < n_layers - 1:
            h = jax.nn.leaky_relu(h, negative_slope=0.01)
        
    # Output denormalization: decoder_params = network_out * latent_std + latent_mean
    y_mean_key = f'{prefix}_norm_y_mean'
    y_std_key = f'{prefix}_norm_y_std'
    if y_mean_key in bundle:
        h = h * bundle[y_std_key] + bundle[y_mean_key]
    
    return h

def _jax_sigmoid_asinh(x, lo, hi):
    """JAX version of specsimile sigmoid_asinh: R -> (lo, hi)."""
    spread = hi - lo
    mid = (lo + hi) / 2.0
    return mid + (spread / 4.0) * jnp.arcsinh(x)


def _jax_cutoffpl_latent_to_params(z_phys, gamma_lo=0.5, gamma_hi=6.0,
                                     logxcut_lo=-1.0, logxcut_hi=6.0):
    """
    JAX version of CutoffPLDecoder.latent_to_params.
    
    Input z_phys: (N, 3) = [logA, gamma_raw, logxcut_raw]
    Output: (N, 3) = [logA, gamma, xcut]
    """
    logA = z_phys[:, 0]
    gamma = _jax_sigmoid_asinh(z_phys[:, 1], gamma_lo, gamma_hi)
    logxcut = _jax_sigmoid_asinh(z_phys[:, 2], logxcut_lo, logxcut_hi)
    xcut = 10.0 ** logxcut
    return jnp.column_stack([logA, gamma, xcut])


def _jax_cutoffpl_latent_to_params_warm(z_phys, gamma_lo=0.5, gamma_hi=6.0,
                                          logxcut_lo=-1.0, logxcut_hi=1.0):
    """
    JAX version of CutoffPLDecoder.latent_to_params for warm corona.
    xcut_lo=1e-1, xcut_hi=10 => logxcut_lo=-1, logxcut_hi=1
    """
    logA = z_phys[:, 0]
    gamma = _jax_sigmoid_asinh(z_phys[:, 1], gamma_lo, gamma_hi)
    logxcut = _jax_sigmoid_asinh(z_phys[:, 2], logxcut_lo, logxcut_hi)
    xcut = 10.0 ** logxcut
    return jnp.column_stack([logA, gamma, xcut])


def _jax_sbpl_latent_to_params(z_phys, slope_scale=2.5,
                                 logx_lo=1.0, logx_hi=3.0,
                                 logw_lo=-1.0, logw_hi=2.0):
    """
    JAX version of SBPLDecoder.latent_to_params.
    
    Input z_phys: (N, 5) = [lognorm, s1_raw, s2_raw, logxbrk_raw, logw_raw]
    Output: (N, 5) = [lognorm, uvslope, plslope, plbendloc, plbendwidth]
    
    Note: duallike.py names these as:
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


def _jax_quadratic_latent_to_params(z_phys, logx_lo=-1.0, logx_hi=2.0,
                                      logw_lo=-1.0, logw_hi=1.0):
    """
    JAX version of QuadraticDecoder.latent_to_params.
    
    Input z_phys: (N, 3) = [logA, logxpeak_raw, logw_raw]
    Output: (N, 3) = [logA, xpeak, w]
    
    Note: emuloader.py then converts:
      out[:, 0] *= ln10    (logA -> lnA)
      out[:, 2] /= sqrt(ln10)  (sigma log10 -> sigma ln)
    """
    logA = z_phys[:, 0]
    logxpeak = _jax_sigmoid_asinh(z_phys[:, 1], logx_lo, logx_hi)
    xpeak = 10.0 ** logxpeak
    logw = _jax_sigmoid_asinh(z_phys[:, 2], logw_lo, logw_hi)
    w = 10.0 ** logw
    return jnp.column_stack([logA, xpeak, w])

def _jax_torus_postprocess(q):
    """
    Apply the emuloader.py post-processing for torus quadratic params:
      logA -> lnA (multiply by ln10)
      sigma -> sigma_ln (divide by sqrt(ln10))
    """
    ln10 = jnp.log(10.0)
    lnA = q[:, 0] * ln10
    xpeak = q[:, 1]
    w = q[:, 2] / jnp.sqrt(ln10)
    return jnp.column_stack([lnA, xpeak, w])



# ============================================================================
# Part 4: specsimile emulator loading and forward pass
# ============================================================================

def load_emulator(path: str, verbose: bool=False) -> dict:
    bundle = {}
        
    emu_weights = extract_pytorch_emulator_weights(path, verbose=verbose)
    
    # Extract Linear layer weights
    weight_keys = sorted([k for k in emu_weights
                          if isinstance(emu_weights[k], np.ndarray)
                          and k.endswith('.weight')
                          and emu_weights[k].ndim == 2])
    
    jax_layer_idx = 0
    for w_key in weight_keys:
        W = emu_weights[w_key]
        b_key = w_key.replace('.weight', '.bias')
        if b_key not in emu_weights:
            continue
        b = emu_weights[b_key]
        bundle[f'W{jax_layer_idx}'] = jnp.array(W.T.astype(np.float64))
        bundle[f'b{jax_layer_idx}'] = jnp.array(b.astype(np.float64))
        jax_layer_idx += 1
    bundle['n_layers'] = jax_layer_idx
    
    if jax_layer_idx > 0 and verbose:
        shapes = " -> ".join(str(tuple(bundle[f'W{j}'].shape)) for j in range(jax_layer_idx))
        if verbose:
            print(f"  extracted {jax_layer_idx} layers: {shapes}")
    
    # Parse normalization from specsimile checkpoint
    # Structure: norm['params'] = input normalization (mean, std, eps, log_mask)
    #            norm['latent'] = output denormalization (mean, std, log_mask)
    #            norm['y'] = {log: bool, eps: float} for final output transform
    if '_norm' in emu_weights:
        norm = emu_weights['_norm']
        
        # Input normalization: norm['params']
        if 'params' in norm and isinstance(norm['params'], dict):
            p_norm = norm['params']
            if 'mean' in p_norm:
                bundle['norm_x_mean'] = jnp.array(
                    np.asarray(p_norm['mean'], dtype=np.float64))
            if 'std' in p_norm:
                bundle['norm_x_std'] = jnp.array(
                    np.asarray(p_norm['std'], dtype=np.float64))
            if 'log_mask' in p_norm:
                bundle['norm_x_log_mask'] = np.asarray(
                    p_norm['log_mask'], dtype=bool)
            if 'eps' in p_norm:
                bundle['norm_x_eps'] = float(p_norm['eps'])
        # Fallback: norm['x']
        elif 'x' in norm and isinstance(norm['x'], dict):
            x_norm = norm['x']
            if 'mean' in x_norm:
                bundle['norm_x_mean'] = jnp.array(
                    np.asarray(x_norm['mean'], dtype=np.float64))
            if 'std' in x_norm:
                bundle['norm_x_std'] = jnp.array(
                    np.asarray(x_norm['std'], dtype=np.float64))
        
        # Output denormalization: norm['latent']
        if 'latent' in norm and isinstance(norm['latent'], dict):
            l_norm = norm['latent']
            if 'mean' in l_norm:
                bundle['norm_y_mean'] = jnp.array(
                    np.asarray(l_norm['mean'], dtype=np.float64))
            if 'std' in l_norm:
                bundle['norm_y_std'] = jnp.array(
                    np.asarray(l_norm['std'], dtype=np.float64))
            if 'log_mask' in l_norm:
                bundle['norm_y_log_mask'] = np.asarray(
                    l_norm['log_mask'], dtype=bool)
        # Fallback: norm['y'] for mean/std
        elif 'y' in norm and isinstance(norm['y'], dict):
            y_norm = norm['y']
            if 'mean' in y_norm:
                bundle['norm_y_mean'] = jnp.array(
                    np.asarray(y_norm['mean'], dtype=np.float64))
            if 'std' in y_norm:
                bundle['norm_y_std'] = jnp.array(
                    np.asarray(y_norm['std'], dtype=np.float64))
        
        # Final output transform: norm['y']['log'] and norm['y']['eps']
        if 'y' in norm and isinstance(norm['y'], dict):
            y_norm = norm['y']
            if 'log' in y_norm:
                bundle['norm_y_log'] = bool(y_norm['log'])
            if 'eps' in y_norm:
                bundle['norm_y_eps'] = float(y_norm['eps'])
        
        if verbose:
            print(f"  norm: x_mean={'yes' if 'norm_x_mean' in bundle else 'no'}"
                  f" x_std={'yes' if 'norm_x_std' in bundle else 'no'}"
                  f" y_mean={'yes' if 'norm_y_mean' in bundle else 'no'}"
                  f" y_std={'yes' if 'norm_y_std' in bundle else 'no'}"
                  f" y_log={bundle.get('norm_y_log', False)}"
                  f" y_eps={bundle.get('norm_y_eps', 'none')}")
            if 'norm_x_mean' in bundle:
                print(f"    x_mean={np.array(bundle['norm_x_mean'])[:5]}")
                print(f"    x_std ={np.array(bundle['norm_x_std'])[:5]}")
            if 'norm_y_mean' in bundle:
                print(f"    y_mean={np.array(bundle['norm_y_mean'])[:5]}")
                print(f"    y_std ={np.array(bundle['norm_y_std'])[:5]}")

    return bundle
