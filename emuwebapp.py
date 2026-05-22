#!/usr/bin/env python3
import argparse
import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# ---------------------------------------------------------------------------
# CLI argument parsing – must happen before the app state is populated
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic SED emulator web-app. Pass a trained .pt model file as the "
                    "only required argument; all other options tune the Flask server."
    )
    parser.add_argument(
        "model_path",
        help="Path to the trained emulator checkpoint (.pt file).",
    )
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on (default: 5000).")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1).")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Emulator loader – generic, reads everything from the checkpoint
# ---------------------------------------------------------------------------

def load_emulator_from_checkpoint(model_path: str):
    """
    Load an emulator purely from its checkpoint file.

    The checkpoint stores:
      - shape, acti, n_context_params, embed_dim, context_acti, dropout
      - nin, nout
      - norm  (includes decoder class name / config)
      - meta  (paramnames, xlabel, ylabel, xunit, yunit, train_param_bounds …)
      - encoder_state

    We reconstruct the decoder from the stored class name + config so that
    the caller does not need to know in advance which decoder was used.
    """
    import torch
    from specsimile import decoders as dec_module
    from specsimile.emulator import Emulator

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    # --- reconstruct decoder ---
    dmeta = ckpt.get("meta", {}).get("decoder", None)
    if dmeta is None:
        raise ValueError(
            "Checkpoint does not contain decoder metadata. "
            "Please pass a model trained with a version that saves decoder info."
        )
    class_name = dmeta.get("class_name")
    config = dmeta.get("config") or {}

    decoder_cls = getattr(dec_module, class_name, None)
    if decoder_cls is None:
        raise ValueError(
            f"Decoder class '{class_name}' not found in specsimile.decoders. "
            "Ensure specsimile is up to date."
        )

    # Instantiate without positional args; configure() handles everything
    decoder = decoder_cls()
    if config and hasattr(decoder, "configure"):
        decoder.configure(**config)
    elif config:
        # Fallback: try keyword constructor
        decoder = decoder_cls(**config)

    shape = ckpt["shape"]
    acti = ckpt.get("acti", "LeakyReLU")
    n_context_params = int(ckpt.get("n_context_params", 0))
    embed_dim = int(ckpt.get("embed_dim", 16))
    context_acti = ckpt.get("context_acti", "ReLU")
    dropout = float(ckpt.get("dropout", 0.0))

    dynamic_range = ckpt.get("meta", {}).get("dynamic_range", None)
    func_floor_loss = ckpt.get("meta", {}).get("func_floor_loss", None)

    emu = Emulator(
        model_path=model_path,
        decoder=decoder,
        shape=shape,
        acti=acti,
        n_context_params=n_context_params,
        embed_dim=embed_dim,
        context_acti=context_acti,
        device=None,
        dynamic_range=dynamic_range,
        func_floor_loss=func_floor_loss,
        dropout=dropout,
    )
    emu.load(verbose=True)
    return emu


# ---------------------------------------------------------------------------
# App-level state (populated in main() before app.run())
# ---------------------------------------------------------------------------

class AppState:
    emulator = None
    model_path: str = ""
    paramnames: list = []
    sliders: list = []          # list of dicts consumed by the Jinja template
    defaults: dict = {}         # name -> default value
    xlabel: str = "x"
    xunit: str = ""
    ylabel: str = "y"
    yunit: str = ""
    title: str = "SED Emulator"
    x_grid: np.ndarray = None   # stored x grid from the checkpoint
    y_range: tuple = None       # (y_min, y_max) for fixed axis range across plots


STATE = AppState()


def _build_sliders_from_emulator(emu) -> list:
    """
    Build slider defs from the emulator's stored metadata.

    Falls back to symmetric ±3σ around the training mean when explicit
    bounds are not available.
    """
    meta = emu._meta or {}
    paramnames = emu.paramnames or meta.get("paramnames") or []
    bounds = meta.get("train_param_bounds") or {}
    pmin_list = bounds.get("min", [])
    pmax_list = bounds.get("max", [])

    sliders = []
    for i, name in enumerate(paramnames):
        lo = float(pmin_list[i]) if i < len(pmin_list) else -10.0
        hi = float(pmax_list[i]) if i < len(pmax_list) else 10.0
        default = float((lo + hi) / 2.0)
        step = round((hi - lo) / 200.0, 6)
        step = max(step, 1e-6)
        sliders.append(
            dict(
                name=name,
                label=name,
                description="",
                min=lo,
                max=hi,
                step=step,
                default=default,
                unit="",
            )
        )
    return sliders


def _infer_x_grid(emu) -> np.ndarray:
    """
    Return the x grid from the decoder buffer if available, else reconstruct
    from the decoder config or fall back to a linear grid of 256 points.
    """
    decoder = emu.decoder
    if hasattr(decoder, "_x") and decoder._x is not None:
        x = decoder._x.detach().cpu().numpy()
        if len(x) > 1:
            return x

    # Try to read from decoder config (SBPLDecoder / QuadraticDecoder store logx bounds)
    cfg = {}
    if hasattr(decoder, "get_config"):
        cfg = decoder.get_config()

    logx_lo = cfg.get("logx_lo", None)
    logx_hi = cfg.get("logx_hi", None)
    if logx_lo is not None and logx_hi is not None:
        return np.geomspace(10 ** logx_lo, 10 ** logx_hi, 256)

    return np.linspace(0.0, 1.0, 256)


def _compute_y_range(emu, x_grid: np.ndarray, n_samples: int = 500) -> tuple:
    """
    Sample the emulator over a range of random parameter combinations to
    determine a stable y-axis range for plots.

    Returns (y_min, y_max) in linear scale, suitable for ax.set_ylim.
    """
    meta = emu._meta or {}
    bounds = meta.get("train_param_bounds") or {}
    pmin_list = bounds.get("min", [])
    pmax_list = bounds.get("max", [])
    paramnames = emu.paramnames or meta.get("paramnames") or []

    if not paramnames or not pmin_list or not pmax_list:
        return None

    rng = np.random.default_rng(42)
    n_params = len(paramnames)
    lo = np.array([float(pmin_list[i]) if i < len(pmin_list) else -10.0 for i in range(n_params)])
    hi = np.array([float(pmax_list[i]) if i < len(pmax_list) else 10.0 for i in range(n_params)])

    samples = rng.uniform(lo, hi, size=(n_samples, n_params))

    emu.decoder.x = x_grid
    all_y = []
    for theta in samples:
        try:
            y = emu.transform(theta[None, :], x=x_grid, return_latent_params=False)
            if y.ndim == 2:
                y = y[0]
            mask = np.isfinite(y) & (y > 0)
            if mask.any():
                all_y.append(y[mask])
        except Exception:
            pass

    if not all_y:
        return None

    all_vals = np.concatenate(all_y)
    # Use robust percentiles to avoid extreme outliers dominating
    y_min = np.percentile(all_vals, 0.5)
    y_max = np.percentile(all_vals, 99.5)
    y_min = max(y_min, y_max / 1e15)

    # Add a small margin (factor of 3 on each side in log space)
    if y_min > 0 and y_max > 0:
        log_min = np.log10(y_min) - 0.5
        log_max = np.log10(y_max) + 0.5
        return (10 ** log_min, 10 ** log_max)

    return None


# ---------------------------------------------------------------------------
# Emulator runner
# ---------------------------------------------------------------------------

def run_emulator_generic(param_values: dict):
    """
    param_values: dict  name -> float

    Returns (x_grid, y, error_msg_or_None)
    """
    emu = STATE.emulator
    if emu is None:
        return STATE.x_grid, np.zeros(len(STATE.x_grid)), "Emulator not loaded"

    try:
        theta = np.array(
            [[param_values[n] for n in STATE.paramnames]], dtype=np.float64
        )
        x = STATE.x_grid
        emu.decoder.x = x
        y = emu.transform(theta, x=x, return_latent_params=False)
        if y.ndim == 2:
            y = y[0]
        return x, y, None
    except Exception as e:
        import traceback
        return STATE.x_grid, np.zeros(len(STATE.x_grid)), str(e) + "\n" + traceback.format_exc()


# ---------------------------------------------------------------------------
# Plot builder
# ---------------------------------------------------------------------------

def make_plot(x, y, param_values: dict, error_msg=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    xlabel = f"{STATE.xlabel} [{STATE.xunit}]" if STATE.xunit else STATE.xlabel
    ylabel = f"{STATE.ylabel} [{STATE.yunit}]" if STATE.yunit else STATE.ylabel

    if error_msg:
        ax.text(
            0.5, 0.5,
            f"Error:\n{error_msg[:400]}",
            transform=ax.transAxes,
            ha='center', va='center', color='red', fontsize=8, wrap=True,
            family='monospace',
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(STATE.title)
    else:
        mask = np.isfinite(y) & (y > 0)
        if mask.any():
            ax.loglog(x[mask], y[mask], color='darkorange', lw=2, label='SED')
        else:
            ax.plot(x, y, color='darkorange', lw=2, label='SED')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        # Apply fixed y-axis range if available
        if STATE.y_range is not None:
            ax.set_ylim(STATE.y_range)

        # Round displayed param values to 2 digits
        param_str = "  ".join(
            f"{k}={round(v, 2)}" for k, v in list(param_values.items())[:4]
        )
        ax.set_title(f"{STATE.title}  |  {param_str}", fontsize=9)
        ax.legend(fontsize=11)
        ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ---------------------------------------------------------------------------
# HTML template (fully generic – uses Jinja vars from STATE)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a2e;
            color: #e0e0e0;
            min-height: 100vh;
        }
        header {
            background: linear-gradient(135deg, #16213e, #0f3460);
            padding: 18px 32px;
            border-bottom: 2px solid #e94560;
            display: flex;
            align-items: center;
            gap: 16px;
        }
        header h1 { font-size: 1.6em; color: #e94560; letter-spacing: 1px; }
        header p { font-size: 0.9em; color: #a0aec0; margin-top: 4px; }
        .main-layout { display: flex; gap: 0; min-height: calc(100vh - 80px); }
        .sidebar {
            width: 360px; min-width: 300px;
            background: #16213e;
            padding: 24px 20px;
            border-right: 1px solid #0f3460;
            overflow-y: auto;
        }
        .sidebar h2 {
            font-size: 1.05em; color: #63b3ed; margin-bottom: 20px;
            text-transform: uppercase; letter-spacing: 1px;
            border-bottom: 1px solid #2d3748; padding-bottom: 8px;
        }
        .param-block {
            background: #0f3460; border-radius: 10px;
            padding: 14px 16px; margin-bottom: 14px;
            border: 1px solid #1a4a7a; transition: border-color 0.2s;
        }
        .param-block:hover { border-color: #e94560; }
        .param-label {
            font-size: 0.88em; color: #90cdf4; margin-bottom: 6px;
            display: flex; justify-content: space-between; align-items: center;
        }
        .param-value-display {
            font-size: 1.1em; font-weight: bold; color: #e94560;
            min-width: 60px; text-align: right;
        }
        .slider-input-row {
            display: flex; align-items: center; gap: 10px; margin: 6px 0 2px 0;
        }
        .slider-input-row input[type=range] {
            flex: 1; margin: 0; accent-color: #e94560; cursor: pointer;
        }
        .number-input {
            width: 90px; flex-shrink: 0;
            background: #1a2a4a; border: 1px solid #2d4a6a;
            color: #e0e0e0; border-radius: 5px; padding: 5px 8px; font-size: 0.92em;
        }
        .number-input:focus { outline: none; border-color: #e94560; }
        .param-minmax {
            display: flex; justify-content: space-between;
            font-size: 0.72em; color: #4a5568; margin-top: 2px;
        }
        .btn-run {
            width: 100%; padding: 12px;
            background: linear-gradient(135deg, #e94560, #c0392b);
            color: white; border: none; border-radius: 8px;
            font-size: 1.05em; font-weight: bold; cursor: pointer;
            letter-spacing: 1px; margin-top: 8px;
            transition: opacity 0.2s, transform 0.1s;
        }
        .btn-run:hover { opacity: 0.9; transform: translateY(-1px); }
        .btn-run:active { transform: translateY(0); }
        .btn-reset {
            width: 100%; padding: 9px;
            background: #2d3748; color: #a0aec0;
            border: 1px solid #4a5568; border-radius: 8px;
            font-size: 0.92em; cursor: pointer; margin-top: 8px;
            transition: background 0.2s;
        }
        .btn-reset:hover { background: #374151; }
        .plot-area {
            flex: 1; display: flex; flex-direction: column;
            align-items: center; justify-content: flex-start;
            padding: 30px 24px; background: #1a1a2e;
        }
        .plot-container {
            background: #16213e; border-radius: 14px;
            border: 1px solid #0f3460; padding: 18px;
            width: 100%; max-width: 900px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        }
        .plot-container img { width: 100%; border-radius: 8px; display: block; }
        .plot-placeholder {
            width: 100%; height: 400px;
            display: flex; align-items: center; justify-content: center;
            color: #4a5568; font-size: 1.1em;
            border-radius: 8px; background: #0f3460;
        }
        .status-bar { margin-top: 14px; font-size: 0.85em; color: #718096; text-align: center; }
        .status-bar.loading { color: #f6ad55; }
        .status-bar.error   { color: #fc8181; }
        .status-bar.success { color: #68d391; }
        .param-info {
            margin-top: 18px; background: #0f3460; border-radius: 8px;
            padding: 12px 14px; font-size: 0.82em; color: #90cdf4;
            border: 1px solid #1a4a7a; max-height: 260px; overflow-y: auto;
        }
        .param-info h3 { color: #63b3ed; margin-bottom: 8px; font-size: 0.95em; }
        .param-info table { width: 100%; border-collapse: collapse; }
        .param-info td { padding: 3px 6px; }
        .param-info td:first-child { color: #a0aec0; }
        .param-info td:last-child { color: #e94560; font-weight: bold; text-align: right; }
        .spinner {
            display: none; width: 22px; height: 22px;
            border: 3px solid #4a5568; border-top: 3px solid #e94560;
            border-radius: 50%; animation: spin 0.8s linear infinite; margin: 0 auto;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .auto-update-row {
            display: flex; align-items: center; gap: 8px;
            margin-top: 10px; font-size: 0.88em; color: #a0aec0;
        }
        .auto-update-row input[type=checkbox] { accent-color: #e94560; }
        .model-info {
            margin-top: 16px; font-size: 0.78em; color: #4a5568;
            word-break: break-all; line-height: 1.5;
        }
    </style>
</head>
<body>
<header>
    <div>
        <h1>&#127756; {{ title }}</h1>
        <p>Neural network emulator &nbsp;|&nbsp; model: <code style="color:#63b3ed;">{{ model_path }}</code></p>
    </div>
</header>
<div class="main-layout">
    <div class="sidebar">
        <h2>Parameters</h2>
        {% for s in sliders %}
        <div class="param-block">
            <div class="param-label">
                <span>{{ s.label | safe }}</span>
                <span class="param-value-display" id="val_{{ s.name }}">{{ "%.4g" | format(s.default) }}</span>
            </div>
            {% if s.description %}
            <span style="font-size:0.78em;color:#718096;display:block;margin-bottom:4px;">{{ s.description }}</span>
            {% endif %}
            <div class="slider-input-row">
                <input type="range"
                    id="slider_{{ s.name }}"
                    min="{{ s.min }}" max="{{ s.max }}" step="{{ s.step }}"
                    value="{{ s.default }}"
                    oninput="onSliderChange('{{ s.name }}', this.value)">
                <input type="number"
                    class="number-input"
                    id="num_{{ s.name }}"
                    min="{{ s.min }}" max="{{ s.max }}" step="{{ s.step }}"
                    value="{{ '%.6g' | format(s.default) }}"
                    oninput="onNumberChange('{{ s.name }}', this.value)">
            </div>
            <div class="param-minmax">
                <span>{{ "%.2g" | format(s.min) }}</span>
                <span>{{ "%.2g" | format(s.max) }}</span>
            </div>
        </div>
        {% endfor %}

        <button class="btn-run" onclick="runEmulator()">&#9654; Compute SED</button>
        <button class="btn-reset" onclick="resetDefaults()">&#8635; Reset to Defaults</button>

        <div class="auto-update-row">
            <input type="checkbox" id="auto_update" checked onchange="onAutoUpdateChange()">
            <label for="auto_update">Auto-update on parameter change</label>
        </div>

        <div class="param-info" id="current_params_display">
            <h3>Current Values</h3>
            <table id="phys_table">
                {% for s in sliders %}
                <tr>
                    <td>{{ s.name }}</td>
                    <td id="phys_{{ s.name }}">{{ "%.4g" | format(s.default) }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>

        <div class="model-info">
            Model: {{ model_path }}<br>
            Parameters: {{ sliders | length }}
        </div>
    </div>

    <div class="plot-area">
        <div class="plot-container">
            <div id="plot_img_container">
                <div class="plot-placeholder" id="plot_placeholder">
                    Click <b>&nbsp;&#9654; Compute SED&nbsp;</b> to generate the spectrum
                </div>
                <img id="plot_img" src="" alt="SED Plot" style="display:none;">
            </div>
        </div>
        <div class="spinner" id="spinner"></div>
        <div class="status-bar" id="status_bar">Ready. Set parameters and click Compute SED.</div>
    </div>
</div>

<script>
const defaults = {
    {% for s in sliders %}
    "{{ s.name }}": {{ s.default }},
    {% endfor %}
};
const sliderDefs = {
    {% for s in sliders %}
    "{{ s.name }}": { min: {{ s.min }}, max: {{ s.max }}, step: {{ s.step }} },
    {% endfor %}
};
const paramNames = [{% for s in sliders %}"{{ s.name }}"{% if not loop.last %}, {% endif %}{% endfor %}];

let isComputing = false;
let pendingUpdate = false;
let autoUpdate = true;
let autoUpdateTimer = null;

function onAutoUpdateChange() {
    autoUpdate = document.getElementById('auto_update').checked;
    if (autoUpdate) scheduleAutoUpdate();
}

function fmtVal(v) {
    if (Math.abs(v) < 1e4 && Math.abs(v) >= 1e-3 || v === 0)
        return parseFloat(v.toFixed(6)).toString();
    return v.toExponential(4);
}

function onSliderChange(name, val) {
    val = parseFloat(val);
    document.getElementById('val_' + name).textContent = fmtVal(val);
    document.getElementById('num_' + name).value = fmtVal(val);
    updatePhysValues();
    if (autoUpdate) scheduleAutoUpdate();
}

function onNumberChange(name, val) {
    val = parseFloat(val);
    const def = sliderDefs[name];
    if (isNaN(val)) return;
    val = Math.max(def.min, Math.min(def.max, val));
    document.getElementById('slider_' + name).value = val;
    document.getElementById('val_' + name).textContent = fmtVal(val);
    updatePhysValues();
    if (autoUpdate) scheduleAutoUpdate();
}

function scheduleAutoUpdate() {
    if (autoUpdateTimer) clearTimeout(autoUpdateTimer);
    autoUpdateTimer = setTimeout(runEmulator, 400);
}

function getParams() {
    const params = {};
    for (const name of paramNames) {
        params[name] = parseFloat(document.getElementById('slider_' + name).value);
    }
    return params;
}

function updatePhysValues() {
    const p = getParams();
    for (const name of paramNames) {
        const el = document.getElementById('phys_' + name);
        if (el) el.textContent = fmtVal(p[name]);
    }
}

function resetDefaults() {
    for (const name of paramNames) {
        document.getElementById('slider_' + name).value = defaults[name];
        document.getElementById('num_' + name).value = fmtVal(defaults[name]);
        document.getElementById('val_' + name).textContent = fmtVal(defaults[name]);
    }
    updatePhysValues();
    if (autoUpdate) scheduleAutoUpdate();
}

function setStatus(msg, cls) {
    const el = document.getElementById('status_bar');
    el.textContent = msg;
    el.className = 'status-bar ' + (cls || '');
}

function runEmulator() {
    if (isComputing) { pendingUpdate = true; return; }
    const params = getParams();
    isComputing = true;
    pendingUpdate = false;
    setStatus('Computing SED...', 'loading');
    document.getElementById('spinner').style.display = 'block';

    fetch('/compute', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params)
    })
    .then(r => r.json())
    .then(data => {
        document.getElementById('spinner').style.display = 'none';
        isComputing = false;
        if (data.error) {
            setStatus('Error: ' + data.error.slice(0, 200), 'error');
        } else {
            const img = document.getElementById('plot_img');
            const ph  = document.getElementById('plot_placeholder');
            img.src = 'data:image/png;base64,' + data.img;
            img.style.display = 'block';
            if (ph) ph.style.display = 'none';
            setStatus(
                'Done. Peak x: ' + data.peak_x + ' ' + data.xunit +
                '  Max y: ' + data.max_y + ' ' + data.yunit,
                'success'
            );
        }
        if (pendingUpdate) { pendingUpdate = false; scheduleAutoUpdate(); }
    })
    .catch(err => {
        document.getElementById('spinner').style.display = 'none';
        isComputing = false;
        setStatus('Request failed: ' + err, 'error');
        if (pendingUpdate) { pendingUpdate = false; scheduleAutoUpdate(); }
    });
}

updatePhysValues();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        title=STATE.title,
        model_path=STATE.model_path,
        sliders=STATE.sliders,
    )


@app.route("/compute", methods=["POST"])
def compute():
    try:
        data = request.json or {}
        param_values = {}
        for s in STATE.sliders:
            name = s["name"]
            param_values[name] = float(data.get(name, s["default"]))

        x, y, error_msg = run_emulator_generic(param_values)

        img_b64 = make_plot(x, y, param_values, error_msg=error_msg)

        peak_x = "N/A"
        max_y = "N/A"
        if not error_msg and np.any(np.isfinite(y) & (y > 0)):
            idx = int(np.argmax(y))
            peak_x = f"{round(x[idx], 2)}"
            max_y = f"{y[idx]:.2e}"

        return jsonify(
            dict(
                img=img_b64,
                error=error_msg,
                peak_x=peak_x,
                max_y=max_y,
                xunit=STATE.xunit,
                yunit=STATE.yunit,
            )
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return jsonify(
            dict(img="", error=str(e) + "\n" + tb,
                 peak_x="N/A", max_y="N/A",
                 xunit="", yunit="")
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    STATE.model_path = args.model_path

    print(f"Loading emulator from: {args.model_path}")
    try:
        emu = load_emulator_from_checkpoint(args.model_path)
    except Exception as e:
        print(f"ERROR: Could not load emulator: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    STATE.emulator = emu

    meta = emu._meta or {}
    STATE.paramnames = emu.paramnames or meta.get("paramnames") or []
    STATE.xlabel = meta.get("xlabel") or "x"
    STATE.xunit  = meta.get("xunit")  or ""
    STATE.ylabel = meta.get("ylabel") or "y"
    STATE.yunit  = meta.get("yunit")  or ""

    # Derive a human-readable title from the model file name
    stem = os.path.splitext(os.path.basename(args.model_path))[0]
    STATE.title = f"SED Emulator – {stem}"

    STATE.sliders = _build_sliders_from_emulator(emu)
    STATE.defaults = {s["name"]: s["default"] for s in STATE.sliders}

    STATE.x_grid = _infer_x_grid(emu)
    print(f"x grid: {len(STATE.x_grid)} points  [{STATE.x_grid.min():.4g} .. {STATE.x_grid.max():.4g}] {STATE.xunit}")
    print(f"Parameters ({len(STATE.paramnames)}): {STATE.paramnames}")

    # Pre-compute y range for consistent axis limits across all plots
    print("Computing y-axis range from emulator samples (for consistent plot scaling)...")
    STATE.y_range = _compute_y_range(emu, STATE.x_grid, n_samples=500)
    if STATE.y_range is not None:
        print(f"y range: [{STATE.y_range[0]:.2e} .. {STATE.y_range[1]:.2e}] {STATE.yunit}")
    else:
        print("y range: could not determine, will use auto-scaling")

    print(f"Serving on http://{args.host}:{args.port}/")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
