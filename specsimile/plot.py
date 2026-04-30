# specsimile/plot.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional


def plot_error_hist(
    eval_out: dict,
    bins: int = 50,
    logx: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot a histogram of RMS errors from emulator evaluation output.

    Parameters
    ----------
    eval_out : dict
        Output dictionary from Emulator.evaluate(), must contain key ``"err"``
        with an array of per-sample RMS errors.
    bins : int, optional
        Number of histogram bins. Default is 50.
    logx : bool, optional
        If True, use a log scale on the x-axis. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object.
    """
    err = np.asarray(eval_out["err"], float)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    e = err[np.isfinite(err)]
    if len(e) == 0:
        ax.text(0.5, 0.5, "no finite errors", ha="center", va="center")
        return fig, ax

    ax.hist(e, bins=bins, color="0.6", edgecolor="0.2")
    p95 = np.percentile(e, 95)
    mean_rms = np.mean(e)
    ax.axvline(mean_rms, color="C4", lw=2, ls="--", label=f"mean={mean_rms:.3g}")
    ax.axvline(p95, color="C3", lw=2, label=f"p95={p95:.3g}")
    ax.set_xlabel("RMS error (tolerance space)")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig, ax


def plot_fit_examples(
    eval_out: dict,
    n_first: int = 12,
    n_worst: int = 12,
    yscale: str = "log",
    legend: bool = False,
) -> tuple[Figure, np.ndarray]:
    """
    Plot example spectral fits for the first and worst-error samples.

    Parameters
    ----------
    eval_out : dict
        Output dictionary from Emulator.evaluate(), must contain keys
        ``"x"``, ``"y_true"``, ``"yhat"``, ``"err"``, and optionally
        ``"xnames"``, ``"z_norm"``, ``"z_phys"``, ``"dec_params"``.
    n_first : int, optional
        Number of first (by index) samples to plot. Default is 12.
    n_worst : int, optional
        Number of worst-error samples to plot. Default is 12.
    yscale : str, optional
        Scale for the y-axis; ``"log"`` for logarithmic, ``"linear"`` for
        linear. Default is ``"log"``.
    legend : bool, optional
        If True, display a legend with latent and physical parameter values
        on each panel. Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    axes : numpy.ndarray of matplotlib.axes.Axes
        2-D array of Axes objects with shape ``(4, ncol2)``.
    """
    x = eval_out["x"]
    y = eval_out["y_true"]
    yhat = eval_out["yhat"]
    err = eval_out["err"]
    xnames = eval_out.get("xnames", None)

    use_xnames = xnames is not None and len(xnames) == len(x)

    n = len(err)
    n_first = min(int(n_first), n)
    n_worst = min(int(n_worst), n)

    idx_first = np.arange(n_first)
    idx_worst = np.argsort(err)[-n_worst:][::-1]

    nshow = n_first + n_worst
    ncol = max(1, int(np.ceil(nshow / 2)))
    ncol2 = max(1, ncol // 2)

    fig, axes = plt.subplots(
        4, ncol2, figsize=(3.2 * ncol2, 6 * 2),
        squeeze=False, sharex=True,
    )

    def _plot(ax: Axes, i: int) -> None:
        yt = np.asarray(y[i], float)
        yp = np.asarray(yhat[i], float)
        if use_xnames:
            ax.plot(x, np.clip(yt, 1e-300, None), "k-", lw=1.5, label="true")
            ax.plot(x, np.clip(yp, 1e-300, None), "r--", lw=1.2, label="pred")
            if yscale == "log":
                ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels(xnames, rotation=90, fontsize=6)
        else:
            if yscale == "log":
                ax.loglog(x, np.clip(yt, 1e-300, None), "k-", lw=1.5, label="true")
                ax.loglog(x, np.clip(yp, 1e-300, None), "r--", lw=1.2, label="pred")
            else:
                ax.plot(x, yt, "k-", lw=1.5, label="true")
                ax.plot(x, yp, "r--", lw=1.2, label="pred")

        ax.set_title(f"i={i}  err={err[i]:.3g}", fontsize=9)
        ax.grid(True, which="both", alpha=0.2)
        if legend:
            ax.legend(title=f"""z_norm={" ".join(['%.2f' % v for v in eval_out["z_norm"][i]]  if eval_out.get("z_norm") is not None else '' )}
z_phys={" ".join(['%.2f' % v for v in eval_out["z_phys"][i]])  if eval_out.get("z_phys") is not None else ''}
params={" ".join(['%.2f' % v for v in eval_out["dec_params"][i]]) if eval_out.get("dec_params") is not None else '?'}
""")

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


def plot_residual_hist(
    eval_out: dict,
    bins: int = 50,
    yscale: str = "log",
) -> tuple[Figure, Axes]:
    """
    One panel with one histogram per x-grid point showing (pred - true).

    Parameters
    ----------
    eval_out : dict
        Output of Emulator.evaluate(), must contain keys ``"x"``,
        ``"yhat"``, ``"y_true"``, and optionally ``"xnames"``,
        ``"xlabel"``, ``"ylabel"``, ``"yunit"``.
    bins : int, optional
        Number of histogram bins. Default is 50.
    yscale : str, optional
        If ``"log"``, use log10(yhat / y_true) (log10-space residuals).
        Otherwise, use yhat - y_true (linear residuals). Default is ``"log"``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object.
    """
    x = np.asarray(eval_out["x"], float)
    ncol = len(x)

    ylabel  = eval_out.get("ylabel", "")
    yunit   = eval_out.get("yunit", "")
    if eval_out.get('xnames', None) is not None and len(eval_out['xnames']) == ncol:
        xnames = eval_out['xnames']
    else:
        xlabel  = eval_out.get("xlabel", "")
        xnames = [f"{xlabel} {x[i]:.3g}" for i in range(ncol)]

    if yscale == 'log':
        pred = np.asarray(eval_out["yhat"], float)
        true = np.asarray(eval_out["y_true"], float)
        residuals = np.log10((pred + 1e-60) / (true + 1e-50))
        res_label = r"$\log(\hat{y} / y)$" + f' {ylabel} [{yunit}]'
    else:
        pred = np.asarray(eval_out["yhat"], float)
        true = np.asarray(eval_out["y_true"], float)
        residuals = pred - true
        res_label = r"$\hat{y} - y$" + f' {ylabel} [{yunit}]'

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (xname, res_k) in enumerate(zip(xnames, residuals.transpose())):
        finite = res_k[np.isfinite(res_k)]
        ax.hist(finite, bins=bins, label=xname, linewidth=0.4, histtype='step')

    ax.legend(ncol=4)
    ax.axvline(0.0, color="k", lw=1.2, ls="--")
    ax.set_xlabel(res_label)
    ax.set_yticks([])
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_fit_diff(
    eval_out: dict,
    yscale: str = "log",
    min_ratio: float = 1.0,
    min_diff: float = 0.0,
) -> tuple[Figure, np.ndarray]:
    """
    Scatter plots of predicted vs. true values and their residuals per output
    dimension.

    Parameters
    ----------
    eval_out : dict
        Output dictionary from Emulator.evaluate(), must contain keys
        ``"x"``, ``"y_true"``, ``"yhat"``, ``"err"``, and optionally
        ``"xnames"``, ``"labels"``, ``"ylabel"``, ``"yunit"``.
    yscale : str, optional
        Scale for axes; ``"log"`` for log-log scatter and ratio residuals,
        anything else for linear scatter and difference residuals.
        Default is ``"log"``.
    min_ratio : float, optional
        Minimum symmetric y-axis range for ratio (pred/true) panels when
        ``yscale="log"``. Default is 1.0.
    min_diff : float, optional
        Minimum symmetric y-axis range for difference (pred-true) panels
        when ``yscale`` is not ``"log"``. Default is 0.0.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    axes : numpy.ndarray of matplotlib.axes.Axes
        2-D array of Axes objects with shape ``(2, ncol)`` where ``ncol``
        equals the number of output dimensions.
    """
    x = eval_out["x"]
    y = eval_out["y_true"]
    yhat = eval_out["yhat"]
    err = eval_out['err']
    xnames = eval_out.get("xnames", None)

    use_xnames = xnames is not None and len(xnames) == len(x)

    ncol = len(x)

    fig, axes = plt.subplots(2, ncol, figsize=(3.2 * ncol, 6.0), squeeze=False)

    for i in range(ncol):
        if use_xnames:
            ylabel = xnames[i]
        elif len(eval_out.get("labels", [])) == ncol:
            ylabel = eval_out["labels"][i]
        else:
            ylabel = f"{eval_out['ylabel']} [{i+1}]"
            if eval_out['yunit'] != '':
                ylabel += f" [{eval_out['yunit']}]"

        ax = axes[0, i]
        axdiff = axes[1, i]
        yt = y[:,i]
        yp = yhat[:,i]
        ylo = min(yt.min(), yp.min())
        yhi = max(yt.max(), yp.max())
        if yscale == "log":
            ax.scatter(np.clip(yt, 1e-300, None), np.clip(yp, 1e-300, None), c=err)
            axdiff.scatter(np.clip(yt, 1e-300, None), np.clip(yp, 1e-300, None) / np.clip(yt, 1e-300, None), c=err)
            ax.set_yscale('log')
            ax.set_xscale('log')
            axdiff.set_yscale('log')
            axdiff.set_xscale('log')
        else:
            ax.scatter(yt, yp, c=err)
            axdiff.scatter(yt, yp - yt, c=err)
        ax.set_xlabel(f"true {ylabel}")
        ax.set_ylabel(f"pred {ylabel}")
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(ylo, yhi)
        ax.plot([ylo, yhi], [ylo, yhi], '--', color='k')
        ax.grid(True, which="both", alpha=0.2)

        axdiff.set_xlabel(f"true {ylabel}")
        if yscale == "log":
            axdiff.set_ylabel("pred / true")
        else:
            axdiff.set_ylabel("pred - true")
        ydifflo, ydiffhi = axdiff.get_ylim()
        if yscale == "log":
            ydiffhi = max(1. / ydifflo, ydiffhi, min_ratio)
            axdiff.set_ylim(1. / ydiffhi, ydiffhi)
            axdiff.plot([ylo, yhi], [1., 1.], '--', color='k')
        else:
            ydiffhi = max(abs(ydifflo), ydiffhi, min_diff)
            axdiff.set_ylim(-ydiffhi, ydiffhi)
            axdiff.plot([ylo, yhi], [0, 0], '--', color='k')

    fig.tight_layout()
    return fig, axes


def plot_param_corner_scatter(
    eval_out: dict,
    param_names: Optional[list[str]] = None,
    max_points: int = 5000,
) -> tuple[Figure, np.ndarray]:
    """
    Corner scatter plot of input parameters coloured by log10 RMS error.

    Parameters
    ----------
    eval_out : dict
        Output dictionary from Emulator.evaluate(), must contain keys
        ``"params"`` with shape ``(N, D)`` and ``"err"`` with shape ``(N,)``.
    param_names : list of str, optional
        Names for each of the D parameter dimensions. If None or of wrong
        length, defaults to ``["p0", "p1", ...]``. Default is None.
    max_points : int, optional
        Maximum number of points to plot; a random subset is drawn when the
        dataset is larger. Default is 5000.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    axes : numpy.ndarray of matplotlib.axes.Axes
        2-D array of Axes objects with shape ``(D, D)``.
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

    if param_names is None or len(param_names) != D:
        param_names = [f"p{i}" for i in range(D)]

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
                ax.set_xlabel(param_names[j], fontsize=8, rotation=45)
            else:
                ax.set_xticks([])
            if j == 0 and i > 0:
                ax.set_ylabel(param_names[i], fontsize=8)
            else:
                ax.set_yticks([])

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label("log10(L2 error)")

    return fig, axes


def plot_decoder_param_corner_scatter(
    eval_out: dict,
    param_names: Optional[list[str]] = None,
    max_points: int = 5000,
) -> tuple[Figure, np.ndarray]:
    """
    Corner scatter plot of decoder (latent-to-params) parameters coloured by
    log10 RMS error.

    Parameters
    ----------
    eval_out : dict
        Output dictionary from Emulator.evaluate(), must contain key
        ``"dec_params"`` with shape ``(N, D)`` and ``"err"`` with shape
        ``(N,)``. Raises ``RuntimeError`` if ``"dec_params"`` is absent.
    param_names : list of str, optional
        Names for each of the D decoder-parameter dimensions. If None or of
        wrong length, defaults to ``["q0", "q1", ...]``. Default is None.
    max_points : int, optional
        Maximum number of points to plot; a random subset is drawn when the
        dataset is larger. Default is 5000.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object.
    axes : numpy.ndarray of matplotlib.axes.Axes
        2-D array of Axes objects with shape ``(D, D)``.
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
