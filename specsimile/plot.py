# specsimile/plot.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_error_hist(eval_out, bins=50, logx=True):
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


def plot_fit_examples(eval_out, n_first=12, n_worst=12, yscale="log", legend=False):
    x = eval_out["x"]
    y = eval_out["y_true"]
    yhat = eval_out["yhat"]
    err = eval_out["err"]

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

def plot_fit_diff(eval_out, yscale="log", min_ratio=1.0, min_diff=0.0):
    x = eval_out["x"]
    y = eval_out["y_true"]
    yhat = eval_out["yhat"]
    err = eval_out['err']

    ncol = len(x)

    fig, axes = plt.subplots(2, ncol, figsize=(3.2 * ncol, 6.0), squeeze=False)

    for i in range(ncol):
        if len(eval_out.get("labels", [])) == ncol:
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
        #axdiff.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    return fig, axes


def plot_param_corner_scatter(eval_out, param_names=None, max_points=5000):
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

    fig.tight_layout()
    return fig, axes


def plot_decoder_param_corner_scatter(eval_out, param_names=None, max_points=5000):
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
