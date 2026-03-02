import numpy as np


def as_2d(a: np.ndarray) -> tuple[np.ndarray, bool]:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        return a[None, :], True
    if a.ndim == 2:
        return a, False
    raise ValueError(f"expected 1D or 2D array, got shape {a.shape}")


def ensure_finite(name: str, arr: np.ndarray):
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")


def apply_log_mask(X: np.ndarray, log_mask: np.ndarray, eps: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    log_mask = np.asarray(log_mask, dtype=bool)
    if log_mask.shape != (X.shape[1],):
        raise ValueError(f"log_mask shape {log_mask.shape} != ({X.shape[1]},)")
    out = X.copy()
    if np.any(log_mask):
        out[:, log_mask] = np.log10(np.clip(out[:, log_mask], eps, None))
    return out


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean, dtype=np.float64).reshape(1, -1)
    std = np.asarray(std, dtype=np.float64).reshape(1, -1)
    return (X - mean) / std


def denormalize(Xn: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean, dtype=np.float64).reshape(1, -1)
    std = np.asarray(std, dtype=np.float64).reshape(1, -1)
    return Xn * std + mean


def _p95_l2(yhat: np.ndarray, y: np.ndarray, percentile: float=95) -> float:
    err = np.linalg.norm(yhat - y, axis=1)
    return float(np.percentile(err, percentile))


