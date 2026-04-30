import numpy as np


def as_2d(a: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Convert an array to 2D, tracking whether it was originally 1D.

    Parameters
    ----------
    a : np.ndarray
        Input array of shape (n,) or (m, n).

    Returns
    -------
    out : np.ndarray
        2D array of shape (1, n) if input was 1D, or (m, n) if input was 2D.
    was_1d : bool
        True if the input array was 1D, False if it was 2D.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        return a[None, :], True
    if a.ndim == 2:
        return a, False
    raise ValueError(f"expected 1D or 2D array, got shape {a.shape}")


def ensure_finite(name: str, arr: np.ndarray) -> None:
    """
    Raise an error if the array contains non-finite values.

    Parameters
    ----------
    name : str
        Name of the array, used in the error message.
    arr : np.ndarray
        Array to check for non-finite values.

    Returns
    -------
    None
    """
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")


def apply_log_mask(X: np.ndarray, log_mask: np.ndarray, eps: float) -> np.ndarray:
    """
    Apply a base-10 logarithm to selected columns of X.

    Parameters
    ----------
    X : np.ndarray
        2D input array of shape (m, n).
    log_mask : np.ndarray
        Boolean array of shape (n,) indicating which columns to log-transform.
    eps : float
        Small positive value used to clip column values before applying log,
        preventing log(0) or log of negative numbers.

    Returns
    -------
    out : np.ndarray
        Copy of X with log10 applied to the columns indicated by log_mask.
    """
    X = np.asarray(X, dtype=np.float64)
    log_mask = np.asarray(log_mask, dtype=bool)
    if log_mask.shape != (X.shape[1],):
        raise ValueError(f"log_mask shape {log_mask.shape} != ({X.shape[1]},)")
    out = X.copy()
    if np.any(log_mask):
        out[:, log_mask] = np.log10(np.clip(out[:, log_mask], eps, None))
    return out


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Normalize an array by subtracting the mean and dividing by the standard deviation.

    Parameters
    ----------
    X : np.ndarray
        2D input array of shape (m, n) to normalize.
    mean : np.ndarray
        Array of shape (n,) containing the mean for each feature.
    std : np.ndarray
        Array of shape (n,) containing the standard deviation for each feature.

    Returns
    -------
    out : np.ndarray
        Normalized array of shape (m, n).
    """
    mean = np.asarray(mean, dtype=np.float64).reshape(1, -1)
    std = np.asarray(std, dtype=np.float64).reshape(1, -1)
    return (X - mean) / std


def denormalize(Xn: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Reverse a previous normalization by multiplying by std and adding the mean.

    Parameters
    ----------
    Xn : np.ndarray
        2D normalized array of shape (m, n) to denormalize.
    mean : np.ndarray
        Array of shape (n,) containing the mean for each feature.
    std : np.ndarray
        Array of shape (n,) containing the standard deviation for each feature.

    Returns
    -------
    out : np.ndarray
        Denormalized array of shape (m, n).
    """
    mean = np.asarray(mean, dtype=np.float64).reshape(1, -1)
    std = np.asarray(std, dtype=np.float64).reshape(1, -1)
    return Xn * std + mean


def _p95_l2(yhat: np.ndarray, y: np.ndarray, percentile: float = 95) -> float:
    """
    Compute the specified percentile of L2 (Euclidean) errors between predictions and targets.

    Parameters
    ----------
    yhat : np.ndarray
        2D array of predicted values with shape (m, n).
    y : np.ndarray
        2D array of target values with shape (m, n).
    percentile : float, optional
        Percentile to compute, by default 95.

    Returns
    -------
    out : float
        The specified percentile of the row-wise L2 errors.
    """
    err = np.linalg.norm(yhat - y, axis=1)
    return float(np.percentile(err, percentile))
