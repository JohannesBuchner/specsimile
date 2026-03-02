# specsimile/store.py
from __future__ import annotations

import os
import h5py
import numpy as np

try:
    import astropy.units as u
except Exception:  # astropy optional
    u = None


def _unit_to_string(unit):
    """Accept astropy Unit, Quantity, str, or None; return str or ''."""
    if unit is None:
        return ""
    if isinstance(unit, str):
        return unit
    if u is not None:
        if isinstance(unit, u.UnitBase):
            return unit.to_string()
        if isinstance(unit, u.Quantity):
            return unit.unit.to_string()
    # fallback
    return str(unit)


def _as_bytes_array(strings):
    if strings is None:
        return None
    return np.asarray([s.encode("utf-8") if isinstance(s, str) else s for s in strings])


def _as_str_list(arr):
    if arr is None:
        return None
    out = []
    for v in arr:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8"))
        else:
            out.append(str(v))
    return out


class DatasetWriter:
    """
    Create + append an emulator training dataset stored in HDF5.

    Layout:
      /x              (X,) float64
      /y              (N,Y) float64
      /params         (N,P) float64
      /paramnames     (P,) bytes (utf-8)
    Attributes:
      num_data_training, N, xlabel, ylabel, xunit, yunit
    """

    def __init__(
        self,
        filename: str,
        N: int,
        x,
        y,
        params,
        *,
        paramnames=None,
        xlabel: str = "",
        ylabel: str = "",
        xunit=None,
        yunit=None,
        mode: str = "w",
    ):
        self.filename = str(filename)
        self._f = h5py.File(self.filename, mode=mode)

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
            if key in self._f:
                raise RuntimeError(f"Dataset '{key}' already exists in {self.filename}")

        self.N = N
        self.num_data_training = 0

        self._f.create_dataset("x", data=x)
        self._f.create_dataset("y", dtype=np.float64, shape=(N, len(y)))
        self._f.create_dataset("params", dtype=np.float64, shape=(N, len(params)))

        if paramnames is None:
            paramnames = [f"p{i}" for i in range(len(params))]
        paramnames_b = _as_bytes_array(paramnames)
        self._f.create_dataset("paramnames", data=paramnames_b)

        self._f.attrs["num_data_training"] = self.num_data_training
        self._f.attrs["N"] = self.N

        self._f.attrs["xlabel"] = str(xlabel)
        self._f.attrs["ylabel"] = str(ylabel)
        self._f.attrs["xunit"] = _unit_to_string(xunit)
        self._f.attrs["yunit"] = _unit_to_string(yunit)

        self._f.flush()

    def append(self, x, y, params) -> int:
        if self._f is None:
            raise RuntimeError("Writer is closed")

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        params = np.asarray(params, dtype=np.float64)

        if x.ndim != 1 or y.ndim != 1 or params.ndim != 1:
            raise ValueError("x,y,params must be 1D arrays")

        if len(y) != self._f["y"].shape[1]:
            raise ValueError("y length mismatch")
        if len(params) != self._f["params"].shape[1]:
            raise ValueError("params length mismatch")

        x0 = self._f["x"][()]
        if x.shape != x0.shape or not np.allclose(x, x0, rtol=0, atol=0):
            raise ValueError("x does not match stored x grid")

        i = int(self.num_data_training)
        if i >= int(self.N):
            raise RuntimeError(f"Training file full (N={self.N}).")

        self._f["y"][i, :] = y
        self._f["params"][i, :] = params

        self.num_data_training += 1
        self._f.attrs["num_data_training"] = int(self.num_data_training)
        self._f.flush()

        # auto-close when full (keeps old behavior)
        if self.num_data_training == self._f["y"].shape[0]:
            self.close()

        return i

    def close(self):
        if self._f is not None:
            try:
                self._f.close()
            finally:
                self._f = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class DatasetReader:
    """
    Read emulator training dataset from HDF5.
    """

    def __init__(self, filename: str, mode: str = "r"):
        self.filename = str(filename)
        if not os.path.exists(self.filename) and "r" in mode:
            raise FileNotFoundError(self.filename)
        self._f = h5py.File(self.filename, mode=mode)

    @property
    def N(self) -> int:
        return int(self._f.attrs.get("N", self._f["y"].shape[0]))

    @property
    def num_data_training(self) -> int:
        return int(self._f.attrs.get("num_data_training", self._f["y"].shape[0]))

    @property
    def xlabel(self) -> str:
        return str(self._f.attrs.get("xlabel", ""))

    @property
    def ylabel(self) -> str:
        return str(self._f.attrs.get("ylabel", ""))

    @property
    def xunit(self) -> str:
        return str(self._f.attrs.get("xunit", ""))

    @property
    def yunit(self) -> str:
        return str(self._f.attrs.get("yunit", ""))

    @property
    def paramnames(self):
        if "paramnames" not in self._f:
            return None
        return _as_str_list(self._f["paramnames"][()])

    @property
    def x(self) -> np.ndarray:
        return self._f["x"][()].astype(np.float64)

    @property
    def y(self) -> np.ndarray:
        return self._f["y"][()].astype(np.float64)

    @property
    def params(self) -> np.ndarray:
        return self._f["params"][()].astype(np.float64)

    def close(self):
        if self._f is not None:
            try:
                self._f.close()
            finally:
                self._f = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
