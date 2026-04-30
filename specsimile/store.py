# specsimile/store.py
from __future__ import annotations

import os
import h5py
import numpy as np
from typing import Optional, Union, List

try:
    import astropy.units as u
except Exception:  # astropy optional
    u = None


def _unit_to_string(unit: Optional[Union[str, "u.UnitBase", "u.Quantity"]]) -> str:
    """
    Accept astropy Unit, Quantity, str, or None; return str or ''.

    Parameters
    ----------
    unit : str, astropy.units.UnitBase, astropy.units.Quantity, or None
        The unit to convert to a string representation.

    Returns
    -------
    str
        String representation of the unit, or empty string if None.
    """
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


def _as_bytes_array(strings: Optional[List[Union[str, bytes]]]) -> Optional[np.ndarray]:
    """
    Convert a list of strings or bytes to a numpy array of bytes.

    Parameters
    ----------
    strings : list of str or bytes, or None
        The strings to convert.

    Returns
    -------
    numpy.ndarray or None
        Array of utf-8 encoded bytes, or None if input is None.
    """
    if strings is None:
        return None
    return np.asarray([s.encode("utf-8") if isinstance(s, str) else s for s in strings])


def _as_str_list(arr: Optional[Union[np.ndarray, List]]) -> Optional[List[str]]:
    """
    Convert a numpy array or list of bytes/strings to a list of str.

    Parameters
    ----------
    arr : numpy.ndarray, list, or None
        Array of bytes or strings to convert.

    Returns
    -------
    list of str or None
        List of decoded strings, or None if input is None.
    """
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
      /xnames         (Y,) bytes (utf-8)   [optional, may be absent]
    Attributes:
      num_data_training, N, xlabel, ylabel, xunit, yunit
    """

    def __init__(
        self,
        filename: str,
        N: int,
        x: Union[np.ndarray, List[float]],
        paramnames: List[str],
        *,
        xnames: Optional[List[str]] = None,
        xlabel: str = "",
        ylabel: str = "",
        xunit: Optional[Union[str, "u.UnitBase", "u.Quantity"]] = None,
        yunit: Optional[Union[str, "u.UnitBase", "u.Quantity"]] = None,
        mode: str = "w",
    ):
        """
        Parameters
        ----------
        filename : str
            Path to the HDF5 file to create (a '.tmp' suffix is used during writing).
        N : int
            Total number of training rows to allocate.
        x : array_like of shape (X,)
            The x-axis grid, must be 1D.
        paramnames : list of str
            Names of the parameters, length P.
        xnames : list of str, optional
            Names for each x-axis point, length must match x. If provided,
            stored as a 'xnames' dataset.
        xlabel : str, optional
            Label for the x-axis. Default is ''.
        ylabel : str, optional
            Label for the y-axis. Default is ''.
        xunit : str, astropy.units.UnitBase, astropy.units.Quantity, or None, optional
            Unit for the x-axis.
        yunit : str, astropy.units.UnitBase, astropy.units.Quantity, or None, optional
            Unit for the y-axis.
        mode : str, optional
            HDF5 file open mode. Default is 'w'.
        """
        self.filename = str(filename)
        self._f = h5py.File(self.filename + ".tmp", mode=mode)

        N = int(N)
        x = np.asarray(x, dtype=np.float64)
        num_params = len(paramnames)

        if x.ndim != 1:
            raise ValueError(f"x must be 1D, got {x.shape}")

        for key in ("x", "params"):
            if key in self._f:
                raise RuntimeError(f"Dataset '{key}' already exists in {self.filename}")

        self.N = N
        self.num_data_training = 0

        self._f.create_dataset("x", data=x)
        self._f.create_dataset("y", dtype=np.float64, shape=(N, len(x)))
        self._f.create_dataset("params", dtype=np.float64, shape=(N, num_params))

        paramnames_b = _as_bytes_array(paramnames)
        self._f.create_dataset("paramnames", data=paramnames_b)

        if xnames is not None:
            if len(xnames) != len(x):
                raise ValueError(
                    f"xnames length {len(xnames)} must match x length {len(x)}"
                )
            self._f.create_dataset("xnames", data=_as_bytes_array(xnames))

        self._f.attrs["num_data_training"] = self.num_data_training
        self._f.attrs["N"] = self.N

        self._f.attrs["xlabel"] = str(xlabel)
        self._f.attrs["ylabel"] = str(ylabel)
        self._f.attrs["xunit"] = _unit_to_string(xunit)
        self._f.attrs["yunit"] = _unit_to_string(yunit)

        self._f.flush()

    def _check_open(self) -> None:
        if self._f is None:
            raise RuntimeError("Writer is closed")

    def _check_x(self, x: np.ndarray) -> None:
        x0 = self._f["x"][()]
        if x.shape != x0.shape or not np.allclose(x, x0, rtol=0, atol=0):
            raise ValueError("x does not match stored x grid")

    def append(
        self,
        x: Union[np.ndarray, List[float]],
        y: Union[np.ndarray, List[float]],
        params: Union[np.ndarray, List[float]],
    ) -> int:
        """
        Write a single row.

        Parameters
        ----------
        x : array_like of shape (X,)
            The x-axis grid; must match the stored x grid exactly.
        y : array_like of shape (Y,)
            The y values for this row.
        params : array_like of shape (P,)
            The parameter values for this row.

        Returns
        -------
        int
            The row index that was written.
        """
        self._check_open()

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        params = np.asarray(params, dtype=np.float64)

        if x.ndim != 1 or y.ndim != 1 or params.ndim != 1:
            raise ValueError("x, y, params must be 1D arrays")

        self._check_x(x)

        if len(y) != self._f["y"].shape[1]:
            raise ValueError(
                f"y length mismatch: got {len(y)}, "
                f"expected {self._f['y'].shape[1]}"
            )
        if len(params) != self._f["params"].shape[1]:
            raise ValueError(
                f"params length mismatch: got {len(params)}, "
                f"expected {self._f['params'].shape[1]}"
            )

        i = int(self.num_data_training)
        if i >= int(self.N):
            raise RuntimeError(f"Training file already completely filled (N={self.N}).")

        self._f["y"][i, :] = y
        self._f["params"][i, :] = params

        self.num_data_training += 1
        self._f.attrs["num_data_training"] = int(self.num_data_training)
        self._f.flush()

        # auto-close when full (keeps old behavior)
        if self.num_data_training == self._f["y"].shape[0]:
            self.close()

        return i

    def set(
        self,
        x: Union[np.ndarray, List[float]],
        y: Union[np.ndarray, List[List[float]]],
        params: Union[np.ndarray, List[List[float]]],
    ) -> slice:
        """
        Write all rows at once.

        Parameters
        ----------
        x : array_like of shape (X,)
            The x-axis grid; must match the stored x grid exactly.
        y : array_like of shape (N, Y)
            All y values to write.
        params : array_like of shape (N, P)
            All parameter values to write.

        Returns
        -------
        slice
            The slice written, i.e. ``slice(i0, i0 + n_rows)``.

        Raises
        ------
        RuntimeError
            If the writer is closed or there is insufficient space.
        ValueError
            If shapes are inconsistent.
        """
        self._check_open()

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        params = np.asarray(params, dtype=np.float64)

        if x.ndim != 1:
            raise ValueError(f"x must be 1D, got shape {x.shape}")
        if y.ndim != 2:
            raise ValueError(f"y must be 2D (N, Y), got shape {y.shape}")
        if params.ndim != 2:
            raise ValueError(f"params must be 2D (N, P), got shape {params.shape}")

        n_rows = y.shape[0]
        if params.shape[0] != n_rows:
            raise ValueError(
                f"y and params must have the same number of rows, "
                f"got y={y.shape}, params={params.shape}"
            )

        self._check_x(x)

        if y.shape[1] != self._f["y"].shape[1]:
            raise ValueError(
                f"y column count mismatch: got {y.shape[1]}, "
                f"expected {self._f['y'].shape[1]}"
            )
        if params.shape[1] != self._f["params"].shape[1]:
            raise ValueError(
                f"params column count mismatch: got {params.shape[1]}, "
                f"expected {self._f['params'].shape[1]}"
            )

        i0 = int(self.num_data_training)
        i1 = i0 + n_rows
        if i1 > int(self.N):
            raise RuntimeError(
                f"Not enough space: trying to write rows {i0}..{i1-1} "
                f"but dataset capacity is N={self.N}."
            )

        self._f["y"][i0:i1, :] = y
        self._f["params"][i0:i1, :] = params

        self.num_data_training += n_rows
        self._f.attrs["num_data_training"] = int(self.num_data_training)
        self._f.flush()

        if self.num_data_training == self._f["y"].shape[0]:
            self.close()

        return slice(i0, i1)

    def close(self) -> None:
        """
        Close the writer and atomically rename the temporary file to the final filename.

        If the final file already exists, it is removed before renaming.
        """
        if self._f is not None:
            try:
                # remove any previous file if it exists, to not leave broken files
                os.unlink(self.filename)
            except IOError:
                pass
            try:
                self._f.close()
                # atomic write to final filename by move
                os.rename(self.filename + ".tmp", self.filename)
            finally:
                self._f = None

    def __enter__(self) -> "DatasetWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class DatasetReader:
    """
    Read emulator training dataset from HDF5.
    """

    def __init__(
        self,
        filename: str,
        mode: str = "r",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xunit: Optional[str] = None,
        yunit: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        filename : str
            Path to the HDF5 file to read.
        mode : str, optional
            HDF5 file open mode. Default is 'r'.
        xlabel : str, optional
            Override for the x-axis label stored in the file. If None, reads
            from file attributes.
        ylabel : str, optional
            Override for the y-axis label stored in the file. If None, reads
            from file attributes.
        xunit : str, optional
            Override for the x-axis unit stored in the file. If None, reads
            from file attributes.
        yunit : str, optional
            Override for the y-axis unit stored in the file. If None, reads
            from file attributes.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        self.filename = filename
        self._f = h5py.File(self.filename, mode=mode)
        self.xlabel: str = xlabel or str(self._f.attrs.get("xlabel", ""))
        self.ylabel: str = ylabel or str(self._f.attrs.get("ylabel", ""))
        self.xunit: str = xunit or str(self._f.attrs.get("xunit", ""))
        self.yunit: str = yunit or str(self._f.attrs.get("yunit", ""))
        self.params: np.ndarray = self._f["params"][()].astype(np.float64)
        self.paramnames: List[str] = _as_str_list(self._f["paramnames"][()])
        self.x: np.ndarray = self._f["x"][()].astype(np.float64)
        self.y: np.ndarray = self._f["y"][()].astype(np.float64)
        self.xnames: Optional[List[str]] = (
            _as_str_list(self._f["xnames"][()])
            if "xnames" in self._f else None
        )

    def __str__(self) -> str:
        return (
            f"Dataset(filename={self.filename},\n"
            f"    xlabel={self.xlabel}, xunit={self.xunit}, x={self.x.min()}..{self.x.max()},\n"
            f"    ylabel={self.ylabel}, yunit={self.yunit}, y={self.y.min()}..{self.y.max()},\n"
            f"    xnames={self.xnames},\n"
            f")"
        )

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def close(self) -> None:
        """
        Close the underlying HDF5 file.
        """
        if self._f is not None:
            try:
                self._f.close()
            finally:
                self._f = None

    def __enter__(self) -> "DatasetReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
