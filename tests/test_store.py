import os
import tempfile
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from specsimile.store import (
    _unit_to_string,
    _as_bytes_array,
    _as_str_list,
    DatasetWriter,
    DatasetReader,
)


# ---------------------------------------------------------------------------
# _unit_to_string
# ---------------------------------------------------------------------------

def test_unit_to_string_none():
    assert _unit_to_string(None) == ""


def test_unit_to_string_empty_string():
    assert _unit_to_string("") == ""


def test_unit_to_string_plain_string():
    assert _unit_to_string("km/s") == "km/s"


@given(st.text(min_size=1, max_size=40))
def test_unit_to_string_arbitrary_string(s):
    assert _unit_to_string(s) == s


# ---------------------------------------------------------------------------
# _as_bytes_array
# ---------------------------------------------------------------------------

def test_as_bytes_array_none():
    assert _as_bytes_array(None) is None


def test_as_bytes_array_strings():
    result = _as_bytes_array(["a", "bb", "ccc"])
    assert list(result) == [b"a", b"bb", b"ccc"]


def test_as_bytes_array_bytes_passthrough():
    result = _as_bytes_array([b"x", b"y"])
    assert list(result) == [b"x", b"y"]


def test_as_bytes_array_mixed():
    result = _as_bytes_array(["hello", b"world"])
    assert list(result) == [b"hello", b"world"]


def test_as_bytes_array_empty():
    result = _as_bytes_array([])
    assert len(result) == 0


@given(st.lists(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=20), max_size=10))
def test_as_bytes_array_roundtrip(strings):
    result = _as_bytes_array(strings)
    for original, encoded in zip(strings, result):
        assert encoded == original.encode("utf-8")


# ---------------------------------------------------------------------------
# _as_str_list
# ---------------------------------------------------------------------------

def test_as_str_list_none():
    assert _as_str_list(None) is None


def test_as_str_list_bytes():
    result = _as_str_list([b"foo", b"bar"])
    assert result == ["foo", "bar"]


def test_as_str_list_strings():
    result = _as_str_list(["foo", "bar"])
    assert result == ["foo", "bar"]


def test_as_str_list_empty():
    assert _as_str_list([]) == []


def test_as_str_list_numpy_bytes():
    arr = np.array([b"alpha", b"beta"])
    result = _as_str_list(arr)
    assert result == ["alpha", "beta"]


@given(st.lists(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=20), max_size=10))
def test_as_str_list_roundtrip(strings):
    encoded = [s.encode("utf-8") for s in strings]
    assert _as_str_list(encoded) == strings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tmp_path():
    fd, path = tempfile.mkstemp(suffix=".hdf5")
    os.close(fd)
    os.unlink(path)
    return path


def write_and_read(x, y, params, paramnames, **writer_kwargs):
    path = make_tmp_path()
    with DatasetWriter(path, N=len(y), x=x, paramnames=paramnames, **writer_kwargs) as w:
        w.set(x, y, params)
    return DatasetReader(path)


# ---------------------------------------------------------------------------
# DatasetWriter basic creation
# ---------------------------------------------------------------------------

def test_writer_creates_file():
    path = make_tmp_path()
    x = np.linspace(0, 1, 5)
    with DatasetWriter(path, N=2, x=x, paramnames=["a", "b"]) as w:
        y = np.ones((2, 5))
        p = np.ones((2, 2))
        w.set(x, y, p)
    assert os.path.exists(path)
    os.unlink(path)


def test_writer_x_must_be_1d():
    path = make_tmp_path()
    with pytest.raises(ValueError):
        DatasetWriter(path, N=2, x=np.ones((3, 3)), paramnames=["a"])


def test_writer_xnames_length_mismatch():
    path = make_tmp_path()
    x = np.linspace(0, 1, 4)
    with pytest.raises(ValueError):
        DatasetWriter(path, N=2, x=x, paramnames=["a"], xnames=["only_one"])


def test_writer_closed_raises():
    path = make_tmp_path()
    x = np.linspace(0, 1, 3)
    w = DatasetWriter(path, N=1, x=x, paramnames=["a"])
    y = np.ones((1, 3))
    p = np.ones((1, 1))
    w.set(x, y, p)
    with pytest.raises(RuntimeError):
        w.append(x, y[0], p[0])


# ---------------------------------------------------------------------------
# DatasetWriter.append
# ---------------------------------------------------------------------------

def test_append_single_row():
    path = make_tmp_path()
    x = np.array([1.0, 2.0, 3.0])
    with DatasetWriter(path, N=1, x=x, paramnames=["p1"]) as w:
        idx = w.append(x, np.array([10.0, 20.0, 30.0]), np.array([0.5]))
    assert idx == 0
    r = DatasetReader(path)
    np.testing.assert_allclose(r.y[0], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(r.params[0], [0.5])
    r.close()
    os.unlink(path)


def test_append_overflow_raises():
    path = make_tmp_path()
    x = np.array([1.0, 2.0])
    w = DatasetWriter(path, N=1, x=x, paramnames=["p"])
    w.append(x, np.array([1.0, 2.0]), np.array([0.1]))
    with pytest.raises(RuntimeError):
        w.append(x, np.array([1.0, 2.0]), np.array([0.2]))
    try:
        w.close()
    except Exception:
        pass
    if os.path.exists(path):
        os.unlink(path)


def test_append_wrong_x_raises():
    path = make_tmp_path()
    x = np.array([1.0, 2.0, 3.0])
    w = DatasetWriter(path, N=2, x=x, paramnames=["p"])
    with pytest.raises(ValueError):
        w.append(np.array([1.0, 2.0, 4.0]), np.array([1.0, 2.0, 3.0]), np.array([0.1]))
    w.close()
    if os.path.exists(path):
        os.unlink(path)


def test_append_wrong_y_length_raises():
    path = make_tmp_path()
    x = np.array([1.0, 2.0, 3.0])
    w = DatasetWriter(path, N=2, x=x, paramnames=["p"])
    with pytest.raises(ValueError):
        w.append(x, np.array([1.0, 2.0]), np.array([0.1]))
    w.close()
    if os.path.exists(path):
        os.unlink(path)


def test_append_wrong_params_length_raises():
    path = make_tmp_path()
    x = np.array([1.0, 2.0])
    w = DatasetWriter(path, N=2, x=x, paramnames=["p1", "p2"])
    with pytest.raises(ValueError):
        w.append(x, np.array([1.0, 2.0]), np.array([0.1]))
    w.close()
    if os.path.exists(path):
        os.unlink(path)


def test_append_increments_num_data_training():
    path = make_tmp_path()
    x = np.array([0.0, 1.0])
    with DatasetWriter(path, N=3, x=x, paramnames=["p"]) as w:
        w.append(x, np.array([1.0, 2.0]), np.array([0.1]))
        assert w.num_data_training == 1
        w.append(x, np.array([3.0, 4.0]), np.array([0.2]))
        assert w.num_data_training == 2
        w.append(x, np.array([5.0, 6.0]), np.array([0.3]))
    if os.path.exists(path):
        os.unlink(path)


# ---------------------------------------------------------------------------
# DatasetWriter.set
# ---------------------------------------------------------------------------

def test_set_all_rows():
    path = make_tmp_path()
    x = np.linspace(0, 1, 10)
    y = np.random.default_rng(0).random((5, 10))
    p = np.random.default_rng(1).random((5, 3))
    with DatasetWriter(path, N=5, x=x, paramnames=["a", "b", "c"]) as w:
        sl = w.set(x, y, p)
    assert sl == slice(0, 5)
    r = DatasetReader(path)
    np.testing.assert_allclose(r.y, y)
    np.testing.assert_allclose(r.params, p)
    r.close()
    os.unlink(path)


def test_set_overflow_raises():
    path = make_tmp_path()
    x = np.linspace(0, 1, 4)
    y = np.ones((3, 4))
    p = np.ones((3, 2))
    w = DatasetWriter(path, N=2, x=x, paramnames=["a", "b"])
    with pytest.raises(RuntimeError):
        w.set(x, y, p)
    w.close()
    if os.path.exists(path):
        os.unlink(path)


def test_set_y_params_row_mismatch_raises():
    path = make_tmp_path()
    x = np.linspace(0, 1, 4)
    y = np.ones((3, 4))
    p = np.ones((2, 2))
    w = DatasetWriter(path, N=3, x=x, paramnames=["a", "b"])
    with pytest.raises(ValueError):
        w.set(x, y, p)
    w.close()
    if os.path.exists(path):
        os.unlink(path)


def test_set_wrong_y_columns_raises():
    path = make_tmp_path()
    x = np.linspace(0, 1, 4)
    y = np.ones((2, 3))
    p = np.ones((2, 1))
    w = DatasetWriter(path, N=2, x=x, paramnames=["a"])
    with pytest.raises(ValueError):
        w.set(x, y, p)
    w.close()
    if os.path.exists(path):
        os.unlink(path)


def test_set_2d_x_raises():
    path = make_tmp_path()
    x = np.linspace(0, 1, 4)
    w = DatasetWriter(path, N=2, x=x, paramnames=["a"])
    with pytest.raises(ValueError):
        w.set(np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 1)))
    w.close()
    if os.path.exists(path):
        os.unlink(path)


# ---------------------------------------------------------------------------
# DatasetReader
# ---------------------------------------------------------------------------

def test_reader_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        DatasetReader("/nonexistent/path/file.hdf5")


def test_reader_x_y_params():
    path = make_tmp_path()
    x = np.array([0.5, 1.5, 2.5])
    y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    p = np.array([[0.1, 0.2], [0.3, 0.4]])
    with DatasetWriter(path, N=2, x=x, paramnames=["p1", "p2"]) as w:
        w.set(x, y, p)
    r = DatasetReader(path)
    np.testing.assert_allclose(r.x, x)
    np.testing.assert_allclose(r.y, y)
    np.testing.assert_allclose(r.params, p)
    assert r.paramnames == ["p1", "p2"]
    r.close()
    os.unlink(path)


def test_reader_len():
    path = make_tmp_path()
    x = np.linspace(0, 1, 5)
    y = np.ones((7, 5))
    p = np.ones((7, 1))
    with DatasetWriter(path, N=7, x=x, paramnames=["q"]) as w:
        w.set(x, y, p)
    r = DatasetReader(path)
    assert len(r) == 7
    r.close()
    os.unlink(path)


def test_reader_xnames_stored_and_read():
    path = make_tmp_path()
    x = np.array([1.0, 2.0, 3.0])
    xnames = ["alpha", "beta", "gamma"]
    with DatasetWriter(path, N=1, x=x, paramnames=["p"], xnames=xnames) as w:
        w.append(x, np.array([0.0, 0.0, 0.0]), np.array([1.0]))
    r = DatasetReader(path)
    assert r.xnames == xnames
    r.close()
    os.unlink(path)


def test_reader_xnames_absent_is_none():
    path = make_tmp_path()
    x = np.array([1.0, 2.0])
    with DatasetWriter(path, N=1, x=x, paramnames=["p"]) as w:
        w.append(x, np.array([0.0, 0.0]), np.array([1.0]))
    r = DatasetReader(path)
    assert r.xnames is None
    r.close()
    os.unlink(path)


def test_reader_labels_and_units():
    path = make_tmp_path()
    x = np.array([1.0, 2.0])
    with DatasetWriter(
        path, N=1, x=x, paramnames=["p"],
        xlabel="wavelength", ylabel="flux",
        xunit="nm", yunit="erg/s"
    ) as w:
        w.append(x, np.array([0.0, 0.0]), np.array([1.0]))
    r = DatasetReader(path)
    assert r.xlabel == "wavelength"
    assert r.ylabel == "flux"
    assert r.xunit == "nm"
    assert r.yunit == "erg/s"
    r.close()
    os.unlink(path)


def test_reader_context_manager():
    path = make_tmp_path()
    x = np.array([1.0])
    with DatasetWriter(path, N=1, x=x, paramnames=["p"]) as w:
        w.append(x, np.array([1.0]), np.array([0.5]))
    with DatasetReader(path) as r:
        assert r.x[0] == pytest.approx(1.0)
    os.unlink(path)


# ---------------------------------------------------------------------------
# Round-trip property tests via hypothesis
# ---------------------------------------------------------------------------

@given(
    n=st.integers(min_value=1, max_value=8),
    xlen=st.integers(min_value=1, max_value=10),
    plen=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=30, deadline=None)
def test_roundtrip_shapes(n, xlen, plen):
    path = make_tmp_path()
    rng = np.random.default_rng(42)
    x = np.sort(rng.random(xlen))
    y = rng.random((n, xlen))
    p = rng.random((n, plen))
    paramnames = [f"p{i}" for i in range(plen)]
    try:
        with DatasetWriter(path, N=n, x=x, paramnames=paramnames) as w:
            w.set(x, y, p)
        r = DatasetReader(path)
        assert r.y.shape == (n, xlen)
        assert r.params.shape == (n, plen)
        np.testing.assert_allclose(r.x, x)
        np.testing.assert_allclose(r.y, y)
        np.testing.assert_allclose(r.params, p)
        r.close()
    finally:
        if os.path.exists(path):
            os.unlink(path)


@given(
    paramnames=st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=10),
        min_size=1, max_size=6, unique=True,
    )
)
@settings(max_examples=20, deadline=None)
def test_roundtrip_paramnames(paramnames):
    path = make_tmp_path()
    x = np.linspace(0, 1, 4)
    n = 2
    y = np.zeros((n, 4))
    p = np.zeros((n, len(paramnames)))
    try:
        with DatasetWriter(path, N=n, x=x, paramnames=paramnames) as w:
            w.set(x, y, p)
        r = DatasetReader(path)
        assert r.paramnames == paramnames
        r.close()
    finally:
        if os.path.exists(path):
            os.unlink(path)


@given(
    values=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=15),
        elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
    )
)
@settings(max_examples=20, deadline=None)
def test_roundtrip_x_values(values):
    path = make_tmp_path()
    x = values
    y = np.zeros((1, len(x)))
    p = np.zeros((1, 1))
    try:
        with DatasetWriter(path, N=1, x=x, paramnames=["q"]) as w:
            w.set(x, y, p)
        r = DatasetReader(path)
        np.testing.assert_allclose(r.x, x)
        r.close()
    finally:
        if os.path.exists(path):
            os.unlink(path)
