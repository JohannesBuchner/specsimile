import os
import tempfile
import numpy as np
import pytest
from hypothesis import given, settings, assume
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
    for s in strings:
        if not isinstance(s, str) or '\x00' in s:
            assume(False)
        if len(s) == 0:
            assume(False)
        try:
            s.encode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            assume(False)
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


def test_append_single_nan_row():
    path = make_tmp_path()
    x = np.array([1.0, 2.0, 3.0])
    with DatasetWriter(path, N=1, x=x, paramnames=["p1"]) as w:
        with pytest.raises(ValueError):
            w.append(x, np.array([np.nan, 20.0, 30.0]), np.array([0.5]))
        with pytest.raises(ValueError):
            w.append(x, np.array([10, 20.0, 30.0]), np.array([np.nan]))
        with pytest.raises(ValueError):
            w.append(x * np.nan, np.array([10, 20.0, 30.0]), np.array([0.5]))
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


# ---------------------------------------------------------------------------
# append_chunk equivalence to append
# ---------------------------------------------------------------------------

def test_append_chunk_single_row_equivalent_to_append():
    """append_chunk with one row should produce identical results to append."""
    x = np.array([1.0, 2.0, 3.0])
    y_row = np.array([10.0, 20.0, 30.0])
    p_row = np.array([0.5, 1.5])

    path_append = make_tmp_path()
    path_chunk = make_tmp_path()

    try:
        with DatasetWriter(path_append, N=1, x=x, paramnames=["p1", "p2"]) as w:
            w.append(x, y_row, p_row)

        with DatasetWriter(path_chunk, N=1, x=x, paramnames=["p1", "p2"]) as w:
            w.append_chunk(x, y_row[np.newaxis, :], p_row[np.newaxis, :])

        r_append = DatasetReader(path_append)
        r_chunk = DatasetReader(path_chunk)
        np.testing.assert_allclose(r_chunk.y, r_append.y)
        np.testing.assert_allclose(r_chunk.params, r_append.params)
        np.testing.assert_allclose(r_chunk.x, r_append.x)
        r_append.close()
        r_chunk.close()
    finally:
        for p in (path_append, path_chunk):
            if os.path.exists(p):
                os.unlink(p)


def test_append_chunk_multiple_rows_equivalent_to_append():
    """append_chunk with multiple rows should match row-by-row append."""
    rng = np.random.default_rng(7)
    x = np.linspace(0, 1, 8)
    n = 5
    plen = 3
    y = rng.random((n, len(x)))
    params = rng.random((n, plen))
    paramnames = ["a", "b", "c"]

    path_append = make_tmp_path()
    path_chunk = make_tmp_path()

    try:
        with DatasetWriter(path_append, N=n, x=x, paramnames=paramnames) as w:
            for i in range(n):
                w.append(x, y[i], params[i])

        with DatasetWriter(path_chunk, N=n, x=x, paramnames=paramnames) as w:
            w.append_chunk(x, y, params)

        r_append = DatasetReader(path_append)
        r_chunk = DatasetReader(path_chunk)
        np.testing.assert_allclose(r_chunk.y, r_append.y)
        np.testing.assert_allclose(r_chunk.params, r_append.params)
        assert len(r_chunk) == len(r_append) == n
        r_append.close()
        r_chunk.close()
    finally:
        for p in (path_append, path_chunk):
            if os.path.exists(p):
                os.unlink(p)


def test_append_chunk_returns_correct_slice():
    """append_chunk should return the correct slice of written rows."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.ones((4, 3))
    params = np.zeros((4, 2))

    path = make_tmp_path()
    try:
        with DatasetWriter(path, N=4, x=x, paramnames=["p1", "p2"]) as w:
            sl = w.append_chunk(x, y, params)
        assert sl == slice(0, 4)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_append_chunk_incremental_slices():
    """Multiple append_chunk calls should return consecutive slices."""
    x = np.array([1.0, 2.0])
    y1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    p1 = np.array([[0.1, 0.2], [0.3, 0.4]])
    y2 = np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    p2 = np.array([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]])

    path = make_tmp_path()
    try:
        with DatasetWriter(path, N=5, x=x, paramnames=["p1", "p2"]) as w:
            sl1 = w.append_chunk(x, y1, p1)
            sl2 = w.append_chunk(x, y2, p2)
        assert sl1 == slice(0, 2)
        assert sl2 == slice(2, 5)

        r = DatasetReader(path)
        np.testing.assert_allclose(r.y[:2], y1)
        np.testing.assert_allclose(r.y[2:], y2)
        np.testing.assert_allclose(r.params[:2], p1)
        np.testing.assert_allclose(r.params[2:], p2)
        r.close()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_append_chunk_and_append_interleaved():
    """Interleaving append_chunk and append should produce correct results."""
    x = np.array([1.0, 2.0, 3.0])
    paramnames = ["q"]

    y_chunk = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    p_chunk = np.array([[0.1], [0.2]])
    y_single = np.array([7.0, 8.0, 9.0])
    p_single = np.array([0.3])

    path_interleaved = make_tmp_path()
    path_append_only = make_tmp_path()

    try:
        with DatasetWriter(path_interleaved, N=3, x=x, paramnames=paramnames) as w:
            w.append_chunk(x, y_chunk, p_chunk)
            w.append(x, y_single, p_single)

        with DatasetWriter(path_append_only, N=3, x=x, paramnames=paramnames) as w:
            w.append(x, y_chunk[0], p_chunk[0])
            w.append(x, y_chunk[1], p_chunk[1])
            w.append(x, y_single, p_single)

        r_i = DatasetReader(path_interleaved)
        r_a = DatasetReader(path_append_only)
        np.testing.assert_allclose(r_i.y, r_a.y)
        np.testing.assert_allclose(r_i.params, r_a.params)
        r_i.close()
        r_a.close()
    finally:
        for p in (path_interleaved, path_append_only):
            if os.path.exists(p):
                os.unlink(p)


def test_append_chunk_num_data_training_updated():
    """num_data_training should be correctly updated after append_chunk."""
    x = np.array([0.0, 1.0])
    y = np.ones((3, 2))
    params = np.zeros((3, 1))

    path = make_tmp_path()
    try:
        w = DatasetWriter(path, N=5, x=x, paramnames=["p"])
        assert w.num_data_training == 0
        w.append_chunk(x, y, params)
        assert w.num_data_training == 3
        w.append_chunk(x, y[:2], params[:2])
        assert w.num_data_training == 5
        w.close()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_append_chunk_overflow_raises():
    """append_chunk should raise RuntimeError when chunk exceeds capacity."""
    x = np.array([1.0, 2.0])
    y = np.ones((5, 2))
    params = np.zeros((5, 1))

    path = make_tmp_path()
    try:
        w = DatasetWriter(path, N=3, x=x, paramnames=["p"])
        with pytest.raises(RuntimeError):
            w.append_chunk(x, y, params)
        w.close()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_append_chunk_wrong_x_raises():
    """append_chunk should raise ValueError when x does not match stored grid."""
    x = np.array([1.0, 2.0, 3.0])
    wrong_x = np.array([1.0, 2.0, 4.0])
    y = np.ones((2, 3))
    params = np.zeros((2, 1))

    path = make_tmp_path()
    try:
        w = DatasetWriter(path, N=2, x=x, paramnames=["p"])
        with pytest.raises(ValueError):
            w.append_chunk(wrong_x, y, params)
        w.close()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_append_chunk_wrong_y_columns_raises():
    """append_chunk should raise ValueError when y has wrong number of columns."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.ones((2, 2))  # should be (2, 3)
    params = np.zeros((2, 1))

    path = make_tmp_path()
    try:
        w = DatasetWriter(path, N=2, x=x, paramnames=["p"])
        with pytest.raises(ValueError):
            w.append_chunk(x, y, params)
        w.close()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_append_chunk_wrong_params_columns_raises():
    """append_chunk should raise ValueError when params has wrong number of columns."""
    x = np.array([1.0, 2.0])
    y = np.ones((2, 2))
    params = np.zeros((2, 3))  # should be (2, 1)

    path = make_tmp_path()
    try:
        w = DatasetWriter(path, N=2, x=x, paramnames=["p"])
        with pytest.raises(ValueError):
            w.append_chunk(x, y, params)
        w.close()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_append_chunk_nan_y_raises():
    """append_chunk should raise ValueError when y contains NaN."""
    x = np.array([1.0, 2.0])
    y = np.array([[1.0, np.nan], [2.0, 3.0]])
    params = np.zeros((2, 1))

    path = make_tmp_path()
    try:
        w = DatasetWriter(path, N=2, x=x, paramnames=["p"])
        with pytest.raises(ValueError):
            w.append_chunk(x, y, params)
        w.close()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_append_chunk_nan_params_raises():
    """append_chunk should raise ValueError when params contains NaN."""
    x = np.array([1.0, 2.0])
    y = np.ones((2, 2))
    params = np.array([[np.nan], [0.5]])

    path = make_tmp_path()
    try:
        w = DatasetWriter(path, N=2, x=x, paramnames=["p"])
        with pytest.raises(ValueError):
            w.append_chunk(x, y, params)
        w.close()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_append_chunk_row_mismatch_raises():
    """append_chunk should raise ValueError when y and params have different row counts."""
    x = np.array([1.0, 2.0])
    y = np.ones((3, 2))
    params = np.zeros((2, 1))

    path = make_tmp_path()
    try:
        w = DatasetWriter(path, N=3, x=x, paramnames=["p"])
        with pytest.raises(ValueError):
            w.append_chunk(x, y, params)
        w.close()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_append_chunk_1d_y_raises():
    """append_chunk should raise ValueError when y is 1D instead of 2D."""
    x = np.array([1.0, 2.0])
    y = np.ones(2)  # 1D, should be 2D
    params = np.zeros((1, 1))

    path = make_tmp_path()
    try:
        w = DatasetWriter(path, N=1, x=x, paramnames=["p"])
        with pytest.raises(ValueError):
            w.append_chunk(x, y, params)
        w.close()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_append_chunk_1d_params_raises():
    """append_chunk should raise ValueError when params is 1D instead of 2D."""
    x = np.array([1.0, 2.0])
    y = np.ones((1, 2))
    params = np.zeros(1)  # 1D, should be 2D

    path = make_tmp_path()
    try:
        w = DatasetWriter(path, N=1, x=x, paramnames=["p"])
        with pytest.raises(ValueError):
            w.append_chunk(x, y, params)
        w.close()
    finally:
        if os.path.exists(path):
            os.unlink(path)


@given(
    n=st.integers(min_value=1, max_value=10),
    xlen=st.integers(min_value=1, max_value=8),
    plen=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=30, deadline=None)
def test_append_chunk_equivalent_to_append_hypothesis(n, xlen, plen):
    """Property: append_chunk and row-by-row append always produce identical files."""
    rng = np.random.default_rng(n * 1000 + xlen * 100 + plen)
    x = np.sort(rng.random(xlen))
    y = rng.random((n, xlen))
    params = rng.random((n, plen))
    paramnames = [f"p{i}" for i in range(plen)]

    path_append = make_tmp_path()
    path_chunk = make_tmp_path()

    try:
        with DatasetWriter(path_append, N=n, x=x, paramnames=paramnames) as w:
            for i in range(n):
                w.append(x, y[i], params[i])

        with DatasetWriter(path_chunk, N=n, x=x, paramnames=paramnames) as w:
            w.append_chunk(x, y, params)

        r_append = DatasetReader(path_append)
        r_chunk = DatasetReader(path_chunk)

        np.testing.assert_allclose(r_chunk.x, r_append.x)
        np.testing.assert_allclose(r_chunk.y, r_append.y)
        np.testing.assert_allclose(r_chunk.params, r_append.params)
        assert len(r_chunk) == len(r_append) == n

        r_append.close()
        r_chunk.close()
    finally:
        for p in (path_append, path_chunk):
            if os.path.exists(p):
                os.unlink(p)


@given(
    chunk_sizes=st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=5),
    xlen=st.integers(min_value=1, max_value=6),
    plen=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=20, deadline=None)
def test_append_chunk_multiple_chunks_equivalent_to_append_hypothesis(chunk_sizes, xlen, plen):
    """Property: multiple append_chunk calls equal the same rows written with append."""
    rng = np.random.default_rng(sum(chunk_sizes) + xlen * 17 + plen * 31)
    n = sum(chunk_sizes)
    x = np.sort(rng.random(xlen))
    y = rng.random((n, xlen))
    params = rng.random((n, plen))
    paramnames = [f"q{i}" for i in range(plen)]

    path_append = make_tmp_path()
    path_chunk = make_tmp_path()

    try:
        with DatasetWriter(path_append, N=n, x=x, paramnames=paramnames) as w:
            for i in range(n):
                w.append(x, y[i], params[i])

        with DatasetWriter(path_chunk, N=n, x=x, paramnames=paramnames) as w:
            offset = 0
            for size in chunk_sizes:
                w.append_chunk(x, y[offset:offset + size], params[offset:offset + size])
                offset += size

        r_append = DatasetReader(path_append)
        r_chunk = DatasetReader(path_chunk)

        np.testing.assert_allclose(r_chunk.x, r_append.x)
        np.testing.assert_allclose(r_chunk.y, r_append.y)
        np.testing.assert_allclose(r_chunk.params, r_append.params)
        assert len(r_chunk) == len(r_append) == n

        r_append.close()
        r_chunk.close()
    finally:
        for p in (path_append, path_chunk):
            if os.path.exists(p):
                os.unlink(p)
