# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines functions that import external libraries used by Uproot, but not
required by an Uproot installation. (Uproot only requires NumPy).

If a library cannot be imported, these functions raise ``ImportError`` with
error messages containing instructions on how to install the library.
"""

from __future__ import absolute_import

import os


def awkward1():
    """
    Imports and returns ``awkward1``.
    """
    try:
        import awkward1
    except ImportError:
        raise ImportError(
            """install the 'awkward1' package with:

    pip install awkward1"""
        )
    else:
        return awkward1


def pandas():
    """
    Imports and returns ``pandas``.
    """
    try:
        import pandas
    except ImportError:
        raise ImportError(
            """install the 'pandas' package with:

    pip install pandas

or

    conda install pandas"""
        )
    else:
        return pandas


def cupy():
    """
    Imports and returns ``cupy``.
    """
    try:
        import cupy
    except ImportError:
        raise ImportError(
            """install the 'cupy' package with:

    pip install cupy

or

    conda install cupy"""
        )
    else:
        return cupy


def XRootD_client():
    """
    Imports and returns ``XRootD.client`` (after setting the
    ```XRD_RUNFORKHANDLER`` environment variable to ``"1"``, to allow
    multiprocessing).
    """
    os.environ["XRD_RUNFORKHANDLER"] = "1"  # set multiprocessing flag
    try:
        import XRootD
        import XRootD.client

    except ImportError:
        raise ImportError(
            """Install XRootD python bindings with:

    conda install -c conda-forge xrootd

(or download from http://xrootd.org/dload.html and manually compile with """
            """cmake; setting PYTHONPATH and LD_LIBRARY_PATH appropriately)."""
        )

    import atexit

    # TODO: When fixed this should only be used for buggy XRootD versions
    # This is registered after calling "import XRootD.client" so it is ran
    # before XRootD.client.finalize.finalize()
    @atexit.register
    def cleanup_open_files():
        """Clean up any open xrootd file objects at exit

        Required to avoid deadlocks from XRootD, for details see:
        * https://github.com/scikit-hep/uproot/issues/504
        * https://github.com/xrootd/xrootd/pull/1260
        """
        import gc

        for obj in gc.get_objects():
            if isinstance(obj, XRootD.client.file.File) and obj.is_open():
                obj.close()

    return XRootD.client


def lzma():
    """
    Imports and returns ``lzma`` (which is part of the Python 3 standard
    library, but not Python 2).
    """
    try:
        import lzma
    except ImportError:
        try:
            import backports.lzma as lzma
        except ImportError:
            raise ImportError(
                """install the 'lzma' package with:

    pip install backports.lzma

or

    conda install backports.lzma

or use Python >= 3.3."""
            )
        else:
            return lzma
    else:
        return lzma


def lz4_block():
    """
    Imports and returns ``lz4``.

    Attempts to import ``xxhash`` as well.
    """
    try:
        import lz4.block
        import xxhash  # noqa: F401
    except ImportError:
        raise ImportError(
            """install the 'lz4' and `xxhash` packages with:

    pip install lz4 xxhash

or

    conda install lz4 python-xxhash"""
        )
    else:
        return lz4.block


def xxhash():
    """
    Imports and returns ``xxhash``.

    Attempts to import ``lz4`` as well.
    """
    try:
        import xxhash
        import lz4.block  # noqa: F401
    except ImportError:
        raise ImportError(
            """install the 'lz4' and `xxhash` packages with:

    pip install lz4 xxhash

or

    conda install lz4 python-xxhash"""
        )
    else:
        return xxhash


def zstandard():
    """
    Imports and returns ``zstandard``.
    """
    try:
        import zstandard
    except ImportError:
        raise ImportError(
            """install the 'zstandard' package with:

    pip install zstandard

or

    conda install zstandard"""
        )
    else:
        return zstandard


def boost_histogram():
    """
    Imports and returns ``boost-histogram``.
    """
    try:
        import boost_histogram
    except ImportError:
        raise ImportError(
            """install the 'boost-histogram' package with:

    pip install boost-histogram

or

    conda install -c conda-forge boost-histogram"""
        )
    else:
        return boost_histogram


def hist():
    """
    Imports and returns ``hist``.
    """
    try:
        import hist
    except ImportError:
        raise ImportError(
            """install the 'hist' package with:

    pip install hist"""
        )
    else:
        return hist
