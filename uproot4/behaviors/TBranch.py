# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines behaviors for :py:class:`~uproot4.behaviors.TBranch.TBranch` and
:py:class:`~uproot4.behaviors.TBranch.HasBranches` (both ``TBranch`` and
``TTree``).

Most of the functionality of TTree-reading is implemented here.

See :py:class:`~uproot4.models.TBranch` for deserialization of the ``TBranch``
objects themselves.
"""

from __future__ import absolute_import

import os
import glob
import sys
import re
import threading
import itertools

try:
    from collections.abc import Mapping
    from collections.abc import MutableMapping
    from collections.abc import Iterable
except ImportError:
    from collections import Mapping
    from collections import MutableMapping
    from collections import Iterable
try:
    import queue
except ImportError:
    import Queue as queue
try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse

import numpy

import uproot4.cache
import uproot4.source.cursor
import uproot4.streamers
import uproot4.containers
import uproot4.interpretation
import uproot4.interpretation.numerical
import uproot4.interpretation.jagged
import uproot4.interpretation.library
import uproot4.interpretation.identify
import uproot4.reading
import uproot4.language.python
import uproot4.models.TBasket
import uproot4.models.TObjArray
import uproot4._util
from uproot4._util import no_filter


class _NoClose(object):
    def __init__(self, hasbranches):
        self.hasbranches = hasbranches

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def __getattr__(self, attr):
        return getattr(self.hasbranches, attr)

    def __getitem__(self, where):
        return self.hasbranches[where]


def iterate(
    files,
    expressions=None,
    cut=None,
    filter_name=no_filter,
    filter_typename=no_filter,
    filter_branch=no_filter,
    aliases=None,
    language=uproot4.language.python.PythonLanguage(),
    step_size="100 MB",
    decompression_executor=None,
    interpretation_executor=None,
    library="ak",
    how=None,
    report=False,
    custom_classes=None,
    allow_missing=False,
    **options  # NOTE: a comma after **options breaks Python 2
):
    u"""
    Args:
        files: See below.
        expressions (None, str, or list of str): Names of ``TBranches`` or
            aliases to convert to arrays or mathematical expressions of them.
            Uses the ``language`` to evaluate. If None, all ``TBranches``
            selected by the filters are included.
        cut (None or str): If not None, this expression filters all of the
            ``expressions``.
        filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
            filter to select ``TBranches`` by name.
        filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
            filter to select ``TBranches`` by type.
        filter_branch (None or function of :py:class:`~uproot4.behaviors.TBranch.TBranch` \u2192 bool, :py:class:`~uproot4.interpretation.Interpretation`, or None): A
            filter to select ``TBranches`` using the full
            :py:class:`~uproot4.behaviors.TBranch.TBranch` object. If the function
            returns False or None, the ``TBranch`` is excluded; if the function
            returns True, it is included with its standard
            :py:class:`~uproot4.behaviors.TBranch.TBranch.interpretation`; if an
            :py:class:`~uproot4.interpretation.Interpretation`, this interpretation
            overrules the standard one.
        aliases (None or dict of str \u2192 str): Mathematical expressions that
            can be used in ``expressions`` or other aliases (without cycles).
            Uses the ``language`` engine to evaluate. If None, only the
            :py:attr:`~uproot4.behaviors.TBranch.TBranch.aliases` are available.
        language (:py:class:`~uproot4.language.Language`): Language used to interpret
            the ``expressions`` and ``aliases``.
        step_size (int or str): If an integer, the maximum number of entries to
            include in each iteration step; if a string, the maximum memory size
            to include. The string must be a number followed by a memory unit,
            such as "100 MB".
        decompression_executor (None or Executor with a ``submit`` method): The
            executor that is used to decompress ``TBaskets``; if None, the
            global ``uproot4.decompression_executor`` is used.
        interpretation_executor (None or Executor with a ``submit`` method): The
            executor that is used to interpret uncompressed ``TBasket`` data as
            arrays; if None, the global ``uproot4.interpretation_executor`` is
            used.
        library (str or :py:class:`~uproot4.interpretation.library.Library`): The library
            that is used to represent arrays. Options are ``"np"`` for NumPy,
            ``"ak"`` for Awkward Array, ``"pd"`` for Pandas, and ``"cp"`` for
            CuPy.
        how (None, str, or container type): Library-dependent instructions
            for grouping. The only recognized container types are ``tuple``,
            ``list``, and ``dict``. Note that the container *type itself*
            must be passed as ``how``, not an instance of that type (i.e.
            ``how=tuple``, not ``how=()``).
        report (bool): If True, this generator yields
            (arrays, :py:class:`~uproot4.behaviors.TBranch.Report`) pairs; if False,
            it only yields arrays. The report has data about the ``TFile``,
            ``TTree``, and global and local entry ranges.
        custom_classes (None or dict): If a dict, override the classes from
            the :py:class:`~uproot4.reading.ReadOnlyFile` or ``uproot4.classes``.
        allow_missing (bool): If True, skip over any files that do not contain
            the specified ``TTree``.
        options: See below.

    Iterates through contiguous chunks of entries from a set of files.

    For example:

    .. code-block:: python

        >>> for array in uproot4.iterate("files*.root:tree", ["x", "y"], step_size=100):
        ...     # each of the following have 100 entries
        ...     array["x"], array["y"]

    Allowed types for the ``files`` parameter:

    * str/bytes: relative or absolute filesystem path or URL, without any colons
      other than Windows drive letter or URL schema.
      Examples: ``"rel/file.root"``, ``"C:\\abs\\file.root"``, ``"http://where/what.root"``
    * str/bytes: same with an object-within-ROOT path, separated by a colon.
      Example: ``"rel/file.root:tdirectory/ttree"``
    * pathlib.Path: always interpreted as a filesystem path or URL only (no
      object-within-ROOT path), regardless of whether there are any colons.
      Examples: ``Path("rel:/file.root")``, ``Path("/abs/path:stuff.root")``
    * glob syntax in str/bytes and pathlib.Path.
      Examples: ``Path("rel/*.root")``, ``"/abs/*.root:tdirectory/ttree"``
    * dict: keys are filesystem paths, values are objects-within-ROOT paths.
      Example: ``{{"/data_v1/*.root": "ttree_v1", "/data_v2/*.root": "ttree_v2"}}``
    * already-open TTree objects.
    * iterables of the above.

    Options (type; default):

    * file_handler (:py:class:`~uproot4.source.chunk.Source` class; :py:class:`~uproot4.source.file.MemmapSource`)
    * xrootd_handler (:py:class:`~uproot4.source.chunk.Source` class; :py:class:`~uproot4.source.xrootd.XRootDSource`)
    * http_handler (:py:class:`~uproot4.source.chunk.Source` class; :py:class:`~uproot4.source.http.HTTPSource`)
    * object_handler (:py:class:`~uproot4.source.chunk.Source` class; :py:class:`~uproot4.source.object.ObjectSource`)
    * timeout (float for HTTP, int for XRootD; 30)
    * max_num_elements (None or int; None)
    * num_workers (int; 1)
    * num_fallback_workers (int; 10)
    * begin_chunk_size (memory_size; 512)
    * minimal_ttree_metadata (bool; True)

    See also :py:meth:`~uproot4.behavior.TBranch.HasBranches.iterate` to iterate
    within a single file.

    Other file entry points:

    * :py:func:`~uproot4.reading.open`: opens one file to read any of its objects.
    * :py:func:`~uproot4.behaviors.TBranch.iterate` (this function): iterates through
      chunks of contiguous entries in ``TTrees``.
    * :py:func:`~uproot4.behaviors.TBranch.concatenate`: returns a single concatenated
      array from ``TTrees``.
    * :py:func:`~uproot4.behaviors.TBranch.lazy`: returns a lazily read array from
      ``TTrees``.
    """
    files = _regularize_files(files)
    decompression_executor, interpretation_executor = _regularize_executors(
        decompression_executor, interpretation_executor
    )
    library = uproot4.interpretation.library._regularize_library(library)

    global_offset = 0
    for file_path, object_path in files:
        print(file_path, global_offset)

        hasbranches = _regularize_object_path(
            file_path, object_path, custom_classes, allow_missing, options
        )

        if hasbranches is not None:
            with hasbranches:
                for item in hasbranches.iterate(
                    expressions=expressions,
                    cut=cut,
                    filter_name=filter_name,
                    filter_typename=filter_typename,
                    filter_branch=filter_branch,
                    aliases=aliases,
                    language=language,
                    step_size=step_size,
                    decompression_executor=decompression_executor,
                    interpretation_executor=interpretation_executor,
                    library=library,
                    how=how,
                    report=report,
                ):
                    if report:
                        arrays, report = item
                        arrays = library.global_index(arrays, global_offset)
                        report = report.to_global(global_offset)
                        yield arrays, report
                    else:
                        arrays = library.global_index(item, global_offset)
                        yield arrays

                global_offset += hasbranches.num_entries


def concatenate(
    files,
    expressions=None,
    cut=None,
    filter_name=no_filter,
    filter_typename=no_filter,
    filter_branch=no_filter,
    aliases=None,
    language=uproot4.language.python.PythonLanguage(),
    decompression_executor=None,
    interpretation_executor=None,
    library="ak",
    how=None,
    custom_classes=None,
    allow_missing=False,
    **options  # NOTE: a comma after **options breaks Python 2
):
    u"""
    Args:
        files: See below.
        expressions (None, str, or list of str): Names of ``TBranches`` or
            aliases to convert to arrays or mathematical expressions of them.
            Uses the ``language`` to evaluate. If None, all ``TBranches``
            selected by the filters are included.
        cut (None or str): If not None, this expression filters all of the
            ``expressions``.
        filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
            filter to select ``TBranches`` by name.
        filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
            filter to select ``TBranches`` by type.
        filter_branch (None or function of :py:class:`~uproot4.behaviors.TBranch.TBranch` \u2192 bool, :py:class:`~uproot4.interpretation.Interpretation`, or None): A
            filter to select ``TBranches`` using the full
            :py:class:`~uproot4.behaviors.TBranch.TBranch` object. If the function
            returns False or None, the ``TBranch`` is excluded; if the function
            returns True, it is included with its standard
            :py:attr:`~uproot4.behaviors.TBranch.TBranch.interpretation`; if an
            :py:class:`~uproot4.interpretation.Interpretation`, this interpretation
            overrules the standard one.
        aliases (None or dict of str \u2192 str): Mathematical expressions that
            can be used in ``expressions`` or other aliases (without cycles).
            Uses the ``language`` engine to evaluate. If None, only the
            :py:attr:`~uproot4.behaviors.TBranch.TBranch.aliases` are available.
        language (:py:class:`~uproot4.language.Language`): Language used to interpret
            the ``expressions`` and ``aliases``.
        decompression_executor (None or Executor with a ``submit`` method): The
            executor that is used to decompress ``TBaskets``; if None, the
            global ``uproot4.decompression_executor`` is used.
        interpretation_executor (None or Executor with a ``submit`` method): The
            executor that is used to interpret uncompressed ``TBasket`` data as
            arrays; if None, the global ``uproot4.interpretation_executor`` is
            used.
        library (str or :py:class:`~uproot4.interpretation.library.Library`): The library
            that is used to represent arrays. Options are ``"np"`` for NumPy,
            ``"ak"`` for Awkward Array, ``"pd"`` for Pandas, and ``"cp"`` for
            CuPy.
        how (None, str, or container type): Library-dependent instructions
            for grouping. The only recognized container types are ``tuple``,
            ``list``, and ``dict``. Note that the container *type itself*
            must be passed as ``how``, not an instance of that type (i.e.
            ``how=tuple``, not ``how=()``).
        custom_classes (None or dict): If a dict, override the classes from
            the :py:class:`~uproot4.reading.ReadOnlyFile` or ``uproot4.classes``.
        allow_missing (bool): If True, skip over any files that do not contain
            the specified ``TTree``.
        options: See below.

    Returns an array with data from a set of files concatenated into one.

    For example:

    .. code-block:: python

        >>> array = uproot4.concatenate("files*.root:tree", ["x", "y"])

    Depending on the number of files, the number of selected ``TBranches``, and
    the size of your computer's memory, this function might not have enough
    memory to run.

    Allowed types for the ``files`` parameter:

    * str/bytes: relative or absolute filesystem path or URL, without any colons
      other than Windows drive letter or URL schema.
      Examples: ``"rel/file.root"``, ``"C:\\abs\\file.root"``, ``"http://where/what.root"``
    * str/bytes: same with an object-within-ROOT path, separated by a colon.
      Example: ``"rel/file.root:tdirectory/ttree"``
    * pathlib.Path: always interpreted as a filesystem path or URL only (no
      object-within-ROOT path), regardless of whether there are any colons.
      Examples: ``Path("rel:/file.root")``, ``Path("/abs/path:stuff.root")``
    * glob syntax in str/bytes and pathlib.Path.
      Examples: ``Path("rel/*.root")``, ``"/abs/*.root:tdirectory/ttree"``
    * dict: keys are filesystem paths, values are objects-within-ROOT paths.
      Example: ``{{"/data_v1/*.root": "ttree_v1", "/data_v2/*.root": "ttree_v2"}}``
    * already-open TTree objects.
    * iterables of the above.

    Options (type; default):

    * file_handler (:py:class:`~uproot4.source.chunk.Source` class; :py:class:`~uproot4.source.file.MemmapSource`)
    * xrootd_handler (:py:class:`~uproot4.source.chunk.Source` class; :py:class:`~uproot4.source.xrootd.XRootDSource`)
    * http_handler (:py:class:`~uproot4.source.chunk.Source` class; :py:class:`~uproot4.source.http.HTTPSource`)
    * object_handler (:py:class:`~uproot4.source.chunk.Source` class; :py:class:`~uproot4.source.object.ObjectSource`)
    * timeout (float for HTTP, int for XRootD; 30)
    * max_num_elements (None or int; None)
    * num_workers (int; 1)
    * num_fallback_workers (int; 10)
    * begin_chunk_size (memory_size; 512)
    * minimal_ttree_metadata (bool; True)

    Other file entry points:

    * :py:func:`~uproot4.reading.open`: opens one file to read any of its objects.
    * :py:func:`~uproot4.behaviors.TBranch.iterate`: iterates through chunks of
      contiguous entries in ``TTrees``.
    * :py:func:`~uproot4.behaviors.TBranch.concatenate` (this function): returns a
      single concatenated array from ``TTrees``.
    * :py:func:`~uproot4.behaviors.TBranch.lazy`: returns a lazily read array from
      ``TTrees``.
    """
    files = _regularize_files(files)
    decompression_executor, interpretation_executor = _regularize_executors(
        decompression_executor, interpretation_executor
    )
    library = uproot4.interpretation.library._regularize_library(library)

    all_arrays = []
    global_start = 0
    for file_path, object_path in files:
        hasbranches = _regularize_object_path(
            file_path, object_path, custom_classes, allow_missing, options
        )

        if hasbranches is not None:
            with hasbranches:
                arrays = hasbranches.arrays(
                    expressions=expressions,
                    cut=cut,
                    filter_name=filter_name,
                    filter_typename=filter_typename,
                    filter_branch=filter_branch,
                    aliases=aliases,
                    language=language,
                    decompression_executor=decompression_executor,
                    interpretation_executor=interpretation_executor,
                    array_cache=None,
                    library=library,
                    how=how,
                )
                arrays = library.global_index(arrays, global_start)
                all_arrays.append(arrays)

                global_start += hasbranches.num_entries

    return library.concatenate(all_arrays)


def lazy(
    files,
    filter_name=no_filter,
    filter_typename=no_filter,
    filter_branch=no_filter,
    recursive=True,
    full_paths=False,
    step_size="100 MB",
    decompression_executor=None,
    interpretation_executor=None,
    array_cache="100 MB",
    library="ak",
    custom_classes=None,
    allow_missing=False,
    **options  # NOTE: a comma after **options breaks Python 2
):
    u"""
    Args:
        files: See below.
        filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
            filter to select ``TBranches`` by name.
        filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
            filter to select ``TBranches`` by type.
        filter_branch (None or function of :py:class:`~uproot4.behaviors.TBranch.TBranch` \u2192 bool, :py:class:`~uproot4.interpretation.Interpretation`, or None): A
            filter to select ``TBranches`` using the full
            :py:class:`~uproot4.behaviors.TBranch.TBranch` object. If the function
            returns False or None, the ``TBranch`` is excluded; if the function
            returns True, it is included with its standard
            :py:attr:`~uproot4.behaviors.TBranch.TBranch.interpretation`; if an
            :py:class:`~uproot4.interpretation.Interpretation`, this interpretation
            overrules the standard one.
        recursive (bool): If True, include all subbranches of branches as
            separate fields; otherwise, only search one level deep.
        full_paths (bool): If True, include the full path to each subbranch
            with slashes (``/``); otherwise, use the descendant's name as
            the field name.
        decompression_executor (None or Executor with a ``submit`` method): The
            executor that is used to decompress ``TBaskets``; if None, the
            global ``uproot4.decompression_executor`` is used.
        interpretation_executor (None or Executor with a ``submit`` method): The
            executor that is used to interpret uncompressed ``TBasket`` data as
            arrays; if None, the global ``uproot4.interpretation_executor`` is
            used.
        array_cache (None, MutableMapping, or memory size): Cache of arrays;
            if None, do not use a cache; if a memory size, create a new cache
            of this size.
        library (str or :py:class:`~uproot4.interpretation.library.Library`): The library
            that is used to represent arrays. For lazy arrays, only ``"ak"``
            for Awkward Array is allowed.
        custom_classes (None or dict): If a dict, override the classes from
            the :py:class:`~uproot4.reading.ReadOnlyFile` or ``uproot4.classes``.
        allow_missing (bool): If True, skip over any files that do not contain
            the specified ``TTree``.
        options: See below.

    Returns a lazy array, which loads data on demand. Only the files and
    ``TTree`` metadata are opened when this function is invoked.

    For example:

    .. code-block:: python

        >>> lazyarray = uproot4.lazy("files*.root:tree")
        >>> # all files*.root have been opened and the "tree" from each is examined

        >>> lazyarray
        <Array [{Type: 'GT', Run: 148031, ... M: 96.7}] type='23047295 * {"Type": string...'>
        >>> # the first TBasket of "Type" and "Run" have now been read, as well as the last
        >>> # TBasket of "M", to print these values on the screen.

        >>> np.sqrt(lazyarray.px1**2 + lazyarray.py1**2)
        <Array [44.7, 38.8, 38.8, ... 32.4, 32.4, 32.5] type='23047295 * float64'>
        >>> # the entirety of "px1" and "py1" have now been read, to perform a calculation.

    If the size of the fields used in a calculation do not fit into ``array_cache``,
    lazy arrays may be inefficient, repeatedly rereading data that could be read
    once by iterating through the calculation with
    :py:func:`~uproot4.behavior.TBranch.iterate`.

    Allowed types for the ``files`` parameter:

    * str/bytes: relative or absolute filesystem path or URL, without any colons
      other than Windows drive letter or URL schema.
      Examples: ``"rel/file.root"``, ``"C:\\abs\\file.root"``, ``"http://where/what.root"``
    * str/bytes: same with an object-within-ROOT path, separated by a colon.
      Example: ``"rel/file.root:tdirectory/ttree"``
    * pathlib.Path: always interpreted as a filesystem path or URL only (no
      object-within-ROOT path), regardless of whether there are any colons.
      Examples: ``Path("rel:/file.root")``, ``Path("/abs/path:stuff.root")``
    * glob syntax in str/bytes and pathlib.Path.
      Examples: ``Path("rel/*.root")``, ``"/abs/*.root:tdirectory/ttree"``
    * dict: keys are filesystem paths, values are objects-within-ROOT paths.
      Example: ``{{"/data_v1/*.root": "ttree_v1", "/data_v2/*.root": "ttree_v2"}}``
    * already-open TTree objects.
    * iterables of the above.

    Options (type; default):

    * file_handler (:py:class:`~uproot4.source.chunk.Source` class; :py:class:`~uproot4.source.file.MemmapSource`)
    * xrootd_handler (:py:class:`~uproot4.source.chunk.Source` class; :py:class:`~uproot4.source.xrootd.XRootDSource`)
    * http_handler (:py:class:`~uproot4.source.chunk.Source` class; :py:class:`~uproot4.source.http.HTTPSource`)
    * object_handler (:py:class:`~uproot4.source.chunk.Source` class; :py:class:`~uproot4.source.object.ObjectSource`)
    * timeout (float for HTTP, int for XRootD; 30)
    * max_num_elements (None or int; None)
    * num_workers (int; 1)
    * num_fallback_workers (int; 10)
    * begin_chunk_size (memory_size; 512)
    * minimal_ttree_metadata (bool; True)

    Other file entry points:

    * :py:func:`~uproot4.reading.open`: opens one file to read any of its objects.
    * :py:func:`~uproot4.behaviors.TBranch.iterate`: iterates through chunks of
      contiguous entries in ``TTrees``.
    * :py:func:`~uproot4.behaviors.TBranch.concatenate`: returns a single
      concatenated array from ``TTrees``.
    * :py:func:`~uproot4.behaviors.TBranch.lazy` (this function): returns a lazily
      read array from ``TTrees``.
    """
    import awkward1

    files = _regularize_files(files)
    decompression_executor, interpretation_executor = _regularize_executors(
        decompression_executor, interpretation_executor
    )
    array_cache = _regularize_array_cache(array_cache, None)
    library = uproot4.interpretation.library._regularize_library_lazy(library)

    if type(array_cache) == dict:
        array_cache = _WrapDict(array_cache)

    if array_cache is not None:
        layout_array_cache = awkward1.layout.ArrayCache(array_cache)

    real_options = dict(options)
    if "num_workers" not in real_options:
        real_options["num_workers"] = 1
    if "num_fallback_workers" not in real_options:
        real_options["num_fallback_workers"] = 1

    filter_branch = uproot4._util.regularize_filter(filter_branch)

    hasbranches = []
    common_keys = None
    is_self = []

    count = 0
    for file_path, object_path in files:
        obj = _regularize_object_path(
            file_path, object_path, custom_classes, allow_missing, real_options
        )

        if obj is not None:
            count += 1

            if isinstance(obj, TBranch) and len(obj.keys(recursive=True)) == 0:
                original = obj
                obj = obj.parent
                is_self.append(True)

                def real_filter_branch(branch):
                    return branch is original and filter_branch(branch)

            else:
                is_self.append(False)
                real_filter_branch = filter_branch

            hasbranches.append(obj)

            new_keys = obj.keys(
                recursive=recursive,
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=real_filter_branch,
                full_paths=full_paths,
            )

            if common_keys is None:
                common_keys = new_keys
            else:
                new_keys = set(new_keys)
                common_keys = [key for key in common_keys if key in new_keys]

    if count == 0:
        raise ValueError(
            "allow_missing=True and no TTrees found in\n\n    {0}".format(
                "\n    ".join(
                    "{"
                    + "{0}: {1}".format(
                        repr(f.file_path if isinstance(f, HasBranches) else f),
                        repr(f.object_path if isinstance(f, HasBranches) else o),
                    )
                    + "}"
                    for f, o in files
                )
            )
        )

    if len(common_keys) == 0 or not (all(is_self) or not any(is_self)):
        raise ValueError(
            "TTrees in\n\n    {0}\n\nhave no TBranches in common".format(
                "\n    ".join(
                    "{"
                    + "{0}: {1}".format(
                        repr(f.file_path if isinstance(f, HasBranches) else f),
                        repr(f.object_path if isinstance(f, HasBranches) else o),
                    )
                    + "}"
                    for f, o in files
                )
            )
        )

    partitions = []
    global_offsets = [0]
    global_cache_key = []
    for obj in hasbranches:
        entry_start, entry_stop = _regularize_entries_start_stop(
            obj.tree.num_entries, None, None
        )
        branchid_interpretation = {}
        for key in common_keys:
            branch = obj[key]
            branchid_interpretation[branch.cache_key] = branch.interpretation
        entry_step = _regularize_step_size(
            obj, step_size, entry_start, entry_stop, branchid_interpretation
        )

        for start in uproot4._util.range(entry_start, entry_stop, entry_step):
            stop = min(start + entry_step, entry_stop)
            length = stop - start

            fields = []
            names = []
            for key in common_keys:
                branch = obj[key]

                interpretation = branchid_interpretation[branch.cache_key]
                form = interpretation.awkward_form(obj.file, index_format="i64")
                if isinstance(interpretation, uproot4.interpretation.objects.AsObjects):
                    if not uproot4.interpretation.objects.awkward_can_optimize(
                        interpretation, form
                    ):
                        form = uproot4._util.awkward_form_of_iter(awkward1, form)

                generator = awkward1.layout.ArrayGenerator(
                    branch.array,
                    (
                        None,
                        start,
                        stop,
                        decompression_executor,
                        interpretation_executor,
                        None,
                        "ak",
                    ),
                    {},
                    uproot4._util.awkward_form_remove_uproot(
                        awkward1, form
                    ),  # , interpretation
                    length,
                )
                cache_key = "{0}:{1}:{2}-{3}:{4}".format(
                    branch.cache_key,
                    interpretation.cache_key,
                    start,
                    stop,
                    library.name,
                )
                global_cache_key.append(cache_key)
                virtualarray = awkward1.layout.VirtualArray(
                    generator, cache=layout_array_cache, cache_key=cache_key
                )
                fields.append(virtualarray)
                names.append(key)

            recordarray = awkward1.layout.RecordArray(fields, names, length)
            partitions.append(recordarray)
            global_offsets.append(global_offsets[-1] + length)

    out = awkward1.partition.IrregularlyPartitionedArray(partitions, global_offsets[1:])
    return awkward1.Array(out, cache=array_cache)


class Report(object):
    """
    Args:
        source (:py:class:`~uproot4.behaviors.TBranch.HasBranches`): The
            object (:py:class:`~uproot4.behaviors.TBranch.TBranch` or
            :py:class:`~uproot4.behaviors.TTree.TTree`) that this batch of data
            came from.
        tree_entry_start (int): First entry in the batch, counting zero at
            the start of the current ``TTree`` (current file).
        tree_entry_stop (int): First entry *after* the batch (last entry plus
            one), counting zero at the start of the ``TTree`` (current file).
        global_offset (int): Number of entries between the start of iteration
            and the start of this ``TTree``. The
            :py:attr:`~uproot4.behaviors.TBranch.Report.global_entry_start` and
            :py:attr:`~uproot4.behaviors.TBranch.Report.global_entry_stop` are
            equal to :py:attr:`~uproot4.behaviors.TBranch.Report.tree_entry_start`
            and :py:attr:`~uproot4.behaviors.TBranch.Report.tree_entry_stop` plus
            ``global_offset``.

    Information about the current iteration of
    :py:meth:`~uproot4.behaviors.TBranch.HasBranches.iterate` (the method) or
    :py:func:`~uproot4.behaviors.TBranch.iterate` (the function).

    Since the :py:meth:`~uproot4.behaviors.TBranch.HasBranches.iterate` method
    only iterates over data from one ``TTree``, its ``global_offset`` is always
    zero; :py:attr:`~uproot4.behaviors.TBranch.Report.global_entry_start` and
    :py:attr:`~uproot4.behaviors.TBranch.Report.global_entry_stop` are equal to
    :py:attr:`~uproot4.behaviors.TBranch.Report.tree_entry_start` and
    :py:attr:`~uproot4.behaviors.TBranch.Report.tree_entry_stop`, respectively.

    """

    def __init__(self, source, tree_entry_start, tree_entry_stop, global_offset=0):
        self._source = source
        self._tree_entry_start = tree_entry_start
        self._tree_entry_stop = tree_entry_stop
        self._global_offset = global_offset

    def __repr__(self):
        if self._global_offset == 0:
            return "Report({0}, {1}, {2})".format(
                self._source, self._tree_entry_start, self._tree_entry_stop
            )
        else:
            return "Report({0}, {1}, {2}, global_offset={3})".format(
                self._source,
                self._tree_entry_start,
                self._tree_entry_stop,
                self._global_offset,
            )

    @property
    def source(self):
        """
        The object (:py:class:`~uproot4.behaviors.TBranch.TBranch` or
        :py:class:`~uproot4.behaviors.TTree.TTree`) that this batch of data
        came from.
        """
        return self._source

    @property
    def tree(self):
        """
        The :py:class:`~uproot4.behaviors.TTree.TTree` that this batch of data
        came from.
        """
        return self._source.tree

    @property
    def file(self):
        """
        The :py:class:`~uproot4.reading.ReadOnlyFile` that this batch of data
        came from.
        """
        return self._source.file

    @property
    def file_path(self):
        """
        The path/name of the :py:class:`~uproot4.reading.ReadOnlyFile` that
        this batch of data came from.
        """
        return self._source.file.file_path

    @property
    def tree_entry_start(self):
        """
        First entry in the batch, counting zero at the start of the current
        ``TTree`` (current file).
        """
        return self._tree_entry_start

    @property
    def tree_entry_stop(self):
        """
        First entry *after* the batch (last entry plus one), counting zero at
        the start of the ``TTree`` (current file).
        """
        return self._tree_entry_stop

    @property
    def global_entry_start(self):
        """
        First entry in the batch, counting zero at the start of iteration
        (potentially over many files).
        """
        return self._tree_entry_start + self._global_offset

    @property
    def global_entry_stop(self):
        """
        First entry *after* the batch (last entry plust one), counting zero at
        the start of iteration (potentially over many files).
        """
        return self._tree_entry_stop + self._global_offset

    @property
    def start(self):
        """
        A synonym for
        :py:attr:`~uproot4.behaviors.TBranch.Report.global_entry_start`.
        """
        return self._tree_entry_start + self._global_offset

    @property
    def stop(self):
        """
        A synonym for
        :py:attr:`~uproot4.behaviors.TBranch.Report.global_entry_stop`.
        """
        return self._tree_entry_stop + self._global_offset

    @property
    def global_offset(self):
        """
        Number of entries between the start of iteration and the start of this
        ``TTree``. The
        :py:attr:`~uproot4.behaviors.TBranch.Report.global_entry_start` and
        :py:attr:`~uproot4.behaviors.TBranch.Report.global_entry_stop` are
        equal to :py:attr:`~uproot4.behaviors.TBranch.Report.tree_entry_start`
        and :py:attr:`~uproot4.behaviors.TBranch.Report.tree_entry_stop` plus
        ``global_offset``.
        """
        return self._global_offset

    def to_global(self, global_offset):
        """
        Copies the data in this :py:class:`~uproot4.branches.TBranch.Report` to
        another with a new
        :py:attr:`~uproot4.branches.TBranch.Report.global_offset`.
        """
        return Report(
            self._source, self._tree_entry_start, self._tree_entry_stop, global_offset
        )


class HasBranches(Mapping):
    """
    Abstract class of behaviors for anything that "has branches," namely
    :py:class:`~uproot4.behavior.TTree.TTree` and
    :py:class:`~uproot4.behavior.TBranch.TBranch`, which mostly consist of array-reading
    methods.

    A :py:class:`~uproot4.behavior.TBranch.HasBranches` is a Python ``Mapping``, which
    uses square bracket syntax to extract subbranches:

    .. code-block:: python

        my_tree["branch"]
        my_tree["branch"]["subbranch"]
        my_tree["branch/subbranch"]
        my_tree["branch/subbranch/subsubbranch"]
    """

    @property
    def branches(self):
        """
        The list of :py:class:`~uproot4.behavior.TBranch.TBranch` directly under
        this :py:class:`~uproot4.behavior.TTree.TTree` or
        :py:class:`~uproot4.behavior.TBranch.TBranch` (i.e. not recursive).
        """
        return self.member("fBranches")

    def show(
        self,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,
        full_paths=True,
        name_width=20,
        typename_width=24,
        interpretation_width=30,
        stream=sys.stdout,
    ):
        """
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :py:class:`~uproot4.behaviors.TBranch.TBranch` \u2192 bool, :py:class:`~uproot4.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :py:class:`~uproot4.behaviors.TBranch.TBranch` object. The ``TBranch`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, recursively descend into the branches'
                branches.
            full_paths (bool): If True, include the full path to each subbranch
                with slashes (``/``); otherwise, use the descendant's name as
                the display name.
            name_width (int): Number of characters to reserve for the ``TBranch``
                names.
            typename_width (int): Number of characters to reserve for the C++
                typenames.
            interpretation_width (int): Number of characters to reserve for the
                :py:class:`~uproot4.interpretation.Interpretation` displays.
            stream (object with a ``write(str)`` method): Stream to write the
                output to.

        Interactively display the ``TBranches``.

        Example:

        .. code-block::

            name                 | typename             | interpretation
            ---------------------+----------------------+-----------------------------------
            event_number         | int32_t              | AsDtype('>i4')
            trigger_isomu24      | bool                 | AsDtype('bool')
            eventweight          | float                | AsDtype('>f4')
            MET                  | TVector2             | AsStridedObjects(Model_TVector2_v3
            jetp4                | std::vector<TLorentz | AsJagged(AsStridedObjects(Model_TL
            jetbtag              | std::vector<float>   | AsJagged(AsDtype('>f4'), header_by
            jetid                | std::vector<bool>    | AsJagged(AsDtype('bool'), header_b
        """
        if name_width < 3:
            raise ValueError("'name_width' must be at least 3")
        if typename_width < 3:
            raise ValueError("'typename_width' must be at least 3")
        if interpretation_width < 3:
            raise ValueError("'interpretation_width' must be at least 3")

        formatter = "{{0:{0}.{0}}} | {{1:{1}.{1}}} | {{2:{2}.{2}}}".format(
            name_width, typename_width, interpretation_width,
        )

        stream.write(formatter.format("name", "typename", "interpretation"))
        stream.write(
            "\n"
            + "-" * name_width
            + "-+-"
            + "-" * typename_width
            + "-+-"
            + "-" * interpretation_width
            + "\n"
        )

        if isinstance(self, TBranch):
            stream.write(
                formatter.format(self.name, self.typename, repr(self.interpretation))
            )

        for name, branch in self.iteritems(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            recursive=recursive,
            full_paths=full_paths,
        ):
            typename = branch.typename
            interp = repr(branch.interpretation)

            if len(name) > name_width:
                name = name[: name_width - 3] + "..."
            if len(typename) > typename_width:
                typename = typename[: typename_width - 3] + "..."
            if len(interp) > interpretation_width:
                interp = interp[: interpretation_width - 3] + "..."

            stream.write(formatter.format(name, typename, interp).rstrip(" ") + "\n")

    def arrays(
        self,
        expressions=None,
        cut=None,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        aliases=None,
        language=uproot4.language.python.PythonLanguage(),
        entry_start=None,
        entry_stop=None,
        decompression_executor=None,
        interpretation_executor=None,
        array_cache="inherit",
        library="ak",
        how=None,
    ):
        u"""
        Args:
            expressions (None, str, or list of str): Names of ``TBranches`` or
                aliases to convert to arrays or mathematical expressions of them.
                Uses the ``language`` to evaluate. If None, all ``TBranches``
                selected by the filters are included.
            cut (None or str): If not None, this expression filters all of the
                ``expressions``.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :py:class:`~uproot4.behaviors.TBranch.TBranch` \u2192 bool, :py:class:`~uproot4.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :py:class:`~uproot4.behaviors.TBranch.TBranch` object. If the function
                returns False or None, the ``TBranch`` is excluded; if the function
                returns True, it is included with its standard
                :py:class:`~uproot4.behaviors.TBranch.TBranch.interpretation`; if an
                :py:class:`~uproot4.interpretation.Interpretation`, this interpretation
                overrules the standard one.
            aliases (None or dict of str \u2192 str): Mathematical expressions that
                can be used in ``expressions`` or other aliases (without cycles).
                Uses the ``language`` engine to evaluate. If None, only the
                :py:attr:`~uproot4.behaviors.TBranch.TBranch.aliases` are available.
            language (:py:class:`~uproot4.language.Language`): Language used to interpret
                the ``expressions`` and ``aliases``.
            entry_start (None or int): The first entry to include. If None, start
                at zero. If negative, count from the end, like a Python slice.
            entry_stop (None or int): The first entry to exclude (i.e. one greater
                than the last entry to include). If None, stop at
                :py:attr:`~uproot4.behavior.TTree.TTree.num_entries`. If negative,
                count from the end, like a Python slice.
            decompression_executor (None or Executor with a ``submit`` method): The
                executor that is used to decompress ``TBaskets``; if None, the
                global ``uproot4.decompression_executor`` is used.
            interpretation_executor (None or Executor with a ``submit`` method): The
                executor that is used to interpret uncompressed ``TBasket`` data as
                arrays; if None, the global ``uproot4.interpretation_executor`` is
                used.
            array_cache ("inherit", None, MutableMapping, or memory size): Cache of arrays;
                if "inherit", use the file's cache; if None, do not use a cache;
                if a memory size, create a new cache of this size.
            library (str or :py:class:`~uproot4.interpretation.library.Library`): The library
                that is used to represent arrays. Options are ``"np"`` for NumPy,
                ``"ak"`` for Awkward Array, ``"pd"`` for Pandas, and ``"cp"`` for
                CuPy.
            how (None, str, or container type): Library-dependent instructions
                for grouping. The only recognized container types are ``tuple``,
                ``list``, and ``dict``. Note that the container *type itself*
                must be passed as ``how``, not an instance of that type (i.e.
                ``how=tuple``, not ``how=()``).

        Returns a group of arrays from the ``TTree``.

        For example:

        .. code-block:: python

            >>> my_tree["x"].array()
            <Array [-41.2, 35.1, 35.1, ... 32.4, 32.5] type='2304 * float64'>
            >>> my_tree["y"].array()
            <Array [17.4, -16.6, -16.6, ... 1.2, 1.2, 1.2] type='2304 * float64'>

        See also :py:meth:`~uproot4.behavior.TBranch.TBranch.array` to read a single
        ``TBranch`` as an array.

        See also :py:meth:`~uproot4.behavior.TBranch.HasBranches.iterate` to iterate over
        the array in contiguous ranges of entries.
        """
        keys = set(self.keys(recursive=True, full_paths=False))
        if isinstance(self, TBranch) and expressions is None and len(keys) == 0:
            filter_branch = uproot4._util.regularize_filter(filter_branch)
            return self.parent.arrays(
                expressions=expressions,
                cut=cut,
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=lambda branch: branch is self and filter_branch(branch),
                aliases=aliases,
                language=language,
                entry_start=entry_start,
                entry_stop=entry_stop,
                decompression_executor=decompression_executor,
                interpretation_executor=interpretation_executor,
                array_cache=array_cache,
                library=library,
                how=how,
            )

        entry_start, entry_stop = _regularize_entries_start_stop(
            self.tree.num_entries, entry_start, entry_stop
        )
        decompression_executor, interpretation_executor = _regularize_executors(
            decompression_executor, interpretation_executor
        )
        array_cache = _regularize_array_cache(array_cache, self._file)
        library = uproot4.interpretation.library._regularize_library(library)

        def get_from_cache(branchname, interpretation):
            if array_cache is not None:
                cache_key = "{0}:{1}:{2}:{3}-{4}:{5}".format(
                    self.cache_key,
                    branchname,
                    interpretation.cache_key,
                    entry_start,
                    entry_stop,
                    library.name,
                )
                return array_cache.get(cache_key)
            else:
                return None

        aliases = _regularize_aliases(self, aliases)
        arrays, expression_context, branchid_interpretation = _regularize_expressions(
            self,
            expressions,
            cut,
            filter_name,
            filter_typename,
            filter_branch,
            keys,
            aliases,
            language,
            get_from_cache,
        )

        ranges_or_baskets = []
        for expression, context in expression_context:
            branch = context.get("branch")
            if branch is not None and not context["is_duplicate"]:
                for basket_num, range_or_basket in branch.entries_to_ranges_or_baskets(
                    entry_start, entry_stop
                ):
                    ranges_or_baskets.append((branch, basket_num, range_or_basket))

        _ranges_or_baskets_to_arrays(
            self,
            ranges_or_baskets,
            branchid_interpretation,
            entry_start,
            entry_stop,
            decompression_executor,
            interpretation_executor,
            library,
            arrays,
        )

        if array_cache is not None:
            for expression, context in expression_context:
                branch = context.get("branch")
                if branch is not None:
                    interpretation = branchid_interpretation[branch.cache_key]
                    if branch is not None:
                        cache_key = "{0}:{1}:{2}:{3}-{4}:{5}".format(
                            self.cache_key,
                            expression,
                            interpretation.cache_key,
                            entry_start,
                            entry_stop,
                            library.name,
                        )
                    array_cache[cache_key] = arrays[branch.cache_key]

        output = language.compute_expressions(
            arrays,
            expression_context,
            keys,
            aliases,
            self.file.file_path,
            self.object_path,
        )

        expression_context = [
            (e, c) for e, c in expression_context if c["is_primary"] and not c["is_cut"]
        ]

        return library.group(output, expression_context, how)

    def iterate(
        self,
        expressions=None,
        cut=None,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        aliases=None,
        language=uproot4.language.python.PythonLanguage(),
        entry_start=None,
        entry_stop=None,
        step_size="100 MB",
        decompression_executor=None,
        interpretation_executor=None,
        library="ak",
        how=None,
        report=False,
    ):
        u"""
        Args:
            expressions (None, str, or list of str): Names of ``TBranches`` or
                aliases to convert to arrays or mathematical expressions of them.
                Uses the ``language`` to evaluate. If None, all ``TBranches``
                selected by the filters are included.
            cut (None or str): If not None, this expression filters all of the
                ``expressions``.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :py:class:`~uproot4.behaviors.TBranch.TBranch` \u2192 bool, :py:class:`~uproot4.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :py:class:`~uproot4.behaviors.TBranch.TBranch` object. If the function
                returns False or None, the ``TBranch`` is excluded; if the function
                returns True, it is included with its standard
                :py:attr:`~uproot4.behaviors.TBranch.TBranch.interpretation`; if an
                :py:class:`~uproot4.interpretation.Interpretation`, this interpretation
                overrules the standard one.
            aliases (None or dict of str \u2192 str): Mathematical expressions that
                can be used in ``expressions`` or other aliases (without cycles).
                Uses the ``language`` engine to evaluate. If None, only the
                :py:attr:`~uproot4.behaviors.TBranch.TBranch.aliases` are available.
            language (:py:class:`~uproot4.language.Language`): Language used to interpret
                the ``expressions`` and ``aliases``.
            entry_start (None or int): The first entry to include. If None, start
                at zero. If negative, count from the end, like a Python slice.
            entry_stop (None or int): The first entry to exclude (i.e. one greater
                than the last entry to include). If None, stop at
                :py:attr:`~uproot4.behavior.TTree.TTree.num_entries`. If negative,
                count from the end, like a Python slice.
            step_size (int or str): If an integer, the maximum number of entries to
                include in each iteration step; if a string, the maximum memory size
                to include. The string must be a number followed by a memory unit,
                such as "100 MB".
            decompression_executor (None or Executor with a ``submit`` method): The
                executor that is used to decompress ``TBaskets``; if None, the
                global ``uproot4.decompression_executor`` is used.
            interpretation_executor (None or Executor with a ``submit`` method): The
                executor that is used to interpret uncompressed ``TBasket`` data as
                arrays; if None, the global ``uproot4.interpretation_executor`` is
                used.
            library (str or :py:class:`~uproot4.interpretation.library.Library`): The library
                that is used to represent arrays. Options are ``"np"`` for NumPy,
                ``"ak"`` for Awkward Array, ``"pd"`` for Pandas, and ``"cp"`` for
                CuPy.
            how (None, str, or container type): Library-dependent instructions
                for grouping. The only recognized container types are ``tuple``,
                ``list``, and ``dict``. Note that the container *type itself*
                must be passed as ``how``, not an instance of that type (i.e.
                ``how=tuple``, not ``how=()``).
            report (bool): If True, this generator yields
                (arrays, :py:class:`~uproot4.behaviors.TBranch.Report`) pairs; if False,
                it only yields arrays. The report has data about the ``TFile``,
                ``TTree``, and global and local entry ranges.

        Iterates through contiguous chunks of entries from the ``TTree``.

        For example:

        .. code-block:: python

            >>> for array in tree.iterate(["x", "y"], step_size=100):
            ...     # each of the following have 100 entries
            ...     array["x"], array["y"]

        See also :py:meth:`~uproot4.behavior.TBranch.HasBranches.arrays` to read
        everything in a single step, without iteration.

        See also :py:func:`~uproot4.behavior.TBranch.iterate` to iterate over many
        files.
        """
        keys = set(self.keys(recursive=True, full_paths=False))
        if isinstance(self, TBranch) and expressions is None and len(keys) == 0:
            filter_branch = uproot4._util.regularize_filter(filter_branch)
            for x in self.parent.iterate(
                expressions=expressions,
                cut=cut,
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=lambda branch: branch is self and filter_branch(branch),
                aliases=aliases,
                language=language,
                entry_start=entry_start,
                entry_stop=entry_stop,
                step_size=step_size,
                decompression_executor=decompression_executor,
                interpretation_executor=interpretation_executor,
                library=library,
                how=how,
                report=report,
            ):
                yield x

        else:
            entry_start, entry_stop = _regularize_entries_start_stop(
                self.tree.num_entries, entry_start, entry_stop
            )
            decompression_executor, interpretation_executor = _regularize_executors(
                decompression_executor, interpretation_executor
            )
            library = uproot4.interpretation.library._regularize_library(library)

            aliases = _regularize_aliases(self, aliases)
            (
                arrays,
                expression_context,
                branchid_interpretation,
            ) = _regularize_expressions(
                self,
                expressions,
                cut,
                filter_name,
                filter_typename,
                filter_branch,
                keys,
                aliases,
                language,
                (lambda branchname, interpretation: None),
            )

            entry_step = _regularize_step_size(
                self, step_size, entry_start, entry_stop, branchid_interpretation
            )

            previous_baskets = {}
            for sub_entry_start in uproot4._util.range(
                entry_start, entry_stop, entry_step
            ):
                sub_entry_stop = min(sub_entry_start + entry_step, entry_stop)
                if sub_entry_stop - sub_entry_start == 0:
                    continue

                ranges_or_baskets = []
                for expression, context in expression_context:
                    branch = context.get("branch")
                    if branch is not None and not context["is_duplicate"]:
                        for (
                            basket_num,
                            range_or_basket,
                        ) in branch.entries_to_ranges_or_baskets(
                            sub_entry_start, sub_entry_stop
                        ):
                            previous_basket = previous_baskets.get(
                                (branch.cache_key, basket_num)
                            )
                            if previous_basket is None:
                                ranges_or_baskets.append(
                                    (branch, basket_num, range_or_basket)
                                )
                            else:
                                ranges_or_baskets.append(
                                    (branch, basket_num, previous_basket)
                                )

                arrays = {}
                _ranges_or_baskets_to_arrays(
                    self,
                    ranges_or_baskets,
                    branchid_interpretation,
                    sub_entry_start,
                    sub_entry_stop,
                    decompression_executor,
                    interpretation_executor,
                    library,
                    arrays,
                )

                output = language.compute_expressions(
                    arrays,
                    expression_context,
                    keys,
                    aliases,
                    self.file.file_path,
                    self.object_path,
                )

                expression_context = [
                    (e, c)
                    for e, c in expression_context
                    if c["is_primary"] and not c["is_cut"]
                ]

                arrays = library.group(output, expression_context, how)

                if report:
                    yield arrays, Report(self, sub_entry_start, sub_entry_stop)
                else:
                    yield arrays

                for branch, basket_num, basket in ranges_or_baskets:
                    previous_baskets[branch.cache_key, basket_num] = basket

    def keys(
        self,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,
        full_paths=True,
    ):
        u"""
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :py:class:`~uproot4.behaviors.TBranch.TBranch` \u2192 bool, :py:class:`~uproot4.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :py:class:`~uproot4.behaviors.TBranch.TBranch` object. The ``TBranch`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subbranches.
                If False, only return the names of branches directly accessible
                under this object.
            full_paths (bool): If True, include the full path to each subbranch
                with slashes (``/``); otherwise, use the descendant's name as
                the output name.

        Returns the names of the subbranches as a list of strings.
        """
        return list(
            self.iterkeys(
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=filter_branch,
                recursive=recursive,
                full_paths=full_paths,
            )
        )

    def values(
        self,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,
    ):
        u"""
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :py:class:`~uproot4.behaviors.TBranch.TBranch` \u2192 bool, :py:class:`~uproot4.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :py:class:`~uproot4.behaviors.TBranch.TBranch` object. The ``TBranch`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subbranches.
                If False, only return branches that are directly accessible
                under this object.

        Returns the subbranches as a list of
        :py:class:`~uproot4.behavior.TBranch.TBranch`.

        (Note: with ``recursive=False``, this is the same as
        :py:attr:`~uproot4.behavior.TBranch.HasBranches.branches`.)
        """
        return list(
            self.itervalues(
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=filter_branch,
                recursive=recursive,
            )
        )

    def items(
        self,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,
        full_paths=True,
    ):
        u"""
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :py:class:`~uproot4.behaviors.TBranch.TBranch` \u2192 bool, :py:class:`~uproot4.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :py:class:`~uproot4.behaviors.TBranch.TBranch` object. The ``TBranch`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subbranches.
                If False, only return (name, branch) pairs for branches
                directly accessible under this object.
            full_paths (bool): If True, include the full path to each subbranch
                with slashes (``/``) in the name; otherwise, use the descendant's
                name as the name without modification.

        Returns (name, branch) pairs of the subbranches as a list of 2-tuples
        of (str, :py:class:`~uproot4.behavior.TBranch.TBranch`).
        """
        return list(
            self.iteritems(
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=filter_branch,
                recursive=recursive,
                full_paths=full_paths,
            )
        )

    def typenames(
        self,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,
        full_paths=True,
    ):
        u"""
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :py:class:`~uproot4.behaviors.TBranch.TBranch` \u2192 bool, :py:class:`~uproot4.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :py:class:`~uproot4.behaviors.TBranch.TBranch` object. The ``TBranch`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subbranches.
                If False, only return (name, typename) pairs for branches
                directly accessible under this object.
            full_paths (bool): If True, include the full path to each subbranch
                with slashes (``/``) in the name; otherwise, use the descendant's
                name as the name without modification.

        Returns (name, typename) pairs of the subbranches as a dict of
        str \u2192 str.
        """
        return dict(
            self.itertypenames(
                filter_name=filter_name,
                filter_typename=filter_typename,
                filter_branch=filter_branch,
                recursive=recursive,
                full_paths=full_paths,
            )
        )

    def iterkeys(
        self,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,
        full_paths=True,
    ):
        u"""
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :py:class:`~uproot4.behaviors.TBranch.TBranch` \u2192 bool, :py:class:`~uproot4.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :py:class:`~uproot4.behaviors.TBranch.TBranch` object. The ``TBranch`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subbranches.
                If False, only return the names of branches directly accessible
                under this object.
            full_paths (bool): If True, include the full path to each subbranch
                with slashes (``/``); otherwise, use the descendant's name as
                the output name.

        Returns the names of the subbranches as an iterator over strings.
        """
        for k, v in self.iteritems(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            recursive=recursive,
            full_paths=full_paths,
        ):
            yield k

    def itervalues(
        self,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,
    ):
        u"""
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :py:class:`~uproot4.behaviors.TBranch.TBranch` \u2192 bool, :py:class:`~uproot4.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :py:class:`~uproot4.behaviors.TBranch.TBranch` object. The ``TBranch`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subbranches.
                If False, only return branches that are directly accessible
                under this object.

        Returns the subbranches as an iterator over
        :py:class:`~uproot4.behavior.TBranch.TBranch`.

        (Note: with ``recursive=False``, this is the same as
        :py:attr:`~uproot4.behavior.TBranch.HasBranches.branches`.)
        """
        for k, v in self.iteritems(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            recursive=recursive,
            full_paths=False,
        ):
            yield v

    def iteritems(
        self,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,
        full_paths=True,
    ):
        u"""
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :py:class:`~uproot4.behaviors.TBranch.TBranch` \u2192 bool, :py:class:`~uproot4.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :py:class:`~uproot4.behaviors.TBranch.TBranch` object. The ``TBranch`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subbranches.
                If False, only return (name, branch) pairs for branches
                directly accessible under this object.
            full_paths (bool): If True, include the full path to each subbranch
                with slashes (``/``) in the name; otherwise, use the descendant's
                name as the name without modification.

        Returns (name, branch) pairs of the subbranches as an iterator over
        2-tuples of (str, :py:class:`~uproot4.behavior.TBranch.TBranch`).
        """
        filter_name = uproot4._util.regularize_filter(filter_name)
        filter_typename = uproot4._util.regularize_filter(filter_typename)
        if filter_branch is None:
            filter_branch = no_filter
        elif callable(filter_branch):
            pass
        else:
            raise TypeError(
                "filter_branch must be None or a function: TBranch -> bool, not {0}".format(
                    repr(filter_branch)
                )
            )
        for branch in self.branches:
            if (
                (filter_name is no_filter or filter_name(branch.name))
                and (filter_typename is no_filter or filter_typename(branch.typename))
                and (filter_branch is no_filter or filter_branch(branch))
            ):
                yield branch.name, branch

            if recursive:
                for k1, v in branch.iteritems(
                    recursive=recursive,
                    filter_name=no_filter,
                    filter_typename=filter_typename,
                    filter_branch=filter_branch,
                    full_paths=full_paths,
                ):
                    if full_paths:
                        k2 = "{0}/{1}".format(branch.name, k1)
                    else:
                        k2 = k1
                    if filter_name is no_filter or filter_name(k2):
                        yield k2, v

    def itertypenames(
        self,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,
        full_paths=True,
    ):
        u"""
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :py:class:`~uproot4.behaviors.TBranch.TBranch` \u2192 bool, :py:class:`~uproot4.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :py:class:`~uproot4.behaviors.TBranch.TBranch` object. The ``TBranch`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subbranches.
                If False, only return (name, typename) pairs for branches
                directly accessible under this object.
            full_paths (bool): If True, include the full path to each subbranch
                with slashes (``/``) in the name; otherwise, use the descendant's
                name as the name without modification.

        Returns (name, typename) pairs of the subbranches as an iterator over
        2-tuples of (str, str).
        """
        for k, v in self.iteritems(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            recursive=recursive,
            full_paths=full_paths,
        ):
            yield k, v.typename

    def _ipython_key_completions_(self):
        """
        Supports key-completion in an IPython or Jupyter kernel.
        """
        return self.iterkeys()

    def num_entries_for(
        self,
        memory_size,
        expressions=None,
        cut=None,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        aliases=None,
        language=uproot4.language.python.PythonLanguage(),
        entry_start=None,
        entry_stop=None,
    ):
        """
        Args:
            memory_size (int or str): An integer is interpreted as a number of
                bytes and a string must be a number followed by a unit, such as
                "100 MB".
            expressions (None, str, or list of str): Names of ``TBranches`` or
                aliases to convert to arrays or mathematical expressions of them.
                Uses the ``language`` to evaluate. If None, all ``TBranches``
                selected by the filters are included.
            cut (None or str): If not None, this expression filters all of the
                ``expressions``.
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :py:class:`~uproot4.behaviors.TBranch.TBranch` \u2192 bool, :py:class:`~uproot4.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :py:class:`~uproot4.behaviors.TBranch.TBranch` object. The ``TBranch`` is
                included if the function returns True, excluded if it returns False.
            aliases (None or dict of str \u2192 str): Mathematical expressions that
                can be used in ``expressions`` or other aliases (without cycles).
                Uses the ``language`` engine to evaluate. If None, only the
                :py:attr:`~uproot4.behaviors.TBranch.TBranch.aliases` are available.
            language (:py:class:`~uproot4.language.Language`): Language used to interpret
                the ``expressions`` and ``aliases``.
            entry_start (None or int): The first entry to include. If None, start
                at zero. If negative, count from the end, like a Python slice.
            entry_stop (None or int): The first entry to exclude (i.e. one greater
                than the last entry to include). If None, stop at
                :py:attr:`~uproot4.behavior.TTree.TTree.num_entries`. If negative,
                count from the end, like a Python slice.

        Returns an *approximate* step size as a number of entries to read
        a given ``memory_size`` in each step.

        This method does not actually read the ``TBranch`` data or compute any
        expressions to arrive at its estimate. It only uses metadata from the
        already-loaded ``TTree``; it only needs ``language`` to parse the
        expressions, not to evaluate them.

        In addition, the estimate is based on compressed ``TBasket`` sizes
        (the amount of data that would have to be read), not uncompressed
        ``TBasket`` sizes (the amount of data that the final arrays would use
        in memory, without considering ``cuts``).

        This is the algorithm that
        :py:meth:`~uproot4.behavior.TBranch.HasBranches.iterate` uses to convert a
        ``step_size`` expressed in memory units into a number of entries.
        """
        target_num_bytes = uproot4._util.memory_size(memory_size)

        entry_start, entry_stop = _regularize_entries_start_stop(
            self.tree.num_entries, entry_start, entry_stop
        )

        keys = set(self.keys(recursive=True, full_paths=False))
        aliases = _regularize_aliases(self, aliases)
        arrays, expression_context, branchid_interpretation = _regularize_expressions(
            self,
            expressions,
            cut,
            filter_name,
            filter_typename,
            filter_branch,
            keys,
            aliases,
            language,
            (lambda branchname, interpretation: None),
        )

        return _hasbranches_num_entries_for(
            self, target_num_bytes, entry_start, entry_stop, branchid_interpretation
        )

    def common_entry_offsets(
        self,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=no_filter,
        recursive=True,
    ):
        u"""
        Args:
            filter_name (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by name.
            filter_typename (None, glob string, regex string in ``"/pattern/i"`` syntax, function of str \u2192 bool, or iterable of the above): A
                filter to select ``TBranches`` by type.
            filter_branch (None or function of :py:class:`~uproot4.behaviors.TBranch.TBranch` \u2192 bool, :py:class:`~uproot4.interpretation.Interpretation`, or None): A
                filter to select ``TBranches`` using the full
                :py:class:`~uproot4.behaviors.TBranch.TBranch` object. The ``TBranch`` is
                included if the function returns True, excluded if it returns False.
            recursive (bool): If True, descend into any nested subbranches.
                If False, only consider branches directly accessible under this
                object. (Only applies when ``branches=None``.)

        Returns entry offsets in which ``TBasket`` boundaries align in the
        specified set of branches.

        If this :py:class:`~uproot4.behaviors.TBranch.TBranch` has no subbranches,
        the output is identical to
        :py:attr:`~uproot4.behaviors.TBranch.TBranch.entry_offsets`.
        """
        common_offsets = None
        for branch in self.itervalues(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            recursive=recursive,
        ):
            if common_offsets is None:
                common_offsets = set(branch.entry_offsets)
            else:
                common_offsets = common_offsets.intersection(set(branch.entry_offsets))
        return sorted(common_offsets)

    def __getitem__(self, where):
        original_where = where

        got = self._lookup.get(original_where)
        if got is not None:
            return got

        if uproot4._util.isint(where):
            return self.branches[where]
        elif uproot4._util.isstr(where):
            where = uproot4._util.ensure_str(where)
        else:
            raise TypeError(
                "where must be an integer or a string, not {0}".format(repr(where))
            )

        if where.startswith("/"):
            recursive = False
            where = where[1:]
        else:
            recursive = True

        if "/" in where:
            where = "/".join([x for x in where.split("/") if x != ""])
            for k, v in self.iteritems(recursive=True, full_paths=True):
                if where == k:
                    self._lookup[original_where] = v
                    return v
            else:
                raise uproot4.KeyInFileError(
                    original_where,
                    keys=self.keys(recursive=recursive),
                    file_path=self._file.file_path,
                    object_path=self.object_path,
                )

        elif recursive:
            got = _get_recursive(self, where)
            if got is not None:
                self._lookup[original_where] = got
                return got
            else:
                raise uproot4.KeyInFileError(
                    original_where,
                    keys=self.keys(recursive=recursive),
                    file_path=self._file.file_path,
                    object_path=self.object_path,
                )

        else:
            for branch in self.branches:
                if branch.name == where:
                    self._lookup[original_where] = branch
                    return branch
            else:
                raise uproot4.KeyInFileError(
                    original_where,
                    keys=self.keys(recursive=recursive),
                    file_path=self._file.file_path,
                    object_path=self.object_path,
                )

    def __iter__(self):
        for x in self.branches:
            yield x

    def __len__(self):
        return len(self.branches)


_branch_clean_name = re.compile(r"(.*\.)*([^\.\[\]]*)(\[.*\])*")
_branch_clean_parent_name = re.compile(r"(.*\.)*([^\.\[\]]*)\.([^\.\[\]]*)(\[.*\])*")


class TBranch(HasBranches):
    """
    Behaviors for a ``TBranch``, which mostly consist of array-reading methods.

    Since a :py:class:`~uproot4.behavior.TBranch.TBranch` is a
    :py:class:`~uproot4.behavior.TBranch.HasBranches`, it is also a Python
    ``Mapping``, which uses square bracket syntax to extract subbranches:

    .. code-block:: python

        my_branch["subbranch"]
        my_branch["subbranch"]["subsubbranch"]
        my_branch["subbranch/subsubbranch"]
    """

    def __repr__(self):
        if len(self) == 0:
            return "<{0} {1} at 0x{2:012x}>".format(
                self.classname, repr(self.name), id(self)
            )
        else:
            return "<{0} {1} ({2} subbranches) at 0x{3:012x}>".format(
                self.classname, repr(self.name), len(self), id(self)
            )

    def array(
        self,
        interpretation=None,
        entry_start=None,
        entry_stop=None,
        decompression_executor=None,
        interpretation_executor=None,
        array_cache="inherit",
        library="ak",
    ):
        u"""
        Args:
            interpretation (None or :py:class:`~uproot4.interpretation.Interpretation`): An
                interpretation of the ``TBranch`` data as an array. If None, the
                standard :py:attr:`~uproot4.behavior.TBranch.TBranch.interpretation`
                is used, which is derived from
                :py:func:`~uproot4.interpretation.identify.interpretation_of`.
            entry_start (None or int): The first entry to include. If None, start
                at zero. If negative, count from the end, like a Python slice.
            entry_stop (None or int): The first entry to exclude (i.e. one greater
                than the last entry to include). If None, stop at
                :py:attr:`~uproot4.behavior.TTree.TTree.num_entries`. If negative,
                count from the end, like a Python slice.
            decompression_executor (None or Executor with a ``submit`` method): The
                executor that is used to decompress ``TBaskets``; if None, the
                global ``uproot4.decompression_executor`` is used.
            interpretation_executor (None or Executor with a ``submit`` method): The
                executor that is used to interpret uncompressed ``TBasket`` data as
                arrays; if None, the global ``uproot4.interpretation_executor`` is
                used.
            array_cache ("inherit", None, MutableMapping, or memory size): Cache of arrays;
                if "inherit", use the file's cache; if None, do not use a cache;
                if a memory size, create a new cache of this size.
            library (str or :py:class:`~uproot4.interpretation.library.Library`): The library
                that is used to represent arrays. Options are ``"np"`` for NumPy,
                ``"ak"`` for Awkward Array, ``"pd"`` for Pandas, and ``"cp"`` for
                CuPy.

        Returns the ``TBranch`` data as an array.

        For example:

        .. code-block:: python

            >>> array = tree
            >>> array = tree.arrays(["x", "y"])    # only reads branches "x" and "y"
            >>> array
            <Array [{x: -41.2, y: 17.4}, ... {x: 32.5, y: 1.2}], type='2304 * {"x": float64,...'>
            >>> array["x"]
            <Array [-41.2, 35.1, 35.1, ... 32.4, 32.5] type='2304 * float64'>
            >>> array["y"]
            <Array [17.4, -16.6, -16.6, ... 1.2, 1.2, 1.2] type='2304 * float64'>

        See also :py:meth:`~uproot4.behavior.TBranch.HasBranches.arrays` to read
        multiple ``TBranches`` into a group of arrays or an array-group.
        """
        if interpretation is None:
            interpretation = self.interpretation
        else:
            interpretation = _regularize_interpretation(interpretation)
        branchid_interpretation = {self.cache_key: interpretation}

        entry_start, entry_stop = _regularize_entries_start_stop(
            self.num_entries, entry_start, entry_stop
        )
        decompression_executor, interpretation_executor = _regularize_executors(
            decompression_executor, interpretation_executor
        )
        array_cache = _regularize_array_cache(array_cache, self._file)
        library = uproot4.interpretation.library._regularize_library(library)

        cache_key = "{0}:{1}:{2}-{3}:{4}".format(
            self.cache_key,
            interpretation.cache_key,
            entry_start,
            entry_stop,
            library.name,
        )
        if array_cache is not None:
            got = array_cache.get(cache_key)
            if got is not None:
                return got

        ranges_or_baskets = []
        for basket_num, range_or_basket in self.entries_to_ranges_or_baskets(
            entry_start, entry_stop
        ):
            ranges_or_baskets.append((self, basket_num, range_or_basket))

        arrays = {}
        _ranges_or_baskets_to_arrays(
            self,
            ranges_or_baskets,
            branchid_interpretation,
            entry_start,
            entry_stop,
            decompression_executor,
            interpretation_executor,
            library,
            arrays,
        )

        if array_cache is not None:
            array_cache[cache_key] = arrays[self.cache_key]

        return arrays[self.cache_key]

    def __array__(self, *args, **kwargs):
        out = self.array(library="np")
        if args == () and kwargs == {}:
            return out
        else:
            return numpy.array(out, *args, **kwargs)

    @property
    def name(self):
        """
        Name of the ``TBranch``.

        Note that ``TBranch`` names are not guaranteed to be unique; it is
        sometimes necessary to address a branch by its
        :py:attr:`~uproot4.behavior.TBranch.TBranch.index`.
        """
        return self.member("fName")

    @property
    def title(self):
        """
        Title of the ``TBranch``.
        """
        return self.member("fTitle")

    @property
    def object_path(self):
        """
        Object path of the ``TBranch``.
        """
        if isinstance(self._parent, uproot4.behaviors.TTree.TTree):
            sep = ":"
        else:
            sep = "/"
        return "{0}{1}{2}".format(self.parent.object_path, sep, self.name)

    @property
    def cache_key(self):
        """
        String that uniquely specifies this ``TBranch`` in its path, to use as
        part of object and array cache keys.
        """
        if self._cache_key is None:
            if isinstance(self._parent, uproot4.behaviors.TTree.TTree):
                sep = ":"
            else:
                sep = "/"
            self._cache_key = "{0}{1}{2}({3})".format(
                self.parent.cache_key, sep, self.name, self.index
            )
        return self._cache_key

    @property
    def index(self):
        """
        Integer position of this ``TBranch`` in its parent's list of branches.

        Useful for cases in which the
        :py:attr:`~uproot4.behavior.TBranch.TBranch.name` is not unique: the
        non-recursive index is always unique.
        """
        for i, branch in enumerate(self.parent.branches):
            if branch is self:
                return i
        else:
            raise AssertionError

    @property
    def interpretation(self):
        """
        The standard :py:class:`~uproot4.interpretation.Interpretation` of this
        ``TBranch`` as an array, derived from
        :py:func:`~uproot4.interpretation.identify.interpretation_of`.

        If no interpretation could be found, the value of this property
        would be a :py:exc:`~uproot4.interpretation.identify.UnknownInterpretation`,
        which is a Python ``Exception``. Since the exception is *returned*,
        rather than *raised*, a branch lacking an interpretation is not a fatal
        error.

        However, any attempt to use this exception object as an
        :py:class:`~uproot4.interpretation.Interpretation` causes it to raise itself:
        attempting to read a branch lacking an interpretation is a fatal error.
        """
        if self._interpretation is None:
            try:
                self._interpretation = uproot4.interpretation.identify.interpretation_of(
                    self, {}
                )
            except uproot4.interpretation.identify.UnknownInterpretation as err:
                self._interpretation = err
        return self._interpretation

    @property
    def typename(self):
        """
        The C++ typename of the ``TBranch``, derived from its
        :py:attr:`~uproot4.behavior.TBranch.TBranch.interpretation`. If the
        interpretation is
        :py:exc:`~uproot4.interpretation.identify.UnknownInterpretation`, the
        typename is ``"unknown"``.
        """
        if self.interpretation is None:
            return "unknown"
        else:
            return self.interpretation.typename

    @property
    def num_entries(self):
        """
        The number of entries in the ``TBranch``, as reported by ``fEntries``.

        In principle, this could disagree with the
        :py:attr:`~uproot4.behaviors.TTree.TTree.num_entries`, which is from the
        ``TTree``'s ``fEntries``.

        The ``TBranch`` also has a ``fEntryNumber``, which ought to be equal to
        the ``TBranch`` and ``TTree``'s ``fEntries``, and the last value of
        :py:attr:`~uproot4.behaviors.TBranch.TBranch.entry_offsets` ought to be
        equal to the number of entries as well.
        """
        return int(self.member("fEntries"))  # or fEntryNumber?

    @property
    def entry_offsets(self):
        """
        The starting and stopping entry numbers for all the ``TBaskets`` in the
        ``TBranch`` as a list of increasing, non-negative integers.

        The number of ``entry_offsets`` in this list of integers is one more
        than the number of ``TBaskets``. The first is ``0`` and the last is
        the number of entries
        (:py:attr:`~uproot4.behaviors.TBranch.TBranch.num_entries`).
        """
        if self._num_normal_baskets == 0:
            out = [0]
        else:
            out = self.member("fBasketEntry")[: self._num_normal_baskets + 1].tolist()
        num_entries_normal = out[-1]

        for basket in self.embedded_baskets:
            out.append(out[-1] + basket.num_entries)

        if out[-1] != self.num_entries and self.interpretation is not None:
            raise ValueError(
                """entries in normal baskets ({0}) plus embedded baskets ({1}) """
                """don't add up to expected number of entries ({2})
in file {3}""".format(
                    num_entries_normal,
                    sum(basket.num_entries for basket in self.embedded_baskets),
                    self.num_entries,
                    self._file.file_path,
                )
            )
        else:
            return out

    @property
    def tree(self):
        """
        The ``TTree`` to which this ``TBranch`` belongs. The branch might be
        deeply nested; this property ascends all the way to the top.
        """
        out = self
        while not isinstance(out, uproot4.behaviors.TTree.TTree):
            out = out.parent
        return out

    @property
    def top_level(self):
        """
        True if the immediate :py:attr:`~uproot4.behaviors.TBranch.TBranch.parent`
        is the ``TTree``; False otherwise.
        """
        return isinstance(self.parent, uproot4.behaviors.TTree.TTree)

    @property
    def streamer(self):
        """
        The ``TStreamerInfo`` or ``TStreamerElement`` for this ``TBranch``,
        which may be None.

        If the :py:attr:`~uproot4.reading.ReadOnlyFile.streamers` have not yet been
        read, this method *might* cause them to be read. (Only
        ``TBranchElements`` can have streamers.)
        """
        if self._streamer is None:
            clean_name = _branch_clean_name.match(self.name).group(2)
            clean_parentname = _branch_clean_parent_name.match(self.name)
            fParentName = self.member("fParentName", none_if_missing=True)
            fClassName = self.member("fClassName", none_if_missing=True)

            if fParentName is not None and fParentName != "":
                matches = self._file.streamers.get(fParentName)

                if matches is not None:
                    streamerinfo = matches[max(matches)]

                    for element in streamerinfo.walk_members(self._file.streamers):
                        if element.name == clean_name:
                            self._streamer = element
                            break

                    if self._streamer is None and clean_parentname is not None:
                        clean_parentname = clean_parentname.group(2)
                        for element in streamerinfo.walk_members(self._file.streamers):
                            if element.name == clean_parentname:
                                substreamerinfo = self._file.streamer_named(
                                    element.typename
                                )
                                for subelement in substreamerinfo.walk_members(
                                    self._file.streamers
                                ):
                                    if subelement.name == clean_name:
                                        self._streamer = subelement
                                        break
                                break

                    if (
                        self.parent.member("fClassName") == "TClonesArray"
                        or self.parent.member("fClonesName", none_if_missing=True)
                        == fParentName
                    ):
                        self._streamer_isTClonesArray = True

            elif fClassName is not None and fClassName != "":
                if fClassName == "TClonesArray":
                    self._streamer_isTClonesArray = True
                    matches = self._file.streamers.get(
                        self.member("fClonesName", none_if_missing=True)
                    )
                else:
                    matches = self._file.streamers.get(fClassName)

                if matches is not None:
                    self._streamer = matches[max(matches)]

        return self._streamer

    @property
    def context(self):
        """
        Auxiliary data used in deserialization. This is a *copy* of the
        ``context`` dict at the time that the ``TBranch`` was deserialized
        with ``"in_TBranch": True`` added.
        """
        return self._context

    @property
    def aliases(self):
        """
        The :py:attr:`~uproot4.behaviors.TTree.TTree.aliases`, which are used as the
        ``aliases`` argument to
        :py:meth:`~uproot4.behaviors.TBranch.HasBranches.arrays`,
        :py:meth:`~uproot4.behaviors.TBranch.HasBranches.iterate`,
        :py:func:`~uproot4.behaviors.TBranch.iterate`, and
        :py:func:`~uproot4.behaviors.TBranch.concatenate` if one is not given.

        The return type is always a dict of str \u2192 str, even if there
        are no aliases (an empty dict).
        """
        return self.tree.aliases

    @property
    def count_branch(self):
        """
        The ``TBranch`` object in which this branch's "counts" reside or None
        if this branch has no "counts".
        """
        leaf = self.count_leaf
        if leaf is None:
            return None
        else:
            return leaf.parent

    @property
    def count_leaf(self):
        """
        The ``TLeaf`` object of this branch's "counts" or None if this branch
        has no "counts".
        """
        leaves = self.member("fLeaves")
        if len(leaves) != 1:
            return None
        return leaves[0].member("fLeafCount")

    @property
    def compressed_bytes(self):
        """
        The number of compressed bytes in all ``TBaskets`` of this ``TBranch``.

        The number of compressed bytes is specified in the ``TBranch`` metadata
        and can be determined without reading any additional data. The
        uncompressed bytes requires reading all of the ``TBasket`` ``TKeys`` at
        least.
        """
        return sum(self.basket_compressed_bytes(i) for i in range(self.num_baskets))

    @property
    def uncompressed_bytes(self):
        """
        The number of uncompressed bytes in all ``TBaskets`` of this ``TBranch``.

        The number of uncompressed bytes cannot be determined without reading a
        ``TKey``, which are small, but may be slow for remote connections because
        of the latency of round-trip requests.
        """
        return sum(self.basket_uncompressed_bytes(i) for i in range(self.num_baskets))

    @property
    def compression_ratio(self):
        """
        The number of uncompressed bytes divided by the number of compressed
        bytes for this ``TBranch``.

        See :py:attr:`~uproot4.behaviors.TBranch.TBranch.compressed_bytes` and
        :py:attr:`~uproot4.behaviors.TBranch.TBranch.uncompressed_bytes`.
        """
        return float(self.uncompressed_bytes) / float(self.compressed_bytes)

    @property
    def num_baskets(self):
        """
        The number of ``TBaskets`` in this ``TBranch``, including both normal
        (free) ``TBaskets`` and
        :py:attr:`~uproot4.behaviors.TBranch.TBranch.embedded_baskets`.
        """
        return self._num_normal_baskets + len(self.embedded_baskets)

    def basket(self, basket_num):
        """
        The :py:class:`~uproot4.models.TBasket.Model_TBasket` at index ``basket_num``.

        It may be a normal (free) ``TBasket`` or one of the
        :py:attr:`~uproot4.behaviors.TBranch.TBranch.embedded_baskets`.
        """
        if 0 <= basket_num < self._num_normal_baskets:
            chunk, cursor = self.basket_chunk_cursor(basket_num)
            return uproot4.models.TBasket.Model_TBasket.read(
                chunk, cursor, {"basket_num": basket_num}, self._file, self._file, self
            )
        elif 0 <= basket_num < self.num_baskets:
            return self.embedded_baskets[basket_num - self._num_normal_baskets]
        else:
            raise IndexError(
                """branch {0} has {1} baskets; cannot get basket {2}
in file {3}""".format(
                    repr(self.name), self.num_baskets, basket_num, self._file.file_path
                )
            )

    def basket_chunk_cursor(self, basket_num):
        """
        Returns a :py:class:`~uproot4.source.chunk.Chunk` and
        :py:class:`~uproot4.source.cursor.Cursor` as a 2-tuple for a given
        ``basket_num``.
        """
        if 0 <= basket_num < self._num_normal_baskets:
            start = self.member("fBasketSeek")[basket_num]
            stop = start + self.basket_compressed_bytes(basket_num)
            cursor = uproot4.source.cursor.Cursor(start)
            chunk = self._file.source.chunk(start, stop)
            return chunk, cursor
        elif 0 <= basket_num < self.num_baskets:
            raise IndexError(
                """branch {0} has {1} normal baskets; cannot get chunk and """
                """cursor for basket {2} because only normal baskets have cursors
in file {3}""".format(
                    repr(self.name),
                    self._num_normal_baskets,
                    basket_num,
                    self._file.file_path,
                )
            )
        else:
            raise IndexError(
                """branch {0} has {1} baskets; cannot get cursor and chunk """
                """for basket {2}
in file {3}""".format(
                    repr(self.name), self.num_baskets, basket_num, self._file.file_path
                )
            )

    def basket_compressed_bytes(self, basket_num):
        """
        The number of compressed bytes for the ``TBasket`` at ``basket_num``.

        The number of compressed bytes is specified in the ``TBranch`` metadata
        and can be determined without reading any additional data. The
        uncompressed bytes requires reading the ``TBasket``'s ``TKey`` at least.
        """
        if 0 <= basket_num < self._num_normal_baskets:
            return int(self.member("fBasketBytes")[basket_num])
        elif 0 <= basket_num < self.num_baskets:
            return self.embedded_baskets[
                basket_num - self._num_normal_baskets
            ].compressed_bytes
        else:
            raise IndexError(
                """branch {0} has {1} baskets; cannot get basket chunk {2}
in file {3}""".format(
                    repr(self.name), self.num_baskets, basket_num, self._file.file_path
                )
            )

    def basket_uncompressed_bytes(self, basket_num):
        """
        The number of uncompressed bytes for the ``TBasket`` at ``basket_num``.

        The number of uncompressed bytes cannot be determined without reading a
        ``TKey``, which are small, but may be slow for remote connections because
        of the latency of round-trip requests.
        """
        if 0 <= basket_num < self.num_baskets:
            return self.basket(basket_num).uncompressed_bytes
        else:
            return self.basket_key(basket_num).data_uncompressed_bytes

    def basket_key(self, basket_num):
        """
        The ``TKey`` (:py:class:`~uproot4.reading.ReadOnlyKey`) for the ``TBasket``
        at ``basket_num``.

        Only applies to normal (free) ``TBaskets``, not
        :py:attr:`~uproot4.behaviors.TBranch.TBranch.embedded_baskets`.
        """
        if 0 <= basket_num < self._num_normal_baskets:
            start = self.member("fBasketSeek")[basket_num]
            stop = start + uproot4.reading._key_format_big.size
            cursor = uproot4.source.cursor.Cursor(start)
            chunk = self._file.source.chunk(start, stop)
            return uproot4.reading.ReadOnlyKey(
                chunk, cursor, {}, self._file, self, read_strings=False
            )
        elif 0 <= basket_num < self.num_baskets:
            raise ValueError(
                "branch {0} basket {1} is an embedded basket, which has no TKey".format(
                    repr(self.name), basket_num
                )
            )
        else:
            raise IndexError(
                """branch {0} has {1} baskets; cannot get basket chunk {2}
in file {3}""".format(
                    repr(self.name), self.num_baskets, basket_num, self._file.file_path
                )
            )

    @property
    def embedded_baskets(self):
        """
        The ``TBaskets`` that are embedded within the ``TBranch`` metadata,
        usually because the ROOT process that was writing the file closed
        unexpectedly.
        """
        if self._embedded_baskets is None:
            cursor = self._cursor_baskets.copy()
            baskets = uproot4.models.TObjArray.Model_TObjArrayOfTBaskets.read(
                self.tree.chunk, cursor, {}, self._file, self._file, self
            )
            with self._embedded_baskets_lock:
                self._embedded_baskets = []
                for basket in baskets:
                    if basket is not None:
                        basket._basket_num = self._num_normal_baskets + len(
                            self._embedded_baskets
                        )
                        self._embedded_baskets.append(basket)

        return self._embedded_baskets

    def entries_to_ranges_or_baskets(self, entry_start, entry_stop):
        """
        Returns a list of (start, stop) integer pairs for free (normal)
        ``TBaskets`` and :py:class:`~uproot4.models.TBasket.Model_TBasket` objects
        for embedded ``TBaskets``.

        The intention is for this list to be updated in place, replacing
        (start, stop) integer pairs with
        :py:class:`~uproot4.models.TBasket.Model_TBasket` objects as they get
        read and interpreted.
        """
        entry_offsets = self.entry_offsets
        out = []
        start = entry_offsets[0]
        for basket_num, stop in enumerate(entry_offsets[1:]):
            if entry_start < stop and start <= entry_stop:
                if 0 <= basket_num < self._num_normal_baskets:
                    byte_start = self.member("fBasketSeek")[basket_num]
                    byte_stop = byte_start + self.basket_compressed_bytes(basket_num)
                    out.append((basket_num, (byte_start, byte_stop)))
                elif 0 <= basket_num < self.num_baskets:
                    out.append((basket_num, self.basket(basket_num)))
                else:
                    raise AssertionError((self.name, basket_num))
            start = stop
        return out

    def postprocess(self, chunk, cursor, context, file):
        fWriteBasket = self.member("fWriteBasket")

        self._lookup = {}
        self._interpretation = None
        self._typename = None
        self._streamer = None
        self._streamer_isTClonesArray = False
        self._cache_key = None
        self._context = dict(context)
        self._context["breadcrumbs"] = ()
        self._context["in_TBranch"] = True

        self._num_normal_baskets = 0
        for i, x in enumerate(self.member("fBasketSeek")):
            if x == 0 or i == fWriteBasket:
                break
            self._num_normal_baskets += 1

        if (
            self.member("fEntries")
            == self.member("fBasketEntry")[self._num_normal_baskets]
        ):
            self._embedded_baskets = []
            self._embedded_baskets_lock = None

        elif self.has_member("fBaskets"):
            self._embedded_baskets = []
            for basket in self.member("fBaskets"):
                if basket is not None:
                    basket._basket_num = self._num_normal_baskets + len(
                        self._embedded_baskets
                    )
                    self._embedded_baskets.append(basket)
            self._embedded_baskets_lock = None

        else:
            self._embedded_baskets = None
            self._embedded_baskets_lock = threading.Lock()

        if "fIOFeatures" in self._parent.members:
            self._tree_iofeatures = self._parent.member("fIOFeatures").member("fIOBits")

        return self

    def debug(
        self,
        entry,
        skip_bytes=None,
        limit_bytes=None,
        dtype=None,
        offset=0,
        stream=sys.stdout,
    ):
        """
        Args:
            entry (int): Entry number to inspect. Note: this debugging routine
                is not applicable to data without entry offsets (nor would it
                be needed).
            skip_bytes (int): Number of bytes to skip before presenting the
                remainder of the ``entry``. May be negative, to examine the
                byte stream before the ``entry``.
            limit_bytes (None or int): Number of bytes to limit the output to.
                A line of debugging output (without any ``offset``) is 20 bytes,
                so multiples of 20 show full lines. If None, everything is
                shown to the end of the ``entry``, which might be large.
            dtype (None, ``numpy.dtype``, or its constructor argument): If None,
                present only the bytes as decimal values (0-255). Otherwise,
                also interpret them as an array of a given NumPy type.
            offset (int): Number of bytes to skip before interpreting a ``dtype``;
                can be helpful if the numerical values are out of phase with
                the first byte shown. Not to be confused with ``skip_bytes``,
                which determines which bytes are shown at all. Any ``offset``
                values that are equivalent modulo ``dtype.itemsize`` show
                equivalent interpretations.
            stream (object with a ``write(str)`` method): Stream to write the
                debugging output to.

        Presents the data for one entry as raw bytes.

        Example output with ``dtype=">f4"`` and ``offset=3``.

        .. code-block::

            --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
            123 123 123  63 140 204 205  64  12 204 205  64  83  51  51  64 140 204 205  64
              {   {   {   ? --- --- ---   @ --- --- ---   @   S   3   3   @ --- --- ---   @
                                    1.1             2.2             3.3             4.4
                --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
                176   0   0  64 211  51  51  64 246 102 102  65  12 204 205  65  30 102 102  66
                --- --- ---   @ ---   3   3   @ ---   f   f   A --- --- ---   A ---   f   f   B
                        5.5             6.6             7.7             8.8             9.9
                --+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+-
                202   0   0  67  74   0   0  67 151 128   0 123 123
                --- --- ---   C   J --- ---   C --- --- ---   {   {
                      101.0           202.0           303.0
        """
        data = self.debug_array(entry)
        chunk = uproot4.source.chunk.Chunk.wrap(self._file.source, data)
        if skip_bytes is None:
            cursor = uproot4.source.cursor.Cursor(0)
        else:
            cursor = uproot4.source.cursor.Cursor(skip_bytes)
        cursor.debug(
            chunk, limit_bytes=limit_bytes, dtype=dtype, offset=offset, stream=stream
        )

    def debug_array(self, entry, skip_bytes=0, dtype=numpy.dtype("u1")):
        """
        Args:
            entry (int): Entry number to inspect. Note: this debugging routine
                is not applicable to data without entry offsets (nor would it
                be needed).
            skip_bytes (int): Number of bytes to skip before presenting the
                remainder of the ``entry``. May be negative, to examine the
                byte stream before the ``entry``.
            dtype (``numpy.dtype`` or its constructor argument): Data type in
                which to interpret the data. (The size of the array returned is
                truncated to this ``dtype.itemsize``.)

        Like :py:meth:`~uproot4.behaviors.TBranch.TBranch.debug`, but returns a
        NumPy array for further inspection.
        """
        dtype = numpy.dtype(dtype)
        interpretation = uproot4.interpretation.jagged.AsJagged(
            uproot4.interpretation.numerical.AsDtype("u1")
        )
        out = self.array(
            interpretation, entry_start=entry, entry_stop=entry + 1, library="np"
        )[0][skip_bytes:]
        return out[: (len(out) // dtype.itemsize) * dtype.itemsize].view(dtype)


_regularize_files_braces = re.compile(r"{([^}]*,)*([^}]*)}")


def _regularize_files_inner(files, parse_colon):
    files2 = uproot4._util.regularize_path(files)

    if uproot4._util.isstr(files2) and not uproot4._util.isstr(files):
        parse_colon = False
        files = files2

    if uproot4._util.isstr(files):
        if parse_colon:
            file_path, object_path = uproot4._util.file_object_path_split(files)
        else:
            file_path, object_path = files, None

        parsed_url = urlparse(file_path)

        if parsed_url.scheme.upper() in uproot4._util._remote_schemes:
            yield file_path, object_path

        else:
            expanded = os.path.expanduser(file_path)
            matches = list(_regularize_files_braces.finditer(expanded))
            if len(matches) == 0:
                results = [expanded]
            else:
                results = []
                for combination in itertools.product(
                    *[match.group(0)[1:-1].split(",") for match in matches]
                ):
                    tmp = expanded
                    for c, m in list(zip(combination, matches))[::-1]:
                        tmp = tmp[: m.span()[0]] + c + tmp[m.span()[1] :]
                    results.append(tmp)

            seen = set()
            for result in results:
                for match in glob.glob(result):
                    if match not in seen:
                        yield match, object_path
                        seen.add(match)

    elif isinstance(files, HasBranches):
        yield files, None

    elif isinstance(files, dict):
        for key, object_path in files.items():
            for file_path, _ in _regularize_files_inner(key, False):
                yield file_path, object_path

    elif isinstance(files, Iterable):
        for file in files:
            for file_path, object_path in _regularize_files_inner(file, parse_colon):
                yield file_path, object_path

    else:
        raise TypeError(
            "'files' must be a file path/URL (string or Path), possibly with "
            "a glob pattern (for local files), a dict of "
            "{{path/URL: TTree/TBranch name}}, actual TTree/TBranch objects, or "
            "an iterable of such things, not {0}".format(repr(files))
        )


def _regularize_files(files):
    out = []
    seen = set()
    for file_path, object_path in _regularize_files_inner(files, True):
        if uproot4._util.isstr(file_path):
            if (file_path, object_path) not in seen:
                out.append((file_path, object_path))
                seen.add((file_path, object_path))
        else:
            out.append((file_path, object_path))

    if len(out) == 0:
        uproot4._util._file_not_found(files)

    return out


def _regularize_object_path(
    file_path, object_path, custom_classes, allow_missing, options
):
    if isinstance(file_path, HasBranches):
        return _NoClose(file_path)

    else:
        file = uproot4.reading.ReadOnlyFile(
            file_path,
            object_cache=None,
            array_cache=None,
            custom_classes=custom_classes,
            **options  # NOTE: a comma after **options breaks Python 2
        ).root_directory
        if object_path is None:
            trees = [k for k, v in file.classnames().items() if v == "TTree"]
            if len(trees) == 0:
                if allow_missing:
                    return None
                else:
                    raise ValueError(
                        """no TTrees found
in file {0}""".format(
                            file_path
                        )
                    )
            elif len(trees) == 1:
                return file[trees[0]]
            else:
                raise ValueError(
                    """TTree object paths must be specified in the 'files' """
                    """as {{\"filenames*.root\": \"path\"}} if any files have """
                    """more than one TTree

    TTrees: {0}

in file {1}""".format(
                        ", ".join(repr(x) for x in trees), file_path
                    )
                )

        else:
            if allow_missing and object_path not in file:
                return None
            return file[object_path]


def _get_recursive(hasbranches, where):
    for branch in hasbranches.branches:
        if branch.name == where:
            return branch
        got = _get_recursive(branch, where)
        if got is not None:
            return got
    else:
        return None


def _regularize_entries_start_stop(num_entries, entry_start, entry_stop):
    if entry_start is None:
        entry_start = 0
    elif entry_start < 0:
        entry_start += num_entries
    entry_start = min(num_entries, max(0, entry_start))

    if entry_stop is None:
        entry_stop = num_entries
    elif entry_stop < 0:
        entry_stop += num_entries
    entry_stop = min(num_entries, max(0, entry_stop))

    if entry_stop < entry_start:
        entry_stop = entry_start

    return int(entry_start), int(entry_stop)


def _regularize_executors(decompression_executor, interpretation_executor):
    if decompression_executor is None:
        decompression_executor = uproot4.decompression_executor
    if interpretation_executor is None:
        interpretation_executor = uproot4.interpretation_executor
    return decompression_executor, interpretation_executor


def _regularize_array_cache(array_cache, file):
    if isinstance(array_cache, MutableMapping):
        return array_cache
    elif uproot4._util.isstr(array_cache) and array_cache == "inherit":
        return file._array_cache
    elif array_cache is None:
        return None
    elif uproot4._util.isint(array_cache) or uproot4._util.isstr(array_cache):
        return uproot4.cache.LRUArrayCache(array_cache)
    else:
        raise TypeError("array_cache must be None, a MutableMapping, or a memory size")


def _regularize_aliases(hasbranches, aliases):
    if aliases is None:
        return hasbranches.aliases
    else:
        new_aliases = dict(hasbranches.aliases)
        new_aliases.update(aliases)
        return new_aliases


def _regularize_interpretation(interpretation):
    if isinstance(interpretation, uproot4.interpretation.Interpretation):
        return interpretation
    elif isinstance(interpretation, numpy.dtype):
        return uproot4.interpretation.numerical.AsDtype(interpretation)
    else:
        dtype = numpy.dtype(interpretation)
        dtype = dtype.newbyteorder(">")
        return uproot4.interpretation.numerical.AsDtype(interpretation)


def _regularize_branchname(
    hasbranches,
    branchname,
    branch,
    interpretation,
    get_from_cache,
    arrays,
    expression_context,
    branchid_interpretation,
    is_primary,
    is_cut,
):
    got = get_from_cache(branchname, interpretation)
    if got is not None:
        arrays[branch.cache_key] = got

    is_jagged = isinstance(interpretation, uproot4.interpretation.jagged.AsJagged)

    if branch.cache_key in branchid_interpretation:
        if (
            branchid_interpretation[branch.cache_key].cache_key
            != interpretation.cache_key
        ):
            raise ValueError(
                "a branch cannot be loaded with multiple interpretations: "
                "{0} and {1}".format(
                    repr(branchid_interpretation[branch.cache_key]),
                    repr(interpretation),
                )
            )
        else:
            expression_context.append(
                (
                    branchname,
                    {
                        "is_primary": is_primary,
                        "is_cut": is_cut,
                        "is_duplicate": True,
                        "is_jagged": is_jagged,
                        "branch": branch,
                    },
                )
            )

    else:
        expression_context.append(
            (
                branchname,
                {
                    "is_primary": is_primary,
                    "is_cut": is_cut,
                    "is_duplicate": False,
                    "is_jagged": is_jagged,
                    "branch": branch,
                },
            )
        )
        branchid_interpretation[branch.cache_key] = interpretation


def _regularize_expression(
    hasbranches,
    expression,
    keys,
    aliases,
    language,
    get_from_cache,
    arrays,
    expression_context,
    branchid_interpretation,
    symbol_path,
    is_cut,
):
    is_primary = symbol_path == ()

    branch = hasbranches.get(expression)
    if branch is not None:
        _regularize_branchname(
            hasbranches,
            expression,
            branch,
            branch.interpretation,
            get_from_cache,
            arrays,
            expression_context,
            branchid_interpretation,
            is_primary,
            is_cut,
        )

    else:
        if expression in aliases:
            to_compute = aliases[expression]
        else:
            to_compute = expression

        is_jagged = False
        for symbol in language.free_symbols(
            to_compute,
            keys,
            aliases,
            hasbranches.file.file_path,
            hasbranches.object_path,
        ):
            if symbol in symbol_path:
                raise ValueError(
                    """symbol {0} is recursively defined with aliases:

    {1}

in file {2} at {3}""".format(
                        repr(symbol),
                        "\n    ".join(
                            "{0}: {1}".format(k, v) for k, v in aliases.items()
                        ),
                        hasbranches.file.file_path,
                        hasbranches.object_path,
                    )
                )

            _regularize_expression(
                hasbranches,
                symbol,
                keys,
                aliases,
                language,
                get_from_cache,
                arrays,
                expression_context,
                branchid_interpretation,
                symbol_path + (symbol,),
                False,
            )
            if expression_context[-1][1]["is_jagged"]:
                is_jagged = True

        expression_context.append(
            (
                expression,
                {"is_primary": is_primary, "is_cut": is_cut, "is_jagged": is_jagged},
            )
        )


def _regularize_expressions(
    hasbranches,
    expressions,
    cut,
    filter_name,
    filter_typename,
    filter_branch,
    keys,
    aliases,
    language,
    get_from_cache,
):
    arrays = {}
    expression_context = []
    branchid_interpretation = {}

    if expressions is None:
        for branchname, branch in hasbranches.iteritems(
            filter_name=filter_name,
            filter_typename=filter_typename,
            filter_branch=filter_branch,
            recursive=True,
            full_paths=False,
        ):
            if not isinstance(
                branch.interpretation,
                (
                    uproot4.interpretation.identify.UnknownInterpretation,
                    uproot4.interpretation.grouped.AsGrouped,
                ),
            ):
                _regularize_branchname(
                    hasbranches,
                    branchname,
                    branch,
                    branch.interpretation,
                    get_from_cache,
                    arrays,
                    expression_context,
                    branchid_interpretation,
                    True,
                    False,
                )

    elif uproot4._util.isstr(expressions):
        _regularize_expression(
            hasbranches,
            expressions,
            keys,
            aliases,
            language,
            get_from_cache,
            arrays,
            expression_context,
            branchid_interpretation,
            (),
            False,
        )

    elif isinstance(expressions, Iterable):
        if isinstance(expressions, dict):
            items = expressions.items()
        else:
            items = []
            for expression in expressions:
                if uproot4._util.isstr(expression):
                    items.append((expression, None))
                elif isinstance(expression, tuple) and len(expression) == 2:
                    items.append(expression)
                else:
                    raise TypeError(
                        "iterable of expressions must be strings or "
                        "name, Interpretation pairs (length-2 tuples), not "
                        + repr(expression)
                    )

        for expression, interp in items:
            if interp is None:
                _regularize_expression(
                    hasbranches,
                    expression,
                    keys,
                    aliases,
                    language,
                    get_from_cache,
                    arrays,
                    expression_context,
                    branchid_interpretation,
                    (),
                    False,
                )
            else:
                branch = hasbranches[expression]
                interp = _regularize_interpretation(interp)
                _regularize_branchname(
                    hasbranches,
                    expression,
                    branch,
                    interp,
                    get_from_cache,
                    arrays,
                    expression_context,
                    branchid_interpretation,
                    True,
                    False,
                )

    else:
        raise TypeError(
            "expressions must be None (for all branches), a string (single "
            "branch or expression), a list of strings (multiple), or a dict "
            "or list of name, Interpretation pairs (branch names and their "
            "new Interpretation), not {0}".format(repr(expressions))
        )

    if cut is None:
        pass
    elif uproot4._util.isstr(cut):
        _regularize_expression(
            hasbranches,
            cut,
            keys,
            aliases,
            language,
            get_from_cache,
            arrays,
            expression_context,
            branchid_interpretation,
            (),
            True,
        )

    return arrays, expression_context, branchid_interpretation


def _ranges_or_baskets_to_arrays(
    hasbranches,
    ranges_or_baskets,
    branchid_interpretation,
    entry_start,
    entry_stop,
    decompression_executor,
    interpretation_executor,
    library,
    arrays,
):
    notifications = queue.Queue()

    branchid_arrays = {}
    branchid_num_baskets = {}
    ranges = []
    range_args = {}
    range_original_index = {}
    original_index = 0

    for branch, basket_num, range_or_basket in ranges_or_baskets:
        if branch.cache_key not in branchid_arrays:
            branchid_arrays[branch.cache_key] = {}
            branchid_num_baskets[branch.cache_key] = 0
        branchid_num_baskets[branch.cache_key] += 1

        if isinstance(range_or_basket, tuple) and len(range_or_basket) == 2:
            range_or_basket = (int(range_or_basket[0]), int(range_or_basket[1]))
            ranges.append(range_or_basket)
            range_args[range_or_basket] = (branch, basket_num)
            range_original_index[range_or_basket] = original_index
        else:
            notifications.put(range_or_basket)

        original_index += 1

    hasbranches._file.source.chunks(ranges, notifications=notifications)

    def replace(ranges_or_baskets, original_index, basket):
        branch, basket_num, range_or_basket = ranges_or_baskets[original_index]
        ranges_or_baskets[original_index] = branch, basket_num, basket

    def chunk_to_basket(chunk, branch, basket_num):
        try:
            cursor = uproot4.source.cursor.Cursor(chunk.start)
            basket = uproot4.models.TBasket.Model_TBasket.read(
                chunk,
                cursor,
                {"basket_num": basket_num},
                hasbranches._file,
                hasbranches._file,
                branch,
            )
            original_index = range_original_index[(chunk.start, chunk.stop)]
            replace(ranges_or_baskets, original_index, basket)
        except Exception:
            notifications.put(sys.exc_info())
        else:
            notifications.put(basket)

    def basket_to_array(basket):
        try:
            assert basket.basket_num is not None
            branch = basket.parent
            interpretation = branchid_interpretation[branch.cache_key]
            basket_arrays = branchid_arrays[branch.cache_key]

            basket_arrays[basket.basket_num] = interpretation.basket_array(
                basket.data,
                basket.byte_offsets,
                basket,
                branch,
                branch.context,
                basket.member("fKeylen"),
                library,
            )
            if basket.num_entries != len(basket_arrays[basket.basket_num]):
                raise ValueError(
                    """basket {0} in tree/branch {1} has the wrong number of entries """
                    """(expected {2}, obtained {3}) when interpreted as {4}
    in file {5}""".format(
                        basket.basket_num,
                        branch.object_path,
                        basket.num_entries,
                        len(basket_arrays[basket.basket_num]),
                        interpretation,
                        branch.file.file_path,
                    )
                )

            if len(basket_arrays) == branchid_num_baskets[branch.cache_key]:
                arrays[branch.cache_key] = interpretation.final_array(
                    basket_arrays,
                    entry_start,
                    entry_stop,
                    branch.entry_offsets,
                    library,
                    branch,
                )
        except Exception:
            notifications.put(sys.exc_info())
        else:
            notifications.put(None)

    while len(arrays) < len(branchid_interpretation):
        obj = notifications.get()

        if isinstance(obj, uproot4.source.chunk.Chunk):
            chunk = obj
            args = range_args[(chunk.start, chunk.stop)]
            decompression_executor.submit(chunk_to_basket, chunk, *args)

        elif isinstance(obj, uproot4.models.TBasket.Model_TBasket):
            basket = obj
            interpretation_executor.submit(basket_to_array, basket)

        elif obj is None:
            pass

        elif isinstance(obj, tuple) and len(obj) == 3:
            uproot4.source.futures.delayed_raise(*obj)

        else:
            raise AssertionError(obj)


def _hasbranches_num_entries_for(
    hasbranches, target_num_bytes, entry_start, entry_stop, branchid_interpretation
):
    total_bytes = 0.0
    for branch in hasbranches.itervalues(recursive=True):
        if branch.cache_key in branchid_interpretation:
            entry_offsets = branch.entry_offsets
            start = entry_offsets[0]
            for basket_num, stop in enumerate(entry_offsets[1:]):
                if entry_start < stop and start <= entry_stop:
                    total_bytes += branch.basket_compressed_bytes(basket_num)
                start = stop

    total_entries = entry_stop - entry_start
    num_entries = int(round(target_num_bytes * total_entries / total_bytes))
    if num_entries <= 0:
        return 1
    else:
        return num_entries


def _regularize_step_size(
    hasbranches, step_size, entry_start, entry_stop, branchid_interpretation
):
    if uproot4._util.isint(step_size):
        return step_size
    target_num_bytes = uproot4._util.memory_size(
        step_size,
        "number of entries or memory size string with units "
        "(such as '100 MB') required, not {0}".format(repr(step_size)),
    )
    return _hasbranches_num_entries_for(
        hasbranches, target_num_bytes, entry_start, entry_stop, branchid_interpretation
    )


class _WrapDict(MutableMapping):
    def __init__(self, dict):
        self.dict = dict

    def __str__(self):
        return str(self.dict)

    def __repr__(self):
        return repr(self.dict)

    def __getitem__(self, where):
        return self.dict[where]

    def __setitem__(self, where, what):
        self.dict[where] = what

    def __delitem__(self, where):
        del self.dict[where]

    def __iter__(self, where):
        for x in self.dict:
            yield x

    def __len__(self):
        return len(self.dict)
