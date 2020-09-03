# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines the entry-point for opening a file, :doc:`uproot4.reading.open`, and
the classes that are too fundamental to be models:
:doc:`uproot4.reading.ReadOnlyFile` (``TFile``),
:doc:`uproot4.reading.ReadOnlyDirectory` (``TDirectory`` or ``TDirectoryFile``),
and :doc:`uproot4.reading.ReadOnlyKey` (``TKey``).
"""

from __future__ import absolute_import

import sys
import struct
import uuid

try:
    from collections.abc import Mapping
    from collections.abc import MutableMapping
except ImportError:
    from collections import Mapping
    from collections import MutableMapping

import uproot4.compression
import uproot4.cache
import uproot4.source.cursor
import uproot4.source.chunk
import uproot4.source.http
import uproot4.source.xrootd
import uproot4.streamers
import uproot4.model
import uproot4.behaviors.TBranch
import uproot4._util
from uproot4._util import no_filter


def open(
    path,
    object_cache=100,
    array_cache="100 MB",
    custom_classes=None,
    **options  # NOTE: a comma after **options breaks Python 2
):
    """
    Args:
        path (str or ``pathlib.Path``): The filesystem path or remote URL of
            the file to open. If a string, it may be followed by a colon (``:``)
            and an object path within the ROOT file, to return an object,
            rather than a file. Path objects are interpreted strictly as
            filesystem paths or URLs.
            Examples: "rel/file.root", "C:\abs\file.root", "http://where/what.root",
                      "rel/file.root:tdirectory/ttree",
                      Path("rel:/file.root"), Path("/abs/path:stuff.root")
        object_cache (None, MutableMapping, or int): Cache of objects drawn
            from ROOT directories (e.g histograms, TTrees, other directories);
            if None, do not use a cache; if an int, create a new cache of this
            size.
        array_cache (None, MutableMapping, or memory size): Cache of arrays
            drawn from TTrees; if None, do not use a cache; if a memory size,
            create a new cache of this size.
        custom_classes (None or MutableMapping): If None, classes come from
            uproot4.classes; otherwise, a container of class definitions that
            is both used to fill with new classes and search for dependencies.
        options: see below.

    Opens a ROOT file, possibly through a remote protocol.

    Options (type; default):

        * file_handler (:doc:`uproot4.source.chunk.Source` class; :doc:`uproot4.source.file.MemmapSource`)
        * xrootd_handler (:doc:`uproot4.source.chunk.Source` class; :doc:`uproot4.source.xrootd.XRootDSource`)
        * http_handler (:doc:`uproot4.source.chunk.Source` class; :doc:`uproot4.source.http.HTTPSource`)
        * timeout (float for HTTP, int for XRootD; 30)
        * max_num_elements (None or int; None)
        * num_workers (int; 1)
        * num_fallback_workers (int; 10)
        * begin_chunk_size (memory_size; 512)
        * minimal_ttree_metadata (bool; True)
    """

    if isinstance(path, dict) and len(path) == 1:
        ((file_path, object_path),) = path.items()

    elif uproot4._util.isstr(path):
        file_path, object_path = uproot4._util.file_object_path_split(path)

    else:
        file_path = path
        object_path = None

    file_path = uproot4._util.regularize_path(file_path)

    if not uproot4._util.isstr(file_path):
        raise ValueError(
            "'path' must be a string, pathlib.Path, or a length-1 dict of "
            "{{file_path: object_path}}, not {0}".format(repr(path))
        )

    file = ReadOnlyFile(
        file_path,
        object_cache=object_cache,
        array_cache=array_cache,
        custom_classes=custom_classes,
        **options  # NOTE: a comma after **options breaks Python 2
    )

    if object_path is None:
        return file.root_directory
    else:
        return file.root_directory[object_path]


open.defaults = {
    "file_handler": uproot4.source.file.MemmapSource,
    "xrootd_handler": uproot4.source.xrootd.XRootDSource,
    "http_handler": uproot4.source.http.HTTPSource,
    "timeout": 30,
    "max_num_elements": None,
    "num_workers": 1,
    "num_fallback_workers": 10,
    "begin_chunk_size": 512,
    "minimal_ttree_metadata": True,
}


must_be_attached = [
    "TROOT",
    "TDirectory",
    "TDirectoryFile",
    "RooWorkspace::WSDir",
    "TTree",
    "TChain",
    "TProofChain",
    "THbookTree",
    "TNtuple",
    "TNtupleD",
    "TTreeSQL",
]


class CommonFileMethods(object):
    """
    Abstract class for :doc:`uproot4.reading.ReadOnlyFile` and
    :doc:`uproot4.reading.DetachedFile`. The latter is a placeholder for file
    information, such as the :doc:`uproot4.reading.CommonFileMethods.file_path`
    used in many error messages, without holding a reference to the active
    :doc:`uproot4.source.chunk.Source`.

    This allows the file to be closed and deleted while objects that were read
    from it still exist. Also, only objects that hold detached file references,
    rather than active ones, can be pickled.

    The (unpickleable) objects that must hold a reference to an active
    :doc:`uproot4.reading.ReadOnlyFile` are listed by C++ (decoded) classname
    in ``uproot4.must_be_attached``.
    """
    @property
    def file_path(self):
        """
        The original path to the file (converted to ``str`` if it was originally
        a ``pathlib.Path``).
        """
        return self._file_path

    @property
    def options(self):
        """
        The dict of ``options`` originally passed to the
        :doc:`uproot4.reading.ReadOnlyFile` constructor.
        """
        return self._options

    @property
    def root_version(self):
        """
        Version of ROOT used to write the file as a string.

        See :doc:`uproot4.reading.CommonFileMethods.root_version_tuple` and
        :doc:`uproot4.reading.CommonFileMethods.fVersion`.
        """
        return "{0}.{1:02d}/{2:02d}".format(*self.root_version_tuple)

    @property
    def root_version_tuple(self):
        """
        Version of ROOT used to write teh file as a tuple.

        See :doc:`uproot4.reading.CommonFileMethods.root_version` and
        :doc:`uproot4.reading.CommonFileMethods.fVersion`.
        """
        version = self._fVersion
        if version >= 1000000:
            version -= 1000000

        major = version // 10000
        version %= 10000
        minor = version // 100
        version %= 100

        return major, minor, version

    @property
    def is_64bit(self):
        """
        True if the ROOT file contains 64-bit seek points; False otherwise.

        A file that is larger than 4 GiB must have 64-bit seek points, though
        any file might.
        """
        return self._fVersion >= 1000000

    @property
    def compression(self):
        """
        A :doc:`uproot4.compression.Compression` object describing the
        compression setting for the ROOT file.

        Note that different objects (even different ``TBranches`` within a
        ``TTree``) can be compressed differently, so this file-level
        compression is only a strong hint of how the objects are likely to
        be compressed.

        For some versions of ROOT ``TStreamerInfo`` is always compressed with
        :doc:`uproot4.compression.ZLIB`, even if the compression is set to a
        different algorithm.

        See :doc:`uproot4.reading.CommonFileMethods.fCompress`.
        """
        return uproot4.compression.Compression.from_code(self._fCompress)

    @property
    def hex_uuid(self):
        """
        The unique identifier (UUID) of the ROOT file expressed as a hexadecimal
        string.

        See :doc:`uproot4.reading.CommonFileMethods.uuid` and
        :doc:`uproot4.reading.CommonFileMethods.fUUID`.
        """
        if uproot4._util.py2:
            out = "".join("{0:02x}".format(ord(x)) for x in self._fUUID)
        else:
            out = "".join("{0:02x}".format(x) for x in self._fUUID)
        return "-".join([out[0:8], out[8:12], out[12:16], out[16:20], out[20:32]])

    @property
    def uuid(self):
        """
        The unique identifier (UUID) of the ROOT file expressed as a Python
        ``uuid.UUID`` object.

        See :doc:`uproot4.reading.CommonFileMethods.hex_uuid` and
        :doc:`uproot4.reading.CommonFileMethods.fUUID`.
        """
        return uuid.UUID(self.hex_uuid.replace("-", ""))

    @property
    def fVersion(self):
        """
        Raw version information for the ROOT file; this number is used to derive
        :doc:`uproot4.reading.CommonFileMethods.root_version`,
        :doc:`uproot4.reading.CommonFileMethods.root_version_tuple`, and
        :doc:`uproot4.reading.CommonFileMethods.is_64bit`.
        """
        return self._fVersion

    @property
    def fBEGIN(self):
        """
        The seek point (int) for the first data record, past the TFile header.

        Usually 100.
        """
        return self._fBEGIN

    @property
    def fEND(self):
        """
        The seek point (int) to the last free word at the end of the ROOT file.
        """
        return self._fEND

    @property
    def fSeekFree(self):
        """
        The seek point (int) to the ``TFree`` data, for managing empty spaces
        in a ROOT file (filesystem-like fragmentation).
        """
        return self._fSeekFree

    @property
    def fNbytesFree(self):
        """
        The number of bytes in the ``TFree` data, for managing empty spaces
        in a ROOT file (filesystem-like fragmentation).
        """
        return self._fNbytesFree

    @property
    def nfree(self):
        """
        The number of objects in the ``TFree`` data, for managing empty spaces
        in a ROOT file (filesystem-like fragmentation).
        """
        return self._nfree

    @property
    def fNbytesName(self):
        """
        The number of bytes in the filename (``TNamed``) that is embedded in
        the ROOT file.
        """
        return self._fNbytesName

    @property
    def fUnits(self):
        """
        Number of bytes in the serialization of file seek points.

        Usually 4 or 8.
        """
        return self._fUnits

    @property
    def fCompress(self):
        """
        The raw integer describing the compression setting for the ROOT file.

        Note that different objects (even different ``TBranches`` within a
        ``TTree``) can be compressed differently, so this file-level
        compression is only a strong hint of how the objects are likely to
        be compressed.

        For some versions of ROOT ``TStreamerInfo`` is always compressed with
        :doc:`uproot4.compression.ZLIB`, even if the compression is set to a
        different algorithm.

        See :doc:`uproot4.reading.CommonFileMethods.compression`.
        """
        return self._fCompress

    @property
    def fSeekInfo(self):
        """
        The seek point (int) to the ``TStreamerInfo`` data, where
        :doc:`uproot4.reading.ReadOnlyFile.streamers` are located.
        """
        return self._fSeekInfo

    @property
    def fNbytesInfo(self):
        """
        The number of bytes in the ``TStreamerInfo`` data, where
        :doc:`uproot4.reading.ReadOnlyFile.streamers` are located.
        """
        return self._fNbytesInfo

    @property
    def fUUID(self):
        """
        The unique identifier (UUID) of the ROOT file as a raw bytestring
        (Python ``bytes``).

        See :doc:`uproot4.reading.CommonFileMethods.hex_uuid` and
        :doc:`uproot4.reading.CommonFileMethods.uuid`.
        """
        return self._fUUID


class DetachedFile(CommonFileMethods):
    """
    Args:
        file (:doc:`uproot4.reading.ReadOnlyFile`): The active file object to
            convert into a detached file.

    A placeholder for a :doc:`uproot4.reading.ReadOnlyFile` with useful
    information, such as the :doc:`uproot4.reading.CommonFileMethods.file_path`
    used in many error messages, without holding a reference to the active
    :doc:`uproot4.source.chunk.Source`.

    This allows the file to be closed and deleted while objects that were read
    from it still exist. Also, only objects that hold detached file references,
    rather than active ones, can be pickled.

    The (unpickleable) objects that must hold a reference to an active
    :doc:`uproot4.reading.ReadOnlyFile` are listed by C++ (decoded) classname
    in ``uproot4.must_be_attached``.
    """
    def __init__(self, file):
        self._file_path = file._file_path
        self._options = file._options
        self._fVersion = file._fVersion
        self._fBEGIN = file._fBEGIN
        self._fEND = file._fEND
        self._fSeekFree = file._fSeekFree
        self._fNbytesFree = file._fNbytesFree
        self._nfree = file._nfree
        self._fNbytesName = file._fNbytesName
        self._fUnits = file._fUnits
        self._fCompress = file._fCompress
        self._fSeekInfo = file._fSeekInfo
        self._fNbytesInfo = file._fNbytesInfo
        self._fUUID_version = file._fUUID_version
        self._fUUID = file._fUUID


_file_header_fields_small = struct.Struct(">4siiiiiiiBiiiH16s")
_file_header_fields_big = struct.Struct(">4siiqqiiiBiqiH16s")


class ReadOnlyFile(CommonFileMethods):
    """
    Args:
        file_path (str or ``pathlib.Path``): The filesystem path or remote URL
            of the file to open. Unlike :doc:`uproot4.reading.open`, it cannot
            be followed by a colon (``:``) and an object path within the ROOT
            file.
        object_cache (None, MutableMapping, or int): Cache of objects drawn
            from ROOT directories (e.g histograms, TTrees, other directories);
            if None, do not use a cache; if an int, create a new cache of this
            size.
        array_cache (None, MutableMapping, or memory size): Cache of arrays
            drawn from TTrees; if None, do not use a cache; if a memory size,
            create a new cache of this size.
        custom_classes (None or MutableMapping): If None, classes come from
            uproot4.classes; otherwise, a container of class definitions that
            is both used to fill with new classes and search for dependencies.
        options: see below.

    Handle to an open ROOT file, the way to access data in ``TDirectories``
    (:doc:`uproot4.reading.ReadOnlyDirectory`) and create new classes from
    ``TStreamerInfo`` (:doc:`uproot4.reading.ReadOnlyFile.streamers`).

    All objects derived from ROOT files have a pointer back to the file,
    though this is a :doc:`uproot4.reading.DetachedFile` (no active connection,
    cannot read more data) if the object's :doc:`uproot4.model.Model.classname`
    is not in ``uproot4.reading.must_be_attached``: objects that can read
    more data and need to have an active connection (like ``TTree``,
    ``TBranch``, and ``TDirectory``).

    Options (type; default):

        * file_handler (:doc:`uproot4.source.chunk.Source` class; :doc:`uproot4.source.file.MemmapSource`)
        * xrootd_handler (:doc:`uproot4.source.chunk.Source` class; :doc:`uproot4.source.xrootd.XRootDSource`)
        * http_handler (:doc:`uproot4.source.chunk.Source` class; :doc:`uproot4.source.http.HTTPSource`)
        * timeout (float for HTTP, int for XRootD; 30)
        * max_num_elements (None or int; None)
        * num_workers (int; 1)
        * num_fallback_workers (int; 10)
        * begin_chunk_size (memory_size; 512)
        * minimal_ttree_metadata (bool; True)
    """
    def __init__(
        self,
        file_path,
        object_cache=100,
        array_cache="100 MB",
        custom_classes=None,
        **options  # NOTE: a comma after **options breaks Python 2
    ):
        self._file_path = file_path
        self.object_cache = object_cache
        self.array_cache = array_cache
        self.custom_classes = custom_classes

        self._options = dict(open.defaults)
        self._options.update(options)
        for option in ["begin_chunk_size"]:
            self._options[option] = uproot4._util.memory_size(self._options[option])

        self._streamers = None
        self._streamer_rules = None

        self.hook_before_create_source()

        Source, file_path = uproot4._util.file_path_to_source_class(
            file_path, self._options
        )
        self._source = Source(
            file_path, **self._options  # NOTE: a comma after **options breaks Python 2
        )

        self.hook_before_get_chunks()

        if self._options["begin_chunk_size"] < _file_header_fields_big.size:
            raise ValueError(
                "begin_chunk_size={0} is not enough to read the TFile header ({1})".format(
                    self._options["begin_chunk_size"],
                    self._file_header_fields_big.size,
                )
            )

        self._begin_chunk = self._source.chunk(0, self._options["begin_chunk_size"])

        self.hook_before_interpret()

        (
            magic,
            self._fVersion,
            self._fBEGIN,
            self._fEND,
            self._fSeekFree,
            self._fNbytesFree,
            self._nfree,
            self._fNbytesName,
            self._fUnits,
            self._fCompress,
            self._fSeekInfo,
            self._fNbytesInfo,
            self._fUUID_version,
            self._fUUID,
        ) = uproot4.source.cursor.Cursor(0).fields(
            self._begin_chunk, _file_header_fields_small, {}
        )

        if self.is_64bit:
            (
                magic,
                self._fVersion,
                self._fBEGIN,
                self._fEND,
                self._fSeekFree,
                self._fNbytesFree,
                self._nfree,
                self._fNbytesName,
                self._fUnits,
                self._fCompress,
                self._fSeekInfo,
                self._fNbytesInfo,
                self._fUUID_version,
                self._fUUID,
            ) = uproot4.source.cursor.Cursor(0).fields(
                self._begin_chunk, _file_header_fields_big, {}
            )

        self.hook_after_interpret(magic=magic)

        if magic != b"root":
            raise ValueError(
                """not a ROOT file: first four bytes are {0}
in file {1}""".format(
                    repr(magic), file_path
                )
            )

    def __repr__(self):
        return "<ReadOnlyFile {0} at 0x{1:012x}>".format(
            repr(self._file_path), id(self)
        )

    @property
    def detached(self):
        """
        A :doc:`uproot4.reading.DetachedFile` version of this file.
        """
        return DetachedFile(self)

    def close(self):
        """
        Explicitly close the file.

        (Files can also be closed with the Python ``with`` statement, as context
        managers.)

        After closing, new objects and classes cannot be extracted from the file,
        but objects with :doc:`uproot4.reading.DetachedFile` references instead
        of :doc:`uproot4.reading.ReadOnlyFile` that are still in the
        :doc:`uproot4.reading.ReadOnlyFile.object_cache` would still be
        accessible.
        """
        self._source.close()

    @property
    def closed(self):
        """
        True if the file has been closed; False otherwise.

        The file may have been closed explicitly with
        :doc:`uproot4.reading.ReadOnlyFile.close` or implicitly in the Python
        ``with`` statement, as a context manager.

        After closing, new objects and classes cannot be extracted from the file,
        but objects with :doc:`uproot4.reading.DetachedFile` references instead
        of :doc:`uproot4.reading.ReadOnlyFile` that are still in the
        :doc:`uproot4.reading.ReadOnlyFile.object_cache` would still be
        accessible.
        """
        return self._source.closed

    def __enter__(self):
        self._source.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._source.__exit__(exception_type, exception_value, traceback)

    @property
    def source(self):
        """
        The :doc:`uproot4.source.chunk.Source` associated with this file, which
        is the "physical layer" that knows how to communicate with local file
        systems or through remote protocols like HTTP(S) or XRootD, but does not
        know what the bytes mean.
        """
        return self._source

    @property
    def object_cache(self):
        """
        A cache used to hold previously extracted objects, so that code like

        .. code-block:: python

            h = my_file["histogram"]
            h = my_file["histogram"]
            h = my_file["histogram"]

        only reads the ``"histogram"`` once.

        Any Python ``MutableMapping`` can be used as a cache (i.e. a Python
        dict would be a cache that never evicts old objects), though
        :doc:`uproot4.cache.LRUCache` is a good choice because it is thread-safe
        and evicts least-recently used objects when a maximum number of objects
        is reached.
        """
        return self._object_cache

    @object_cache.setter
    def object_cache(self, value):
        if value is None or isinstance(value, MutableMapping):
            self._object_cache = value
        elif uproot4._util.isint(value):
            self._object_cache = uproot4.cache.LRUCache(value)
        else:
            raise TypeError("object_cache must be None, a MutableMapping, or an int")

    @property
    def array_cache(self):
        """
        A cache used to hold previously extracted arrays, so that code like

        .. code-block:: python

            a = my_tree["branch"].array()
            a = my_tree["branch"].array()
            a = my_tree["branch"].array()

        only reads the ``"branch"`` once.

        Any Python ``MutableMapping`` can be used as a cache (i.e. a Python
        dict would be a cache that never evicts old objects), though
        :doc:`uproot4.cache.LRUArrayCache` is a good choice because it is
        thread-safe and evicts least-recently used objects when a size limit is
        reached.
        """
        return self._array_cache

    @array_cache.setter
    def array_cache(self, value):
        if value is None or isinstance(value, MutableMapping):
            self._array_cache = value
        elif uproot4._util.isint(value) or uproot4._util.isstr(value):
            self._array_cache = uproot4.cache.LRUArrayCache(value)
        else:
            raise TypeError(
                "array_cache must be None, a MutableMapping, or a memory size"
            )

    @property
    def root_directory(self):
        """
        The root ``TDirectory`` of the file
        (:doc:`uproot4.reading.ReadOnlyDirectory`).
        """
        return ReadOnlyDirectory(
            (),
            uproot4.source.cursor.Cursor(self._fBEGIN + self._fNbytesName),
            {},
            self,
            self,
        )

    def show_streamers(self, classname=None, version="max", stream=sys.stdout):
        """
        Args:
            classname (None or str): If None, all streamers that are
                defined in the file are shown; if a class name, only
                this class and its dependencies are shown.
            version (int, "min", or "max"): Version number of the desired
                class; "min" or "max" returns the minimum or maximum version
                number, respectively.
            stream: Object with a `write` method for writing the output.
        """
        if classname is None:
            names = []
            for name, streamer_versions in self.streamers.items():
                for version in streamer_versions:
                    names.append((name, version))
        else:
            names = self.streamer_dependencies(classname, version=version)
        first = True
        for name, version in names:
            for v, streamer in self.streamers[name].items():
                if v == version:
                    if not first:
                        stream.write(u"\n")
                    streamer.show(stream=stream)
                    first = False

    @property
    def streamers(self):
        """
        A list of :doc:`uproot4.streamers.Model_TStreamerInfo` objects
        representing the ``TStreamerInfos`` in the ROOT file.

        A file's ``TStreamerInfos`` are only read the first time they are needed.
        Uproot has a suite of predefined models in ``uproot4.models`` to reduce
        the probability that ``TStreamerInfos`` will need to be read (depending
        on the choice of classes or versions of the classes that are accessed).

        See also :doc:`uproot4.reading.ReadOnlyFile.streamer_rules`, which are
        read in the same pass with ``TStreamerInfos``.
        """
        import uproot4.streamers
        import uproot4.models.TList
        import uproot4.models.TObjArray
        import uproot4.models.TObjString

        if self._streamers is None:
            if self._fSeekInfo == 0:
                self._streamers = {}

            else:
                key_cursor = uproot4.source.cursor.Cursor(self._fSeekInfo)
                key_start = self._fSeekInfo
                key_stop = min(
                    self._fSeekInfo + ReadOnlyKey._format_big.size, self._fEND
                )
                key_chunk = self.chunk(key_start, key_stop)

                self.hook_before_read_streamer_key(
                    key_chunk=key_chunk, key_cursor=key_cursor,
                )

                streamer_key = ReadOnlyKey(key_chunk, key_cursor, {}, self, self)

                self.hook_before_read_decompress_streamers(
                    key_chunk=key_chunk,
                    key_cursor=key_cursor,
                    streamer_key=streamer_key,
                )

                (
                    streamer_chunk,
                    streamer_cursor,
                ) = streamer_key.get_uncompressed_chunk_cursor()

                self.hook_before_interpret_streamers(
                    key_chunk=key_chunk,
                    key_cursor=key_cursor,
                    streamer_key=streamer_key,
                    streamer_cursor=streamer_cursor,
                    streamer_chunk=streamer_chunk,
                )

                classes = uproot4.model.maybe_custom_classes(self._custom_classes)
                tlist = classes["TList"].read(
                    streamer_chunk, streamer_cursor, {}, self, self.detached, None
                )

                self._streamers = {}
                self._streamer_rules = []

                for x in tlist:
                    if isinstance(x, uproot4.streamers.Model_TStreamerInfo):
                        if x.name not in self._streamers:
                            self._streamers[x.name] = {}
                        self._streamers[x.name][x.class_version] = x

                    elif isinstance(x, uproot4.models.TList.Model_TList) and all(
                        isinstance(y, uproot4.models.TObjString.Model_TObjString)
                        for y in x
                    ):
                        self._streamer_rules.extend([str(y) for y in x])

                    else:
                        raise ValueError(
                            """unexpected type in TList of streamers and streamer rules: {0}
in file {1}""".format(
                                type(x), self._file_path
                            )
                        )

                self.hook_after_interpret_streamers(
                    key_chunk=key_chunk,
                    key_cursor=key_cursor,
                    streamer_key=streamer_key,
                    streamer_cursor=streamer_cursor,
                    streamer_chunk=streamer_chunk,
                )

        return self._streamers

    @property
    def streamer_rules(self):
        """
        A list of strings of C++ code that help schema evolution of
        ``TStreamerInfo`` by providing rules to evaluate when new objects are
        accessed by old ROOT versions.

        Uproot does not evaluate these rules because they are written in C++ and
        Uproot does not have access to a C++ compiler.

        These rules are read in the same pass that produces
        :doc:`uproot4.reading.ReadOnlyFile.streamers`.
        """
        if self._streamer_rules is None:
            self.streamers
        return self._streamer_rules

    def streamers_named(self, classname):
        """
        Returns a list of :doc:`uproot4.streamers.Model_TStreamerInfo` objects
        that match C++ (decoded) ``classname``.

        More that one streamer matching a given name is unlikely, but possible
        because there may be different versions of the same class. (Perhaps such
        files can be created by merging data from different ROOT versions with
        hadd?)

        See also :doc:`uproot4.reading.ReadOnlyFile.streamer_named` (singular).
        """
        streamer_versions = self.streamers.get(classname)
        if streamer_versions is None:
            return []
        else:
            return list(streamer_versions.values())

    def streamer_named(self, classname, version="max"):
        """
        Returns a single :doc:`uproot4.streamers.Model_TStreamerInfo` object
        that matches C++ (decoded) ``classname`` and ``version``.

        The ``version`` can be an integer or ``"min"`` or ``"max"`` for the
        minimum and maximum version numbers available in the file. The default
        is ``"max"`` because there's usually only one.

        See also :doc:`uproot4.reading.ReadOnlyFile.streamers_named` (plural).
        """
        streamer_versions = self.streamers.get(classname)
        if streamer_versions is None or len(streamer_versions) == 0:
            return None
        elif version == "min":
            return streamer_versions[min(streamer_versions)]
        elif version == "max":
            return streamer_versions[max(streamer_versions)]
        else:
            return streamer_versions.get(version)

    def streamer_dependencies(self, classname, version="max"):
        """
        Returns a list of :doc:`uproot4.streamers.Model_TStreamerInfo` objects
        that depend on the one that matches C++ (decoded) ``classname`` and
        ``version``.

        The ``classname`` and ``version`` are interpreted the same way as
        :doc:`uproot4.reading.ReadOnlyFile.streamer_named`.
        """
        streamer = self.streamer_named(classname, version=version)
        out = []
        streamer._dependencies(self.streamers, out)
        return out[::-1]

    @property
    def custom_classes(self):
        """
        Either a dict of class objects specific to this file or None if it uses
        the common ``uproot4.classes`` pool.
        """
        return self._custom_classes

    @custom_classes.setter
    def custom_classes(self, value):
        if value is None or isinstance(value, MutableMapping):
            self._custom_classes = value
        else:
            raise TypeError("custom_classes must be None or a MutableMapping")

    def remove_class_definition(self, classname):
        """
        Removes all versions of a class, specified by C++ (decoded)
        ``classname``, from the :doc:`uproot4.reading.ReadOnlyFile.custom_classes`.

        If the file doesn't have a
        :doc:`uproot4.reading.ReadOnlyFile.custom_classes`, this function adds
        one, so it does not remove the class from the common pool.

        If you want to remove a class from the common pool, you can do so with

        .. code-block:: python

            del uproot4.classes[classname]
        """
        if self._custom_classes is None:
            self._custom_classes = dict(uproot4.classes)
        if classname in self._custom_classes:
            del self._custom_classes[classname]

    def class_named(self, classname, version=None):
        """
        Returns or creates a class with a given C++ (decoded) ``classname``
        and possible ``version``.

        * If the ``version`` is None, this function may return a
          :doc:`uproot4.model.DispatchByVersion`.
        * If the ``version`` is an integer, ``"min"`` or ``"max"``, then it
          returns a :doc:`uproot4.model.VersionedModel`. Using ``"min"`` or
          ``"max"`` specifies the minium or maximum version ``TStreamerInfo``
          defined by the file; most files define only one so ``"max"`` is
          usually safe.

        If this file has :doc:`uproot4.reading.ReadOnlyFile.custom_classes`,
        the new class is added to that dict; otherwise, it is added to the
        global ``uproot4.classes``.
        """
        classes = uproot4.model.maybe_custom_classes(self._custom_classes)
        cls = classes.get(classname)

        if cls is None:
            streamers = self.streamers_named(classname)

            if len(streamers) == 0:
                unknown_cls = uproot4.unknown_classes.get(classname)
                if unknown_cls is None:
                    unknown_cls = uproot4._util.new_class(
                        uproot4.model.classname_encode(classname, unknown=True),
                        (uproot4.model.UnknownClass,),
                        {},
                    )
                    uproot4.unknown_classes[classname] = unknown_cls
                return unknown_cls

            else:
                cls = uproot4._util.new_class(
                    uproot4._util.ensure_str(uproot4.model.classname_encode(classname)),
                    (uproot4.model.DispatchByVersion,),
                    {"known_versions": {}},
                )
                classes[classname] = cls

        if version is not None and issubclass(cls, uproot4.model.DispatchByVersion):
            if not uproot4._util.isint(version):
                streamer = self.streamer_named(classname, version)
                if streamer is not None:
                    version = streamer.class_version
                elif version == "max" and len(cls.known_versions) != 0:
                    version = max(cls.known_versions)
                elif version == "min" and len(cls.known_versions) != 0:
                    version = min(cls.known_versions)
                else:
                    unknown_cls = uproot4.unknown_classes.get(classname)
                    if unknown_cls is None:
                        unknown_cls = uproot4._util.new_class(
                            uproot4.model.classname_encode(
                                classname, version, unknown=True
                            ),
                            (uproot4.model.UnknownClassVersion,),
                            {},
                        )
                        uproot4.unknown_classes[classname] = unknown_cls
                    return unknown_cls

            versioned_cls = cls.class_of_version(version)
            if versioned_cls is None:
                cls = cls.new_class(self, version)
            else:
                cls = versioned_cls

        return cls

    def chunk(self, start, stop):
        """
        Returns a :doc:`uproot4.source.chunk.Chunk` from the
        :doc:`uproot4.source.chunk.Source` that is guaranteed to include bytes
        from ``start`` up to ``stop`` seek points in the file.

        If the desired range is satisfied by a previously saved chunk, such as
        :doc:`uproot4.reading.ReadOnlyFile.begin_chunk`, then that is returned.
        Hence, the returned chunk may include more data than the range from
        ``start`` up to ``stop``.
        """
        if self.closed:
            raise OSError("file {0} is closed".format(repr(self._file_path)))
        elif (start, stop) in self._begin_chunk:
            return self._begin_chunk
        else:
            return self._source.chunk(start, stop)

    @property
    def begin_chunk(self):
        """
        A special :doc:`uproot4.source.chunk.Chunk` corresponding to the
        beginning of the file, from seek point ``0`` up to
        ``options["begin_chunk_size"]``.
        """
        return self._begin_chunk

    def hook_before_create_source(self, **kwargs):
        """
        Called in the :doc:`uproot4.reading.ReadOnlyFile` constructor before the
        :doc:`uproot4.source.chunk.Source` is created.

        This is the first hook called in the :doc:`uproot4.reading.ReadOnlyFile`
        constructor.
        """
        pass

    def hook_before_get_chunks(self, **kwargs):
        """
        Called in the :doc:`uproot4.reading.ReadOnlyFile` constructor after the
        :doc:`uproot4.source.chunk.Source` is created but before attempting to
        get any :doc:`uproot4.source.chunk.Chunk`, specifically the
        :doc:`uproot4.reading.ReadOnlyFile.begin_chunk`.
        """
        pass

    def hook_before_interpret(self, **kwargs):
        """
        Called in the :doc:`uproot4.reading.ReadOnlyFile` constructor after
        loading the :doc:`uproot4.reading.ReadOnlyFile.begin_chunk` and before
        interpreting its ``TFile`` header.
        """
        pass

    def hook_after_interpret(self, **kwargs):
        """
        Called in the :doc:`uproot4.reading.ReadOnlyFile` constructor after
        interpreting the ``TFile`` header and before raising an error if
        the first four bytes are not ``b"root"``.

        This is the last hook called in the :doc:`uproot4.reading.ReadOnlyFile`
        constructor.
        """
        pass

    def hook_before_read_streamer_key(self, **kwargs):
        """
        Called in :doc:`uproot4.reading.ReadOnlyFile.streamers` before reading
        the ``TKey`` associated with the ``TStreamerInfo``.

        This is the first hook called in
        :doc:`uproot4.reading.ReadOnlyFile.streamers`.
        """
        pass

    def hook_before_read_decompress_streamers(self, **kwargs):
        """
        Called in :doc:`uproot4.reading.ReadOnlyFile.streamers` after reading
        the ``TKey`` associated with the ``TStreamerInfo`` and before reading
        and decompressing the ``TStreamerInfo`` data.
        """
        pass

    def hook_before_interpret_streamers(self, **kwargs):
        """
        Called in :doc:`uproot4.reading.ReadOnlyFile.streamers` after reading
        and decompressing the ``TStreamerInfo`` data, but before interpreting
        it.
        """
        pass

    def hook_after_interpret_streamers(self, **kwargs):
        """
        Called in :doc:`uproot4.reading.ReadOnlyFile.streamers` after
        interpreting the ``TStreamerInfo`` data.

        This is the last hook called in
        :doc:`uproot4.reading.ReadOnlyFile.streamers`.
        """
        pass


class ReadOnlyKey(object):
    _format_small = struct.Struct(">ihiIhhii")
    _format_big = struct.Struct(">ihiIhhqq")

    def __init__(self, chunk, cursor, context, file, parent, read_strings=False):
        self._cursor = cursor.copy()
        self._file = file
        self._parent = parent

        self.hook_before_read(
            chunk=chunk,
            cursor=cursor,
            context=context,
            file=file,
            parent=parent,
            read_strings=read_strings,
        )

        (
            self._fNbytes,
            self._fVersion,
            self._fObjlen,
            self._fDatime,
            self._fKeylen,
            self._fCycle,
            self._fSeekKey,
            self._fSeekPdir,
        ) = cursor.fields(chunk, self._format_small, context, move=False)

        if self.is_64bit:
            (
                self._fNbytes,
                self._fVersion,
                self._fObjlen,
                self._fDatime,
                self._fKeylen,
                self._fCycle,
                self._fSeekKey,
                self._fSeekPdir,
            ) = cursor.fields(chunk, self._format_big, context)

        else:
            cursor.skip(self._format_small.size)

        if read_strings:
            self.hook_before_read_strings(
                chunk=chunk,
                cursor=cursor,
                context=context,
                file=file,
                parent=parent,
                read_strings=read_strings,
            )

            self._fClassName = cursor.string(chunk, context)
            self._fName = cursor.string(chunk, context)
            self._fTitle = cursor.string(chunk, context)

        else:
            self._fClassName = None
            self._fName = None
            self._fTitle = None

        self.hook_after_read(
            chunk=chunk,
            cursor=cursor,
            context=context,
            file=file,
            parent=parent,
            read_strings=read_strings,
        )

    def __repr__(self):
        if self._fName is None or self._fClassName is None:
            nameclass = ""
        else:
            nameclass = " {0}: {1}".format(self.name(cycle=True), self.classname())
        return "<ReadOnlyKey{0} (seek pos {1}) at 0x{2:012x}>".format(
            nameclass, self.data_cursor.index, id(self)
        )

    def hook_before_read(self, **kwargs):
        pass

    def hook_before_read_strings(self, **kwargs):
        pass

    def hook_after_read(self, **kwargs):
        pass

    @property
    def cursor(self):
        return self._cursor

    @property
    def file(self):
        return self._file

    @property
    def parent(self):
        return self._parent

    @property
    def data_compressed_bytes(self):
        return self._fNbytes - self._fKeylen

    @property
    def data_uncompressed_bytes(self):
        return self._fObjlen

    @property
    def is_compressed(self):
        return self.data_compressed_bytes != self.data_uncompressed_bytes

    @property
    def is_64bit(self):
        return self._fVersion > 1000

    def name(self, cycle=False):
        if cycle:
            return "{0};{1}".format(self.fName, self.fCycle)
        else:
            return self.fName

    def classname(self, encoded=False, version=None):
        if encoded:
            return uproot4.model.classname_encode(self.fClassName, version=version)
        else:
            return self.fClassName

    @property
    def fNbytes(self):
        return self._fNbytes

    @property
    def fVersion(self):
        return self._fVersion

    @property
    def fObjlen(self):
        return self._fObjlen

    @property
    def fDatime(self):
        return self._fDatime

    @property
    def fKeylen(self):
        return self._fKeylen

    @property
    def fCycle(self):
        return self._fCycle

    @property
    def fSeekKey(self):
        return self._fSeekKey

    @property
    def fSeekPdir(self):
        return self._fSeekPdir

    @property
    def fClassName(self):
        return self._fClassName

    @property
    def fName(self):
        return self._fName

    @property
    def fTitle(self):
        return self._fTitle

    @property
    def data_cursor(self):
        return uproot4.source.cursor.Cursor(self._fSeekKey + self._fKeylen)

    def get_uncompressed_chunk_cursor(self):
        cursor = uproot4.source.cursor.Cursor(0, origin=-self._fKeylen)

        data_start = self.data_cursor.index
        data_stop = data_start + self.data_compressed_bytes
        chunk = self._file.chunk(data_start, data_stop)

        if self.is_compressed:
            uncompressed_chunk = uproot4.compression.decompress(
                chunk,
                self.data_cursor,
                {},
                self.data_compressed_bytes,
                self.data_uncompressed_bytes,
            )
        else:
            uncompressed_chunk = uproot4.source.chunk.Chunk.wrap(
                chunk.source,
                chunk.get(
                    data_start,
                    data_stop,
                    self.data_cursor,
                    {"breadcrumbs": (), "TKey": self},
                ),
            )

        return uncompressed_chunk, cursor

    @property
    def cache_key(self):
        return "{0}:{1}".format(self._file.hex_uuid, self._fSeekKey)

    @property
    def object_path(self):
        if isinstance(self._parent, ReadOnlyDirectory):
            return "{0}{1};{2}".format(
                self._parent.object_path, self.name(False), self._fCycle
            )
        else:
            return "(seek pos {0})/{1}".format(self.data_cursor.index, self.name(False))

    def get(self):
        if self._file.object_cache is not None:
            out = self._file.object_cache.get(self.cache_key)
            if out is not None:
                if isinstance(out.file, ReadOnlyFile) and out.file.closed:
                    del self._file.object_cache[self.cache_key]
                else:
                    return out

        if self._fClassName in must_be_attached:
            selffile = self._file
            parent = self
        else:
            selffile = self._file.detached
            parent = None

        if isinstance(self._parent, ReadOnlyDirectory) and self._fClassName in (
            "TDirectory",
            "TDirectoryFile",
        ):
            out = ReadOnlyDirectory(
                self._parent.path + (self.fName,),
                self.data_cursor,
                {},
                self._file,
                self,
            )

        else:
            chunk, cursor = self.get_uncompressed_chunk_cursor()
            start_cursor = cursor.copy()
            cls = self._file.class_named(self._fClassName)
            context = {"breadcrumbs": (), "TKey": self}

            try:
                out = cls.read(chunk, cursor, context, self._file, selffile, parent)

            except uproot4.deserialization.DeserializationError:
                breadcrumbs = context.get("breadcrumbs")

                if breadcrumbs is None or all(
                    breadcrumb_cls.classname in uproot4.model.bootstrap_classnames
                    or isinstance(breadcrumb_cls, uproot4.containers.AsContainer)
                    or getattr(breadcrumb_cls.class_streamer, "file_uuid", None)
                    == self._file.uuid
                    for breadcrumb_cls in breadcrumbs
                ):
                    # we're already using the most specialized versions of each class
                    raise

                for breadcrumb_cls in breadcrumbs:
                    if (
                        breadcrumb_cls.classname
                        not in uproot4.model.bootstrap_classnames
                    ):
                        self._file.remove_class_definition(breadcrumb_cls.classname)

                cursor = start_cursor
                cls = self._file.class_named(self._fClassName)
                context = {"breadcrumbs": (), "TKey": self}

                out = cls.read(chunk, cursor, context, self._file, selffile, parent)

        if self._fClassName not in must_be_attached:
            out._file = self._file.detached
            out._parent = None

        if self._file.object_cache is not None:
            self._file.object_cache[self.cache_key] = out
        return out


class ReadOnlyDirectory(Mapping):
    _format_small = struct.Struct(">hIIiiiii")
    _format_big = struct.Struct(">hIIiiqqq")
    _format_num_keys = struct.Struct(">i")

    def __init__(self, path, cursor, context, file, parent):
        self._path = path
        self._cursor = cursor.copy()
        self._file = file
        self._parent = parent

        directory_start = cursor.index
        directory_stop = min(directory_start + self._format_big.size, file.fEND)
        chunk = file.chunk(directory_start, directory_stop)

        self.hook_before_read(
            path=path, chunk=chunk, cursor=cursor, file=file, parent=parent,
        )

        (
            self._fVersion,
            self._fDatimeC,
            self._fDatimeM,
            self._fNbytesKeys,
            self._fNbytesName,
            self._fSeekDir,
            self._fSeekParent,
            self._fSeekKeys,
        ) = cursor.fields(chunk, self._format_small, context, move=False)

        if self.is_64bit:
            (
                self._fVersion,
                self._fDatimeC,
                self._fDatimeM,
                self._fNbytesKeys,
                self._fNbytesName,
                self._fSeekDir,
                self._fSeekParent,
                self._fSeekKeys,
            ) = cursor.fields(chunk, self._format_big, context)

        else:
            cursor.skip(self._format_small.size)

        if self._fSeekKeys == 0:
            self._header_key = None
            self._keys = []

        else:
            keys_start = self._fSeekKeys
            keys_stop = min(keys_start + self._fNbytesKeys + 8, file.fEND)

            if (keys_start, keys_stop) in chunk:
                keys_chunk = chunk
            else:
                keys_chunk = file.chunk(keys_start, keys_stop)

            keys_cursor = uproot4.source.cursor.Cursor(self._fSeekKeys)

            self.hook_before_header_key(
                path=path,
                chunk=chunk,
                cursor=cursor,
                file=file,
                parent=parent,
                keys_chunk=keys_chunk,
                keys_cursor=keys_cursor,
            )

            self._header_key = ReadOnlyKey(
                keys_chunk, keys_cursor, {}, file, self, read_strings=True
            )

            num_keys = keys_cursor.field(keys_chunk, self._format_num_keys, context)

            self.hook_before_keys(
                path=path,
                chunk=chunk,
                cursor=cursor,
                file=file,
                parent=parent,
                keys_chunk=keys_chunk,
                keys_cursor=keys_cursor,
                num_keys=num_keys,
            )

            self._keys = []
            for i in uproot4._util.range(num_keys):
                key = ReadOnlyKey(
                    keys_chunk, keys_cursor, {}, file, self, read_strings=True
                )
                self._keys.append(key)

            self.hook_after_read(
                path=path,
                chunk=chunk,
                cursor=cursor,
                file=file,
                parent=parent,
                keys_chunk=keys_chunk,
                keys_cursor=keys_cursor,
                num_keys=num_keys,
            )

    def __repr__(self):
        return "<ReadOnlyDirectory {0} at 0x{1:012x}>".format(
            repr("/" + "/".join(self._path)), id(self)
        )

    def hook_before_read(self, **kwargs):
        pass

    def hook_before_header_key(self, **kwargs):
        pass

    def hook_before_keys(self, **kwargs):
        pass

    def hook_after_read(self, **kwargs):
        pass

    @property
    def path(self):
        return self._path

    @property
    def cursor(self):
        return self._cursor

    @property
    def file(self):
        return self._file

    @property
    def parent(self):
        return self._parent

    @property
    def header_key(self):
        return self._header_key

    @property
    def is_64bit(self):
        return self._fVersion > 1000

    @property
    def fVersion(self):
        return self._fVersion

    @property
    def fDatimeC(self):
        return self._fDatimeC

    @property
    def fDatimeM(self):
        return self._fDatimeM

    @property
    def fNbytesKeys(self):
        return self._fNbytesKeys

    @property
    def fNbytesName(self):
        return self._fNbytesName

    @property
    def fSeekDir(self):
        return self._fSeekDir

    @property
    def fSeekParent(self):
        return self._fSeekParent

    @property
    def fSeekKeys(self):
        return self._fSeekKeys

    def __enter__(self):
        """
        Passes __enter__ to the file and returns self.
        """
        self._file.source.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """
        Passes __exit__ to the file, which closes physical files and shuts down
        any other resources, such as thread pools for parallel reading.
        """
        self._file.source.__exit__(exception_type, exception_value, traceback)

    def close(self):
        """
        Closes the file from which this object is derived.
        """
        self._file.close()

    @property
    def closed(self):
        """
        True if the associated file is closed; False otherwise.
        """
        return self._file.closed

    def streamer_dependencies(self, classname, version="max"):
        return self._file.streamer_dependencies(classname=classname, version=version)

    def show_streamers(self, classname=None, stream=sys.stdout):
        """
        Args:
            classname (None or str): If None, all streamers that are
                defined in the file are shown; if a class name, only
                this class and its dependencies are shown.
            stream: Object with a `write` method for writing the output.
        """
        self._file.show_streamers(classname=classname, stream=stream)

    @property
    def cache_key(self):
        return self.file.hex_uuid + ":" + self.object_path

    @property
    def object_path(self):
        return "/".join(("",) + self._path + ("",)).replace("//", "/")

    @property
    def object_cache(self):
        return self._file._object_cache

    @object_cache.setter
    def object_cache(self, value):
        if value is None or isinstance(value, MutableMapping):
            self._file._object_cache = value
        elif uproot4._util.isint(value):
            self._file._object_cache = uproot4.cache.LRUCache(value)
        else:
            raise TypeError("object_cache must be None, a MutableMapping, or an int")

    @property
    def array_cache(self):
        return self._file._array_cache

    @array_cache.setter
    def array_cache(self, value):
        if value is None or isinstance(value, MutableMapping):
            self._file._array_cache = value
        elif uproot4._util.isint(value) or uproot4._util.isstr(value):
            self._file._array_cache = uproot4.cache.LRUArrayCache(value)
        else:
            raise TypeError(
                "array_cache must be None, a MutableMapping, or a memory size"
            )

    def iterclassnames(
        self,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        filter_name = uproot4._util.regularize_filter(filter_name)
        filter_classname = uproot4._util.regularize_filter(filter_classname)
        for key in self._keys:
            if (filter_name is no_filter or filter_name(key.fName)) and (
                filter_classname is no_filter or filter_classname(key.fClassName)
            ):
                yield key.name(cycle=cycle), key.fClassName

            if recursive and key.fClassName in ("TDirectory", "TDirectoryFile"):
                for k1, v in key.get().iterclassnames(
                    recursive=recursive,
                    cycle=cycle,
                    filter_name=no_filter,
                    filter_classname=filter_classname,
                ):
                    k2 = "{0}/{1}".format(key.name(cycle=False), k1)
                    k3 = k2[: k2.index(";")] if ";" in k2 else k2
                    if filter_name is no_filter or filter_name(k3):
                        yield k2, v

    def classnames(
        self,
        recursive=True,
        cycle=False,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        return dict(
            self.iterclassnames(
                recursive=recursive,
                cycle=cycle,
                filter_name=filter_name,
                filter_classname=filter_classname,
            )
        )

    def iterkeys(
        self,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        filter_name = uproot4._util.regularize_filter(filter_name)
        filter_classname = uproot4._util.regularize_filter(filter_classname)
        for key in self._keys:
            if (filter_name is no_filter or filter_name(key.fName)) and (
                filter_classname is no_filter or filter_classname(key.fClassName)
            ):
                yield key.name(cycle=cycle)

            if recursive and key.fClassName in ("TDirectory", "TDirectoryFile"):
                for k1 in key.get().iterkeys(
                    recursive=recursive,
                    cycle=cycle,
                    filter_name=no_filter,
                    filter_classname=filter_classname,
                ):
                    k2 = "{0}/{1}".format(key.name(cycle=False), k1)
                    k3 = k2[: k2.index(";")] if ";" in k2 else k2
                    if filter_name is no_filter or filter_name(k3):
                        yield k2

    def keys(
        self,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        return list(
            self.iterkeys(
                recursive=recursive,
                cycle=cycle,
                filter_name=filter_name,
                filter_classname=filter_classname,
            )
        )

    def iteritems(
        self,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        filter_name = uproot4._util.regularize_filter(filter_name)
        filter_classname = uproot4._util.regularize_filter(filter_classname)
        for key in self._keys:
            if (filter_name is no_filter or filter_name(key.fName)) and (
                filter_classname is no_filter or filter_classname(key.fClassName)
            ):
                yield key.name(cycle=cycle), key.get()

            if recursive and key.fClassName in ("TDirectory", "TDirectoryFile"):
                for k1, v in key.get().iteritems(
                    recursive=recursive,
                    cycle=cycle,
                    filter_name=no_filter,
                    filter_classname=filter_classname,
                ):
                    k2 = "{0}/{1}".format(key.name(cycle=False), k1)
                    k3 = k2[: k2.index(";")] if ";" in k2 else k2
                    if filter_name is no_filter or filter_name(k3):
                        yield k2, v

    def items(
        self,
        recursive=True,
        cycle=True,
        filter_name=no_filter,
        filter_classname=no_filter,
    ):
        return list(
            self.iteritems(
                recursive=recursive,
                cycle=cycle,
                filter_name=filter_name,
                filter_classname=filter_classname,
            )
        )

    def itervalues(
        self, recursive=True, filter_name=no_filter, filter_classname=no_filter,
    ):
        for k, v in self.iteritems(
            recursive=recursive,
            cycle=False,
            filter_name=filter_name,
            filter_classname=filter_classname,
        ):
            yield v

    def values(
        self, recursive=True, filter_name=no_filter, filter_classname=no_filter,
    ):
        return list(
            self.itervalues(
                recursive=recursive,
                filter_name=filter_name,
                filter_classname=filter_classname,
            )
        )

    def __len__(self):
        return len(self._keys) + sum(
            len(x.get())
            for x in self._keys
            if x.fClassName in ("TDirectory", "TDirectoryFile")
        )

    def __contains__(self, where):
        try:
            self.key(where)
        except KeyError:
            return False
        else:
            return True

    def __iter__(self):
        return self.iterkeys()

    def _ipython_key_completions_(self):
        "Support key-completion in an IPython or Jupyter kernel."
        return self.iterkeys()

    def __getitem__(self, where):
        if "/" in where or ":" in where:
            items = where.split("/")
            step = last = self

            for i, item in enumerate(items):
                if item != "":
                    if isinstance(step, ReadOnlyDirectory):
                        if ":" in item and item not in step:
                            index = item.index(":")
                            head, tail = item[:index], item[index + 1 :]
                            last = step
                            step = step[head]
                            if isinstance(step, uproot4.behaviors.TBranch.HasBranches):
                                return step["/".join([tail] + items[i + 1 :])]
                            else:
                                raise uproot4.KeyInFileError(
                                    where,
                                    repr(head)
                                    + " is not a TDirectory, TTree, or TBranch",
                                    keys=[key.fName for key in last._keys],
                                    file_path=self._file.file_path,
                                )
                        else:
                            last = step
                            step = step[item]

                    elif isinstance(step, uproot4.behaviors.TBranch.HasBranches):
                        return step["/".join(items[i:])]

                    else:
                        raise uproot4.KeyInFileError(
                            where,
                            repr(item) + " is not a TDirectory, TTree, or TBranch",
                            keys=[key.fName for key in last._keys],
                            file_path=self._file.file_path,
                        )

            return step

        else:
            return self.key(where).get()

    def classname_of(self, where, encoded=False, version=None):
        key = self.key(where)
        return key.classname(encoded=encoded, version=version)

    def streamer_of(self, where, version):
        key = self.key(where)
        return self._file.streamer_named(key.fClassName, version)

    def class_of(self, where, version=None):
        key = self.key(where)
        return self._file.class_named(key.fClassName, version=version)

    def key(self, where):
        where = uproot4._util.ensure_str(where)

        if "/" in where:
            items = where.split("/")
            step = last = self
            for item in items[:-1]:
                if item != "":
                    if isinstance(step, ReadOnlyDirectory):
                        last = step
                        step = step[item]
                    else:
                        raise uproot4.KeyInFileError(
                            where,
                            repr(item) + " is not a TDirectory",
                            keys=[key.fName for key in last._keys],
                            file_path=self._file.file_path,
                        )
            return step.key(items[-1])

        if ";" in where:
            at = where.rindex(";")
            item, cycle = where[:at], where[at + 1 :]
            try:
                cycle = int(cycle)
            except ValueError:
                item, cycle = where, None
        else:
            item, cycle = where, None

        last = None
        for key in self._keys:
            if key.fName == item:
                if cycle == key.fCycle:
                    return key
                elif cycle is None and last is None:
                    last = key
                elif cycle is None and last.fCycle < key.fCycle:
                    last = key

        if last is not None:
            return last
        elif cycle is None:
            raise uproot4.KeyInFileError(
                item, cycle="any", keys=self.keys(), file_path=self._file.file_path
            )
        else:
            raise uproot4.KeyInFileError(
                item, cycle=cycle, keys=self.keys(), file_path=self._file.file_path
            )
