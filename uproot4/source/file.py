# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Physical layer for local files.

Defines a :py:class:`~uproot4.source.file.FileResource` (wrapped Python file handle)
and two sources: :py:class:`~uproot4.source.file.MultithreadedFileSource` and
:py:class:`~uproot4.source.file.MemmapSource`, which provide thread-safe local
file readers using many file handles or a memory-mapped file, respectively.

If the filesystem or operating system does not support memory-mapped files, the
:py:class:`~uproot4.source.file.MultithreadedFileSource` is an automatic fallback.
"""

from __future__ import absolute_import

import os.path

import numpy

import uproot4.source.futures
import uproot4.source.chunk
import uproot4._util


class FileResource(uproot4.source.chunk.Resource):
    """
    Args:
        file_path (str): The filesystem path of the file to open.

    A :py:class:`~uproot4.source.chunk.Resource` for a simple file handle.
    """

    def __init__(self, file_path):
        self._file_path = file_path
        try:
            self._file = open(self._file_path, "rb")
        except uproot4._util._FileNotFoundError:
            raise uproot4._util._file_not_found(file_path)

    @property
    def file(self):
        """
        The Python file handle.
        """
        return self._file

    @property
    def closed(self):
        return self._file.closed

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._file.__exit__(exception_type, exception_value, traceback)

    def get(self, start, stop):
        """
        Args:
            start (int): Seek position of the first byte to include.
            stop (int): Seek position of the first byte to exclude
                (one greater than the last byte to include).

        Returns a Python buffer of data between ``start`` and ``stop``.
        """
        self._file.seek(start)
        return self._file.read(stop - start)

    @staticmethod
    def future(source, start, stop):
        """
        Args:
            source (:py:class:`~uproot4.source.chunk.MultithreadedFileSource`): The
                data source.
            start (int): Seek position of the first byte to include.
            stop (int): Seek position of the first byte to exclude
                (one greater than the last byte to include).

        Returns a :py:class:`~uproot4.source.futures.ResourceFuture` that calls
        :py:meth:`~uproot4.source.file.FileResource.get` with ``start`` and ``stop``.
        """

        def task(resource):
            return resource.get(start, stop)

        return uproot4.source.futures.ResourceFuture(task)


class MemmapSource(uproot4.source.chunk.Source):
    """
    Args:
        file_path (str): The filesystem path of the file to open.
        options: Must include ``"num_fallback_workers"``.

    A :py:class:`~uproot4.source.chunk.Source` that manages one memory-mapped file.
    """

    _dtype = uproot4.source.chunk.Chunk._dtype

    def __init__(self, file_path, **options):
        num_fallback_workers = options["num_fallback_workers"]
        self._num_requests = 0
        self._num_requested_chunks = 0
        self._num_requested_bytes = 0

        self._file_path = file_path

        try:
            self._file = numpy.memmap(self._file_path, dtype=self._dtype, mode="r")
            self._fallback = None
        except (OSError, IOError):
            self._file = None
            opts = dict(options)
            opts["num_workers"] = num_fallback_workers
            self._fallback = uproot4.source.file.MultithreadedFileSource(
                file_path, **opts  # NOTE: a comma after **opts breaks Python 2
            )

    def __repr__(self):
        path = repr(self._file_path)
        if len(self._file_path) > 10:
            path = repr("..." + self._file_path[-10:])
        fallback = ""
        if self._fallback is not None:
            fallback = " with fallback"
        return "<{0} {1}{2} at 0x{3:012x}>".format(
            type(self).__name__, path, fallback, id(self)
        )

    def chunk(self, start, stop):
        if self._fallback is None:
            if self.closed:
                raise OSError("memmap is closed for file {0}".format(self._file_path))

            self._num_requests += 1
            self._num_requested_chunks += 1
            self._num_requested_bytes += stop - start

            data = numpy.array(self._file[start:stop], copy=True)
            future = uproot4.source.futures.NoFuture(data)
            return uproot4.source.chunk.Chunk(self, start, stop, future)

        else:
            return self._fallback.chunk(start, stop)

    def chunks(self, ranges, notifications):
        if self._fallback is None:
            if self.closed:
                raise OSError("memmap is closed for file {0}".format(self._file_path))

            self._num_requests += 1
            self._num_requested_chunks += len(ranges)
            self._num_requested_bytes += sum(stop - start for start, stop in ranges)

            chunks = []
            for start, stop in ranges:
                data = numpy.array(self._file[start:stop], copy=True)
                future = uproot4.source.futures.NoFuture(data)
                chunk = uproot4.source.chunk.Chunk(self, start, stop, future)
                notifications.put(chunk)
                chunks.append(chunk)
            return chunks

        else:
            return self._fallback.chunks(ranges, notifications)

    @property
    def file(self):
        """
        The ``numpy.memmap`` array/file.
        """
        return self._file

    @property
    def fallback(self):
        """
        If None, the :py:attr:`~uproot4.source.file.MemmapSource.file` opened
        successfully and no fallback is needed.

        Otherwise, this is a :py:class:`~uproot4.source.file.MultithreadedFileSource`
        to which all requests are forwarded.
        """
        return self._fallback

    @property
    def closed(self):
        if self._fallback is None:
            if uproot4._util.py2:
                try:
                    self._file._mmap.tell()
                except ValueError:
                    return True
                else:
                    return False
            else:
                return self._file._mmap.closed
        else:
            return self._fallback.closed

    def __enter__(self):
        if self._fallback is None:
            if hasattr(self._file._mmap, "__enter__"):
                self._file._mmap.__enter__()
        else:
            self._fallback.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self._fallback is None:
            if hasattr(self._file._mmap, "__exit__"):
                self._file._mmap.__exit__(exception_type, exception_value, traceback)
            else:
                self._file._mmap.close()
        else:
            self._fallback.__exit__(exception_type, exception_value, traceback)

    @property
    def num_bytes(self):
        if self._fallback is None:
            return self._file._mmap.size()
        else:
            return self._fallback.num_bytes


class MultithreadedFileSource(uproot4.source.chunk.MultithreadedSource):
    """
    Args:
        file_path (str): The filesystem path of the file to open.
        options: Must include ``"num_workers"``.

    A :py:class:`~uproot4.source.chunk.MultithreadedSource` that manages many
    :py:class:`~uproot4.source.file.FileResource` objects.
    """

    ResourceClass = FileResource

    def __init__(self, file_path, **options):
        num_workers = options["num_workers"]
        self._num_requests = 0
        self._num_requested_chunks = 0
        self._num_requested_bytes = 0

        self._file_path = file_path
        self._executor = uproot4.source.futures.ResourceThreadPoolExecutor(
            [FileResource(file_path) for x in uproot4._util.range(num_workers)]
        )
        self._num_bytes = os.path.getsize(self._file_path)
