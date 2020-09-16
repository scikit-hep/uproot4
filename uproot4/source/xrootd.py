# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Physical layer for remote files, accessed via the XRootD protocol.

Defines a :py:class:`~uproot4.source.http.XRootDResource` (``XRootD.File``) and two
sources: :py:class:`~uproot4.source.http.MultithreadedXRootDSource` and
:py:class:`~uproot4.source.http.XRootDSource`. The latter requires the server to
support vector-read requests; if not, it automatically falls back to
:py:class:`~uproot4.source.http.MultithreadedXRootDSource`.
"""

from __future__ import absolute_import

import uproot4.source.chunk
import uproot4.source.futures
import uproot4.extras


def get_server_config(file):
    """
    Args:
        file (``XRootD.client.File``): An XRootD file object.

    Query a XRootD server for its configuration.

    Returns a 2-tuple of integers:

    * ``readv_iov_max``: The maximum number of elements that can be
      requested in a single vector read.
    * ``readv_ior_max``: The maximum number of bytes that can be requested
      per *element* in a vector read.
    """
    # Set from some sensible defaults in case this fails
    readv_iov_max = 1024
    readv_ior_max = 2097136

    XRootD_client = uproot4.extras.XRootD_client()

    # Check if the file is stored locally on this machine
    last_url = file.get_property("LastURL")
    last_url = XRootD_client.URL(last_url)
    if last_url.protocol == "file" and last_url.hostid == "localhost":
        # The URL will redirect to a local file where XRootD will split the
        # vector reads into multiple sequential reads so just use the default
        return readv_iov_max, readv_ior_max

    # Find where the data is actually stored
    data_server = file.get_property("DataServer")
    if data_server == "":
        raise NotImplementedError

    data_server = XRootD_client.URL(data_server)
    data_server = "{0}://{1}/".format(data_server.protocol, data_server.hostid)

    # Use a single query call to avoid doubling the latency
    fs = XRootD_client.FileSystem(data_server)
    status, result = fs.query(
        XRootD_client.flags.QueryCode.CONFIG, "readv_iov_max readv_ior_max"
    )
    if status.error:
        raise OSError(status.message)

    # Result is something like b'178956968\n2097136\n'
    readv_iov_max, readv_ior_max = map(int, result.split(b"\n", 1))

    return readv_iov_max, readv_ior_max


class XRootDResource(uproot4.source.chunk.Resource):
    """
    Args:
        file_path (str): A URL of the file to open.
        timeout (None or float): An optional timeout in seconds.

    A :py:class:`~uproot4.source.chunk.Resource` for XRootD connections.
    """

    def __init__(self, file_path, timeout):
        XRootD_client = uproot4.extras.XRootD_client()
        self._file_path = file_path
        self._timeout = timeout

        self._file = XRootD_client.File()

        status, dummy = self._file.open(self._file_path, timeout=self._xrd_timeout())
        if status.error:
            self._xrd_error(status)

    def _xrd_timeout(self):
        if self._timeout is None:
            return 0
        else:
            return int(self._timeout)

    def _xrd_error(self, status):
        self._file.close(timeout=self._xrd_timeout())

        # https://github.com/xrootd/xrootd/blob/8e91462e76ab969720b40fc324714b84e0b4bd42/src/XrdCl/XrdClStatus.hh#L47-L103
        # https://github.com/xrootd/xrootd/blob/250eced4d3787c2ac5be2c8c922134153bbf7f08/src/XrdCl/XrdClStatus.cc#L34-L74
        if status.code == 101 or status.code == 304 or status.code == 400:
            raise uproot4._util._file_not_found(self._file_path, status.message)

        else:
            raise OSError(
                """XRootD error: {0}
in file {1}""".format(
                    status.message, self._file_path
                )
            )

    @property
    def timeout(self):
        """
        The timeout in seconds or None.
        """
        return self._timeout

    @property
    def file(self):
        """
        The ``XRootD.client.File`` object.
        """
        return self._file

    @property
    def num_bytes(self):
        status, info = self._file.stat(self._xrd_timeout())
        if status.error:
            self._xrd_error(status)
        return info.size

    @property
    def closed(self):
        return not self._file.is_open()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._file.close(timeout=self._xrd_timeout())

    def get(self, start, stop):
        """
        Args:
            start (int): Seek position of the first byte to include.
            stop (int): Seek position of the first byte to exclude
                (one greater than the last byte to include).

        Returns a Python buffer of data between ``start`` and ``stop``.
        """
        status, data = self._file.read(start, stop - start, timeout=self._xrd_timeout())
        if status.error:
            self._xrd_error(status)
        return data

    @staticmethod
    def future(source, start, stop):
        """
        Args:
            source (:py:class:`~uproot4.source.chunk.MultithreadedXRootDSource`): The
                data source.
            start (int): Seek position of the first byte to include.
            stop (int): Seek position of the first byte to exclude
                (one greater than the last byte to include).

        Returns a :py:class:`~uproot4.source.futures.ResourceFuture` that calls
        :py:meth:`~uproot4.source.xrootd.XRootDResource.get` with ``start`` and
        ``stop``.
        """

        def task(resource):
            return resource.get(start, stop)

        return uproot4.source.futures.ResourceFuture(task)

    @staticmethod
    def partfuture(results, start, stop):
        """
        Returns a :py:class:`~uproot4.source.futures.ResourceFuture` to simply select
        the ``(start, stop)`` item from the ``results`` dict.

        In :py:meth:`~uproot4.source.xrootd.XRootDSource.chunks`, each chunk has a
        :py:meth:`~uproot4.source.xrootd.XRootDResource.partfuture` that are collectively
        filled by callbacks from :py:meth:`~uproot4.source.xrootd.XRootDResource.callbacker`.
        """

        def task(resource):
            return results[start, stop]

        return uproot4.source.futures.ResourceFuture(task)

    @staticmethod
    def callbacker(futures, results):
        """
        Returns an XRootD callback function to fill the ``futures`` and
        ``results``.
        """

        def callback(status, response, hosts):
            for chunk in response.chunks:
                start, stop = chunk.offset, chunk.offset + chunk.length
                results[start, stop] = chunk.buffer
                futures[start, stop]._run(None)

        return callback


class XRootDSource(uproot4.source.chunk.Source):
    """
    Args:
        file_path (str): A URL of the file to open.
        options: Must include ``"timeout"`` and ``"max_num_elements"``.

    A :py:class:`~uproot4.source.chunk.Source` that uses XRootD's vector-read
    to get many chunks in one request.
    """

    ResourceClass = XRootDResource

    def __init__(self, file_path, **options):
        timeout = options["timeout"]
        max_num_elements = options["max_num_elements"]
        self._num_requests = 0
        self._num_requested_chunks = 0
        self._num_requested_bytes = 0

        self._file_path = file_path
        self._timeout = timeout
        self._num_bytes = None

        self._resource = XRootDResource(file_path, timeout)

        self._max_num_elements, self._max_element_size = get_server_config(
            self._resource.file
        )
        if max_num_elements is not None:
            self._max_num_elements = min(self._max_num_elements, max_num_elements)

    def __repr__(self):
        path = repr(self._file_path)
        if len(self._file_path) > 10:
            path = repr("..." + self._file_path[-10:])
        return "<{0} {1} at 0x{2:012x}>".format(type(self).__name__, path, id(self))

    def chunk(self, start, stop):
        self._num_requests += 1
        self._num_requested_chunks += 1
        self._num_requested_bytes += stop - start

        data = self._resource.get(start, stop)
        future = uproot4.source.futures.NoFuture(data)
        return uproot4.source.chunk.Chunk(self, start, stop, future)

    def chunks(self, ranges, notifications):
        self._num_requests += 1
        self._num_requested_chunks += len(ranges)
        self._num_requested_bytes += sum(stop - start for start, stop in ranges)

        all_request_ranges = [[]]
        for start, stop in ranges:
            if stop - start > self._max_element_size:
                raise NotImplementedError(
                    "TODO: Probably need to fall back to a non-vector read"
                )
            if len(all_request_ranges[-1]) > self._max_num_elements:
                all_request_ranges.append([])
            all_request_ranges[-1].append((start, stop - start))

        chunks = []
        for i, request_ranges in enumerate(all_request_ranges):
            futures = {}
            results = {}
            for start, size in request_ranges:
                stop = start + size
                partfuture = self.ResourceClass.partfuture(results, start, stop)
                futures[start, stop] = partfuture
                chunk = uproot4.source.chunk.Chunk(self, start, stop, partfuture)
                partfuture._set_notify(
                    uproot4.source.chunk.notifier(chunk, notifications)
                )
                chunks.append(chunk)

            callback = self.ResourceClass.callbacker(futures, results)

            status = self._resource.file.vector_read(
                chunks=request_ranges, callback=callback
            )
            if status.error:
                self._resource._xrd_error(status)

        return chunks

    @property
    def resource(self):
        """
        The :py:class:`~uproot4.source.xrootd.XRootDResource` object.
        """
        return self._resource

    @property
    def timeout(self):
        """
        The timeout in seconds or None.
        """
        return self._timeout

    @property
    def file(self):
        """
        The ``XRootD.client.File`` object.
        """
        return self._resource.file

    @property
    def closed(self):
        return self._resource.closed

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._resource.__exit__(exception_type, exception_value, traceback)

    @property
    def num_bytes(self):
        if self._num_bytes is None:
            self._num_bytes = self._resource.num_bytes
        return self._num_bytes


class MultithreadedXRootDSource(uproot4.source.chunk.MultithreadedSource):
    """
    Args:
        file_path (str): A URL of the file to open.
        options: Must include ``"num_workers"`` and ``"timeout"``.

    A :py:class:`~uproot4.source.chunk.MultithreadedSource` that manages many
    :py:class:`~uproot4.source.xrootd.XRootDResource` objects.
    """

    ResourceClass = XRootDResource

    def __init__(self, file_path, **options):
        num_workers = options["num_workers"]
        timeout = options["timeout"]
        self._num_requests = 0
        self._num_requested_chunks = 0
        self._num_requested_bytes = 0

        self._file_path = file_path
        self._num_bytes = None
        self._timeout = timeout

        self._executor = uproot4.source.futures.ResourceThreadPoolExecutor(
            [
                XRootDResource(file_path, timeout)
                for x in uproot4._util.range(num_workers)
            ]
        )

    @property
    def timeout(self):
        """
        The timeout in seconds or None.
        """
        return self._timeout

    @property
    def num_bytes(self):
        if self._num_bytes is None:
            self._num_bytes = self._executor.workers[0].resource.num_bytes
        return self._num_bytes
