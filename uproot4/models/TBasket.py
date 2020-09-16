# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines a versionless model for ``TBasket``, including much of the functionality
of basket-reading.

Includes both "embedded" ``TBaskets`` (as a member of TBranch) and "free"
``TBaskets`` (top-level objects, located by ``TKeys``).
"""

from __future__ import absolute_import

import struct

import numpy

import uproot4.model
import uproot4.deserialization
import uproot4.compression
import uproot4.behaviors.TBranch
import uproot4.const


_tbasket_format1 = struct.Struct(">ihiIhh")
_tbasket_format2 = struct.Struct(">Hiiii")
_tbasket_offsets_dtype = numpy.dtype(">i4")


class Model_TBasket(uproot4.model.Model):
    """
    A versionless :py:class:`~uproot4.model.Model` for ``TBasket``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    def __repr__(self):
        basket_num = self._basket_num if self._basket_num is not None else "(unknown)"
        return "<TBasket {0} of {1} at 0x{2:012x}>".format(
            basket_num, repr(self._parent.name), id(self)
        )

    @property
    def raw_data(self):
        """
        The raw but uncompressed data in the ``TBasket``, which combines data
        content with entry offsets, if the latter exists.

        If there are no entry offsets, this is identical to
        :py:attr:`~uproot4.models.TBasket.TBasket.data`.
        """
        return self._raw_data

    @property
    def data(self):
        """
        The uncompressed data content in the ``TBasket``, not including any
        entry offsets, if they exist.

        If there are no entry offsets, this is identical to
        :py:attr:`~uproot4.models.TBasket.TBasket.raw_data`.
        """
        return self._data

    @property
    def byte_offsets(self):
        """
        The index where each entry starts and stops in the
        :py:attr:`~uproot4.models.TBasket.TBasket.data`, not including header.

        The first offset is ``0`` and the number of offsets is one greater than
        the number of entries, such that the last offset is the length of
        :py:attr:`~uproot4.models.TBasket.TBasket.data`.
        """
        return self._byte_offsets

    def array(self, interpretation=None, library="ak"):
        """
        The ``TBasket`` data and entry offsets as an array, given an
        :py:class:`~uproot4.interpretation.Interpretation` (or the ``TBranch`` parent's
        :py:class:`~uproot4.behaviors.TBranch.TBranch.interpretation`) and a
        ``library``.
        """
        if interpretation is None:
            interpretation = self._parent.interpretation
        library = uproot4.interpretation.library._regularize_library(library)

        basket_array = interpretation.basket_array(
            self.data,
            self.byte_offsets,
            self,
            self._parent,
            self._parent.context,
            self._members["fKeylen"],
            library,
        )

        return interpretation.final_array(
            [basket_array],
            0,
            self.num_entries,
            [0, self.num_entries],
            library,
            self._parent,
        )

    @property
    def counts(self):
        """
        The number of items in each entry as a NumPy array, derived from the
        parent ``TBranch``'s
        :py:attr:`~uproot4.behavior.TBranch.TBranch.count_branch`. If there is
        no such branch (e.g. the data are ``std::vector``), then this method
        returns None.
        """
        count_branch = self._parent.count_branch
        if count_branch is not None:
            entry_offsets = count_branch.entry_offsets
            entry_start = entry_offsets[self._basket_num]
            entry_stop = entry_offsets[self._basket_num + 1]
            return count_branch.array(
                entry_start=entry_start, entry_stop=entry_stop, library="np"
            )
        else:
            return None

    @property
    def basket_num(self):
        """
        The index of this ``TBasket`` within its ``TBranch``.
        """
        return self._basket_num

    @property
    def key_version(self):
        """
        The instance version of the ``TKey`` for this ``TBasket`` (which is
        deserialized along with the ``TBasket``, unlike normal objects).
        """
        return self._key_version

    @property
    def num_entries(self):
        """
        The number of entries in this ``TBasket``.
        """
        return self._members["fNevBuf"]

    @property
    def is_embedded(self):
        """
        If this ``TBasket`` is embedded within its ``TBranch`` (i.e. must be
        deserialized as part of the ``TBranch``), then ``is_embedded`` is True.

        If this ``TBasket`` is a free-standing object, then ``is_embedded`` is
        False.
        """
        return self._members["fNbytes"] <= self._members["fKeylen"]

    @property
    def uncompressed_bytes(self):
        """
        The number of bytes for the uncompressed data, not including the header.

        If the ``TBasket`` is uncompressed, this is equal to
        :py:attr:`~uproot4.models.TBasket.TBasket.compressed_bytes`.
        """
        if self.is_embedded:
            if self._byte_offsets is None:
                return self._data.nbytes
            else:
                return self._data.nbytes + 4 + self.num_entries * 4
        else:
            return self._members["fObjlen"]

    @property
    def compressed_bytes(self):
        """
        The number of bytes for the compressed data, not including the header
        (which is always uncompressed).

        If the ``TBasket`` is uncompressed, this is equal to
        :py:attr:`~uproot4.models.TBasket.TBasket.uncompressed_bytes`.
        """
        if self.is_embedded:
            if self._byte_offsets is None:
                return self._data.nbytes
            else:
                return self._data.nbytes + 4 + self.num_entries * 4
        else:
            return self._members["fNbytes"] - self._members["fKeylen"]

    @property
    def border(self):
        """
        The byte position of the boundary between data content and entry offsets.

        Equal to ``self.member("fLast") - self.member("fKeylen")``.
        """
        return self._members["fLast"] - self._members["fKeylen"]

    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context, file):
        assert isinstance(self._parent, uproot4.behaviors.TBranch.TBranch)
        self._basket_num = context.get("basket_num")

        (
            self._members["fNbytes"],
            self._key_version,
            self._members["fObjlen"],
            self._members["fDatime"],
            self._members["fKeylen"],
            self._members["fCycle"],
        ) = cursor.fields(chunk, _tbasket_format1, context)

        # skip the class name, name, and title
        cursor.move_to(
            self._cursor.index + self._members["fKeylen"] - _tbasket_format2.size - 1
        )

        (
            self._members["fVersion"],
            self._members["fBufferSize"],
            self._members["fNevBufSize"],
            self._members["fNevBuf"],
            self._members["fLast"],
        ) = cursor.fields(chunk, _tbasket_format2, context)

        cursor.skip(1)

        if self.is_embedded:
            if self._members["fNevBufSize"] > 8:
                raw_byte_offsets = cursor.bytes(
                    chunk, 8 + self.num_entries * 4, context
                ).view(_tbasket_offsets_dtype)
                cursor.skip(-4)

                # subtracting fKeylen makes a new buffer and converts to native endian
                self._byte_offsets = raw_byte_offsets[1:] - self._members["fKeylen"]
                # so modifying it in place doesn't have non-local consequences
                self._byte_offsets[-1] = self.border

            else:
                self._byte_offsets = None

            # second key has no new information
            cursor.skip(self._members["fKeylen"])

            self._raw_data = None
            self._data = cursor.bytes(chunk, self.border, context)

        else:
            if self.compressed_bytes != self.uncompressed_bytes:
                uncompressed = uproot4.compression.decompress(
                    chunk, cursor, {}, self.compressed_bytes, self.uncompressed_bytes,
                )
                self._raw_data = uncompressed.get(
                    0,
                    self.uncompressed_bytes,
                    uproot4.source.cursor.Cursor(0),
                    context,
                )
            else:
                self._raw_data = cursor.bytes(chunk, self.uncompressed_bytes, context)

            if self.border != self.uncompressed_bytes:
                self._data = self._raw_data[: self.border]
                raw_byte_offsets = self._raw_data[self.border :].view(
                    _tbasket_offsets_dtype
                )

                # subtracting fKeylen makes a new buffer and converts to native endian
                self._byte_offsets = raw_byte_offsets[1:] - self._members["fKeylen"]
                # so modifying it in place doesn't have non-local consequences
                self._byte_offsets[-1] = self.border

            else:
                self._data = self._raw_data
                self._byte_offsets = None


uproot4.classes["TBasket"] = Model_TBasket
