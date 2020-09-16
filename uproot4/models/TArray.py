# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines versionless models for ``TArray`` and its subclasses.
"""

from __future__ import absolute_import

import struct

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import numpy

import uproot4.model
import uproot4._util


_tarray_format1 = struct.Struct(">i")


class Model_TArray(uproot4.model.Model, Sequence):
    """
    A versionless :py:class:`~uproot4.model.Model` for ``TArray`` and its subclasses.

    These also satisfy Python's abstract ``Sequence`` protocol.
    """

    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context, file):
        if self.is_memberwise:
            raise NotImplementedError(
                """memberwise serialization of {0}
in file {1}""".format(
                    type(self).__name__, self.file.file_path
                )
            )
        self._members["fN"] = cursor.field(chunk, _tarray_format1, context)
        self._data = cursor.array(chunk, self._members["fN"], self.dtype, context)

    def __array__(self, *args, **kwargs):
        if len(args) == len(kwargs) == 0:
            return self._data
        else:
            return numpy.array(self._data, *args, **kwargs)

    @property
    def nbytes(self):
        return self._data.nbytes

    def __getitem__(self, where):
        return self._data[where]

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        if self.class_version is None:
            version = ""
        else:
            version = " (version {0})".format(self.class_version)
        return "<{0}{1} {2} at 0x{3:012x}>".format(
            self.classname,
            version,
            numpy.array2string(
                self._data,
                max_line_width=numpy.inf,
                separator=", ",
                formatter={"float": lambda x: "%g" % x},
                threshold=6,
            ),
            id(self),
        )

    def tojson(self):
        return self._data.tolist()

    @classmethod
    def awkward_form(cls, file, index_format="i64", header=False, tobject_header=True):
        import awkward1

        return awkward1.forms.ListOffsetForm(
            index_format,
            uproot4._util.awkward_form(
                cls.dtype, file, index_format, header, tobject_header
            ),
            parameters={"uproot": {"as": "TArray"}},
        )


class Model_TArrayC(Model_TArray):
    """
    A versionless :py:class:`~uproot4.model.Model` for ``TArrayC``
    (``dtype(">i1")``).

    It also satisfies Python's abstract ``Sequence`` protocol.
    """

    dtype = numpy.dtype(">i1")


class Model_TArrayS(Model_TArray):
    """
    A versionless :py:class:`~uproot4.model.Model` for ``TArrayS``
    (``dtype(">i2")``).

    It also satisfies Python's abstract ``Sequence`` protocol.
    """

    dtype = numpy.dtype(">i2")


class Model_TArrayI(Model_TArray):
    """
    A versionless :py:class:`~uproot4.model.Model` for ``TArrayI``
    (``dtype(">i4")``).

    It also satisfies Python's abstract ``Sequence`` protocol.
    """

    dtype = numpy.dtype(">i4")


class Model_TArrayL(Model_TArray):
    """
    A versionless :py:class:`~uproot4.model.Model` for ``TArrayL``
    (``dtype(">i8")``).

    It also satisfies Python's abstract ``Sequence`` protocol.
    """

    dtype = numpy.dtype(">i8")


class Model_TArrayL64(Model_TArray):
    """
    A versionless :py:class:`~uproot4.model.Model` for ``TArrayL64``
    (``dtype(">i8")``).

    It also satisfies Python's abstract ``Sequence`` protocol.
    """

    dtype = numpy.dtype(">i8")


class Model_TArrayF(Model_TArray):
    """
    A versionless :py:class:`~uproot4.model.Model` for ``TArrayF``
    (``dtype(">f4")``).

    It also satisfies Python's abstract ``Sequence`` protocol.
    """

    dtype = numpy.dtype(">f4")


class Model_TArrayD(Model_TArray):
    """
    A versionless :py:class:`~uproot4.model.Model` for ``TArrayD``
    (``dtype(">f8")``).

    It also satisfies Python's abstract ``Sequence`` protocol.
    """

    dtype = numpy.dtype(">f8")


uproot4.classes["TArray"] = Model_TArray
uproot4.classes["TArrayC"] = Model_TArrayC
uproot4.classes["TArrayS"] = Model_TArrayS
uproot4.classes["TArrayI"] = Model_TArrayI
uproot4.classes["TArrayL"] = Model_TArrayL
uproot4.classes["TArrayL64"] = Model_TArrayL64
uproot4.classes["TArrayF"] = Model_TArrayF
uproot4.classes["TArrayD"] = Model_TArrayD
