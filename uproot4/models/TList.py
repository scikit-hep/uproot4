# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import uproot4.model
import uproot4.deserialization
import uproot4.models.TObject


_tlist_format1 = struct.Struct(">i")
_tlist_format2 = struct.Struct(">B")


class Model_TList(uproot4.model.Model, Sequence):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            uproot4.models.TObject.Model_TObject.read(
                chunk, cursor, context, self._file, self._parent
            )
        )

        self._members["fName"] = cursor.string(chunk)
        self._members["fSize"] = cursor.field(chunk, _tlist_format1)

        self._data = []
        for i in range(self._members["fSize"]):
            item = uproot4.deserialization.read_object_any(
                chunk, cursor, context, self._file, self._parent
            )
            self._data.append(item)

            # ignore "option"
            n = cursor.field(chunk, _tlist_format2)
            cursor.skip(n)

    def __getitem__(self, where):
        return self._data[where]

    def __len__(self):
        return len(self._data)

    def tojson(self):
        return {
            "_typename": "TList",
            "name": "TList",
            "arr": [x.tojson() for x in self._data],
            "opt": [],
        }


uproot4.classes["TList"] = Model_TList
