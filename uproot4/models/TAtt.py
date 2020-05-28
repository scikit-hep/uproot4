# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct

import uproot4.model


_tattline_format1 = struct.Struct(">hhh")


class Model_TAttLine_v1(uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        (
            self._members["fLineColor"],
            self._members["fLineStyle"],
            self._members["fLineWidth"],
        ) = cursor.fields(chunk, _tattline_format1)

    base_names_versions = []
    member_names = ["fLineColor", "fLineStyle", "fLineWidth"]
    class_flags = {}
    hooks = {}
    class_code = ""


_tattfill_format1 = struct.Struct(">hh")


class Model_TAttFill_v1(uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        self._members["fFillColor"], self._members["fFillStyle"] = cursor.fields(
            chunk, _tattfill_format1
        )

    base_names_versions = []
    member_names = ["fFillColor", "fFillStyle"]
    class_flags = {}
    hooks = {}
    class_code = ""


_tattmarker_format1 = struct.Struct(">hhf")


class Model_TAttMarker_v2(uproot4.model.VersionedModel):
    def read_members(self, chunk, cursor, context):
        (
            self._members["fMarkerColor"],
            self._members["fMarkerStyle"],
            self._members["fMarkerSize"],
        ) = cursor.fields(chunk, _tattmarker_format1)

    base_names_versions = []
    member_names = ["fMarkerColor", "fMarkerStyle", "fMarkserSize"]
    class_flags = {}
    hooks = {}
    class_code = ""


class Model_TAttLine(uproot4.model.DispatchByVersion):
    known_versions = {1: Model_TAttLine_v1}


class Model_TAttFill(uproot4.model.DispatchByVersion):
    known_versions = {1: Model_TAttFill_v1}


class Model_TAttMarker(uproot4.model.DispatchByVersion):
    known_versions = {1: Model_TAttMarker_v2}


uproot4.classes["TAttLine"] = Model_TAttLine
uproot4.classes["TAttFill"] = Model_TAttFill
uproot4.classes["TAttMarker"] = Model_TAttMarker
