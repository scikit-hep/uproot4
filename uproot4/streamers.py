# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys
import struct
import re

import numpy

import uproot4._util
import uproot4.model
import uproot4.const
import uproot4.deserialization
import uproot4.models.TNamed
import uproot4.interpretation.identify


_canonical_typename_patterns = [
    (re.compile(r"\bChar_t\b"), "char"),
    (re.compile(r"\bUChar_t\b"), "unsigned char"),
    (re.compile(r"\bShort_t\b"), "short"),
    (re.compile(r"\bUShort_t\b"), "unsigned short"),
    (re.compile(r"\bInt_t\b"), "int"),
    (re.compile(r"\bUInt_t\b"), "unsigned int"),
    (re.compile(r"\bSeek_t\b"), "int"),  # file pointer
    (re.compile(r"\bLong_t\b"), "long"),
    (re.compile(r"\bULong_t\b"), "unsigned long"),
    (re.compile(r"\bFloat_t\b"), "float"),
    (
        re.compile(r"\bFloat16_t\b"),
        "Float16_t",
    ),  # 32-bit, written as 16, trunc mantissa
    (re.compile(r"\bDouble_t\b"), "double"),
    (re.compile(r"\bDouble32_t\b"), "Double32_t"),  # 64-bit, written as 32
    (re.compile(r"\bLongDouble_t\b"), "long double"),
    (re.compile(r"\bText_t\b"), "char"),
    (re.compile(r"\bBool_t\b"), "bool"),
    (re.compile(r"\bByte_t\b"), "unsigned char"),
    (re.compile(r"\bVersion_t\b"), "short"),  # class version id
    (re.compile(r"\bOption_t\b"), "const char"),  # option string
    (re.compile(r"\bSsiz_t\b"), "int"),  # string size
    (re.compile(r"\bReal_t\b"), "float"),  # TVector/TMatrix element
    (re.compile(r"\bLong64_t\b"), "long long"),  # portable int64
    (re.compile(r"\bULong64_t\b"), "unsigned long long"),  # portable uint64
    (re.compile(r"\bAxis_t\b"), "double"),  # axis values type
    (re.compile(r"\bStat_t\b"), "double"),  # statistics type
    (re.compile(r"\bFont_t\b"), "short"),  # font number
    (re.compile(r"\bStyle_t\b"), "short"),  # style number
    (re.compile(r"\bMarker_t\b"), "short"),  # marker number
    (re.compile(r"\bWidth_t\b"), "short"),  # line width
    (re.compile(r"\bColor_t\b"), "short"),  # color number
    (re.compile(r"\bSCoord_t\b"), "short"),  # screen coordinates
    (re.compile(r"\bCoord_t\b"), "double"),  # pad world coordinates
    (re.compile(r"\bAngle_t\b"), "float"),  # graphics angle
    (re.compile(r"\bSize_t\b"), "float"),  # attribute size
]


def _canonical_typename(name):
    for pattern, replacement in _canonical_typename_patterns:
        name = pattern.sub(replacement, name)
    return name


def _ftype_to_dtype(fType):
    if fType == uproot4.const.kBool:
        return "numpy.dtype(numpy.bool_)"
    elif fType == uproot4.const.kChar:
        return "numpy.dtype('i1')"
    elif fType in (uproot4.const.kUChar, uproot4.const.kCharStar):
        return "numpy.dtype('u1')"
    elif fType == uproot4.const.kShort:
        return "numpy.dtype('>i2')"
    elif fType == uproot4.const.kUShort:
        return "numpy.dtype('>u2')"
    elif fType == uproot4.const.kInt:
        return "numpy.dtype('>i4')"
    elif fType in (uproot4.const.kBits, uproot4.const.kUInt, uproot4.const.kCounter):
        return "numpy.dtype('>u4')"
    elif fType == uproot4.const.kLong:
        return "numpy.dtype('>i8')"
    elif fType == uproot4.const.kULong:
        return "numpy.dtype('>u8')"
    elif fType == uproot4.const.kLong64:
        return "numpy.dtype('>i8')"
    elif fType == uproot4.const.kULong64:
        return "numpy.dtype('>u8')"
    elif fType in (uproot4.const.kFloat, uproot4.const.kFloat16):
        return "numpy.dtype('>f4')"
    elif fType in (uproot4.const.kDouble, uproot4.const.kDouble32):
        return "numpy.dtype('>f8')"
    else:
        return None


def _ftype_to_struct(fType):
    if fType == uproot4.const.kBool:
        return "?"
    elif fType == uproot4.const.kChar:
        return "b"
    elif fType in (uproot4.const.kUChar, uproot4.const.kCharStar):
        return "B"
    elif fType == uproot4.const.kShort:
        return "h"
    elif fType == uproot4.const.kUShort:
        return "H"
    elif fType == uproot4.const.kInt:
        return "i"
    elif fType in (uproot4.const.kBits, uproot4.const.kUInt, uproot4.const.kCounter):
        return "I"
    elif fType == uproot4.const.kLong:
        return "q"
    elif fType == uproot4.const.kULong:
        return "Q"
    elif fType == uproot4.const.kLong64:
        return "q"
    elif fType == uproot4.const.kULong64:
        return "Q"
    elif fType in (uproot4.const.kFloat, uproot4.const.kFloat16):
        return "f"
    elif fType in (uproot4.const.kDouble, uproot4.const.kDouble32):
        return "d"
    else:
        raise NotImplementedError(fType)


_tstreamerinfo_format1 = struct.Struct(">Ii")


class Model_TStreamerInfo(uproot4.model.Model):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            uproot4.models.TNamed.Model_TNamed.read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._bases[0]._members["fName"] = _canonical_typename(
            self._bases[0]._members["fName"]
        )

        self._members["fCheckSum"], self._members["fClassVersion"] = cursor.fields(
            chunk, _tstreamerinfo_format1, context
        )

        self._members["fElements"] = uproot4.deserialization.read_object_any(
            chunk, cursor, context, self._file, self
        )

    def postprocess(self, chunk, cursor, context):
        # prevent circular dependencies and long-lived references to files
        self._file_uuid = self._file.uuid
        self._file = None
        self._parent = None
        return self

    def __repr__(self):
        return "<TStreamerInfo for {0} version {1} at 0x{2:012x}>".format(
            self.name, self.class_version, id(self)
        )

    @property
    def name(self):
        return self.member("fName")

    @property
    def typename(self):
        return self.member("fName")

    @property
    def class_version(self):
        return self._members["fClassVersion"]

    @property
    def elements(self):
        return self._members["fElements"]

    @property
    def file_uuid(self):
        return self._file_uuid

    def _dependencies(self, streamers, out):
        out.append((self.name, self.class_version))
        for element in self.elements:
            element._dependencies(streamers, out)

    def show(self, stream=sys.stdout):
        """
        Args:
            stream: Object with a `write` method for writing the output.
        """
        bases = []
        for element in self.elements:
            if isinstance(element, Model_TStreamerBase):
                bases.append(u"{0} (v{1})".format(element.name, element.base_version))
        if len(bases) == 0:
            bases = u""
        else:
            bases = u": " + u", ".join(bases)
        stream.write(u"{0} (v{1}){2}\n".format(self.name, self.class_version, bases))
        for element in self.elements:
            element.show(stream=stream)

    def new_class(self, file):
        class_code = self.class_code()
        class_name = uproot4.model.classname_encode(self.name, self.class_version)
        classes = uproot4.model.maybe_custom_classes(file.custom_classes)
        return uproot4.deserialization.compile_class(
            file, classes, class_code, class_name
        )

    def class_code(self):
        read_members = ["    def read_members(self, chunk, cursor, context):"]
        strided_interpretation = [
            "    @classmethod",
            "    def strided_interpretation(cls, file, header=False, "
            "tobject_header=True, original=None):",
            "        members = []",
            "        if header:",
            "            members.append(('@num_bytes', numpy.dtype('>u4')))",
            "            members.append(('@instance_version', numpy.dtype('>u2')))",
        ]
        awkward_form = [
            "    @classmethod",
            "    def awkward_form(cls, file, header=False, tobject_header=True):",
            "        from awkward1.forms import NumpyForm, ListOffsetForm, "
            "RegularForm, RecordForm",
            "        contents = {}",
            "        if header:",
            "            contents['@num_bytes'] = "
            "uproot4._util.awkward_form(numpy.dtype('u4'))",
            "            contents['@instance_version'] = "
            "uproot4._util.awkward_form(numpy.dtype('u2'))",
        ]
        fields = []
        formats = []
        dtypes = []
        stl_containers = []
        base_names_versions = []
        member_names = []
        class_flags = {}

        for i in range(len(self._members["fElements"])):
            self._members["fElements"][i].class_code(
                self,
                i,
                self._members["fElements"],
                read_members,
                strided_interpretation,
                awkward_form,
                fields,
                formats,
                dtypes,
                stl_containers,
                base_names_versions,
                member_names,
                class_flags,
            )

        if len(read_members) == 1:
            read_members.append("        pass")
        read_members.append("")

        strided_interpretation.append(
            "        return uproot4.interpretation.objects.AsStridedObjects"
            "(cls, members, original=original)"
        )
        strided_interpretation.append("")

        awkward_form.extend(
            [
                "        return RecordForm(contents, parameters={",
                "            '__record__': {0},".format(repr(self.name)),
                "            '__hidden_prefix__': '@'",
                "        })",
                "",
            ]
        )

        class_data = []

        for i, format in enumerate(formats):
            class_data.append(
                "    _format{0} = struct.Struct('>{1}')".format(i, "".join(format))
            )

        for i, dt in enumerate(dtypes):
            class_data.append("    _dtype{0} = {1}".format(i, dt))

        for i, stl in enumerate(stl_containers):
            class_data.append("    _stl_container{0} = {1}".format(i, stl))

        class_data.append(
            "    base_names_versions = [{0}]".format(
                ", ".join(
                    "({0}, {1})".format(repr(name), version)
                    for name, version in base_names_versions
                )
            )
        )

        class_data.append(
            "    member_names = [{0}]".format(", ".join(repr(x) for x in member_names))
        )

        class_data.append(
            "    class_flags = {{{0}}}".format(
                ", ".join(repr(k) + ": " + repr(v) for k, v in class_flags.items())
            )
        )

        return "\n".join(
            [
                "class {0}(uproot4.model.VersionedModel):".format(
                    uproot4.model.classname_encode(self.name, self.class_version)
                )
            ]
            + read_members
            + strided_interpretation
            + awkward_form
            + class_data
        )


_tstreamerelement_format1 = struct.Struct(">iiii")
_tstreamerelement_format2 = struct.Struct(">i")
_tstreamerelement_format3 = struct.Struct(">ddd")
_tstreamerelement_dtype1 = numpy.dtype(">i4")


class Model_TStreamerElement(uproot4.model.Model):
    def read_members(self, chunk, cursor, context):
        # https://github.com/root-project/root/blob/master/core/meta/src/TStreamerElement.cxx#L505

        self._bases.append(
            uproot4.models.TNamed.Model_TNamed.read(
                chunk, cursor, context, self._file, self._parent
            )
        )

        (
            self._members["fType"],
            self._members["fSize"],
            self._members["fArrayLength"],
            self._members["fArrayDim"],
        ) = cursor.fields(chunk, _tstreamerelement_format1, context)

        if self._instance_version == 1:
            n = cursor.field(chunk, _tstreamerelement_format2, context)
            self._members["fMaxIndex"] = cursor.array(
                chunk, n, _tstreamerelement_dtype1, context
            )
        else:
            self._members["fMaxIndex"] = cursor.array(
                chunk, 5, _tstreamerelement_dtype1, context
            )

        self._members["fTypeName"] = _canonical_typename(cursor.string(chunk, context))

        if self._members["fType"] == 11 and self._members["fTypeName"] in (
            "Bool_t" or "bool"
        ):
            self._members["fType"] = 18

        if self._instance_version <= 2:
            # FIXME
            # self._fSize = self._fArrayLength * gROOT->GetType(GetTypeName())->Size()
            pass

        if self._instance_version > 3:
            # FIXME
            # if (TestBit(kHasRange)) GetRange(GetTitle(),fXmin,fXmax,fFactor)
            pass

    def postprocess(self, chunk, cursor, context):
        # prevent circular dependencies and long-lived references to files
        self._file_uuid = self._file.uuid
        self._file = None
        self._parent = None
        return self

    @property
    def name(self):
        return self.member("fName")

    @property
    def title(self):
        return self.member("fTitle")

    @property
    def typename(self):
        return self.member("fTypeName")

    @property
    def array_length(self):
        return self.member("fArrayLength")

    @property
    def fType(self):
        return self.member("fType")

    @property
    def file_uuid(self):
        return self._file_uuid

    def _dependencies(self, streamers, out):
        pass

    def show(self, stream=sys.stdout):
        """
        Args:
            stream: Object with a `write` method for writing the output.
        """
        stream.write(
            u"    {0}: {1} ({2})\n".format(
                self.name,
                self.typename,
                uproot4.model.classname_decode(type(self).__name__)[0],
            )
        )


class Model_TStreamerArtificial(Model_TStreamerElement):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            Model_TStreamerElement.read(
                chunk, cursor, context, self._file, self._parent
            )
        )

    def class_code(
        self,
        streamerinfo,
        i,
        elements,
        read_members,
        strided_interpretation,
        awkward_form,
        fields,
        formats,
        dtypes,
        stl_containers,
        base_names_versions,
        member_names,
        class_flags,
    ):
        read_members.append(
            "        raise uproot4.deserialization.DeserializationError("
            "'not implemented: class members defined by {0} of type {1} in member "
            "{2} of class {3}', chunk, cursor, context, self._file.file_path)".format(
                type(self).__name__, self.typename, self.name, streamerinfo.name
            )
        )

        strided_interpretation.append(
            "        raise uproot4.deserialization.CannotBeStrided("
            "'not implemented: class members defined by {0} of type {1} in member "
            "{2} of class {3}')".format(
                type(self).__name__, self.typename, self.name, streamerinfo.name
            )
        )

        awkward_form.append(
            "        raise uproot4.deserialization.CannotBeAwkward("
            "'not implemented: class members defined by {0} of type {1} in member "
            "{2} of class {3}')".format(
                type(self).__name__, self.typename, self.name, streamerinfo.name
            )
        )


_tstreamerbase_format1 = struct.Struct(">i")


class Model_TStreamerBase(Model_TStreamerElement):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            Model_TStreamerElement.read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        if self._instance_version >= 2:
            self._members["fBaseVersion"] = cursor.field(
                chunk, _tstreamerbase_format1, context
            )

    @property
    def base_version(self):
        return self._members["fBaseVersion"]

    def _dependencies(self, streamers, out):
        streamer_versions = streamers.get(self.name)
        if streamer_versions is not None:
            streamer = streamer_versions.get(self.base_version)
            if (
                streamer is not None
                and (streamer.name, streamer.class_version) not in out
            ):
                streamer._dependencies(streamers, out)

    def show(self, stream=sys.stdout):
        """
        Args:
            stream: Object with a `write` method for writing the output.
        """

    def class_code(
        self,
        streamerinfo,
        i,
        elements,
        read_members,
        strided_interpretation,
        awkward_form,
        fields,
        formats,
        dtypes,
        stl_containers,
        base_names_versions,
        member_names,
        class_flags,
    ):
        read_members.append(
            "        self._bases.append(c({0}, {1}).read(chunk, cursor, "
            "context, self._file, self._parent))".format(
                repr(self.name), self.base_version
            )
        )
        strided_interpretation.append(
            "        members.extend(file.class_named({0}, {1})."
            "strided_interpretation(file, header, tobject_header).members)".format(
                repr(self.name), self.base_version
            )
        )
        awkward_form.append(
            "        contents.update(file.class_named({0}, {1}).awkward_form(file, "
            "header, tobject_header).contents)".format(
                repr(self.name), self.base_version
            )
        )

        base_names_versions.append((self.name, self.base_version))


_tstreamerbasicpointer_format1 = struct.Struct(">i")


class Model_TStreamerBasicPointer(Model_TStreamerElement):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            Model_TStreamerElement.read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._members["fCountVersion"] = cursor.field(
            chunk, _tstreamerbasicpointer_format1, context
        )
        self._members["fCountName"] = cursor.string(chunk, context)
        self._members["fCountClass"] = cursor.string(chunk, context)

    @property
    def count_name(self):
        return self._members["fCountName"]

    def class_code(
        self,
        streamerinfo,
        i,
        elements,
        read_members,
        strided_interpretation,
        awkward_form,
        fields,
        formats,
        dtypes,
        stl_containers,
        base_names_versions,
        member_names,
        class_flags,
    ):
        read_members.append("        tmp = self._dtype{0}".format(len(dtypes)))

        if streamerinfo.name == "TBranch" and self.name == "fBasketSeek":
            read_members.append("        if context.get('speedbump', True):")
            read_members.append(
                "            if cursor.bytes(chunk, 1, context)[0] == 2:"
            )
            read_members.append("                tmp = numpy.dtype('>i8')")

        else:
            read_members.append("        if context.get('speedbump', True):")
            read_members.append("            cursor.skip(1)")

        read_members.append(
            "        self._members[{0}] = cursor.array(chunk, self.member({1}), tmp, context)".format(
                repr(self.name), repr(self.count_name)
            )
        )

        strided_interpretation.append(
            "        raise uproot4.deserialization.CannotBeStrided("
            "'class members defined by {0} of type {1} in member "
            "{2} of class {3}')".format(
                type(self).__name__, self.typename, self.name, streamerinfo.name
            )
        )

        awkward_form.extend(
            [
                "        contents[{0}] = ListOffsetForm('i32', "
                "uproot4._util.awkward_form(cls._dtype{1}),".format(
                    repr(self.name), len(dtypes)
                ),
                "            parameters={'uproot': {'as': 'TStreamerBasicPointer', "
                "'count_name': " + repr(self.count_name) + "}}",
                "        )",
            ]
        )

        member_names.append(self.name)
        dtypes.append(_ftype_to_dtype(self.fType - uproot4.const.kOffsetP))


class Model_TStreamerBasicType(Model_TStreamerElement):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            Model_TStreamerElement.read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        if (
            uproot4.const.kOffsetL
            < self._bases[0]._members["fType"]
            < uproot4.const.kOffsetP
        ):
            self._bases[0]._members["fType"] -= uproot4.const.kOffsetL

        basic = True

        if self._bases[0]._members["fType"] in (
            uproot4.const.kBool,
            uproot4.const.kUChar,
            uproot4.const.kChar,
        ):
            self._bases[0]._members["fSize"] = 1

        elif self._bases[0]._members["fType"] in (
            uproot4.const.kUShort,
            uproot4.const.kShort,
        ):
            self._bases[0]._members["fSize"] = 2

        elif self._bases[0]._members["fType"] in (
            uproot4.const.kBits,
            uproot4.const.kUInt,
            uproot4.const.kInt,
            uproot4.const.kCounter,
        ):
            self._bases[0]._members["fSize"] = 4

        elif self._bases[0]._members["fType"] in (
            uproot4.const.kULong,
            uproot4.const.kLong,
        ):
            self._bases[0]._members["fSize"] = numpy.dtype(numpy.long).itemsize

        elif self._bases[0]._members["fType"] in (
            uproot4.const.kULong64,
            uproot4.const.kLong64,
        ):
            self._bases[0]._members["fSize"] = 8

        elif self._bases[0]._members["fType"] in (
            uproot4.const.kFloat,
            uproot4.const.kFloat16,
        ):
            self._bases[0]._members["fSize"] = 4

        elif self._bases[0]._members["fType"] in (
            uproot4.const.kDouble,
            uproot4.const.kDouble32,
        ):
            self._bases[0]._members["fSize"] = 8

        elif self._bases[0]._members["fType"] == uproot4.const.kCharStar:
            self._bases[0]._members["fSize"] = numpy.dtype(numpy.intp).itemsize

        else:
            basic = False

        if basic and self._bases[0]._members["fArrayLength"] > 0:
            self._bases[0]._members["fSize"] *= self._bases[0]._members["fArrayLength"]

    def class_code(
        self,
        streamerinfo,
        i,
        elements,
        read_members,
        strided_interpretation,
        awkward_form,
        fields,
        formats,
        dtypes,
        stl_containers,
        base_names_versions,
        member_names,
        class_flags,
    ):
        if self.typename == "Double32_t":
            read_members.append(
                "        self._members[{0}] = cursor.double32(chunk, "
                "context)".format(repr(self.name))
            )

        elif self.typename == "Float16_t":
            read_members.append(
                "        self._members[{0}] = cursor.float16(chunk, 12, "
                "context)".format(repr(self.name))
            )

        elif self.array_length == 0:
            if (
                i == 0
                or not isinstance(elements[i - 1], Model_TStreamerBasicType)
                or elements[i - 1].array_length != 0
                or elements[i - 1].typename in ("Double32_t", "Float16_t")
            ):
                fields.append([])
                formats.append([])

            fields[-1].append(self.name)
            formats[-1].append(_ftype_to_struct(self.fType))

            if (
                i + 1 == len(elements)
                or not isinstance(elements[i + 1], Model_TStreamerBasicType)
                or elements[i + 1].array_length != 0
            ):
                if len(fields[-1]) == 1:
                    read_members.append(
                        "        self._members['{0}'] = cursor.field(chunk, "
                        "self._format{1}, context)".format(
                            fields[-1][0], len(formats) - 1
                        )
                    )
                else:
                    read_members.append(
                        "        {0} = cursor.fields(chunk, self._format{1}, context)".format(
                            ", ".join(
                                "self._members[{0}]".format(repr(x)) for x in fields[-1]
                            ),
                            len(formats) - 1,
                        )
                    )

        else:
            read_members.append(
                "        self._members[{0}] = cursor.array(chunk, {1}, "
                "self._dtype{2}, context)".format(
                    repr(self.name), self.array_length, len(dtypes)
                )
            )
            dtypes.append(_ftype_to_dtype(self.fType))

        if self.array_length == 0 and self.typename not in ("Double32_t", "Float16_t"):
            strided_interpretation.append(
                "        members.append(({0}, {1}))".format(
                    repr(self.name), _ftype_to_dtype(self.fType)
                )
            )
        else:
            strided_interpretation.append(
                "        raise uproot4.deserialization.CannotBeStrided("
                "'class members defined by {0} of type {1} in member "
                "{2} of class {3}')".format(
                    type(self).__name__, self.typename, self.name, streamerinfo.name
                )
            )

        if self.array_length == 0:
            if self.typename == "Double32_t":
                awkward_form.extend(
                    [
                        "        contents[{0}] = NumpyForm((), 8, 'd',".format(
                            repr(self.name)
                        ),
                        "            parameters={'uproot': {'as': 'Double32'}})",
                        "        )",
                    ]
                )

            elif self.typename == "Float16_t":
                awkward_form.extend(
                    [
                        "        contents[{0}] = NumpyForm((), 4, 'f',".format(
                            repr(self.name)
                        ),
                        "            parameters={'uproot': {'as': 'Float16'}})",
                        "        )",
                    ]
                )

            else:
                awkward_form.append(
                    "        contents[{0}] = uproot4._util.awkward_form({1})".format(
                        repr(self.name), _ftype_to_dtype(self.fType)
                    )
                )

        else:
            awkward_form.append(
                "        contents[{0}] = RegularForm(uproot4._util.awkward_form({1}), {2})".format(
                    repr(self.name), _ftype_to_dtype(self.fType), self.array_length
                )
            )

        member_names.append(self.name)


_tstreamerloop_format1 = struct.Struct(">i")


class Model_TStreamerLoop(Model_TStreamerElement):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            Model_TStreamerElement.read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._members["fCountVersion"] = cursor.field(
            chunk, _tstreamerloop_format1, context
        )
        self._members["fCountName"] = cursor.string(chunk, context)
        self._members["fCountClass"] = cursor.string(chunk, context)

    @property
    def count_name(self):
        return self._members["fCountName"]

    def _dependencies(self, streamers, out):
        streamer_versions = streamers.get(self.typename.rstrip("*"))
        if streamer_versions is not None:
            for streamer in streamer_versions.values():
                if (streamer.name, streamer.class_version) not in out:
                    streamer._dependencies(streamers, out)

    def class_code(
        self,
        streamerinfo,
        i,
        elements,
        read_members,
        strided_interpretation,
        awkward_form,
        fields,
        formats,
        dtypes,
        stl_containers,
        base_names_versions,
        member_names,
        class_flags,
    ):
        read_members.extend(
            [
                "        cursor.skip(6)",
                "        for tmp in range(self.member({0})):".format(self.count_name),
                "            self._members[{0}] = c({1}).read(chunk, cursor, "
                "context, self._file, self)".format(
                    repr(self.name), repr(self.typename.rstrip("*"))
                ),
            ]
        )

        strided_interpretation.append(
            "        raise uproot4.deserialization.CannotBeStrided("
            "'class members defined by {0} of type {1} in member "
            "{2} of class {3}')".format(
                type(self).__name__, self.typename, self.name, streamerinfo.name
            )
        )

        awkward_form.extend(
            [
                "        tmp = file.class_named({0}, 'max').awkward_form(file, "
                "header, tobject_header)".format(repr(self.typename.rstrip("*"))),
                "        contents[" + repr(self.name) + "] = ListOffsetForm('i32', "
                "tmp, parameters={'uproot': {'as': TStreamerLoop, 'count_name': "
                + repr(self.count_name)
                + "}})",
            ]
        )

        member_names.append(self.name)


_tstreamerstl_format1 = struct.Struct(">ii")


class Model_TStreamerSTL(Model_TStreamerElement):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            Model_TStreamerElement.read(
                chunk, cursor, context, self._file, self._parent
            )
        )
        self._members["fSTLtype"], self._members["fCtype"] = cursor.fields(
            chunk, _tstreamerstl_format1, context
        )

        if self._members["fSTLtype"] in (
            uproot4.const.kSTLmultimap,
            uproot4.const.kSTLset,
        ):
            if self._bases[0]._members["fTypeName"].startswith(
                "std::set"
            ) or self._bases[0]._members["fTypeName"].startswith("set"):
                self._members["fSTLtype"] = uproot4.const.kSTLset

            elif self._bases[0]._members["fTypeName"].startswith(
                "std::multimap"
            ) or self._bases[0]._members["fTypeName"].startswith("multimap"):
                self._members["fSTLtype"] = uproot4.const.kSTLmultimap

    @property
    def stl_type(self):
        return self._members["fSTLtype"]

    @property
    def fCtype(self):
        return self._members["fCtype"]

    def class_code(
        self,
        streamerinfo,
        i,
        elements,
        read_members,
        strided_interpretation,
        awkward_form,
        fields,
        formats,
        dtypes,
        stl_containers,
        base_names_versions,
        member_names,
        class_flags,
    ):
        stl_container = uproot4.interpretation.identify.parse_typename(
            self.typename,
            quote=True,
            outer_header=True,
            inner_header=False,
            string_header=True,
        )
        read_members.append(
            "        self._members[{0}] = self._stl_container{1}.read("
            "chunk, cursor, context, self._file, self)"
            "".format(repr(self.name), len(stl_containers))
        )

        strided_interpretation.append(
            "        members.append(({0}, cls._stl_container{1}."
            "strided_interpretation(file, header, tobject_header)))".format(
                repr(self.name), len(stl_containers)
            )
        )

        awkward_form.append(
            "        contents[{0}] = cls._stl_container{1}.awkward_form".format(
                repr(self.name), len(stl_containers)
            )
        )

        stl_containers.append(stl_container)
        member_names.append(self.name)


class Model_TStreamerSTLstring(Model_TStreamerSTL):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            Model_TStreamerSTL.read(chunk, cursor, context, self._file, self._parent)
        )


class pointer_types(object):
    def _dependencies(self, streamers, out):
        streamer_versions = streamers.get(self.typename.rstrip("*"))
        if streamer_versions is not None:
            for streamer in streamer_versions.values():
                if (streamer.name, streamer.class_version) not in out:
                    streamer._dependencies(streamers, out)

    def class_code(
        self,
        streamerinfo,
        i,
        elements,
        read_members,
        strided_interpretation,
        awkward_form,
        fields,
        formats,
        dtypes,
        stl_containers,
        base_names_versions,
        member_names,
        class_flags,
    ):
        if self.fType == uproot4.const.kObjectp or self.fType == uproot4.const.kAnyp:
            read_members.append(
                "        self._members[{0}] = c({1}).read(chunk, cursor, context, "
                "self._file, self)".format(
                    repr(self.name), repr(self.typename.rstrip("*"))
                )
            )
            strided_interpretation.append(
                "        members.append(({0}, file.class_named({1}, 'max')."
                "strided_interpretation(file, header, tobject_header)))".format(
                    repr(self.name), repr(self.typename.rstrip("*"))
                )
            )
            awkward_form.append(
                "        contents[{0}] = file.class_named({1}, 'max').awkward_form(file, "
                "header, tobject_header)".format(
                    repr(self.name), repr(self.typename.rstrip("*"))
                )
            )

        elif self.fType == uproot4.const.kObjectP or self.fType == uproot4.const.kAnyP:
            read_members.append(
                "        self._members[{0}] = read_object_any(chunk, cursor, "
                "context, self._file, self)".format(repr(self.name))
            )
            strided_interpretation.append(
                "        raise uproot4.deserialization.CannotBeStrided("
                "'class members defined by {0} of type {1} in member "
                "{2} of class {3}')".format(
                    type(self).__name__, self.typename, self.name, streamerinfo.name
                )
            )
            class_flags["has_read_object_any"] = True

        else:
            read_members.append(
                "        raise uproot4.deserialization.DeserializationError("
                "'not implemented: class members defined by {0} with fType {1}', "
                "chunk, cursor, context, self._file.file_path)".format(
                    type(self).__name__, self.fType,
                )
            )

        member_names.append(self.name)


class Model_TStreamerObjectAnyPointer(pointer_types, Model_TStreamerElement):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            Model_TStreamerElement.read(
                chunk, cursor, context, self._file, self._parent
            )
        )


class Model_TStreamerObjectPointer(pointer_types, Model_TStreamerElement):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            Model_TStreamerElement.read(
                chunk, cursor, context, self._file, self._parent
            )
        )


class object_types(object):
    def _dependencies(self, streamers, out):
        streamer_versions = streamers.get(self.typename.rstrip("*"))
        if streamer_versions is not None:
            for streamer in streamer_versions.values():
                if (streamer.name, streamer.class_version) not in out:
                    streamer._dependencies(streamers, out)

    def class_code(
        self,
        streamerinfo,
        i,
        elements,
        read_members,
        strided_interpretation,
        awkward_form,
        fields,
        formats,
        dtypes,
        stl_containers,
        base_names_versions,
        member_names,
        class_flags,
    ):
        read_members.append(
            "        self._members[{0}] = c({1}).read(chunk, cursor, context, "
            "self._file, self)".format(repr(self.name), repr(self.typename.rstrip("*")))
        )

        strided_interpretation.append(
            "        members.append(({0}, file.class_named({1}, 'max')."
            "strided_interpretation(file, header, tobject_header)))".format(
                repr(self.name), repr(self.typename.rstrip("*"))
            )
        )
        awkward_form.append(
            "        contents[{0}] = file.class_named({1}, 'max').awkward_form(file, "
            "header, tobject_header)".format(
                repr(self.name), repr(self.typename.rstrip("*"))
            )
        )

        member_names.append(self.name)


class Model_TStreamerObject(object_types, Model_TStreamerElement):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            Model_TStreamerElement.read(
                chunk, cursor, context, self._file, self._parent
            )
        )


class Model_TStreamerObjectAny(object_types, Model_TStreamerElement):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            Model_TStreamerElement.read(
                chunk, cursor, context, self._file, self._parent
            )
        )


class Model_TStreamerString(object_types, Model_TStreamerElement):
    def read_members(self, chunk, cursor, context):
        self._bases.append(
            Model_TStreamerElement.read(
                chunk, cursor, context, self._file, self._parent
            )
        )


uproot4.classes["TStreamerInfo"] = Model_TStreamerInfo
uproot4.classes["TStreamerElement"] = Model_TStreamerElement
uproot4.classes["TStreamerArtificial"] = Model_TStreamerArtificial
uproot4.classes["TStreamerBase"] = Model_TStreamerBase
uproot4.classes["TStreamerBasicPointer"] = Model_TStreamerBasicPointer
uproot4.classes["TStreamerBasicType"] = Model_TStreamerBasicType
uproot4.classes["TStreamerLoop"] = Model_TStreamerLoop
uproot4.classes["TStreamerObject"] = Model_TStreamerObject
uproot4.classes["TStreamerObjectAny"] = Model_TStreamerObjectAny
uproot4.classes["TStreamerObjectAnyPointer"] = Model_TStreamerObjectAnyPointer
uproot4.classes["TStreamerObjectPointer"] = Model_TStreamerObjectPointer
uproot4.classes["TStreamerSTL"] = Model_TStreamerSTL
uproot4.classes["TStreamerSTLstring"] = Model_TStreamerSTLstring
uproot4.classes["TStreamerString"] = Model_TStreamerString
