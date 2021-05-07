# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines models for ``TStreamerInfo`` and its elements, as well as
routines for generating Python code for new classes from streamer data.
"""

from __future__ import absolute_import

import re
import struct
import sys

import numpy

import uproot

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
    if fType == uproot.const.kBool:
        return "numpy.dtype(numpy.bool_)"
    elif fType == uproot.const.kChar:
        return "numpy.dtype('i1')"
    elif fType in (uproot.const.kUChar, uproot.const.kCharStar):
        return "numpy.dtype('u1')"
    elif fType == uproot.const.kShort:
        return "numpy.dtype('>i2')"
    elif fType == uproot.const.kUShort:
        return "numpy.dtype('>u2')"
    elif fType == uproot.const.kInt:
        return "numpy.dtype('>i4')"
    elif fType in (uproot.const.kBits, uproot.const.kUInt, uproot.const.kCounter):
        return "numpy.dtype('>u4')"
    elif fType == uproot.const.kLong:
        return "numpy.dtype('>i8')"
    elif fType == uproot.const.kULong:
        return "numpy.dtype('>u8')"
    elif fType == uproot.const.kLong64:
        return "numpy.dtype('>i8')"
    elif fType == uproot.const.kULong64:
        return "numpy.dtype('>u8')"
    elif fType in (uproot.const.kFloat, uproot.const.kFloat16):
        return "numpy.dtype('>f4')"
    elif fType in (uproot.const.kDouble, uproot.const.kDouble32):
        return "numpy.dtype('>f8')"
    else:
        return None


def _ftype_to_struct(fType):
    if fType == uproot.const.kBool:
        return "?"
    elif fType == uproot.const.kChar:
        return "b"
    elif fType in (uproot.const.kUChar, uproot.const.kCharStar):
        return "B"
    elif fType == uproot.const.kShort:
        return "h"
    elif fType == uproot.const.kUShort:
        return "H"
    elif fType == uproot.const.kInt:
        return "i"
    elif fType in (uproot.const.kBits, uproot.const.kUInt, uproot.const.kCounter):
        return "I"
    elif fType == uproot.const.kLong:
        return "q"
    elif fType == uproot.const.kULong:
        return "Q"
    elif fType == uproot.const.kLong64:
        return "q"
    elif fType == uproot.const.kULong64:
        return "Q"
    elif fType in (uproot.const.kFloat, uproot.const.kFloat16):
        return "f"
    elif fType in (uproot.const.kDouble, uproot.const.kDouble32):
        return "d"
    else:
        raise NotImplementedError(fType)


def _copy_bytes(chunk, start, stop, cursor, context):
    out = chunk.get(start, stop, cursor, context)
    if hasattr(out, "tobytes"):
        return out.tobytes()
    else:
        return out.tostring()


_tstreamerinfo_format1 = struct.Struct(">Ii")


class Model_TStreamerInfo(uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerInfo``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    def __repr__(self):
        return "<TStreamerInfo for {0} version {1} at 0x{2:012x}>".format(
            self.name, self.class_version, id(self)
        )

    def show(self, stream=sys.stdout):
        """
        Args:
            stream (object with a ``write(str)`` method): Stream to write the
                output to.

        Interactively display a ``TStreamerInfo``.

        For example,

        .. code-block::

            TLorentzVector (v4): TObject (v1)
                fP: TVector3 (TStreamerObject)
                fE: double (TStreamerBasicType)
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

    @property
    def name(self):
        """
        The name (``fName``) of this ``TStreamerInfo``, passed through
        :doc:`uproot.model.classname_regularize`
        """
        return uproot.model.classname_regularize(self.member("fName"))

    @property
    def typename(self):
        """
        The typename/classname (``fName``) of this ``TStreamerInfo``.
        """
        return self.member("fName")

    @property
    def elements(self):
        """
        This ``TStreamerInfo``'s list of ``TStreamerElements``
        (:doc:`uproot.streamers.Model_TStreamerElement`).
        """
        return self._members["fElements"]

    @property
    def class_version(self):
        """
        The class version (``fClassVersion``) of this ``TStreamerInfo``.
        """
        return self._members["fClassVersion"]

    def class_code(self):
        """
        Returns Python code as a string that, when evaluated, would be a suitable
        :doc:`uproot.model.VersionedModel` for this class and version.
        """
        read_members = [
            "    def read_members(self, chunk, cursor, context, file):",
            "        if self.is_memberwise:",
            "            raise NotImplementedError(",
            '                "memberwise serialization of {0}\\nin file {1}".format('
            "type(self).__name__, self.file.file_path)",
            "            )",
        ]
        read_member_n = [
            "    def read_member_n(self, chunk, cursor, context, file, member_index):"
        ]
        strided_interpretation = [
            "    @classmethod",
            "    def strided_interpretation(cls, file, header=False, "
            "tobject_header=True, breadcrumbs=(), original=None):",
            "        if cls in breadcrumbs:",
            "            raise uproot.interpretation.objects.CannotBeStrided("
            "'classes that can contain members of the same type cannot be strided "
            "because the depth of instances is unbounded')",
            "        breadcrumbs = breadcrumbs + (cls,)",
            "        members = []",
            "        if header:",
            "            members.append(('@num_bytes', numpy.dtype('>u4')))",
            "            members.append(('@instance_version', numpy.dtype('>u2')))",
        ]
        awkward_form = [
            "    @classmethod",
            "    def awkward_form(cls, file, index_format='i64', header=False, "
            "tobject_header=True, breadcrumbs=()):",
            "        from awkward.forms import NumpyForm, ListOffsetForm, "
            "RegularForm, RecordForm",
            "        if cls in breadcrumbs:",
            "            raise uproot.interpretation.objects.CannotBeAwkward("
            "'classes that can contain members of the same type cannot be Awkward "
            "Arrays because the depth of instances is unbounded')",
            "        breadcrumbs = breadcrumbs + (cls,)",
            "        contents = {}",
            "        if header:",
            "            contents['@num_bytes'] = "
            "uproot._util.awkward_form(numpy.dtype('u4'), file, index_format, "
            "header, tobject_header, breadcrumbs)",
            "            contents['@instance_version'] = "
            "uproot._util.awkward_form(numpy.dtype('u2'), file, index_format, "
            "header, tobject_header, breadcrumbs)",
        ]
        fields = []
        formats = []
        dtypes = []
        formats_memberwise = []
        has_memberwise_header = []
        containers = []
        base_names_versions = []
        member_names = []
        class_flags = {}

        for i in uproot._util.range(len(self._members["fElements"])):
            self._members["fElements"][i].class_code(
                self,
                i,
                self._members["fElements"],
                read_members,
                read_member_n,
                strided_interpretation,
                awkward_form,
                fields,
                formats,
                dtypes,
                formats_memberwise,
                has_memberwise_header,
                containers,
                base_names_versions,
                member_names,
                class_flags,
            )

        if len(read_members) == 1:
            read_members.append("        pass")
        if len(read_member_n) == 1:
            read_member_n.append("        pass")

        read_members.append("")
        read_member_n.append("")

        strided_interpretation.append(
            "        return uproot.interpretation.objects.AsStridedObjects"
            "(cls, members, original=original)"
        )
        strided_interpretation.append("")

        awkward_form.append(
            "        return RecordForm(contents, parameters={'__record__': "
            + repr(self.name)
            + "})"
        )
        awkward_form.append("")

        class_data = []

        for i, format in enumerate(formats):
            class_data.append(
                "    _format{0} = struct.Struct('>{1}')".format(i, "".join(format))
            )

        for i, format in enumerate(formats_memberwise):
            class_data.append(
                "    _format_memberwise{0} = struct.Struct('>{1}')".format(
                    i, "".join(format)
                )
            )

        class_data.append("    has_memberwise_header = {0}".format(str(has_memberwise_header)))

        for i, dt in enumerate(dtypes):
            class_data.append("    _dtype{0} = {1}".format(i, dt))

        for i, stl in enumerate(containers):
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

        class_name = uproot.model.classname_encode(self.name, self.class_version)
        return "\n".join(
            ["class {0}(uproot.model.VersionedModel):".format(class_name)]
            + read_members
            + read_member_n
            + strided_interpretation
            + awkward_form
            + class_data
        )

    def new_class(self, file):
        """
        Args:
            file (:doc:`uproot.reading.ReadOnlyFile`): File to use to generate
                :doc:`uproot.model.Model` classes as needed from its
                :ref:`uproot.reading.ReadOnlyFile.streamers` and ``file_path``
                for error messages.

        Returns a new subclass of :doc:`uproot.model.VersionedModel` for this
        class and version.
        """
        class_code = self.class_code()
        class_name = uproot.model.classname_encode(self.name, self.class_version)
        classes = uproot.model.maybe_custom_classes(file.custom_classes)
        return uproot.deserialization.compile_class(
            file, classes, class_code, class_name
        )

    @property
    def file_uuid(self):
        """
        The unique identifier (:doc:`uproot.reading.ReadOnlyFile`) of the file
        from which this ``TStreamerInfo`` was extracted.
        """
        return self._file.uuid

    def walk_members(self, streamers):
        """
        Args:
            streamers (list of :doc:`uproot.streamers.Model_TStreamerInfo`): The
                complete set of ``TStreamerInfos``, probably including this one.

        Generator that yields all ``TStreamerElements``
        (:doc:`uproot.streamers.Model_TStreamerElement`) for this class and its
        superclasses.

        The ``TStreamerBase`` elements (:doc:`uproot.streamers.Model_TStreamerBase`)
        are not yielded, but they are extracted from ``streamers`` to include
        their elements.
        """
        for element in self._members["fElements"]:
            if isinstance(element, Model_TStreamerBase):
                streamers_with_name = streamers[element.name]
                base_version = element.base_version
                if base_version == "max":
                    base = streamers_with_name[max(streamers_with_name)]
                else:
                    base = streamers_with_name[base_version]
                for x in base.walk_members(streamers):
                    yield x
            else:
                yield element

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

        self._bases.append(
            uproot.models.TNamed.Model_TNamed.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._bases[0]._members["fName"] = _canonical_typename(
            self._bases[0]._members["fName"]
        )

        self._members["fCheckSum"], self._members["fClassVersion"] = cursor.fields(
            chunk, _tstreamerinfo_format1, context
        )

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)

        self._members["fElements"] = uproot.deserialization.read_object_any(
            chunk, cursor, context, file, self._file, self.concrete
        )

    def _serialize(self, out, header, name):
        where = len(out)
        out.append(self._serialization)
        uproot.serialization._serialize_object_any(
            out, self._members["fElements"], name
        )
        if header:
            out.insert(
                where,
                uproot.serialization.numbytes_version(
                    sum(len(x) for x in out[where:]), self._instance_version
                ),
            )

    def _dependencies(self, streamers, out):
        out.append((self.name, self.class_version))
        for element in self.elements:
            element._dependencies(streamers, out)


_tstreamerelement_format1 = struct.Struct(">iiii")
_tstreamerelement_format2 = struct.Struct(">i")
_tstreamerelement_format3 = struct.Struct(">ddd")
_tstreamerelement_dtype1 = numpy.dtype(">i4")


class Model_TStreamerElement(uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerElement``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    def show(self, stream=sys.stdout):
        """
        Args:
            stream (object with a ``write(str)`` method): Stream to write the
                output to.

        Interactively display a ``TStreamerElement``.
        """
        stream.write(
            u"    {0}: {1} ({2})\n".format(
                self.name,
                self.typename,
                uproot.model.classname_decode(type(self).__name__)[0],
            )
        )

    @property
    def name(self):
        """
        The name (``fName``) of this ``TStreamerElement``.
        """
        return self.member("fName")

    @property
    def title(self):
        """
        The title (``fTitle``) of this ``TStreamerElement``.
        """
        return self.member("fTitle")

    @property
    def typename(self):
        """
        The typename (``fTypeName``) of this ``TStreamerElement``.
        """
        return self.member("fTypeName")

    @property
    def array_length(self):
        """
        The array length (``fArrayLength``) of this ``TStreamerElement``.
        """
        return self.member("fArrayLength")

    @property
    def file_uuid(self):
        """
        The unique identifier (:doc:`uproot.reading.ReadOnlyFile`) of the file
        from which this ``TStreamerElement`` was extracted.
        """
        return self._file.uuid

    @property
    def fType(self):
        """
        The type code (``fType``) of this ``TStreamerElement``.
        """
        return self.member("fType")

    def read_members(self, chunk, cursor, context, file):
        # https://github.com/root-project/root/blob/master/core/meta/src/TStreamerElement.cxx#L505
        self._bases.append(
            uproot.models.TNamed.Model_TNamed.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
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

    def _serialize(self, out, header, name):
        where = len(out)
        out.append(self._serialization)
        if header:
            out.insert(
                where,
                uproot.serialization.numbytes_version(
                    sum(len(x) for x in out[where:]), self._instance_version
                ),
            )

    def _dependencies(self, streamers, out):
        pass


class Model_TStreamerArtificial(Model_TStreamerElement):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerArtificial``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

        self._bases.append(
            Model_TStreamerElement.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)

    def class_code(
        self,
        streamerinfo,
        i,
        elements,
        read_members,
        read_member_n,
        strided_interpretation,
        awkward_form,
        fields,
        formats,
        dtypes,
        formats_memberwise,
        has_memberwise_header,
        containers,
        base_names_versions,
        member_names,
        class_flags,
    ):
        has_memberwise_header.append(True)
        read_member_n.append("        if member_index == {0}:".format(i))

        read_members.append(
            "        raise uproot.deserialization.DeserializationError("
            "'not implemented: class members defined by {0} of type {1} in member "
            "{2} of class {3}', chunk, cursor, context, file.file_path)".format(
                type(self).__name__, self.typename, self.name, streamerinfo.name
            )
        )
        read_member_n.append("    " + read_members[-1])

        strided_interpretation.append(
            "        raise uproot.interpretation.objects.CannotBeStrided("
            "'not implemented: class members defined by {0} of type {1} in member "
            "{2} of class {3}')".format(
                type(self).__name__, self.typename, self.name, streamerinfo.name
            )
        )

        awkward_form.append(
            "        raise uproot.interpretation.objects.CannotBeAwkward("
            "'not implemented: class members defined by {0} of type {1} in member "
            "{2} of class {3}')".format(
                type(self).__name__, self.typename, self.name, streamerinfo.name
            )
        )


_tstreamerbase_format1 = struct.Struct(">i")


class Model_TStreamerBase(Model_TStreamerElement):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerBase``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    @property
    def name(self):
        """
        The name (``fName``) of this ``TStreamerBase``, passed through
        :doc:`uproot.model.classname_regularize`.
        """
        return uproot.model.classname_regularize(self.member("fName"))

    @property
    def base_version(self):
        """
        The base version (``fBaseVersion``) of this ``TStreamerBase``.
        """
        if self._members["fBaseVersion"] == -1:
            return "max"
        else:
            return self._members["fBaseVersion"]

    def show(self, stream=sys.stdout):
        pass

    def class_code(
        self,
        streamerinfo,
        i,
        elements,
        read_members,
        read_member_n,
        strided_interpretation,
        awkward_form,
        fields,
        formats,
        dtypes,
        formats_memberwise,
        has_memberwise_header,
        containers,
        base_names_versions,
        member_names,
        class_flags,
    ):
        has_memberwise_header.append(True)
        read_member_n.append("        if member_index == {0}:".format(i))

        read_members.append(
            "        self._bases.append(c({0}, {1}).read(chunk, cursor, "
            "context, file, self._file, self._parent, concrete=self.concrete))".format(
                repr(self.name), repr(self.base_version)
            )
        )
        read_member_n.append("    " + read_members[-1])

        strided_interpretation.append(
            "        members.extend(file.class_named({0}, {1})."
            "strided_interpretation(file, header, tobject_header, breadcrumbs).members)".format(
                repr(self.name), repr(self.base_version)
            )
        )
        awkward_form.append(
            "        contents.update(file.class_named({0}, {1}).awkward_form(file, "
            "index_format, header, tobject_header, breadcrumbs).contents)".format(
                repr(self.name), repr(self.base_version)
            )
        )

        base_names_versions.append((self.name, self.base_version))

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

        self._bases.append(
            Model_TStreamerElement.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        if self._instance_version >= 2:
            self._members["fBaseVersion"] = cursor.field(
                chunk, _tstreamerbase_format1, context
            )

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)

    def _dependencies(self, streamers, out):
        streamer_versions = streamers.get(self.name)
        if streamer_versions is not None:
            base_version = self.base_version
            if base_version == "max":
                streamer = streamer_versions[max(streamer_versions)]
            else:
                streamer = streamer_versions[base_version]
            if (
                streamer is not None
                and (streamer.name, streamer.class_version) not in out
            ):
                streamer._dependencies(streamers, out)


_tstreamerbasicpointer_format1 = struct.Struct(">i")


class Model_TStreamerBasicPointer(Model_TStreamerElement):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerBasicPointer``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    @property
    def count_name(self):
        """
        The count name (``fCountName``) of this ``TStreamerBasicPointer``.
        """
        return self._members["fCountName"]

    def class_code(
        self,
        streamerinfo,
        i,
        elements,
        read_members,
        read_member_n,
        strided_interpretation,
        awkward_form,
        fields,
        formats,
        dtypes,
        formats_memberwise,
        has_memberwise_header,
        containers,
        base_names_versions,
        member_names,
        class_flags,
    ):
        has_memberwise_header.append(True)
        read_member_n.append("        if member_index == {0}:".format(i))

        read_members.append("        tmp = self._dtype{0}".format(len(dtypes)))
        read_member_n.append("    " + read_members[-1])

        if streamerinfo.name == "TBranch" and self.name == "fBasketSeek":
            read_members.append(
                "        if context.get('speedbump', True):\n"
                "            if cursor.bytes(chunk, 1, context)[0] == 2:\n"
                "                tmp = numpy.dtype('>i8')"
            )
            read_member_n.append("    " + read_members[-1].replace("\n", "\n    "))

        else:
            read_members.append(
                "        if context.get('speedbump', True):\n"
                "            cursor.skip(1)"
            )
            read_member_n.append("    " + read_members[-1].replace("\n", "\n    "))

        read_members.append(
            "        self._members[{0}] = cursor.array(chunk, self.member({1}), "
            "tmp, context)".format(repr(self.name), repr(self.count_name))
        )
        read_member_n.append("    " + read_members[-1])

        strided_interpretation.append(
            "        raise uproot.interpretation.objects.CannotBeStrided("
            "'class members defined by {0} of type {1} in member "
            "{2} of class {3}')".format(
                type(self).__name__, self.typename, self.name, streamerinfo.name
            )
        )

        awkward_form.extend(
            [
                "        contents[{0}] = ListOffsetForm(index_format, "
                "uproot._util.awkward_form(cls._dtype{1}, file, index_format, header, "
                "tobject_header, breadcrumbs),".format(repr(self.name), len(dtypes)),
                "            parameters={'uproot': {'as': 'TStreamerBasicPointer', "
                "'count_name': " + repr(self.count_name) + "}}",
                "        )",
            ]
        )

        member_names.append(self.name)
        dtypes.append(_ftype_to_dtype(self.fType - uproot.const.kOffsetP))

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

        self._bases.append(
            Model_TStreamerElement.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fCountVersion"] = cursor.field(
            chunk, _tstreamerbasicpointer_format1, context
        )
        self._members["fCountName"] = cursor.string(chunk, context)
        self._members["fCountClass"] = cursor.string(chunk, context)

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)


class Model_TStreamerBasicType(Model_TStreamerElement):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerBasicType``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    def class_code(
        self,
        streamerinfo,
        i,
        elements,
        read_members,
        read_member_n,
        strided_interpretation,
        awkward_form,
        fields,
        formats,
        dtypes,
        formats_memberwise,
        has_memberwise_header,
        containers,
        base_names_versions,
        member_names,
        class_flags,
    ):
        has_memberwise_header.append(False)
        read_member_n.append("        if member_index == {0}:".format(i))

        if self.typename == "Double32_t":
            read_members.append(
                "        self._members[{0}] = cursor.double32(chunk, "
                "context)".format(repr(self.name))
            )
            read_member_n.append("    " + read_members[-1])

        elif self.typename == "Float16_t":
            read_members.append(
                "        self._members[{0}] = cursor.float16(chunk, 12, "
                "context)".format(repr(self.name))
            )
            read_member_n.append("    " + read_members[-1])

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

            formats_memberwise.append(_ftype_to_struct(self.fType))

            if (
                i + 1 == len(elements)
                or not isinstance(elements[i + 1], Model_TStreamerBasicType)
                or elements[i + 1].array_length != 0
            ):
                if len(fields[-1]) == 1:
                    read_members.append(
                        "        self._members[{0}] = cursor.field(chunk, "
                        "self._format{1}, context)".format(
                            repr(fields[-1][0]), len(formats) - 1
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

            read_member_n.append(
                "            self._members[{0}] = cursor.field(chunk, "
                "self._format_memberwise{1}, context)".format(
                    repr(self.name), len(formats_memberwise) - 1
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

            read_member_n.append("    " + read_members[-1])

        if self.array_length == 0 and self.typename not in ("Double32_t", "Float16_t"):
            strided_interpretation.append(
                "        members.append(({0}, {1}))".format(
                    repr(self.name), _ftype_to_dtype(self.fType)
                )
            )
        else:
            strided_interpretation.append(
                "        raise uproot.interpretation.objects.CannotBeStrided("
                "'class members defined by {0} of type {1} in member "
                "{2} of class {3}')".format(
                    type(self).__name__, self.typename, self.name, streamerinfo.name
                )
            )

        if self.array_length == 0:
            if self.typename == "Double32_t":
                awkward_form.append(
                    "        contents["
                    + repr(self.name)
                    + "] = NumpyForm((), 8, 'd', parameters={'uproot': {'as': 'Double32'}})"
                )

            elif self.typename == "Float16_t":
                awkward_form.append(
                    "        contents["
                    + repr(self.name)
                    + "] = NumpyForm((), 4, 'f', parameters={'uproot': {'as': 'Float16'}})"
                )

            else:
                awkward_form.append(
                    "        contents[{0}] = uproot._util.awkward_form({1}, "
                    "file, index_format, header, tobject_header, breadcrumbs)".format(
                        repr(self.name), _ftype_to_dtype(self.fType)
                    )
                )

        else:
            awkward_form.append(
                "        contents[{0}] = RegularForm(uproot._util.awkward_form({1}, "
                "file, index_format, header, tobject_header, breadcrumbs), {2})".format(
                    repr(self.name), _ftype_to_dtype(self.fType), self.array_length
                )
            )

        member_names.append(self.name)

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

        self._bases.append(
            Model_TStreamerElement.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        if (
            uproot.const.kOffsetL
            < self._bases[0]._members["fType"]
            < uproot.const.kOffsetP
        ):
            self._bases[0]._members["fType"] -= uproot.const.kOffsetL

        basic = True

        if self._bases[0]._members["fType"] in (
            uproot.const.kBool,
            uproot.const.kUChar,
            uproot.const.kChar,
        ):
            self._bases[0]._members["fSize"] = 1

        elif self._bases[0]._members["fType"] in (
            uproot.const.kUShort,
            uproot.const.kShort,
        ):
            self._bases[0]._members["fSize"] = 2

        elif self._bases[0]._members["fType"] in (
            uproot.const.kBits,
            uproot.const.kUInt,
            uproot.const.kInt,
            uproot.const.kCounter,
        ):
            self._bases[0]._members["fSize"] = 4

        elif self._bases[0]._members["fType"] in (
            uproot.const.kULong,
            uproot.const.kLong,
        ):
            self._bases[0]._members["fSize"] = numpy.dtype(numpy.compat.long).itemsize

        elif self._bases[0]._members["fType"] in (
            uproot.const.kULong64,
            uproot.const.kLong64,
        ):
            self._bases[0]._members["fSize"] = 8

        elif self._bases[0]._members["fType"] in (
            uproot.const.kFloat,
            uproot.const.kFloat16,
        ):
            self._bases[0]._members["fSize"] = 4

        elif self._bases[0]._members["fType"] in (
            uproot.const.kDouble,
            uproot.const.kDouble32,
        ):
            self._bases[0]._members["fSize"] = 8

        elif self._bases[0]._members["fType"] == uproot.const.kCharStar:
            self._bases[0]._members["fSize"] = numpy.dtype(numpy.intp).itemsize

        else:
            basic = False

        if basic and self._bases[0]._members["fArrayLength"] > 0:
            self._bases[0]._members["fSize"] *= self._bases[0]._members["fArrayLength"]

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)


_tstreamerloop_format1 = struct.Struct(">i")


class Model_TStreamerLoop(Model_TStreamerElement):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerLoop``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    @property
    def count_name(self):
        """
        The count name (``fCountName``) of this ``TStreamerLoop``.
        """
        return self._members["fCountName"]

    def class_code(
        self,
        streamerinfo,
        i,
        elements,
        read_members,
        read_member_n,
        strided_interpretation,
        awkward_form,
        fields,
        formats,
        dtypes,
        formats_memberwise,
        has_memberwise_header,
        containers,
        base_names_versions,
        member_names,
        class_flags,
    ):
        has_memberwise_header.append(True)
        read_members.extend(
            [
                "        cursor.skip(6)",
                "        for tmp in uproot._util.range(self.member({0})):".format(
                    repr(self.count_name)
                ),
                "            self._members[{0}] = c({1}).read(chunk, cursor, "
                "context, file, self._file, self.concrete)".format(
                    repr(self.name), repr(self.typename.rstrip("*"))
                ),
            ]
        )

        strided_interpretation.append(
            "        raise uproot.interpretation.objects.CannotBeStrided("
            "'class members defined by {0} of type {1} in member "
            "{2} of class {3}')".format(
                type(self).__name__, self.typename, self.name, streamerinfo.name
            )
        )

        awkward_form.extend(
            [
                "        tmp = file.class_named({0}, 'max').awkward_form(file, "
                "index_format, header, tobject_header, breadcrumbs)".format(
                    repr(self.typename.rstrip("*"))
                ),
                "        contents["
                + repr(self.name)
                + "] = ListOffsetForm(index_format, "
                "tmp, parameters={'uproot': {'as': TStreamerLoop, 'count_name': "
                + repr(self.count_name)
                + "}})",
            ]
        )

        member_names.append(self.name)

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

        self._bases.append(
            Model_TStreamerElement.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fCountVersion"] = cursor.field(
            chunk, _tstreamerloop_format1, context
        )
        self._members["fCountName"] = cursor.string(chunk, context)
        self._members["fCountClass"] = cursor.string(chunk, context)

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)

    def _dependencies(self, streamers, out):
        streamer_versions = streamers.get(self.typename.rstrip("*"))
        if streamer_versions is not None:
            for streamer in streamer_versions.values():
                if (streamer.name, streamer.class_version) not in out:
                    streamer._dependencies(streamers, out)


_tstreamerstl_format1 = struct.Struct(">ii")


class Model_TStreamerSTL(Model_TStreamerElement):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerSTL``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    @property
    def stl_type(self):
        """
        The STL type code (``fSTLtype``) of this ``TStreamerSTL``.
        """
        return self._members["fSTLtype"]

    @property
    def fCtype(self):
        """
        The type code (``fCtype``) of this ``TStreamerSTL``.
        """
        return self._members["fCtype"]

    def class_code(
        self,
        streamerinfo,
        i,
        elements,
        read_members,
        read_member_n,
        strided_interpretation,
        awkward_form,
        fields,
        formats,
        dtypes,
        formats_memberwise,
        has_memberwise_header,
        containers,
        base_names_versions,
        member_names,
        class_flags,
    ):
        has_memberwise_header.append(True)
        read_member_n.append("        if member_index == {0}:".format(i))

        stl_container = uproot.interpretation.identify.parse_typename(
            self.typename,
            quote=True,
            outer_header=True,
            inner_header=False,
            string_header=True,
        )
        read_members.append(
            "        self._members[{0}] = self._stl_container{1}.read("
            "chunk, cursor, context, file, self._file, self.concrete)"
            "".format(repr(self.name), len(containers))
        )
        read_member_n.append(
            "            self._members[{0}] = self._stl_container{1}.read("
            "chunk, cursor, context, file, self._file, self.concrete, header=False)"
            "".format(repr(self.name), len(containers))
        )

        strided_interpretation.append(
            "        members.append(({0}, cls._stl_container{1}."
            "strided_interpretation(file, header, tobject_header, breadcrumbs)))".format(
                repr(self.name), len(containers)
            )
        )

        awkward_form.append(
            "        contents[{0}] = cls._stl_container{1}.awkward_form(file, "
            "index_format, header, tobject_header, breadcrumbs)".format(
                repr(self.name), len(containers)
            )
        )

        containers.append(stl_container)
        member_names.append(self.name)

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

        self._bases.append(
            Model_TStreamerElement.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )
        self._members["fSTLtype"], self._members["fCtype"] = cursor.fields(
            chunk, _tstreamerstl_format1, context
        )

        if self._members["fSTLtype"] in (
            uproot.const.kSTLmultimap,
            uproot.const.kSTLset,
        ):
            if self._bases[0]._members["fTypeName"].startswith(
                "std::set"
            ) or self._bases[0]._members["fTypeName"].startswith("set"):
                self._members["fSTLtype"] = uproot.const.kSTLset

            elif self._bases[0]._members["fTypeName"].startswith(
                "std::multimap"
            ) or self._bases[0]._members["fTypeName"].startswith("multimap"):
                self._members["fSTLtype"] = uproot.const.kSTLmultimap

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)


class Model_TStreamerSTLstring(Model_TStreamerSTL):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerSTLString``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

        self._bases.append(
            Model_TStreamerSTL.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)


class TStreamerPointerTypes(object):
    """
    A class to share code between
    :doc:`uproot.streamers.Model_TStreamerObjectAnyPointer` and
    :doc:`uproot.streamers.Model_TStreamerObjectPointer`.
    """

    def class_code(
        self,
        streamerinfo,
        i,
        elements,
        read_members,
        read_member_n,
        strided_interpretation,
        awkward_form,
        fields,
        formats,
        dtypes,
        formats_memberwise,
        has_memberwise_header,
        containers,
        base_names_versions,
        member_names,
        class_flags,
    ):
        has_memberwise_header.append(True)
        read_member_n.append("        if member_index == {0}:".format(i))

        if self.fType == uproot.const.kObjectp or self.fType == uproot.const.kAnyp:
            read_members.append(
                "        self._members[{0}] = c({1}).read(chunk, cursor, context, "
                "file, self._file, self.concrete)".format(
                    repr(self.name), repr(self.typename.rstrip("*"))
                )
            )
            read_member_n.append("    " + read_members[-1])
            strided_interpretation.append(
                "        members.append(({0}, file.class_named({1}, 'max')."
                "strided_interpretation(file, header, tobject_header, breadcrumbs)))".format(
                    repr(self.name), repr(self.typename.rstrip("*"))
                )
            )
            awkward_form.append(
                "        contents[{0}] = file.class_named({1}, 'max').awkward_form(file, "
                "index_format, header, tobject_header, breadcrumbs)".format(
                    repr(self.name), repr(self.typename.rstrip("*"))
                )
            )

        elif self.fType == uproot.const.kObjectP or self.fType == uproot.const.kAnyP:
            read_members.append(
                "        self._members[{0}] = read_object_any(chunk, cursor, "
                "context, file, self._file, self)".format(repr(self.name))
            )
            read_member_n.append("    " + read_members[-1])
            strided_interpretation.append(
                "        raise uproot.interpretation.objects.CannotBeStrided("
                "'class members defined by {0} of type {1} in member "
                "{2} of class {3}')".format(
                    type(self).__name__, self.typename, self.name, streamerinfo.name
                )
            )
            class_flags["has_read_object_any"] = True

        else:
            read_members.append(
                "        raise uproot.deserialization.DeserializationError("
                "'not implemented: class members defined by {0} with fType {1}', "
                "chunk, cursor, context, file.file_path)".format(
                    type(self).__name__,
                    self.fType,
                )
            )
            read_member_n.append("    " + read_members[-1])

        member_names.append(self.name)

    def _dependencies(self, streamers, out):
        streamer_versions = streamers.get(self.typename.rstrip("*"))
        if streamer_versions is not None:
            for streamer in streamer_versions.values():
                if (streamer.name, streamer.class_version) not in out:
                    streamer._dependencies(streamers, out)


class Model_TStreamerObjectAnyPointer(TStreamerPointerTypes, Model_TStreamerElement):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerObjectAnyPointer``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

        self._bases.append(
            Model_TStreamerElement.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)


class Model_TStreamerObjectPointer(TStreamerPointerTypes, Model_TStreamerElement):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerObjectPointer``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

        self._bases.append(
            Model_TStreamerElement.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)


class TStreamerObjectTypes(object):
    """
    A class to share code between
    :doc:`uproot.streamers.Model_TStreamerObject`,
    :doc:`uproot.streamers.Model_TStreamerObjectAny`, and
    :doc:`uproot.streamers.Model_TStreamerString`.
    """

    def class_code(
        self,
        streamerinfo,
        i,
        elements,
        read_members,
        read_member_n,
        strided_interpretation,
        awkward_form,
        fields,
        formats,
        dtypes,
        formats_memberwise,
        has_memberwise_header,
        containers,
        base_names_versions,
        member_names,
        class_flags,
    ):
        has_memberwise_header.append(True)
        read_member_n.append("        if member_index == {0}:".format(i))

        read_members.append(
            "        self._members[{0}] = c({1}).read(chunk, cursor, context, "
            "file, self._file, self.concrete)".format(
                repr(self.name), repr(self.typename.rstrip("*"))
            )
        )

        read_member_n.append("""            # uninterpreted header
            #cursor.skip(4)
            breakpoint()
            # no known class version number (maybe in that header? unclear...)
            model = c({0}).new_class(file, "max")""".format(repr(self.typename.rstrip("*"))))

        read_member_n.append(
            "            self._members[{0}] = model.read(chunk, cursor, {{**context, 'numbytes_version': True}}, "
            "file, self._file, self.concrete)".format(
                repr(self.name)
            )
        )

        #read_member_n.append("    " + read_members[-1])

        '''
        read_member_n.append("            model = c({0})".format(repr(self.typename.rstrip("*"))))
        read_member_n.append("            value = model.read(chunk, cursor, {**context, 'reading': False}, "
                             "file, self._file, self.concrete)")

        read_member_n.append("""            _member_names = getattr(value, 'member_names', [])
            _has_memberwise_header = getattr(value, 'has_memberwise_header', [False]*len(_member_names))

            breakpoint()
            print(value)
            print(value.class_code)
            print(member_index)
            # memberwise reading!
            for _member_index, has_header in zip(uproot._util.range(len(_member_names)), _has_memberwise_header):
                cursor.debug(chunk, limit_bytes=160)
                print(member_index, _member_index, has_header)
                #if has_header:
                #    # uninterpreted header
                #    cursor.skip(6)
                value.read_member_n(
                    chunk, cursor, context, file, _member_index
                )""")

        read_member_n.append("            self._members[{0}] = value".format(repr(self.name)))
        '''

        strided_interpretation.append(
            "        members.append(({0}, file.class_named({1}, 'max')."
            "strided_interpretation(file, header, tobject_header, breadcrumbs)))".format(
                repr(self.name), repr(self.typename.rstrip("*"))
            )
        )
        awkward_form.append(
            "        contents[{0}] = file.class_named({1}, 'max').awkward_form(file, "
            "index_format, header, tobject_header, breadcrumbs)".format(
                repr(self.name), repr(self.typename.rstrip("*"))
            )
        )

        member_names.append(self.name)

    def _dependencies(self, streamers, out):
        streamer_versions = streamers.get(self.typename.rstrip("*"))
        if streamer_versions is not None:
            for streamer in streamer_versions.values():
                if (streamer.name, streamer.class_version) not in out:
                    streamer._dependencies(streamers, out)


class Model_TStreamerObject(TStreamerObjectTypes, Model_TStreamerElement):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerObject``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

        self._bases.append(
            Model_TStreamerElement.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)


class Model_TStreamerObjectAny(TStreamerObjectTypes, Model_TStreamerElement):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerObjectAny``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

        self._bases.append(
            Model_TStreamerElement.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)


class Model_TStreamerString(TStreamerObjectTypes, Model_TStreamerElement):
    """
    A versionless :doc:`uproot.model.Model` for ``TStreamerString``.

    Since this model is versionless and most of its functionality is internal
    (not to be directly accessed by most users), it is defined on the model
    instead of creating a behavior class to mix in functionality.
    """

    def read_members(self, chunk, cursor, context, file):
        start = cursor.index

        self._bases.append(
            Model_TStreamerElement.read(
                chunk,
                cursor,
                context,
                file,
                self._file,
                self._parent,
                concrete=self.concrete,
            )
        )

        self._serialization = _copy_bytes(chunk, start, cursor.index, cursor, context)


uproot.classes["TStreamerInfo"] = Model_TStreamerInfo
uproot.classes["TStreamerElement"] = Model_TStreamerElement
uproot.classes["TStreamerArtificial"] = Model_TStreamerArtificial
uproot.classes["TStreamerBase"] = Model_TStreamerBase
uproot.classes["TStreamerBasicPointer"] = Model_TStreamerBasicPointer
uproot.classes["TStreamerBasicType"] = Model_TStreamerBasicType
uproot.classes["TStreamerLoop"] = Model_TStreamerLoop
uproot.classes["TStreamerObject"] = Model_TStreamerObject
uproot.classes["TStreamerObjectAny"] = Model_TStreamerObjectAny
uproot.classes["TStreamerObjectAnyPointer"] = Model_TStreamerObjectAnyPointer
uproot.classes["TStreamerObjectPointer"] = Model_TStreamerObjectPointer
uproot.classes["TStreamerSTL"] = Model_TStreamerSTL
uproot.classes["TStreamerSTLstring"] = Model_TStreamerSTLstring
uproot.classes["TStreamerString"] = Model_TStreamerString
