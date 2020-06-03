# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import re
import ast

import numpy

import uproot4.const
import uproot4.interpretation.numerical
import uproot4.streamers
import uproot4._util


class NotNumerical(Exception):
    pass


class UnknownInterpretation(Exception):
    def __init__(self, reason, file_path, object_path):
        self.reason = reason
        self.file_path = file_path
        self.object_path = object_path

    def __repr__(self):
        return "<UnknownInterpretation {0}>".format(repr(self.reason))

    def __str__(self):
        return """{0}
in file {1} at {2}""".format(
            self.reason, self.file_path, self.object_path
        )


def _normalize_ftype(fType):
    if fType is not None and uproot4.const.kOffsetL < fType < uproot4.const.kOffsetP:
        return fType - uproot4.const.kOffsetL
    else:
        return fType


def _ftype_to_dtype(fType):
    fType = _normalize_ftype(fType)
    if fType == uproot4.const.kBool:
        return numpy.dtype(numpy.bool_)
    elif fType == uproot4.const.kChar:
        return numpy.dtype("i1")
    elif fType == uproot4.const.kUChar:
        return numpy.dtype("u1")
    elif fType == uproot4.const.kShort:
        return numpy.dtype(">i2")
    elif fType == uproot4.const.kUShort:
        return numpy.dtype(">u2")
    elif fType == uproot4.const.kInt:
        return numpy.dtype(">i4")
    elif fType in (uproot4.const.kBits, uproot4.const.kUInt, uproot4.const.kCounter):
        return numpy.dtype(">u4")
    elif fType == uproot4.const.kLong:
        return numpy.dtype(">i8")
    elif fType == uproot4.const.kULong:
        return numpy.dtype(">u8")
    elif fType == uproot4.const.kLong64:
        return numpy.dtype(">i8")
    elif fType == uproot4.const.kULong64:
        return numpy.dtype(">u8")
    elif fType == uproot4.const.kFloat:
        return numpy.dtype(">f4")
    elif fType == uproot4.const.kDouble:
        return numpy.dtype(">f8")
    else:
        raise NotNumerical()


def _leaf_to_dtype(leaf):
    if leaf.classname == "TLeafO":
        return numpy.dtype(numpy.bool_)
    elif leaf.classname == "TLeafB":
        if leaf.member("fIsUnsigned"):
            return numpy.dtype(numpy.uint8)
        else:
            return numpy.dtype(numpy.int8)
    elif leaf.classname == "TLeafS":
        if leaf.member("fIsUnsigned"):
            return numpy.dtype(numpy.uint16)
        else:
            return numpy.dtype(numpy.int16)
    elif leaf.classname == "TLeafI":
        if leaf.member("fIsUnsigned"):
            return numpy.dtype(numpy.uint32)
        else:
            return numpy.dtype(numpy.int32)
    elif leaf.classname == "TLeafL":
        if leaf.member("fIsUnsigned"):
            return numpy.dtype(numpy.uint64)
        else:
            return numpy.dtype(numpy.int64)
    elif leaf.classname == "TLeafF":
        return numpy.dtype(numpy.float32)
    elif leaf.classname == "TLeafD":
        return numpy.dtype(numpy.float64)
    elif leaf.classname == "TLeafElement":
        return _ftype_to_dtype(leaf.member("fType"))
    else:
        raise NotNumerical()


_title_has_dims = re.compile(r"^([^\[\]]+)(\[[^\[\]]+\])+")
_item_dim_pattern = re.compile(r"\[([1-9][0-9]*)\]")
_item_any_pattern = re.compile(r"\[(.*)\]")
_vector_pointer = re.compile(r"vector\<([^<>]*)\*\>")
_pair_second = re.compile(r"pair\<[^<>]*,(.*) \>")


def _from_leaves(branch, context):
    dims, is_jagged = (), False
    if len(branch.member("fLeaves")) == 1:
        leaf = branch.member("fLeaves")[0]
        title = leaf.member("fTitle")

        m = _title_has_dims.match(title)
        if m is not None:
            dims = tuple(int(x) for x in re.findall(_item_dim_pattern, title))
            if dims == ():
                if leaf.member("fLen") > 1:
                    dims = (leaf.member("fLen"),)

            if any(
                _item_dim_pattern.match(x) is None
                for x in re.findall(_item_any_pattern, title)
            ):
                is_jagged = True

    else:
        for leaf in branch.member("fLeaves"):
            if _title_has_dims.match(leaf.member("fTitle")):
                raise UnknownInterpretation(
                    "leaf-list with square brackets in the title",
                    branch.file.file_path,
                    branch.object_path,
                )

    return dims, is_jagged


def _float16_double32_walk_ast(node, branch, source):
    if isinstance(node, ast.AST):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == "pi"
        ):
            out = ast.Num(3.141592653589793)  # TMath::Pi()
        elif isinstance(node, ast.Num):
            out = ast.Num(float(node.n))
        elif isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            out = ast.BinOp(
                _float16_double32_walk_ast(node.left, branch, source),
                node.op,
                _float16_double32_walk_ast(node.right, branch, source),
            )
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            out = ast.UnaryOp(
                node.op, _float16_double32_walk_ast(node.operand, branch, source)
            )
        elif (
            isinstance(node, ast.List)
            and isinstance(node.ctx, ast.Load)
            and len(node.elts) == 2
        ):
            out = ast.List(
                [
                    _float16_double32_walk_ast(node.elts[0], branch, source),
                    _float16_double32_walk_ast(node.elts[1], branch, source),
                ],
                node.ctx,
            )
        elif (
            isinstance(node, ast.List)
            and isinstance(node.ctx, ast.Load)
            and len(node.elts) == 3
            and isinstance(node.elts[2], ast.Num)
        ):
            out = ast.List(
                [
                    _float16_double32_walk_ast(node.elts[0], branch, source),
                    _float16_double32_walk_ast(node.elts[1], branch, source),
                    node.elts[2],
                ],
                node.ctx,
            )
        else:
            raise UnknownInterpretation(
                "cannot compute streamer title {0}".format(repr(source)),
                branch.file.file_path,
                branch.object_path,
            )
        out.lineno, out.col_offset = node.lineno, node.col_offset
        return out

    else:
        raise UnknownInterpretation(
            "cannot compute streamer title {0}".format(repr(source)),
            branch.file.file_path,
            branch.object_path,
        )


def _float16_or_double32(branch, context, leaf, is_float16, dims):
    try:
        left = branch.streamer.title.index("[")
        right = branch.streamer.title.index("]")

    except (ValueError, AttributeError):
        low, high, num_bits = 0, 0, 0

    else:
        source = branch.streamer.title[left : right + 1]
        try:
            parsed = ast.parse(source).body[0].value
        except SyntaxError:
            raise UnknownInterpretation(
                "cannot parse streamer title {0} (as Python)".format(repr(source)),
                branch.file.file_path,
                branch.object_path,
            )

        transformed = ast.Expression(_float16_double32_walk_ast(parsed, branch, source))
        spec = eval(compile(transformed, repr(branch.streamer.title), "eval"))

        if (
            len(spec) == 2
            and uproot4._util.isnum(spec[0])
            and uproot4._util.isnum(spec[1])
        ):
            low, high = spec
            num_bits = None

        elif (
            len(spec) == 3
            and uproot4._util.isnum(spec[0])
            and uproot4._util.isnum(spec[1])
            and uproot4._util.isint(spec[1])
        ):
            low, high, num_bits = spec

        else:
            raise UnknownInterpretation(
                "cannot interpret streamer title {0} as (low, high) or "
                "(low, high, num_bits)".format(repr(source)),
                branch.file.file_path,
                branch.object_path,
            )

    if not is_float16:
        if num_bits == 0:
            return uproot4.interpretation.numerical.AsDtype(
                numpy.dtype((">f4", dims)), numpy.dtype(("f8", dims))
            )
        elif num_bits is None:
            return uproot4.interpretation.numerical.AsDouble32(low, high, 32, dims)
        else:
            return uproot4.interpretation.numerical.AsDouble32(
                low, high, num_bits, dims
            )

    else:
        if num_bits == 0:
            return uproot4.interpretation.numerical.AsFloat16(low, high, 12, dims)
        elif num_bits is None:
            return uproot4.interpretation.numerical.AsFloat16(low, high, 32, dims)
        else:
            return uproot4.interpretation.numerical.AsFloat16(low, high, num_bits, dims)


def interpretation_of(branch, context):
    dims, is_jagged = _from_leaves(branch, context)

    try:
        if len(branch.member("fLeaves")) == 0:
            pass

        elif len(branch.member("fLeaves")) == 1:
            leaf = branch.member("fLeaves")[0]

            if isinstance(
                branch.streamer, uproot4.streamers.Model_TStreamerObjectPointer
            ):
                typename = branch.streamer.typename
                if typename.endswith("*"):
                    typename = typename[:-1]
                raise NotImplementedError("obj_or_genobj")

            leaftype = uproot4.const.kBase
            if leaf.classname == "TLeafElement":
                leaftype = _normalize_ftype(leaf.member("fType"))

            is_float16 = leaftype == uproot4.const.kFloat16
            is_double32 = leaftype == uproot4.const.kDouble32
            if is_float16 or is_double32:
                out = _float16_or_double32(branch, context, leaf, is_float16, dims)

            else:
                from_dtype = _leaf_to_dtype(leaf).newbyteorder(">")

                if context.get("swap_bytes", True):
                    to_dtype = from_dtype.newbyteorder("=")
                else:
                    to_dtype = from_dtype

                out = uproot4.interpretation.numerical.AsDtype(
                    numpy.dtype((from_dtype, dims)), numpy.dtype((to_dtype, dims))
                )

            if leaf.member("fLeafCount") is None:
                return out
            else:
                return uproot4.interpretation.jagged.AsJagged(out)

        else:
            from_dtype = []
            for leaf in branch.member("fLeaves"):
                from_dtype.append(
                    (leaf.member("fName"), _leaf_to_dtype(leaf).newbyteorder(">"))
                )

            if context.get("swap_bytes", True):
                to_dtype = [(name, dt.newbyteorder("=")) for name, dt in from_dtype]
            else:
                to_dtype = from_dtype

            if all(
                leaf.member("fLeafCount") is None for leaf in branch.member("fLeaves")
            ):
                return uproot4.interpretation.numerical.AsDtype(
                    numpy.dtype((from_dtype, dims)), numpy.dtype((to_dtype, dims))
                )
            else:
                raise UnknownInterpretation(
                    "leaf-list with non-null fLeafCount",
                    branch.file.file_path,
                    branch.object_path,
                )

    except NotNumerical:
        raise NotImplementedError
