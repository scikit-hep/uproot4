# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines utilities for identifying the
:doc:`uproot.interpretation.Interpretation` of a
:doc:`uproot.behaviors.TBranch.TBranch`.

This includes a tokenizer/parser for C++ types and heuristics encoded in
:doc:`uproot.interpretation.identify.interpretation_of`. The latter will
need to be tweaked by new types, type combinations, and serialization methods
observed in ROOT files (perhaps forever), unless a systematic study can be
performed to exhaustively discover all cases.
"""

from __future__ import absolute_import

import re
import ast

import numpy

import uproot


def _normalize_ftype(fType):
    if fType is not None and uproot.const.kOffsetL < fType < uproot.const.kOffsetP:
        return fType - uproot.const.kOffsetL
    else:
        return fType


def _ftype_to_dtype(fType):
    fType = _normalize_ftype(fType)
    if fType == uproot.const.kBool:
        return numpy.dtype(numpy.bool_)
    elif fType == uproot.const.kChar:
        return numpy.dtype("i1")
    elif fType == uproot.const.kUChar:
        return numpy.dtype("u1")
    elif fType == uproot.const.kShort:
        return numpy.dtype(">i2")
    elif fType == uproot.const.kUShort:
        return numpy.dtype(">u2")
    elif fType == uproot.const.kInt:
        return numpy.dtype(">i4")
    elif fType in (uproot.const.kBits, uproot.const.kUInt, uproot.const.kCounter):
        return numpy.dtype(">u4")
    elif fType == uproot.const.kLong:
        return numpy.dtype(">i8")
    elif fType == uproot.const.kULong:
        return numpy.dtype(">u8")
    elif fType == uproot.const.kLong64:
        return numpy.dtype(">i8")
    elif fType == uproot.const.kULong64:
        return numpy.dtype(">u8")
    elif fType == uproot.const.kFloat:
        return numpy.dtype(">f4")
    elif fType == uproot.const.kDouble:
        return numpy.dtype(">f8")
    else:
        raise NotNumerical("Unrecognized fType: {}".format(fType))


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


_title_has_dims = re.compile(r"^([^\[\]]*)(\[[^\[\]]+\])+")
_item_dim_pattern = re.compile(r"\[([1-9][0-9]*)\]")
_item_any_pattern = re.compile(r"\[(.*)\]")
_vector_pointer = re.compile(r"vector\<([^<>]*)\*\>")
_pair_second = re.compile(r"pair\<[^<>]*,(.*) \>")


def _from_leaves_one(leaf, title):
    dims, is_jagged = (), False

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

    return dims, is_jagged


def _from_leaves(branch, context):
    if len(branch.member("fLeaves")) == 0:
        raise UnknownInterpretation(
            "leaf-list with zero leaves",
            branch.file.file_path,
            branch.object_path,
        )

    elif len(branch.member("fLeaves")) == 1:
        leaf = branch.member("fLeaves")[0]
        title = leaf.member("fTitle")
        return _from_leaves_one(leaf, title)

    else:
        first = True
        for leaf in branch.member("fLeaves"):
            title = leaf.member("fTitle")
            if first:
                dims, is_jagged = _from_leaves_one(leaf, title)
            else:
                trial_dims, trial_is_jagged = _from_leaves_one(leaf, title)
                if dims != trial_dims or is_jagged != trial_is_jagged:
                    raise UnknownInterpretation(
                        "leaf-list with different dimensions among the leaves",
                        branch.file.file_path,
                        branch.object_path,
                    )
        return dims, is_jagged


def _float16_double32_walk_ast(node, branch, source):
    if isinstance(node, ast.AST):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id.lower() == "pi"
        ):
            out = ast.Num(3.141592653589793)  # TMath::Pi()
        elif (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id.lower() == "twopi"
        ):
            out = ast.Num(6.283185307179586)  # TMath::TwoPi()
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
    if leaf.classname in ("TLeafF16", "TLeafD32"):
        title = leaf.member("fTitle")
    elif branch.streamer is not None:
        title = branch.streamer.title
    else:
        title = ""

    try:
        left = title.index("[")
        right = title.index("]")

    except (ValueError, AttributeError):
        low, high, num_bits = 0, 0, 0

    else:
        source = title[left : right + 1]
        try:
            parsed = ast.parse(source).body[0].value
        except SyntaxError:
            raise UnknownInterpretation(
                "cannot parse streamer title {0} (as Python)".format(repr(source)),
                branch.file.file_path,
                branch.object_path,
            )

        transformed = ast.Expression(_float16_double32_walk_ast(parsed, branch, source))
        spec = eval(compile(transformed, repr(title), "eval"))
        if (
            len(spec) == 2
            and uproot._util.isnum(spec[0])
            and uproot._util.isnum(spec[1])
        ):
            low, high = spec
            num_bits = None

        elif (
            len(spec) == 3
            and uproot._util.isnum(spec[0])
            and uproot._util.isnum(spec[1])
            and uproot._util.isint(spec[2])
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
            return uproot.interpretation.numerical.AsDtype(
                numpy.dtype((">f4", dims)), numpy.dtype(("f8", dims))
            )
        elif num_bits is None:
            return uproot.interpretation.numerical.AsDouble32(low, high, 32, dims)
        else:
            return uproot.interpretation.numerical.AsDouble32(low, high, num_bits, dims)

    else:
        if num_bits == 0:
            return uproot.interpretation.numerical.AsFloat16(low, high, 12, dims)
        elif num_bits is None:
            return uproot.interpretation.numerical.AsFloat16(low, high, 32, dims)
        else:
            return uproot.interpretation.numerical.AsFloat16(low, high, num_bits, dims)


def interpretation_of(branch, context, simplify=True):
    """
    Args:
        branch (:doc:`uproot.behaviors.TBranch.TBranch`): The ``TBranch`` to
            interpret as an array.
        context (dict): Auxiliary data used in deserialization.
        simplify (bool): If True, call
            :ref:`uproot.interpretation.objects.AsObjects.simplify` on any
            :doc:`uproot.interpretation.objects.AsObjects` to try to get a
            more efficient interpretation.

    Attempts to derive an :doc:`uproot.interpretation.Interpretation` of the
    ``branch`` (within some ``context``).

    If no interpretation can be found, it raises
    :doc:`uproot.interpretation.identify.UnknownInterpretation`.
    """
    if len(branch.branches) != 0:
        if branch.top_level and branch.has_member("fClassName"):
            typename = branch.member("fClassName")
        elif branch.streamer is not None:
            typename = branch.streamer.typename
        else:
            typename = None
        subbranches = dict((x.name, x.interpretation) for x in branch.branches)
        return uproot.interpretation.grouped.AsGrouped(
            branch, subbranches, typename=typename
        )

    if branch.classname == "TBranchObject":
        if branch.top_level and branch.has_member("fClassName"):
            model_cls = parse_typename(
                branch.member("fClassName"),
                file=branch.file,
                outer_header=True,
                inner_header=False,
                string_header=False,
            )
            return uproot.interpretation.objects.AsObjects(
                uproot.containers.AsDynamic(model_cls), branch
            )

        if branch.streamer is not None:
            model_cls = parse_typename(
                branch.streamer.typename,
                file=branch.file,
                outer_header=True,
                inner_header=False,
                string_header=True,
            )

            return uproot.interpretation.objects.AsObjects(
                uproot.containers.AsDynamic(model_cls), branch
            )

        return uproot.interpretation.objects.AsObjects(
            uproot.containers.AsDynamic(), branch
        )

    dims, is_jagged = _from_leaves(branch, context)

    try:
        if len(branch.member("fLeaves")) == 0:
            raise NotNumerical()

        elif len(branch.member("fLeaves")) == 1:
            leaf = branch.member("fLeaves")[0]

            leaftype = uproot.const.kBase
            if leaf.classname == "TLeafElement":
                leaftype = _normalize_ftype(leaf.member("fType"))

            is_float16 = (
                leaftype == uproot.const.kFloat16 or leaf.classname == "TLeafF16"
            )
            is_double32 = (
                leaftype == uproot.const.kDouble32 or leaf.classname == "TLeafD32"
            )
            if is_float16 or is_double32:
                out = _float16_or_double32(branch, context, leaf, is_float16, dims)

            else:
                from_dtype = _leaf_to_dtype(leaf).newbyteorder(">")

                if context.get("swap_bytes", True):
                    to_dtype = from_dtype.newbyteorder("=")
                else:
                    to_dtype = from_dtype

                out = uproot.interpretation.numerical.AsDtype(
                    numpy.dtype((from_dtype, dims)), numpy.dtype((to_dtype, dims))
                )

            if leaf.member("fLeafCount") is None:
                return out
            else:
                return uproot.interpretation.jagged.AsJagged(out)

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
                return uproot.interpretation.numerical.AsDtype(
                    numpy.dtype((from_dtype, dims)), numpy.dtype((to_dtype, dims))
                )
            else:
                raise UnknownInterpretation(
                    "leaf-list with non-null fLeafCount",
                    branch.file.file_path,
                    branch.object_path,
                )

    except NotNumerical:
        if (
            branch.member("fStreamerType", none_if_missing=True)
            == uproot.const.kTString
        ):
            return uproot.interpretation.strings.AsStrings(typename="TString")

        if len(branch.member("fLeaves")) == 1:
            leaf = branch.member("fLeaves")[0]

            if leaf.classname == "TLeafC":
                return uproot.interpretation.strings.AsStrings()

        elif len(branch.member("fLeaves")) > 1:
            raise UnknownInterpretation(
                "more than one TLeaf ({0}) in a non-numerical TBranch".format(
                    len(branch.member("fLeaves"))
                ),
                branch.file.file_path,
                branch.object_path,
            )

        if branch.top_level and branch.has_member("fClassName"):
            model_cls = parse_typename(
                branch.member("fClassName"),
                file=branch.file,
                outer_header=True,
                inner_header=False,
                string_header=False,
            )

            out = uproot.interpretation.objects.AsObjects(model_cls, branch)
            if simplify:
                return out.simplify()
            else:
                return out

        if branch.streamer is not None:
            model_cls = parse_typename(
                branch.streamer.typename,
                file=branch.file,
                outer_header=True,
                inner_header=False,
                string_header=True,
            )

            # kObjectp/kAnyp (as opposed to kObjectP/kAnyP) are stored inline
            if isinstance(
                model_cls, uproot.containers.AsPointer
            ) and branch.streamer.member("fType") in (
                uproot.const.kObjectp,
                uproot.const.kAnyp,
            ):
                while isinstance(model_cls, uproot.containers.AsPointer):
                    model_cls = model_cls.pointee

            if branch._streamer_isTClonesArray:
                if isinstance(branch.streamer, uproot.streamers.Model_TStreamerObject):
                    model_cls = uproot.containers.AsArray(False, False, model_cls)
                else:
                    if hasattr(model_cls, "header"):
                        model_cls._header = False
                    model_cls = uproot.containers.AsArray(True, False, model_cls)

            out = uproot.interpretation.objects.AsObjects(model_cls, branch)
            if simplify:
                return out.simplify()
            else:
                return out

    raise UnknownInterpretation(
        "none of the rules matched",
        branch.file.file_path,
        branch.object_path,
    )


_tokenize_typename_pattern = re.compile(
    r"(\b([A-Za-z_0-9]+)(\s*::\s*[A-Za-z_][A-Za-z_0-9]*)*\b(\s*\*)*|<|>|,)"
)

_simplify_token_1 = re.compile(r"\s*\*")
_simplify_token_2 = re.compile(r"\s*::\s*")
_simplify_token_3 = re.compile(r"\s*<\s*")
_simplify_token_4 = re.compile(r"\s*>\s*")


def _simplify_token(token, is_token=True):
    if is_token:
        text = token.group(0)
    else:
        text = token
    text = _simplify_token_1.sub("*", text)
    text = _simplify_token_2.sub("::", text)
    text = _simplify_token_3.sub("<", text)
    text = _simplify_token_4.sub(">", text)
    return text


def _parse_error(pos, typename, file):
    in_file = ""
    if file is not None:
        in_file = "\nin file {0}".format(file.file_path)
    raise ValueError(
        """invalid C++ type name syntax at char {0}

    {1}
{2}{3}""".format(
            pos, typename, "-" * (4 + pos) + "^", in_file
        )
    )


def _parse_expect(what, tokens, i, typename, file):
    if i >= len(tokens):
        _parse_error(len(typename), typename, file)

    if what is not None and tokens[i].group(0) != what:
        _parse_error(tokens[i].start() + 1, typename, file)


def _parse_ignore_extra_arguments(tokens, i, typename, file, at_most):
    while tokens[i].group(0) == ",":
        if at_most == 0:
            _parse_error(tokens[i].start() + 1, typename, file)
        i, values = _parse_node(tokens, i + 1, typename, file, True, False, False)
        at_most -= 1

    return i


def _parse_maybe_quote(quoted, quote):
    if quote:
        return quoted
    else:
        return eval(quoted)


def _parse_node(tokens, i, typename, file, quote, header, inner_header):
    _parse_expect(None, tokens, i, typename, file)

    has2 = i + 1 < len(tokens)

    if tokens[i].group(0) == ",":
        _parse_error(tokens[i].start() + 1, typename, file)

    elif tokens[i].group(0) == "Bool_t":
        return i + 1, _parse_maybe_quote('numpy.dtype("?")', quote)
    elif tokens[i].group(0) == "bool":
        return i + 1, _parse_maybe_quote('numpy.dtype("?")', quote)

    elif _simplify_token(tokens[i]) == "Bool_t*":
        return (
            i + 1,
            _parse_maybe_quote(
                'uproot.containers.AsArray(False, {0}, numpy.dtype("?"))'.format(
                    header
                ),
                quote,
            ),
        )
    elif _simplify_token(tokens[i]) == "bool*":
        return (
            i + 1,
            _parse_maybe_quote(
                'uproot.containers.AsArray(False, {0}, numpy.dtype("?"))'.format(
                    header
                ),
                quote,
            ),
        )

    elif tokens[i].group(0) == "Char_t":
        return i + 1, _parse_maybe_quote('numpy.dtype("i1")', quote)
    elif tokens[i].group(0) == "char":
        return i + 1, _parse_maybe_quote('numpy.dtype("i1")', quote)
    elif tokens[i].group(0) == "UChar_t":
        return i + 1, _parse_maybe_quote('numpy.dtype("u1")', quote)
    elif has2 and tokens[i].group(0) == "unsigned" and tokens[i + 1].group(0) == "char":
        return i + 2, _parse_maybe_quote('numpy.dtype("u1")', quote)

    elif _simplify_token(tokens[i]) == "UChar_t*":
        return (
            i + 1,
            _parse_maybe_quote(
                'uproot.containers.AsArray(False, {0}, numpy.dtype("u1"))'.format(
                    header
                ),
                quote,
            ),
        )
    elif (
        has2
        and tokens[i].group(0) == "unsigned"
        and _simplify_token(tokens[i + 1]) == "char*"
    ):
        return (
            i + 2,
            _parse_maybe_quote(
                'uproot.containers.AsArray(False, {0}, numpy.dtype("u1"))'.format(
                    header
                ),
                quote,
            ),
        )

    elif tokens[i].group(0) == "Short_t":
        return i + 1, _parse_maybe_quote('numpy.dtype(">i2")', quote)
    elif tokens[i].group(0) == "short":
        return i + 1, _parse_maybe_quote('numpy.dtype(">i2")', quote)
    elif tokens[i].group(0) == "UShort_t":
        return i + 1, _parse_maybe_quote('numpy.dtype(">u2")', quote)
    elif (
        has2 and tokens[i].group(0) == "unsigned" and tokens[i + 1].group(0) == "short"
    ):
        return i + 2, _parse_maybe_quote('numpy.dtype(">u2")', quote)

    elif _simplify_token(tokens[i]) == "Short_t*":
        return (
            i + 1,
            _parse_maybe_quote(
                'uproot.containers.AsArray(False, {0}, numpy.dtype(">i2"))'.format(
                    header
                ),
                quote,
            ),
        )
    elif _simplify_token(tokens[i]) == "short*":
        return (
            i + 1,
            _parse_maybe_quote(
                'uproot.containers.AsArray(False, {0}, numpy.dtype(">i2"))'.format(
                    header
                ),
                quote,
            ),
        )
    elif _simplify_token(tokens[i]) == "UShort_t*":
        return (
            i + 1,
            _parse_maybe_quote(
                'uproot.containers.AsArray(False, {0}, numpy.dtype(">u2"))'.format(
                    header
                ),
                quote,
            ),
        )
    elif (
        has2
        and tokens[i].group(0) == "unsigned"
        and _simplify_token(tokens[i + 1]) == "short*"
    ):
        return (
            i + 2,
            _parse_maybe_quote(
                'uproot.containers.AsArray(False, {0}, numpy.dtype(">u2"))'.format(
                    header
                ),
                quote,
            ),
        )

    elif tokens[i].group(0) == "Int_t":
        return i + 1, _parse_maybe_quote('numpy.dtype(">i4")', quote)
    elif tokens[i].group(0) == "int":
        return i + 1, _parse_maybe_quote('numpy.dtype(">i4")', quote)
    elif tokens[i].group(0) == "UInt_t":
        return i + 1, _parse_maybe_quote('numpy.dtype(">u4")', quote)
    elif has2 and tokens[i].group(0) == "unsigned" and tokens[i + 1].group(0) == "int":
        return i + 2, _parse_maybe_quote('numpy.dtype(">u4")', quote)

    elif _simplify_token(tokens[i]) == "Int_t*":
        return (
            i + 1,
            _parse_maybe_quote(
                'uproot.containers.AsArray(False, {0}, numpy.dtype(">i4"))'.format(
                    header
                ),
                quote,
            ),
        )
    elif _simplify_token(tokens[i]) == "int*":
        return (
            i + 1,
            _parse_maybe_quote(
                'uproot.containers.AsArray(False, {0}, numpy.dtype(">i4"))'.format(
                    header
                ),
                quote,
            ),
        )
    elif _simplify_token(tokens[i]) == "UInt_t*":
        return (
            i + 1,
            _parse_maybe_quote(
                'uproot.containers.AsArray(False, {0}, numpy.dtype(">u4"))'.format(
                    header
                ),
                quote,
            ),
        )
    elif (
        has2
        and tokens[i].group(0) == "unsigned"
        and _simplify_token(tokens[i + 1]) == "int*"
    ):
        return (
            i + 2,
            _parse_maybe_quote(
                'uproot.containers.AsArray(False, {0}, numpy.dtype(">u4"))'.format(
                    header
                ),
                quote,
            ),
        )

    elif has2 and tokens[i].group(0) == "long" and tokens[i + 1].group(0) == "long":
        return i + 2, _parse_maybe_quote('numpy.dtype(">i8")', quote)
    elif (
        i + 2 < len(tokens)
        and tokens[i].group(0) == "unsigned"
        and tokens[i + 1].group(0) == "long"
        and tokens[i + 2].group(0) == "long"
    ):
        return i + 3, _parse_maybe_quote('numpy.dtype(">u8")', quote)

    elif tokens[i].group(0) == "Long_t":
        return i + 1, _parse_maybe_quote('numpy.dtype(">i8")', quote)
    elif tokens[i].group(0) == "Long64_t":
        return i + 1, _parse_maybe_quote('numpy.dtype(">i8")', quote)
    elif tokens[i].group(0) == "long":
        return i + 1, _parse_maybe_quote('numpy.dtype(">i8")', quote)
    elif tokens[i].group(0) == "ULong_t":
        return i + 1, _parse_maybe_quote('numpy.dtype(">u8")', quote)
    elif tokens[i].group(0) == "ULong64_t":
        return i + 1, _parse_maybe_quote('numpy.dtype(">u8")', quote)
    elif has2 and tokens[i].group(0) == "unsigned" and tokens[i + 1].group(0) == "long":
        return i + 2, _parse_maybe_quote('numpy.dtype(">u8")', quote)

    elif (
        has2
        and tokens[i].group(0) == "long"
        and _simplify_token(tokens[i + 1]) == "long*"
    ):
        return (
            i + 2,
            _parse_maybe_quote(
                'uproot.containers.AsArray({0}, numpy.dtype(">i8"))'.format(header),
                quote,
            ),
        )
    elif (
        i + 2 < len(tokens)
        and tokens[i].group(0) == "unsigned"
        and _simplify_token(tokens[i + 1]) == "long"
        and _simplify_token(tokens[i + 2]) == "long*"
    ):
        return (
            i + 3,
            _parse_maybe_quote(
                'uproot.containers.AsArray({0}, numpy.dtype(">u8"))'.format(header),
                quote,
            ),
        )

    elif _simplify_token(tokens[i]) == "Long_t*":
        return (
            i + 1,
            _parse_maybe_quote(
                'uproot.containers.AsArray(False, {0}, numpy.dtype(">i8"))'.format(
                    header
                ),
                quote,
            ),
        )
    elif _simplify_token(tokens[i]) == "Long64_t*":
        return (
            i + 1,
            _parse_maybe_quote(
                'uproot.containers.AsArray(False, {0}, numpy.dtype(">i8"))'.format(
                    header
                ),
                quote,
            ),
        )
    elif _simplify_token(tokens[i]) == "long*":
        return (
            i + 1,
            _parse_maybe_quote(
                'uproot.containers.AsArray(False, {0}, numpy.dtype(">i8"))'.format(
                    header
                ),
                quote,
            ),
        )
    elif _simplify_token(tokens[i]) == "ULong_t*":
        return (
            i + 1,
            _parse_maybe_quote(
                'uproot.containers.AsArray(False, {0}, numpy.dtype(">u8"))'.format(
                    header
                ),
                quote,
            ),
        )
    elif _simplify_token(tokens[i]) == "ULong64_t*":
        return (
            i + 1,
            _parse_maybe_quote(
                'uproot.containers.AsArray(False, {0}, numpy.dtype(">u8"))'.format(
                    header
                ),
                quote,
            ),
        )
    elif (
        has2
        and tokens[i].group(0) == "unsigned"
        and _simplify_token(tokens[i + 1]) == "long*"
    ):
        return (
            i + 2,
            _parse_maybe_quote(
                'uproot.containers.AsArray(False, {0}, numpy.dtype(">u8"))'.format(
                    header
                ),
                quote,
            ),
        )

    elif tokens[i].group(0) == "Float_t":
        return i + 1, _parse_maybe_quote('numpy.dtype(">f4")', quote)
    elif tokens[i].group(0) == "float":
        return i + 1, _parse_maybe_quote('numpy.dtype(">f4")', quote)

    elif _simplify_token(tokens[i]) == "Float_t*":
        return (
            i + 1,
            _parse_maybe_quote(
                'uproot.containers.AsArray(False, {0}, numpy.dtype(">f4"))'.format(
                    header
                ),
                quote,
            ),
        )
    elif _simplify_token(tokens[i]) == "float*":
        return (
            i + 1,
            _parse_maybe_quote(
                'uproot.containers.AsArray(False, {0}, numpy.dtype(">f4"))'.format(
                    header
                ),
                quote,
            ),
        )

    elif tokens[i].group(0) == "Double_t":
        return i + 1, _parse_maybe_quote('numpy.dtype(">f8")', quote)
    elif tokens[i].group(0) == "double":
        return i + 1, _parse_maybe_quote('numpy.dtype(">f8")', quote)

    elif _simplify_token(tokens[i]) == "Double_t*":
        return (
            i + 1,
            _parse_maybe_quote(
                'uproot.containers.AsArray(False, {0}, numpy.dtype(">f8"))'.format(
                    header
                ),
                quote,
            ),
        )
    elif _simplify_token(tokens[i]) == "double*":
        return (
            i + 1,
            _parse_maybe_quote(
                'uproot.containers.AsArray(False, {0}, numpy.dtype(">f8"))'.format(
                    header
                ),
                quote,
            ),
        )

    elif tokens[i].group(0) == "Float16_t":
        return (
            i + 1,
            _parse_maybe_quote(
                'uproot.containers.AsFIXME("Float16_t in another context")', quote
            ),
        )

    elif _simplify_token(tokens[i]) == "Float16_t*":
        return (
            i + 1,
            _parse_maybe_quote(
                "uproot.containers.AsArray(False, {0}, "
                'uproot.containers.AsFIXME("Float16_t in array"))'.format(header),
                quote,
            ),
        )

    elif tokens[i].group(0) == "Double32_t":
        return (
            i + 1,
            _parse_maybe_quote(
                'uproot.containers.AsFIXME("Double32_t in another context")', quote
            ),
        )

    elif _simplify_token(tokens[i]) == "Double32_t*":
        return (
            i + 1,
            _parse_maybe_quote(
                "uproot.containers.AsArray(False, {0}, "
                'uproot.containers.AsFIXME("Double32_t in array '
                '(note: Event.root fClosestDistance has an example)"))'.format(header),
                quote,
            ),
        )

    elif tokens[i].group(0) == "string" or _simplify_token(tokens[i]) == "std::string":
        return (
            i + 1,
            _parse_maybe_quote("uproot.containers.AsString({0})".format(header), quote),
        )
    elif tokens[i].group(0) == "TString":
        return (
            i + 1,
            _parse_maybe_quote(
                "uproot.containers.AsString(False, typename='TString')", quote
            ),
        )
    elif _simplify_token(tokens[i]) == "char*":
        return (
            i + 1,
            _parse_maybe_quote(
                "uproot.containers.AsString(False, length_bytes='4', typename='char*')",
                quote,
            ),
        )
    elif (
        has2
        and tokens[i].group(0) == "const"
        and _simplify_token(tokens[i + 1]) == "char*"
    ):
        return (
            i + 2,
            _parse_maybe_quote(
                "uproot.containers.AsString(False, length_bytes='4', typename='char*')",
                quote,
            ),
        )

    elif tokens[i].group(0) == "bitset" or _simplify_token(tokens[i]) == "std::bitset":
        _parse_expect("<", tokens, i + 1, typename, file)
        _parse_expect(None, tokens, i + 2, typename, file)
        try:
            num_bits = int(tokens[i + 2].group(0))
        except ValueError:
            _parse_error(tokens[i + 2].start() + 1, typename, file)
        # std::bitset only ever has one argument
        _parse_expect(">", tokens, i + 3, typename, file)
        return (
            i + 4,
            _parse_maybe_quote(
                'uproot.containers.AsFIXME("std::bitset<{0}>")'.format(num_bits),
                quote,
            ),
        )

    elif tokens[i].group(0) == "vector" or _simplify_token(tokens[i]) == "std::vector":
        _parse_expect("<", tokens, i + 1, typename, file)
        i, values = _parse_node(
            tokens, i + 2, typename, file, quote, inner_header, inner_header
        )
        i = _parse_ignore_extra_arguments(tokens, i, typename, file, 1)
        _parse_expect(">", tokens, i, typename, file)
        if quote:
            return (
                i + 1,
                "uproot.containers.AsVector({0}, {1})".format(header, values),
            )
        else:
            return i + 1, uproot.containers.AsVector(header, values)

    elif tokens[i].group(0) == "set" or _simplify_token(tokens[i]) == "std::set":
        _parse_expect("<", tokens, i + 1, typename, file)
        i, keys = _parse_node(
            tokens, i + 2, typename, file, quote, inner_header, inner_header
        )
        i = _parse_ignore_extra_arguments(tokens, i, typename, file, 2)
        _parse_expect(">", tokens, i, typename, file)
        if quote:
            return i + 1, "uproot.containers.AsSet({0}, {1})".format(header, keys)
        else:
            return i + 1, uproot.containers.AsSet(header, keys)

    elif tokens[i].group(0) == "map" or _simplify_token(tokens[i]) == "std::map":
        _parse_expect("<", tokens, i + 1, typename, file)
        i, keys = _parse_node(
            tokens, i + 2, typename, file, quote, header, inner_header
        )
        _parse_expect(",", tokens, i, typename, file)
        i, values = _parse_node(
            tokens, i + 1, typename, file, quote, header, inner_header
        )
        i = _parse_ignore_extra_arguments(tokens, i, typename, file, 2)
        _parse_expect(">", tokens, i, typename, file)
        if quote:
            return (
                i + 1,
                "uproot.containers.AsMap({0}, {1}, {2})".format(header, keys, values),
            )
        else:
            return i + 1, uproot.containers.AsMap(header, keys, values)

    else:
        start, stop = tokens[i].span()

        if has2 and tokens[i + 1].group(0) == "<":
            i, keys = _parse_node(
                tokens, i + 2, typename, file, quote, inner_header, inner_header
            )
            while tokens[i].group(0) == ",":
                i, keys = _parse_node(
                    tokens, i + 1, typename, file, quote, inner_header, inner_header
                )
            _parse_expect(">", tokens, i, typename, file)
            stop = tokens[i].span()[1]

        classname = _simplify_token(typename[start:stop], is_token=False)
        classname = uproot.model.classname_regularize(classname)

        pointers = 0
        while classname.endswith("*"):
            pointers += 1
            classname = classname[:-1]

        if quote:
            cls = "c({0})".format(repr(classname))
            for x in uproot._util.range(pointers):
                cls = "uproot.containers.AsPointer({0})".format(cls)
        elif file is None:
            cls = uproot.classes[classname]
            for x in uproot._util.range(pointers):
                cls = uproot.containers.AsPointer(cls)
        else:
            cls = file.class_named(classname)
            for x in uproot._util.range(pointers):
                cls = uproot.containers.AsPointer(cls)

        return i + 1, cls


def parse_typename(
    typename,
    file=None,
    quote=False,
    outer_header=True,
    inner_header=False,
    string_header=False,
):
    """
    Args:
        typename (str): The C++ type to parse.
        file (None or :doc:`uproot.reading.CommonFileMethods`): Used to provide
            error messages with the ``file_path``.
        quote (bool): If True, return the output as a string to evaluate. This
            is used to build code for a :doc:`uproot.model.Model`, rather than
            the :doc:`uproot.model.Model` itself.
        outer_header (bool): If True, set the ``header`` flag for the outermost
            :doc:`uproot.containers.AsContainer` to True.
        inner_header (bool): If True, set the ``header`` flag for inner
            :doc:`uproot.containers.AsContainer` objects to True.
        string_header (bool): If True, set the ``header`` flag for
            :doc:`uproot.containers.AsString` objects to True.

    Return a :doc:`uproot.model.Model` or :doc:`uproot.containers.AsContainer`
    for the C++ ``typename``.
    """
    tokens = list(_tokenize_typename_pattern.finditer(typename))

    if (
        not string_header
        and len(tokens) != 0
        and (
            tokens[0].group(0) == "string"
            or _simplify_token(tokens[0]) == "std::string"
        )
    ):
        i, out = 1, _parse_maybe_quote("uproot.containers.AsString(False)", quote)

    else:
        i, out = _parse_node(
            tokens, 0, typename, file, quote, outer_header, inner_header
        )

    if i < len(tokens):
        _parse_error(tokens[i].start(), typename, file)

    return out


class NotNumerical(Exception):
    """
    Exception used to stop searches for a numerical interpretation in
    :doc:`uproot.interpretation.identify.interpretation_of` as soon as a
    non-conforming type is found.
    """

    pass


class UnknownInterpretation(Exception):
    """
    Exception raised by :doc:`uproot.interpretation.identify.interpretation_of`
    if an :doc:`uproot.interpretation.Interpretation` cannot be found.

    The :ref:`uproot.behaviors.TBranch.TBranch.interpretation` property may have
    :doc:`uproot.interpretation.identify.UnknownInterpretation` as a value.

    Any attempts to use this class as a
    :doc:`uproot.interpretation.Interpretation` causes it to raise itself.
    Thus, failing to find an interpretation for a ``TBranch`` is not a fatal
    error, but attempting to use it to deserialize arrays is a fatal error.
    """

    def __init__(self, reason, file_path, object_path):
        self.reason = reason
        self.file_path = file_path
        self.object_path = object_path

    def __repr__(self):
        return "<UnknownInterpretation {0}>".format(repr(self.reason))

    def __str__(self):
        return """{0}
in file {1}
in object {2}""".format(
            self.reason, self.file_path, self.object_path
        )

    @property
    def typename(self):
        return "unknown"

    @property
    def cache_key(self):
        raise self

    @property
    def numpy_dtype(self):
        raise self

    def awkward_form(
        self,
        file,
        index_format="i64",
        header=False,
        tobject_header=True,
        breadcrumbs=(),
    ):
        raise self

    @property
    def basket_array(self):
        raise self

    @property
    def final_array(self):
        raise self

    @property
    def hook_before_basket_array(self):
        raise self

    @property
    def hook_after_basket_array(self):
        raise self

    @property
    def hook_before_final_array(self):
        raise self

    @property
    def hook_before_library_finalize(self):
        raise self

    @property
    def hook_after_final_array(self):
        raise self

    @property
    def itemsize(self):
        raise self

    @property
    def from_dtype(self):
        raise self

    @property
    def to_dtype(self):
        raise self

    @property
    def content(self):
        raise self

    @property
    def header_bytes(self):
        raise self

    @property
    def size_1to5_bytes(self):
        raise self
