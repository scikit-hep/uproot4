# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Represents external libraries that define "array-like" types so that users can
choose an output format.

The :doc:`uproot4.interpretation.library.NumPy` library always works (NumPy is
Uproot's only strict dependency) and outputs NumPy arrays for single arrays
and dict/tuple/list as groups. Objects and jagged arrays are not efficiently
represented, but it provides a zero-dependency least common denominator.

The :doc:`uproot4.interpretation.library.Awkward` library is the default and
depends on Awkward Array (``awkward1``). It is usually the best option, as it
was designed for Uproot.

The :doc:`uproot4.interpretation.library.Pandas` library outputs
``pandas.Series`` for single arrays and ``pandas.DataFrame`` as groups. Objects
are not efficiently represented, but some jagged arrays are encoded as
``pandas.MultiIndex``.

The :doc:`uproot4.interpretation.library.CuPy` library outputs arrays on a
GPU, but the types that it supports are limited. Note that Awkward Arrays can
be GPU-resident as well.

Lazy arrays (:doc:`uproot4.behavior.TBranch.TBranch.lazy`) can only use the
:doc:`uproot4.interpretation.library.Awkward` library.
"""

from __future__ import absolute_import

import itertools
import json

import numpy

import uproot4.interpretation.jagged
import uproot4.interpretation.strings
import uproot4.interpretation.objects
import uproot4.containers
import uproot4.extras


class Library(object):
    """
    Abstract superclass of array-library handlers, for libraries such as NumPy,
    Awkward Array, Pandas, and CuPy.

    A library is used in the finalization and grouping stages of producing an
    array, converting it from internal representations like
    :doc:`uproot4.interpretation.jagged.JaggedArray`,
    :doc:`uproot4.interpretation.strings.StringArray`, and
    :doc:`uproot4.interpretation.objects.ObjectArray` into the library's
    equivalents. It can also be required for concatenation and other late-stage
    operations on the output arrays.

    Libraries are usually selected by a string name. These names are held in a
    private registry in the :doc:`uproot4.interpretation.library` module.
    """

    @property
    def imported(self):
        """
        Attempts to import the library and returns the imported module.
        """
        raise AssertionError

    def empty(self, shape, dtype):
        """
        Args:
            shape (tuple of int): NumPy array ``shape``. (The first item must
                be zero.)
            dtype (``numpy.dtype`` or its constructor argument): NumPy array
                ``dtype``.

        Returns an empty NumPy-like array.
        """
        return numpy.empty(shape, dtype)

    def finalize(self, array, branch, interpretation, entry_start, entry_stop):
        """
        Args:
            array (array): Internal, temporary, trimmed array. If this is a
                NumPy array, it may be identical to the output array.
            branch (:doc:`uproot4.behavior.TBranch.TBranch`): The ``TBranch``
                that is represented by this array.
            interpretation (:doc:`uproot4.interpretation.Interpretation`): The
                interpretation that produced the ``array``.
            entry_start (int): First entry that is included in the output.
            entry_stop (int): FIrst entry that is excluded (one greater than
                the last entry that is included) in the output.

        Create a library-appropriate output array for this temporary ``array``.

        This array would represent one ``TBranch`` (i.e. not a "group").
        """
        raise AssertionError

    def group(self, arrays, expression_context, how):
        u"""
        Args:
            arrays (dict of str \u2192 array): Mapping from names to finalized
                array objets to combine into a group.
            expression_context (list of (str, dict) tuples): Expression strings
                and a dict of metadata about each.
            how (None, str, or container type): Library-dependent instructions
                for grouping. The only recognized container types are ``tuple``,
                ``list``, and ``dict``. Note that the container *type itself*
                must be passed as ``how``, not an instance of that type (i.e.
                ``how=tuple``, not ``how=()``).

        Combine the finalized ``arrays`` into a library-appropriate group type.
        """
        if how is tuple:
            return tuple(arrays[name] for name, _ in expression_context)
        elif how is list:
            return [arrays[name] for name, _ in expression_context]
        elif how is dict or how is None:
            return dict((name, arrays[name]) for name, _ in expression_context)
        else:
            raise TypeError(
                "for library {0}, how must be tuple, list, dict, or None (for "
                "dict)".format(self.name)
            )

    def global_index(self, array, global_start):
        """
        Args:
            array (array): The library-appropriate array whose global index
                needs adjustment.
            global_start (int): A number to add to the global index of
                ``array`` to correct it.

        Apply *in-place* corrections to the global index of ``array`` by adding
        ``global_start``.

        Even though the operation is performed *in-place*, this method returns
        the ``array``.
        """
        return array

    def concatenate(self, all_arrays):
        """
        Args:
            all_arrays (list of arrays): A list of library-appropriate arrays
                that need to be concatenated.

        Returns a concatenated version of ``all_arrays``.
        """
        raise AssertionError

    def __repr__(self):
        return repr(self.name)

    def __eq__(self, other):
        return type(_libraries[self.name]) is type(_libraries[other.name])  # noqa: E721


class NumPy(Library):
    u"""
    A :doc:`uproot4.interpetation.library.Library` that presents ``TBranch``
    data as NumPy arrays. The standard name for this library is ``"np"``.

    The single-``TBranch`` form for this library is a ``numpy.ndarray``. If
    the data are non-numerical, they will be converted into Python objects and
    stored in an array with ``dtype="O"``. This is inefficient, but it is the
    minimal-dependency option for Python.

    The "group" behavior for this library is:

    * ``how=dict`` or ``how=None``: a dict of str \u2192 array, mapping the
      names to arrays.
    * ``how=tuple``: a tuple of arrays, in the order requested. (Names are
      lost.)
    * ``how=list``: a list of arrays, in the order requested. (Names are lost.)

    Since NumPy arrays are not indexed, ``global_index`` has no effect.
    """

    name = "np"

    @property
    def imported(self):
        import numpy

        return numpy

    def finalize(self, array, branch, interpretation, entry_start, entry_stop):
        if isinstance(array, uproot4.interpretation.jagged.JaggedArray) and isinstance(
            array.content, uproot4.interpretation.objects.StridedObjectArray,
        ):
            out = numpy.zeros(len(array), dtype=numpy.object)
            for i, x in enumerate(array):
                out[i] = numpy.zeros(len(x), dtype=numpy.object)
                for j, y in enumerate(x):
                    out[i][j] = y
            return out

        elif isinstance(
            array,
            (
                uproot4.interpretation.jagged.JaggedArray,
                uproot4.interpretation.strings.StringArray,
                uproot4.interpretation.objects.ObjectArray,
                uproot4.interpretation.objects.StridedObjectArray,
            ),
        ):
            out = numpy.zeros(len(array), dtype=numpy.object)
            for i, x in enumerate(array):
                out[i] = x
            return out

        else:
            return array

    def concatenate(self, all_arrays):
        if len(all_arrays) == 0:
            return all_arrays

        if isinstance(all_arrays[0], (tuple, list)):
            keys = uproot4._util.range(len(all_arrays[0]))
        elif isinstance(all_arrays[0], dict):
            keys = list(all_arrays[0])
        else:
            raise AssertionError(repr(all_arrays[0]))

        to_concatenate = dict((k, []) for k in keys)
        for arrays in all_arrays:
            for k in keys:
                to_concatenate[k].append(arrays[k])

        concatenated = dict((k, numpy.concatenate(to_concatenate[k])) for k in keys)

        if isinstance(all_arrays[0], tuple):
            return tuple(concatenated[k] for k in keys)
        elif isinstance(all_arrays[0], list):
            return [concatenated[k] for k in keys]
        elif isinstance(all_arrays[0], dict):
            return concatenated


def _strided_to_awkward(awkward1, path, interpretation, data):
    contents = []
    names = []
    for name, member in interpretation.members:
        if not name.startswith("@"):
            p = name
            if len(path) != 0:
                p = path + "/" + name
            if isinstance(member, uproot4.interpretation.objects.AsStridedObjects):
                contents.append(_strided_to_awkward(awkward1, p, member, data))
            else:
                contents.append(
                    awkward1.from_numpy(
                        numpy.array(data[p]), regulararray=True, highlevel=False
                    )
                )
            names.append(name)
    parameters = {
        "__record__": uproot4.model.classname_decode(interpretation.model.__name__)[0]
    }
    return awkward1.layout.RecordArray(
        contents, names, len(data), parameters=parameters
    )


# FIXME: _object_to_awkward_json and _awkward_json_to_array are slow functions
# with the right outputs to be replaced by compiled versions in awkward1._io.


def _object_to_awkward_json(form, obj):
    if form["class"] == "NumpyArray":
        return obj

    elif form["class"] == "RecordArray":
        out = {}
        for name, subform in form["contents"].items():
            if not name.startswith("@"):
                if obj.has_member(name):
                    out[name] = _object_to_awkward_json(subform, obj.member(name))
                else:
                    out[name] = _object_to_awkward_json(subform, getattr(obj, name))
        return out

    elif form["class"][:15] == "ListOffsetArray":
        if form["parameters"].get("__array__") == "string":
            return obj

        elif form["parameters"].get("__array__") == "sorted_map":
            key_form = form["content"]["contents"][0]
            value_form = form["content"]["contents"][1]
            return [
                (
                    _object_to_awkward_json(key_form, x),
                    _object_to_awkward_json(value_form, y),
                )
                for x, y in obj.items()
            ]

        else:
            subform = form["content"]
            return [_object_to_awkward_json(subform, x) for x in obj]

    elif form["class"] == "RegularArray":
        subform = form["content"]
        return [_object_to_awkward_json(subform, x) for x in obj]

    else:
        raise AssertionError(form["class"])


def _awkward_p(form):
    out = form["parameters"]
    out.pop("uproot", None)
    return out


def _awkward_json_to_array(awkward1, form, array):
    if form["class"] == "NumpyArray":
        return array

    elif form["class"] == "RecordArray":
        contents = []
        names = []
        for name, subform in form["contents"].items():
            if not name.startswith("@"):
                contents.append(_awkward_json_to_array(awkward1, subform, array[name]))
                names.append(name)
        return awkward1.layout.RecordArray(
            contents, names, len(array), parameters=_awkward_p(form)
        )

    elif form["class"][:15] == "ListOffsetArray":
        if form["parameters"].get("__array__") == "string":
            content = _awkward_json_to_array(awkward1, form["content"], array.content)
            return type(array)(array.offsets, content, parameters=_awkward_p(form))

        elif form["parameters"].get("__array__") == "sorted_map":
            key_form = form["content"]["contents"][0]
            value_form = form["content"]["contents"][1]
            keys = _awkward_json_to_array(awkward1, key_form, array.content["0"])
            values = _awkward_json_to_array(awkward1, value_form, array.content["1"])
            content = awkward1.layout.RecordArray(
                (keys, values),
                None,
                len(array.content),
                parameters=_awkward_p(form["content"]),
            )
            return type(array)(array.offsets, content, parameters=_awkward_p(form))

        else:
            content = _awkward_json_to_array(awkward1, form["content"], array.content)
            return type(array)(array.offsets, content, parameters=_awkward_p(form))

    elif form["class"] == "RegularArray":
        content = _awkward_json_to_array(awkward1, form["content"], array.content)
        return awkward1.layout.RegularArray(
            content, form["size"], parameters=_awkward_p(form)
        )

    else:
        raise AssertionError(form["class"])


class Awkward(Library):
    u"""
    A :doc:`uproot4.interpetation.library.Library` that presents ``TBranch``
    data as Awkward Arrays. The standard name for this library is ``"ak"``.

    This is the default for all functions that require a
    :doc:`uproot4.interpetation.library.Library`, though Uproot does not
    explicitly depend on Awkward Array. If you are confronted with a message
    that Awkward Array is not installed, either install ``awkward1`` or
    select another library (likely :doc:`uproot4.interpretation.library.NumPy`).

    Both the single-``TBranch`` and "group" forms for this library are
    ``ak.Array``, though groups are always arrays of records. Awkward Array
    was originally developed for Uproot, so the data structures are usually
    optimial for Uproot data.

    The "group" behavior for this library is:

    * ``how=None``: an array of Awkward records.
    * ``how=dict``: a dict of str \u2192 array, mapping the names to arrays.
    * ``how=tuple``: a tuple of arrays, in the order requested. (Names are
      lost.)
    * ``how=list``: a list of arrays, in the order requested. (Names are lost.)

    Since Awkward arrays are not indexed, ``global_index`` has no effect.
    """

    name = "ak"

    @property
    def imported(self):
        return uproot4.extras.awkward1()

    def finalize(self, array, branch, interpretation, entry_start, entry_stop):
        awkward1 = self.imported

        if isinstance(array, awkward1.layout.Content):
            return awkward1.Array(array)

        elif isinstance(array, uproot4.interpretation.objects.StridedObjectArray):
            return awkward1.Array(
                _strided_to_awkward(awkward1, "", array.interpretation, array.array)
            )

        elif isinstance(
            array, uproot4.interpretation.jagged.JaggedArray
        ) and isinstance(
            array.content, uproot4.interpretation.objects.StridedObjectArray
        ):
            content = _strided_to_awkward(
                awkward1, "", array.content.interpretation, array.content.array
            )
            if issubclass(array.offsets.dtype.type, numpy.int32):
                offsets = awkward1.layout.Index32(array.offsets)
                layout = awkward1.layout.ListOffsetArray32(offsets, content)
            else:
                offsets = awkward1.layout.Index64(array.offsets)
                layout = awkward1.layout.ListOffsetArray64(offsets, content)
            return awkward1.Array(layout)

        elif isinstance(array, uproot4.interpretation.jagged.JaggedArray):
            content = awkward1.from_numpy(
                array.content, regulararray=True, highlevel=False
            )
            if issubclass(array.offsets.dtype.type, numpy.int32):
                offsets = awkward1.layout.Index32(array.offsets)
                layout = awkward1.layout.ListOffsetArray32(offsets, content)
            else:
                offsets = awkward1.layout.Index64(array.offsets)
                layout = awkward1.layout.ListOffsetArray64(offsets, content)
            return awkward1.Array(layout)

        elif isinstance(array, uproot4.interpretation.strings.StringArray):
            content = awkward1.layout.NumpyArray(
                numpy.frombuffer(array.content, dtype=numpy.dtype(numpy.uint8)),
                parameters={"__array__": "char"},
            )
            if issubclass(array.offsets.dtype.type, numpy.int32):
                offsets = awkward1.layout.Index32(array.offsets)
                layout = awkward1.layout.ListOffsetArray32(
                    offsets, content, parameters={"__array__": "string"}
                )
            elif issubclass(array.offsets.dtype.type, numpy.uint32):
                offsets = awkward1.layout.IndexU32(array.offsets)
                layout = awkward1.layout.ListOffsetArrayU32(
                    offsets, content, parameters={"__array__": "string"}
                )
            elif issubclass(array.offsets.dtype.type, numpy.int64):
                offsets = awkward1.layout.Index64(array.offsets)
                layout = awkward1.layout.ListOffsetArray64(
                    offsets, content, parameters={"__array__": "string"}
                )
            else:
                raise AssertionError(repr(array.offsets.dtype))
            return awkward1.Array(layout)

        elif isinstance(interpretation, uproot4.interpretation.objects.AsObjects):
            try:
                form = json.loads(
                    interpretation.awkward_form(interpretation.branch.file).tojson(
                        verbose=True
                    )
                )
            except uproot4.interpretation.objects.CannotBeAwkward as err:
                raise ValueError(
                    """cannot produce Awkward Arrays for interpretation {0} because

    {1}

instead, try library="np" instead of library="ak"

in file {2}
in object {3}""".format(
                        repr(interpretation),
                        err.because,
                        interpretation.branch.file.file_path,
                        interpretation.branch.object_path,
                    )
                )

            unlabeled = awkward1.from_iter(
                (_object_to_awkward_json(form, x) for x in array), highlevel=False
            )
            return awkward1.Array(_awkward_json_to_array(awkward1, form, unlabeled))

        elif array.dtype.names is not None:
            length, shape = array.shape[0], array.shape[1:]
            array = array.reshape(-1)
            contents = []
            for name in array.dtype.names:
                contents.append(
                    awkward1.from_numpy(
                        numpy.array(array[name]), regulararray=True, highlevel=False
                    )
                )
            out = awkward1.layout.RecordArray(contents, array.dtype.names, length)
            for size in shape[::-1]:
                out = awkward1.layout.RegularArray(out, size)
            return awkward1.Array(out)

        else:
            return awkward1.from_numpy(array, regulararray=True)

    def group(self, arrays, expression_context, how):
        awkward1 = self.imported

        if how is tuple:
            return tuple(arrays[name] for name, _ in expression_context)
        elif how is list:
            return [arrays[name] for name, _ in expression_context]
        elif how is dict:
            return dict((name, arrays[name]) for name, _ in expression_context)
        elif how is None:
            return awkward1.zip(
                dict((name, arrays[name]) for name, _ in expression_context),
                depth_limit=1,
            )
        elif how == "zip":
            nonjagged = []
            offsets = []
            jaggeds = []
            for name, context in expression_context:
                array = arrays[name]
                if context["is_jagged"]:
                    if len(offsets) == 0:
                        offsets.append(array.layout.offsets)
                        jaggeds.append([name])
                    else:
                        for o, j in zip(offsets, jaggeds):
                            if numpy.array_equal(array.layout.offsets, o):
                                j.append(name)
                                break
                        else:
                            offsets.append(array.layout.offsets)
                            jaggeds.append([name])
                else:
                    nonjagged.append(name)
            out = None
            if len(nonjagged) != 0:
                out = awkward1.zip(
                    dict((name, arrays[name]) for name in nonjagged), depth_limit=1
                )
            for number, jagged in enumerate(jaggeds):
                cut = len(jagged[0])
                for name in jagged:
                    cut = min(cut, len(name))
                    while cut > 0 and (
                        name[:cut] != jagged[0][:cut]
                        or name[cut - 1] not in ("_", ".", "/")
                    ):
                        cut -= 1
                    if cut == 0:
                        break
                if cut == 0:
                    common = "jagged{0}".format(number)
                    subarray = awkward1.zip(
                        dict((name, arrays[name]) for name in jagged)
                    )
                else:
                    common = jagged[0][:cut].strip("_./")
                    subarray = awkward1.zip(
                        dict((name[cut:].strip("_./"), arrays[name]) for name in jagged)
                    )
                if out is None:
                    out = awkward1.zip({common: subarray}, depth_limit=1)
                else:
                    for name in jagged:
                        out = awkward1.with_field(out, subarray, common)
            return out
        else:
            raise TypeError(
                'for library {0}, how must be tuple, list, dict, "zip" for '
                "a record array with jagged arrays zipped, if possible, or "
                "None, for an unzipped record array".format(self.name)
            )

    def concatenate(self, all_arrays):
        awkward1 = self.imported

        if len(all_arrays) == 0:
            return all_arrays

        if isinstance(all_arrays[0], (tuple, list)):
            keys = uproot4._util.range(len(all_arrays[0]))
        elif isinstance(all_arrays[0], dict):
            keys = list(all_arrays[0])
        else:
            return awkward1.concatenate(all_arrays)

        to_concatenate = dict((k, []) for k in keys)
        for arrays in all_arrays:
            for k in keys:
                to_concatenate[k].append(arrays[k])

        concatenated = dict((k, awkward1.concatenate(to_concatenate[k])) for k in keys)

        if isinstance(all_arrays[0], tuple):
            return tuple(concatenated[k] for k in keys)
        elif isinstance(all_arrays[0], list):
            return [concatenated[k] for k in keys]
        elif isinstance(all_arrays[0], dict):
            return concatenated


def _pandas_rangeindex():
    import pandas

    return getattr(pandas, "RangeIndex", pandas.Int64Index)


def _strided_to_pandas(path, interpretation, data, arrays, columns):
    for name, member in interpretation.members:
        if not name.startswith("@"):
            p = path + (name,)
            if isinstance(member, uproot4.interpretation.objects.AsStridedObjects):
                _strided_to_pandas(p, member, data, arrays, columns)
            else:
                arrays.append(data["/".join(p)])
                columns.append(p)


def _pandas_basic_index(pandas, entry_start, entry_stop):
    if hasattr(pandas, "RangeIndex"):
        return pandas.RangeIndex(entry_start, entry_stop)
    else:
        return pandas.Int64Index(uproot4._util.range(entry_start, entry_stop))


class Pandas(Library):
    u"""
    A :doc:`uproot4.interpetation.library.Library` that presents ``TBranch``
    data as Pandas Series and DataFrames. The standard name for this library is
    ``"pd"``.

    The single-``TBranch`` (with a single ``TLeaf``) form for this library is
    ``pandas.Series``, and the "group" form is ``pandas.DataFrame``.

    The "group" behavior for this library is:

    * ``how=None`` or a string: passed to ``pandas.merge`` as its ``how``
      parameter, which would be relevant if jagged arrays with different
      multiplicity are requested.
    * ``how=dict``: a dict of str \u2192 array, mapping the names to
      ``pandas.Series``.
    * ``how=tuple``: a tuple of ``pandas.Series``, in the order requested.
      (Names are assigned to the ``pandas.Series``.)
    * ``how=list``: a list of ``pandas.Series``, in the order requested.
      (Names are assigned to the ``pandas.Series``.)

    Pandas Series and DataFrames are indexed, so ``global_index`` adjusts them.
    """

    name = "pd"

    @property
    def imported(self):
        return uproot4.extras.pandas()

    def finalize(self, array, branch, interpretation, entry_start, entry_stop):
        pandas = self.imported

        if isinstance(array, uproot4.interpretation.objects.StridedObjectArray):
            arrays = []
            columns = []
            _strided_to_pandas((), array.interpretation, array.array, arrays, columns)
            maxlen = max(len(x) for x in columns)
            if maxlen == 1:
                columns = [x[0] for x in columns]
            else:
                columns = pandas.MultiIndex.from_tuples(
                    [x + ("",) * (maxlen - len(x)) for x in columns]
                )
            index = _pandas_basic_index(pandas, entry_start, entry_stop)
            out = pandas.DataFrame(
                dict(zip(columns, arrays)), columns=columns, index=index
            )
            out.leaflist = maxlen != 1
            return out

        elif isinstance(
            array, uproot4.interpretation.jagged.JaggedArray
        ) and isinstance(
            array.content, uproot4.interpretation.objects.StridedObjectArray
        ):
            index = pandas.MultiIndex.from_arrays(
                array.parents_localindex(entry_start, entry_stop),
                names=["entry", "subentry"],
            )
            arrays = []
            columns = []
            _strided_to_pandas(
                (), array.content.interpretation, array.content.array, arrays, columns
            )
            maxlen = max(len(x) for x in columns)
            if maxlen == 1:
                columns = [x[0] for x in columns]
            else:
                columns = pandas.MultiIndex.from_tuples(
                    [x + ("",) * (maxlen - len(x)) for x in columns]
                )
            out = pandas.DataFrame(
                dict(zip(columns, arrays)), columns=columns, index=index
            )
            out.leaflist = maxlen != 1
            return out

        elif isinstance(array, uproot4.interpretation.jagged.JaggedArray):
            index = pandas.MultiIndex.from_arrays(
                array.parents_localindex(entry_start, entry_stop),
                names=["entry", "subentry"],
            )
            return pandas.Series(array.content, index=index)

        elif isinstance(
            array,
            (
                uproot4.interpretation.strings.StringArray,
                uproot4.interpretation.objects.ObjectArray,
            ),
        ):
            out = numpy.zeros(len(array), dtype=numpy.object)
            for i, x in enumerate(array):
                out[i] = x
            index = _pandas_basic_index(pandas, entry_start, entry_stop)
            return pandas.Series(out, index=index)

        elif array.dtype.names is not None and len(array.shape) != 1:
            names = []
            arrays = {}
            for n in array.dtype.names:
                for tup in itertools.product(
                    *[uproot4._util.range(d) for d in array.shape[1:]]
                ):
                    name = (n + "".join("[" + str(x) + "]" for x in tup),)
                    names.append(name)
                    arrays[name] = array[n][(slice(None),) + tup]
            index = _pandas_basic_index(pandas, entry_start, entry_stop)
            out = pandas.DataFrame(arrays, columns=names, index=index)
            out.leaflist = True
            return out

        elif array.dtype.names is not None:
            columns = pandas.MultiIndex.from_tuples([(x,) for x in array.dtype.names])
            arrays = dict((y, array[x]) for x, y in zip(array.dtype.names, columns))
            index = _pandas_basic_index(pandas, entry_start, entry_stop)
            out = pandas.DataFrame(arrays, columns=columns, index=index)
            out.leaflist = True
            return out

        elif len(array.shape) != 1:
            names = []
            arrays = {}
            for tup in itertools.product(
                *[uproot4._util.range(d) for d in array.shape[1:]]
            ):
                name = "".join("[" + str(x) + "]" for x in tup)
                names.append(name)
                arrays[name] = array[(slice(None),) + tup]
            index = _pandas_basic_index(pandas, entry_start, entry_stop)
            out = pandas.DataFrame(arrays, columns=names, index=index)
            out.leaflist = False
            return out

        else:
            index = _pandas_basic_index(pandas, entry_start, entry_stop)
            return pandas.Series(array, index=index)

    def _only_series(self, original_arrays, original_names):
        pandas = self.imported
        arrays = {}
        names = []
        for name in original_names:
            if isinstance(original_arrays[name], pandas.Series):
                arrays[name] = original_arrays[name]
                names.append(name)
            else:
                df = original_arrays[name]
                for subname in df.columns:
                    if df.leaflist:
                        if isinstance(subname, tuple):
                            path = (name,) + subname
                        else:
                            path = (name, subname)
                    else:
                        path = name + subname
                    arrays[path] = df[subname]
                    names.append(path)
        return arrays, names

    def group(self, arrays, expression_context, how):
        pandas = self.imported
        names = [name for name, _ in expression_context]

        if how is tuple:
            return tuple(arrays[name] for name in names)

        elif how is list:
            return [arrays[name] for name in names]

        elif how is dict:
            return dict((name, arrays[name]) for name in names)

        elif uproot4._util.isstr(how) or how is None:
            arrays, names = self._only_series(arrays, names)

            if any(isinstance(x, tuple) for x in names):
                longest = max(len(x) for x in names if isinstance(x, tuple))
                newarrays, newnames = {}, []
                for x in names:
                    if not isinstance(x, tuple):
                        y = (x,) + (None,) * (longest - 1)
                    else:
                        y = x + (None,) * (longest - len(x))
                    newarrays[y] = arrays[x]
                    newnames.append(y)
                arrays = newarrays
                names = pandas.MultiIndex.from_tuples(newnames)

            if all(isinstance(x.index, _pandas_rangeindex()) for x in arrays.values()):
                return pandas.DataFrame(data=arrays, columns=names)

            indexes = []
            groups = []
            for name in names:
                array = arrays[name]
                if isinstance(array.index, pandas.MultiIndex):
                    for index, group in zip(indexes, groups):
                        if numpy.array_equal(array.index, index):
                            group.append(name)
                            break
                    else:
                        indexes.append(array.index)
                        groups.append([name])
            if how is None:
                flat_index = None
                dfs = [[] for x in indexes]
                group_names = [[] for x in indexes]
                for index, group, df, gn in zip(indexes, groups, dfs, group_names):
                    for name in names:
                        array = arrays[name]
                        if isinstance(array.index, _pandas_rangeindex()):
                            if flat_index is None or len(flat_index) != len(
                                array.index
                            ):
                                flat_index = pandas.MultiIndex.from_arrays(
                                    [array.index]
                                )
                            df.append(
                                pandas.Series(array.values, index=flat_index).reindex(
                                    index
                                )
                            )
                            gn.append(name)
                        elif name in group:
                            df.append(array)
                            gn.append(name)
                out = []
                for index, df, gn in zip(indexes, dfs, group_names):
                    out.append(
                        pandas.DataFrame(
                            data=dict(zip(gn, df)), index=index, columns=gn
                        )
                    )
                if len(out) == 1:
                    return out[0]
                else:
                    return tuple(out)
            else:
                out = None
                for index, group in zip(indexes, groups):
                    only = dict((name, arrays[name]) for name in group)
                    df = pandas.DataFrame(data=only, index=index, columns=group)
                    if out is None:
                        out = df
                    else:
                        out = pandas.merge(
                            out, df, how=how, left_index=True, right_index=True
                        )
                flat_names = [
                    name
                    for name in names
                    if isinstance(arrays[name].index, _pandas_rangeindex())
                ]
                if len(flat_names) > 0:
                    flat_index = pandas.MultiIndex.from_arrays(
                        [arrays[flat_names[0]].index]
                    )
                    only = dict(
                        (name, pandas.Series(arrays[name].values, index=flat_index))
                        for name in flat_names
                    )
                    df = pandas.DataFrame(
                        data=only, index=flat_index, columns=flat_names
                    )
                    out = pandas.merge(
                        df.reindex(out.index),
                        out,
                        how=how,
                        left_index=True,
                        right_index=True,
                    )
                return out

        else:
            raise TypeError(
                "for library {0}, how must be tuple, list, dict, str (for "
                "pandas.merge's 'how' parameter, or None (for one or more"
                "DataFrames without merging)".format(self.name)
            )

    def global_index(self, arrays, global_start):
        if type(arrays.index).__name__ == "MultiIndex":
            if hasattr(arrays.index.levels[0], "arrays"):
                index = arrays.index.levels[0].arrays  # pandas>=0.24.0
            else:
                index = arrays.index.levels[0].values  # pandas<0.24.0
            numpy.add(index, global_start, out=index)

        elif type(arrays.index).__name__ == "RangeIndex":
            if hasattr(arrays.index, "start") and hasattr(arrays.index, "stop"):
                index_start = arrays.index.start  # pandas>=0.25.0
                index_stop = arrays.index.stop
            else:
                index_start = arrays.index._start  # pandas<0.25.0
                index_stop = arrays.index._stop
            arrays.index = type(arrays.index)(
                index_start + global_start, index_stop + global_start
            )

        else:
            if hasattr(arrays.index, "arrays"):
                index = arrays.index.arrays  # pandas>=0.24.0
            else:
                index = arrays.index.values  # pandas<0.24.0
            numpy.add(index, global_start, out=index)

        return arrays

    def concatenate(self, all_arrays):
        pandas = self.imported

        if len(all_arrays) == 0:
            return all_arrays

        if isinstance(all_arrays[0], (tuple, list)):
            keys = uproot4._util.range(len(all_arrays[0]))
        elif isinstance(all_arrays[0], dict):
            keys = list(all_arrays[0])
        else:
            return pandas.concat(all_arrays)

        to_concatenate = dict((k, []) for k in keys)
        for arrays in all_arrays:
            for k in keys:
                to_concatenate[k].append(arrays[k])

        concatenated = dict((k, pandas.concat(to_concatenate[k])) for k in keys)

        if isinstance(all_arrays[0], tuple):
            return tuple(concatenated[k] for k in keys)
        elif isinstance(all_arrays[0], list):
            return [concatenated[k] for k in keys]
        elif isinstance(all_arrays[0], dict):
            return concatenated


class CuPy(Library):
    u"""
    A :doc:`uproot4.interpetation.library.Library` that presents ``TBranch``
    data as CuPy arrays on a GPU. The standard name for this library is ``"cp"``.

    The single-``TBranch`` form for this library is a ``cupy.ndarray``.
    Non-numerical data are not allowed. Note that Awkward Arrays can be moved
    to GPUs with full data structures.

    The "group" behavior for this library is:

    * ``how=dict`` or ``how=None``: a dict of str \u2192 array, mapping the
      names to arrays.
    * ``how=tuple``: a tuple of arrays, in the order requested. (Names are
      lost.)
    * ``how=list``: a list of arrays, in the order requested. (Names are lost.)

    Since CuPy arrays are not indexed, ``global_index`` has no effect.
    """

    name = "cp"

    @property
    def imported(self):
        return uproot4.extras.cupy()

    def empty(self, shape, dtype):
        cupy = self.imported
        return cupy.empty(shape, dtype)

    def finalize(self, array, branch, interpretation, entry_start, entry_stop):
        cupy = self.imported

        if isinstance(
            array,
            (
                uproot4.interpretation.jagged.JaggedArray,
                uproot4.interpretation.strings.StringArray,
                uproot4.interpretation.objects.ObjectArray,
            ),
        ):
            raise TypeError("jagged arrays and objects are not supported by CuPy")

        else:
            assert isinstance(array, cupy.ndarray)
            return array

    def concatenate(self, all_arrays):
        cupy = self.imported

        if len(all_arrays) == 0:
            return all_arrays

        if isinstance(all_arrays[0], (tuple, list)):
            keys = uproot4._util.range(len(all_arrays[0]))
        elif isinstance(all_arrays[0], dict):
            keys = list(all_arrays[0])
        else:
            raise AssertionError(repr(all_arrays[0]))

        to_concatenate = dict((k, []) for k in keys)
        for arrays in all_arrays:
            for k in keys:
                to_concatenate[k].append(arrays[k])

        concatenated = dict((k, cupy.concatenate(to_concatenate[k])) for k in keys)

        if isinstance(all_arrays[0], tuple):
            return tuple(concatenated[k] for k in keys)
        elif isinstance(all_arrays[0], list):
            return [concatenated[k] for k in keys]
        elif isinstance(all_arrays[0], dict):
            return concatenated


_libraries = {
    NumPy.name: NumPy(),
    Awkward.name: Awkward(),
    Pandas.name: Pandas(),
    CuPy.name: CuPy(),
}

_libraries["numpy"] = _libraries[NumPy.name]
_libraries["Numpy"] = _libraries[NumPy.name]
_libraries["NumPy"] = _libraries[NumPy.name]
_libraries["NUMPY"] = _libraries[NumPy.name]

_libraries["awkward1"] = _libraries[Awkward.name]
_libraries["Awkward1"] = _libraries[Awkward.name]
_libraries["AWKWARD1"] = _libraries[Awkward.name]
_libraries["awkward"] = _libraries[Awkward.name]
_libraries["Awkward"] = _libraries[Awkward.name]
_libraries["AWKWARD"] = _libraries[Awkward.name]

_libraries["pandas"] = _libraries[Pandas.name]
_libraries["Pandas"] = _libraries[Pandas.name]
_libraries["PANDAS"] = _libraries[Pandas.name]

_libraries["cupy"] = _libraries[CuPy.name]
_libraries["Cupy"] = _libraries[CuPy.name]
_libraries["CuPy"] = _libraries[CuPy.name]
_libraries["CUPY"] = _libraries[CuPy.name]


def _regularize_library(library):
    if isinstance(library, Library):
        if library.name in _libraries:
            return _libraries[library.name]
        else:
            raise ValueError(
                "library {0} ({1}) cannot be used in this function".format(
                    type(library).__name__, repr(library.name)
                )
            )

    elif isinstance(library, type) and issubclass(library, Library):
        if library().name in _libraries:
            return _libraries[library().name]
        else:
            raise ValueError(
                "library {0} ({1}) cannot be used in this function".format(
                    library.__name__, repr(library().name)
                )
            )

    else:
        try:
            return _libraries[library]
        except KeyError:
            raise ValueError(
                """library {0} not recognized (for this function); """
                """try "np" (NumPy), "ak" (Awkward1), "pd" (Pandas), or "cp" (CuPy) """
                """instead""".format(repr(library))
            )


_libraries_lazy = {Awkward.name: _libraries[Awkward.name]}

_libraries_lazy["awkward1"] = _libraries_lazy[Awkward.name]
_libraries_lazy["Awkward1"] = _libraries_lazy[Awkward.name]
_libraries_lazy["AWKWARD1"] = _libraries_lazy[Awkward.name]
_libraries_lazy["awkward"] = _libraries_lazy[Awkward.name]
_libraries_lazy["Awkward"] = _libraries_lazy[Awkward.name]
_libraries_lazy["AWKWARD"] = _libraries_lazy[Awkward.name]


def _regularize_library_lazy(library):
    if isinstance(library, Library):
        if library.name in _libraries_lazy:
            return _libraries_lazy[library.name]
        else:
            raise ValueError(
                "library {0} ({1}) cannot be used in this function".format(
                    type(library).__name__, repr(library.name)
                )
            )

    elif isinstance(library, type) and issubclass(library, Library):
        if library().name in _libraries_lazy:
            return _libraries_lazy[library().name]
        else:
            raise ValueError(
                "library {0} ({1}) cannot be used in this function".format(
                    library.__name__, repr(library().name)
                )
            )

    else:
        try:
            return _libraries_lazy[library]
        except KeyError:
            raise ValueError(
                """library {0} not recognized (for this function); """
                """try "ak" (Awkward1) """
                """instead""".format(repr(library))
            )
