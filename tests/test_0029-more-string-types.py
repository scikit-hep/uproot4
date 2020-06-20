# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import sys
import json

import numpy
import pytest
import skhep_testdata

import uproot4
from uproot4.stl_containers import parse_typename
from uproot4.stl_containers import AsString
from uproot4.stl_containers import AsVector
from uproot4.stl_containers import AsSet
from uproot4.stl_containers import AsMap


def test_parse_typename():
    assert parse_typename("TTree") is uproot4.classes["TTree"]
    assert parse_typename("string") == AsString()
    assert parse_typename("std::string") == AsString()
    assert parse_typename("std :: string") == AsString()
    assert parse_typename("char*") == AsString(is_stl=False)
    assert parse_typename("char *") == AsString(is_stl=False)
    assert parse_typename("TString") == AsString(is_stl=False)
    assert parse_typename("vector<TTree>") == AsVector(uproot4.classes["TTree"])
    assert parse_typename("vector<int>") == AsVector(">i4")
    assert parse_typename("vector<bool>") == AsVector("?")
    assert parse_typename("vector<string>") == AsVector(AsString())
    assert parse_typename("vector  <   string   >") == AsVector(AsString())
    assert parse_typename("std::vector<std::string>") == AsVector(AsString())
    assert parse_typename("vector<vector<int>>") == AsVector(AsVector(">i4"))
    assert parse_typename("vector<vector<string>>") == AsVector(AsVector(AsString()))
    assert parse_typename("vector<vector<char*>>") == AsVector(
        AsVector(AsString(is_stl=False))
    )
    assert parse_typename("set<unsigned short>") == AsSet(">u2")
    assert parse_typename("std::set<unsigned short>") == AsSet(">u2")
    assert parse_typename("set<string>") == AsSet(AsString())
    assert parse_typename("set<vector<string>>") == AsSet(AsVector(AsString()))
    assert parse_typename("set<vector<string> >") == AsSet(AsVector(AsString()))
    assert parse_typename("map<int, double>") == AsMap(">i4", ">f8")
    assert parse_typename("map<string, double>") == AsMap(AsString(), ">f8")
    assert parse_typename("map<int, string>") == AsMap(">i4", AsString())
    assert parse_typename("map<string, string>") == AsMap(AsString(), AsString())
    assert parse_typename("map<string,string>") == AsMap(AsString(), AsString())
    assert parse_typename("map<   string,string   >") == AsMap(AsString(), AsString())
    assert parse_typename("map<string,vector<int>>") == AsMap(
        AsString(), AsVector(">i4")
    )
    assert parse_typename("map<vector<int>, string>") == AsMap(
        AsVector(">i4"), AsString()
    )
    assert parse_typename("map<vector<int>, set<float>>") == AsMap(
        AsVector(">i4"), AsSet(">f4")
    )
    assert parse_typename("map<vector<int>, set<set<float>>>") == AsMap(
        AsVector(">i4"), AsSet(AsSet(">f4"))
    )

    with pytest.raises(ValueError):
        parse_typename("string  <")

    with pytest.raises(ValueError):
        parse_typename("vector  <")

    with pytest.raises(ValueError):
        parse_typename("map<string<int>>")

    with pytest.raises(ValueError):
        parse_typename("map<string, int>>")


def test_strings1():
    with uproot4.open(
        skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    )["tree"] as tree:
        result = tree["Beg"].array(library="np")
        assert result.tolist() == ["beg-{0:03d}".format(i) for i in range(100)]

        result = tree["End"].array(library="np")
        assert result.tolist() == ["end-{0:03d}".format(i) for i in range(100)]


def test_map_string_string_in_object():
    with uproot4.open(skhep_testdata.data_path("uproot-issue431.root")) as f:
        head = f["Head"]
        assert head.member("map<string,string>") == {
            "DAQ": "394",
            "PDF": "4      58",
            "XSecFile": "",
            "can": "0 1027 888.4",
            "can_user": "0.00 1027.00  888.40",
            "coord_origin": "0 0 0",
            "cut_in": "0 0 0 0",
            "cut_nu": "100 1e+08 -1 1",
            "cut_primary": "0 0 0 0",
            "cut_seamuon": "0 0 0 0",
            "decay": "doesnt happen",
            "detector": "NOT",
            "drawing": "Volume",
            "end_event": "",
            "genhencut": "2000 0",
            "genvol": "0 1027 888.4 2.649e+09 100000",
            "kcut": "2",
            "livetime": "0 0",
            "model": "1       2       0       1      12",
            "muon_desc_file": "",
            "ngen": "0.1000E+06",
            "norma": "0 0",
            "nuflux": "0       3       0 0.500E+00 0.000E+00 0.100E+01 0.300E+01",
            "physics": "GENHEN 7.2-220514 181116 1138",
            "seed": "GENHEN 3  305765867         0         0",
            "simul": "JSirene 11012 11/17/18 07",
            "sourcemode": "diffuse",
            "spectrum": "-1.4",
            "start_run": "1",
            "target": "isoscalar",
            "usedetfile": "false",
            "xlat_user": "0.63297",
            "xparam": "OFF",
            "zed_user": "0.00 3450.00",
        }


@pytest.mark.skip(
    reason="FIXME: test works, but the file is not in scikit-hep-testdata yet"
)
def test_map_long_int_in_object():
    with uproot4.open(
        "/home/pivarski/irishep/scikit-hep-testdata/src/skhep_testdata/data/uproot-issue283.root"
    ) as f:
        print(f["config/detector"])

    # raise Exception


# has STL vectors at top-level:
#
# python -c 'import uproot; t = uproot.open("/home/pivarski/irishep/scikit-hep-testdata/src/skhep_testdata/data/uproot-issue38a.root")["ntupler/tree"]; print("\n".join(str((x._fName, getattr(x, "_fStreamerType", None), getattr(x, "_fClassName", None), getattr(x, "_fType", None), x.interpretation)) for x in t.allvalues()))'

# has STL map<int,struct> as described here:
#
# https://github.com/scikit-hep/uproot/issues/468#issuecomment-646325842
#
# python -c 'import uproot; t = uproot.open("/home/pivarski/irishep/scikit-hep-testdata/src/skhep_testdata/data/uproot-issue468.root")["Geant4Data/Geant4Data./Geant4Data.particles"]; print(t.array(uproot.asdebug)[0][:1000])'

# def test_strings1():
#     with uproot4.open(
#         skhep_testdata.data_path("uproot-issue31.root")
#     )["T/name"] as branch:
#         result = branch.array(library="np")
#         assert result.tolist() == ["one", "two", "three", "four", "five"]


@pytest.mark.skip(reason="FIXME: implement strings specified by a TStreamer")
def test_strings2():
    with uproot4.open(
        skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    )["tree/Str"] as branch:
        result = branch.array(library="np")
        assert result.tolist() == ["evt-{0:03d}".format(i) for i in range(100)]


@pytest.mark.skip(reason="FIXME: implement std::string")
def test_strings3():
    with uproot4.open(
        skhep_testdata.data_path("uproot-small-evnt-tree-fullsplit.root")
    )["tree/StdStr"] as branch:
        result = branch.array(library="np")
        assert result.tolist() == ["std-{0:03d}".format(i) for i in range(100)]
