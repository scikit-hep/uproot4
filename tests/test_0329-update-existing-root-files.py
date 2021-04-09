# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import os

import numpy as np
import pytest
import skhep_testdata

import uproot
import uproot.writing

ROOT = pytest.importorskip("ROOT")


def test(tmp_path):
    filename = os.path.join(tmp_path, "testy.root")

    f = ROOT.TFile(filename, "recreate")
    x = ROOT.TObjString("hello")
    x.Write()
    f.Close()

    assert uproot.open(filename).classnames() == {"hello": "TObjString"}

    with uproot.writing.update(filename) as root_directory:
        root_directory.mkdir("subdir")

    assert uproot.open(filename).classnames() == {
        "hello": "TObjString",
        "subdir": "TDirectory",
    }

    g = ROOT.TFile(filename, "update")
    g.cd("subdir")
    y = ROOT.TObjString("there")
    y.Write()
    g.Close()

    assert uproot.open(filename).classnames() == {
        "hello": "TObjString",
        "subdir": "TDirectory",
        "subdir/there": "TObjString",
    }
