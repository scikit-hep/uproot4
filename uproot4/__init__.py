# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

classes = {}
unknown_classes = {}

from uproot4.cache import LRUCache
from uproot4.cache import LRUArrayCache

from uproot4.source.memmap import MemmapSource
from uproot4.source.file import FileSource
from uproot4.source.http import HTTPSource
from uproot4.source.http import MultithreadedHTTPSource
from uproot4.source.xrootd import XRootDSource
from uproot4.source.xrootd import MultithreadedXRootDSource
from uproot4.source.cursor import Cursor
from uproot4.source.futures import TrivialExecutor
from uproot4.source.futures import ThreadPoolExecutor

decompression_executor = ThreadPoolExecutor()
interpretation_executor = TrivialExecutor()

from uproot4.reading import open
from uproot4.reading import ReadOnlyFile
from uproot4.reading import ReadOnlyDirectory

from uproot4.model import Model
from uproot4.model import classname_decode
from uproot4.model import classname_encode
from uproot4.model import has_class_named
from uproot4.model import class_named

import uproot4.interpretation
import uproot4.interpretation.library

default_library = "ak"

import uproot4.models.TObject
import uproot4.models.TString
import uproot4.models.TArray
import uproot4.models.TNamed
import uproot4.models.TList
import uproot4.models.THashList
import uproot4.models.TObjArray
import uproot4.models.TObjString
import uproot4.models.TAtt

import uproot4.models.TTree
import uproot4.models.TBranch
import uproot4.models.TLeaf
import uproot4.models.TBasket
from uproot4.behaviors.TTree import TTree
from uproot4.behaviors.TBranch import TBranch

import uproot4.models.RNTuple

# FIXME: add uproot4.models.TRef


import pkgutil
import uproot4.behaviors


def behavior_of(classname):
    name = classname_encode(classname)
    assert name.startswith("Model_")
    name = name[6:]

    if name not in globals():
        if name in behavior_of._module_names:
            exec(
                compile(
                    "import uproot4.behaviors.{0}".format(name), "<dynamic>", "exec"
                ),
                globals(),
            )
            module = eval("uproot4.behaviors.{0}".format(name))
            behavior_cls = getattr(module, name)
            if behavior_cls is not None:
                globals()[name] = behavior_cls

    return globals().get(name)


behavior_of._module_names = [
    module_name
    for loader, module_name, is_pkg in pkgutil.walk_packages(uproot4.behaviors.__path__)
]

del pkgutil


class KeyInFileError(KeyError):
    def __init__(self, key, file_path, cycle=None, because="", object_path=None):
        super(KeyInFileError, self).__init__(key)
        self.key = key
        self.file_path = file_path
        self.cycle = cycle
        self.because = because
        self.object_path = object_path

    def __str__(self):
        if self.because == "":
            because = ""
        else:
            because = " because " + self.because

        if self.object_path is None:
            object_path = ""
        else:
            object_path = " at {0}".format(self.object_path)

        if self.cycle == "any":
            return """not found: {0} (with any cycle number){1}
in file {2}{3}""".format(
                repr(self.key), because, self.file_path, object_path
            )
        elif self.cycle is None:
            return """not found: {0}{1}
in file {2}{3}""".format(
                repr(self.key), because, self.file_path, object_path
            )
        else:
            return """not found: {0} with cycle {1}{2}
in file {3}{4}""".format(
                repr(self.key), self.cycle, because, self.file_path, object_path
            )


from uproot4._util import no_filter
