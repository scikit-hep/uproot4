# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines models, which are classes of objects read from ROOT files.

Models must be subclasses of :doc:`uproot4.model.Model`, and models for a
specific version of a ROOT class must be subclasses of
:doc:`uproot4.model.VersionedModel`.

If a C++ class has no associated model, a new model class will be generated
from the ROOT file's ``TStreamerInfo``.

To add a versionless model for a ROOT class:

1. Translate the ROOT class name from C++ to Python with
   :doc:`uproot4.model.classname_encode`. For example,
   ``"ROOT::RThing"`` becomes ``"Model_ROOT_3a3a_RThing"``.
2. Define a class with that name.
3. Explicitly add it to :doc:`uproot4.classes`.

A versionless model is instantiated for any ROOT object with a given class
name, regardless of its version. The deserialization procedure may need to
include version-dependent code.

To add a versioned model for a ROOT class:

1. Translate the ROOT class name from C++ to Python with
   :doc:`uproot4.model.classname_encode` with a specific ``version``.
   For example version ``2`` of ``"ROOT::RThing"`` becomes
   ``"Model_ROOT_3a3a_RThing_v2"``.
2. Define a class with that name.
3. Explicitly add it to a :doc:`uproot4.model.DispatchByVersion` for that
   class. You might also need to add a :doc:`uproot4.model.DispatchByVersion`
   to the :doc:`uproot4.classes`.

A versioned model is only instantiated for a ROOT object with a given class
name and version. Uproot has common versions of :doc:`uproot4.models.TBranch`
and :doc:`uproot4.models.TTree` predefined so that it can usually avoid reading
a ROOT file's ``TStreamerInfo``.

High-level methods and properties should not be defined on the model class;
add them as behavior classes.

See also :doc:`uproot4.behaviors`.
"""

from __future__ import absolute_import
