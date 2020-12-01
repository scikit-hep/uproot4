# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Initially empty submodule into which new classes are dynamically added.

The purpose of this namespace is to allow :py:class:`~uproot.model.VersionedModel`
classes that were automatically generated from ROOT ``TStreamerInfo`` to be
pickled, with the help of :py:class:`~uproot.model.DynamicModel`.

In `Python 3.7 and later <https://www.python.org/dev/peps/pep-0562>`__, attempts
to extract items from this namespace generate new :py:class:`~uproot.model.DynamicModel`
classes, which are used as a container in which data from pickled
:py:class:`~uproot.model.VersionedModel` instances are filled.
"""


def __getattr__(name):
    import uproot

    g = globals()
    if name not in g:
        g[name] = uproot._util.new_class(name, (uproot.model.DynamicModel,), {})

    return g[name]
