# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

import importlib
import inspect
import pkgutil
import os.path
import sys


order = [
    "uproot",
    "uproot.reading",
    "uproot.behaviors",
    "uproot.model",
    "uproot.streamers",
    "uproot.cache",
    "uproot.compression",
    "uproot.deserialization",
    "uproot.source",
    "uproot.interpretation",
    "uproot.containers",
    "uproot.language",
    "uproot.models",
    "uproot.const",
    "uproot.exceptions",
    "uproot.extras",
]

toctree = open("uproot.toctree", "w")
toctree.write(
    """.. toctree::
    :caption: Reference
    :hidden:

"""
)


def ensure(objname, filename, content):
    overwrite = not os.path.exists(filename)
    if not overwrite:
        overwrite = open(filename, "r").read() != content
    if overwrite:
        open(filename, "w").write(content)
        sys.stderr.write(objname + " (OVERWRITTEN)\n")
    else:
        sys.stderr.write(objname + "\n")


def handle_module(modulename, module):
    content = """{0}
{1}

.. automodule:: {0}
""".format(
        modulename, "=" * len(modulename)
    )
    ensure(modulename, modulename + ".rst", content)
    toctree.write("    " + modulename + "\n")

    if modulename != "uproot" and all(
        not x.startswith("_") for x in modulename.split(".")
    ):

        def good(obj):
            if inspect.isfunction(obj) or inspect.isclass(obj):
                if obj.__module__ == modulename:
                    return True
            return False

        def line_order(pair):
            name, obj = pair
            return inspect.getsourcelines(obj)[1]

        for pair in sorted(inspect.getmembers(module, good), key=line_order):
            name, obj = pair
            if not name.startswith("_"):
                if inspect.isclass(obj):
                    handle_class(modulename + "." + name, obj)
                elif inspect.isfunction(obj):
                    handle_function(modulename + "." + name, obj)


def handle_class(classname, cls):
    def line_order(obj):
        if isinstance(obj, property):
            obj = obj.fget
        return inspect.getsourcelines(obj)[1]

    methods = {}
    mro = list(cls.__mro__)

    import uproot
    if hasattr(uproot, cls.__name__):
        title = "uproot." + cls.__name__
    else:
        title = classname

    for index, basecls in enumerate(mro):
        if basecls.__module__.startswith("uproot."):

            def good(obj):
                if inspect.ismethod(obj) or inspect.isfunction(obj):
                    module, name = obj.__module__, obj.__name__
                elif isinstance(obj, property):
                    module, name = obj.fget.__module__, obj.fget.__name__
                else:
                    module, name = "", ""
                if module.startswith("uproot."):
                    if index + 1 >= len(mro) or obj is not getattr(
                        mro[index + 1], name, None
                    ):
                        return True
                return False

            for name, obj in inspect.getmembers(basecls, good):
                if name in basecls.__dict__ and not name.startswith("_"):
                    fill = []
                    fill.append(name)
                    fill.append("-" * len(fill[-1]))
                    fill.append("")
                    if basecls is not cls:
                        fill.append(
                            "Inherited from :doc:`{0}`.".format(
                                basecls.__module__ + "." + basecls.__name__
                            )
                        )
                        fill.append("")
                    if isinstance(obj, property):
                        fill.append(".. autoattribute:: " + classname + "." + name)
                    else:
                        fill.append(".. automethod:: " + classname + "." + name)
                    fill.append("")
                    methods[name] = (index, line_order(obj), "\n".join(fill))

    def prettymro(c):
        fullname = c.__module__ + "." + c.__name__
        if c.__module__.startswith("uproot."):
            return "#. :doc:`" + fullname + "`"
        else:
            return "#. ``" + fullname + "``"

    content = """{0}
{1}

{2}

.. autoclass:: {3}

{4}
""".format(
        title,
        "=" * len(title),
        "\n".join(prettymro(c) for c in cls.__mro__[1:] if c is not object),
        classname,
        "\n".join([text for index, line, text in sorted(methods.values())]),
    )

    ensure(classname, classname + ".rst", content)
    toctree.write("    " + classname + "\n")


def handle_function(functionname, cls):
    import uproot
    if hasattr(uproot, cls.__name__):
        title = "uproot." + cls.__name__
    else:
        title = functionname

    content = """{0}
{1}

.. autofunction:: {2}
""".format(title, "=" * len(title), functionname)
    ensure(functionname, functionname + ".rst", content)
    toctree.write("    " + functionname + "\n")


for modulename in order:
    module = importlib.import_module(modulename)

    if modulename != "uproot":
        toctree = open(modulename + ".toctree", "w")
        toctree.write(
            """.. toctree::
    :hidden:

"""
        )

    handle_module(modulename, module)
    if module.__file__.endswith("__init__.py") and modulename != "uproot":
        for submodulename in sorted(
            [
                modulename + "." + name
                for loader, name, is_pkg in pkgutil.walk_packages(module.__path__)
            ]
        ):
            submodule = importlib.import_module(submodulename)
            handle_module(submodulename, submodule)
