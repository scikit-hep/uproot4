# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
This module defines the version number string and tuple for this version of Uproot.

The project's ``setup.py`` inspects this file for a version number.

The version number of Uproot's ``master`` branch in GitHub is usually one ahead
of the latest release on PyPI.
"""

from __future__ import absolute_import

import re

__version__ = "4.0.0rc2"
version = __version__
version_info = tuple(re.split(r"[-\.]", __version__))

del re
