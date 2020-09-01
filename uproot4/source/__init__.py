# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Defines the "physical layer" of file-reading, which interacts with local
filesystems or remote protocols like HTTP(S) and XRootD. The "physical layer"
is distinguished from "interpretation" in that the meaning of the bytes that
have been read are not relevant in this layer. The "interpretation layer"
interacts with the "physical layer" by requesting a
:doc:`uproot4.source.chunk.Chunk` from a :doc:`uproot4.source.chunk.Source` and
inspecting it with a :doc:`uproot4.source.cursor.Cursor`.

Any threads used for parallel reading are launched and shut down with the file
handle or handles themselves. Context management (Python's ``with`` statement)
controls both I/O resources and threads.

This module includes a :doc:`uproot4.source.futures` implementation that
connects file handles with threads.
"""

from __future__ import absolute_import
