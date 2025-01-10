.. ensight-reader documentation master file, created by
   sphinx-quickstart on Sun Mar 27 20:08:09 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ensight-reader
==============

This library provides a pure Python reader (with some writing capability) for the EnSight Gold data format,
a common format for results of computational fluid dynamics (CFD) simulations.
It also comes with a few CLI tools, notably ``ensight_transform`` which
allows you to perform in-place scaling/translation/etc. of the geometry in your case.

It's designed for efficient, selective, memory-mapped access to data from EnSight Gold case --
something that would be useful when importing the data into other systems. Primary focus
is on reading existing EnSight Gold cases, but it can also be used to modify existing data
files or even create files from scratch.

To get the most out of this library, familiarity with the EnSight Gold format structure
is beneficial. If you're looking for a more "batteries included" solution, look at
`vtkEnSightGoldBinaryReader <https://vtk.org/doc/nightly/html/classvtkEnSightGoldBinaryReader.html>`_
from the VTK library. For more information, see :ref:`Comparison with VTK library`.

::

   import ensightreader
   import numpy as np

   case = ensightreader.read_case("example.case")
   geofile = case.get_geometry_model()

   part_names = geofile.get_part_names()           # ["internalMesh", ...]
   part = geofile.get_part_by_name(part_names[0])
   N = part.number_of_nodes

   with geofile.open() as fp_geo:
       node_coordinates = part.read_nodes(fp_geo)  # np.ndarray((N, 3), dtype=np.float32)

   variable = case.get_variable("UMean")

   with variable.mmap_writable() as mm_var:
       data = variable.read_node_data(mm_var, part.part_id)
       data[:] = np.sqrt(data)                     # transform variable data in-place

Installation
------------

::

   pip install ensight-reader

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   design-howto
   api-reference
   ensight-transform-cli
   changelog

License
-------

.. code-block:: text

    Copyright (c) 2022-2025 Tomas Karabela

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
