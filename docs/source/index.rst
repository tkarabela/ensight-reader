.. ensight-reader documentation master file, created by
   sphinx-quickstart on Sun Mar 27 20:08:09 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ensight-reader
==============

This library provides a pure Python reader for the EnSight Gold data format,
a common format for results of computational fluid dynamics (CFD) simulations.

It's designed for efficient, selective, memory-mapped access to data from EnSight Gold case --
something that would be useful when importing the data into other systems.

If you're looking for a more "batteries included" solution, look at
`vtkEnSightGoldBinaryReader <https://vtk.org/doc/nightly/html/classvtkEnSightGoldBinaryReader.html>`_
from the VTK library. For more information, see :ref:`Comparison with VTK library`.

::

   import ensightreader

   case = ensightreader.read_case("example.case")
   geofile = case.get_geometry_model()

   part_names = geofile.get_part_names()                # ["internalMesh", ...]
   part = geofile.get_part_by_name(part_names[0])
   N = part.number_of_nodes

   with open(geofile.file_path, "rb") as fp_geo:
      node_coordinates = part.read_coordinates(fp_geo)  # np.ndarray((N, 3), dtype=np.float32)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   design-howto
   api-reference

License
-------

.. code-block:: text

    Copyright (c) 2022 Tomas Karabela

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
