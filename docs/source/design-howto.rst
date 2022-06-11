Library design & HOWTO
======================

Code examples
-------------

The library comes with scripts that use it to convert EnSight Gold into
other data formats. These are mostly useful for reference as they
output data in text format, making them unsuitable for production use
with large models.

.. automodule:: ensight2obj
    :members:

.. automodule:: ensight2vtk
    :members:

Memory and file management
--------------------------

The library is designed to give you control over memory and file management:
you have to open the data files yourself and pass these opened files to the
library to get back NumPy arrays with the data. This adds some bookkeeping
overhead for you, but it enables you to choose between:

  1. traditional I/O and getting arrays which own a copy of the data, or
  2. memory-mapped I/O and getting arrays that are views into
     the binary EnSight files on disk.

Depending on your use-case, using **memory-mapped I/O** can be very
efficient.

.. seealso::
    The ``ensight2vtk.py`` script for example how to use memory-mapped I/O
    with the library. Simply pass ``mmap`` objects wherever the library
    expects opened file objects. Make sure to keep the ``mmap`` objects
    opened for as long as you need the arrays that are read from them.

Comparison with VTK library
---------------------------

`The Visualisation Toolkit (VTK) <https://vtk.org/>`_ is a great library
for handling scientific and engineering data. Among its *many* features
it comes with a reader (and a somewhat less featureful writer) for the EnSight Gold format.

Both libraries are quite different in focus. Put shortly, **ensight-reader** is a library
you would use if you wanted to implement an EnSight Gold reader in a high-level library
such as **VTK**.

**Common points**

- both can be used from Python (VTK is a C++ library with official Python bindings)
- both give you access to node coordinates, cell connectivity, variable data as Numpy arrays

**ensight-reader specifics**

- supports partial reading (you can read just one variable for one part, etc.)
- supports memory-mapped access (to minimize copying)
- doesn't do any conversion/interpretation of the data (useful if you're doing something low-level yourself)

**VTK specifics**

- supports some features that ensight-reader currently does not (eg. structured parts)
- has more high-level API, can do much more with the data (eg. interpolate node/element data, save to different formats, show in 3D viewport, ...)
- does not support partial reading of parts
- does not support memory-mapped access
- converts the data into VTK datastructures (``vtkUnstructuredGrid``, etc.)

Code example
~~~~~~~~~~~~

**VTK library**

::

    >>> import vtk
    >>> from vtk.numpy_interface import dataset_adapter as dsa

    >>> reader = vtk.vtkEnSightGoldBinaryReader()
    >>> reader.SetCaseFileName("data/sphere/sphere.case")
    >>> reader.Update()
    >>> case = reader.GetOutput()

    >>> part = case.GetBlock(0)
    >>> part_ = dsa.WrapDataObject(part)
    >>> part_.Points
    VTKArray([[ 0.0000000e+00,  0.0000000e+00,  5.0000000e+00],
              [ 0.0000000e+00,  0.0000000e+00, -5.0000000e+00],
              ...
              [ 1.5340106e+00, -1.5340106e+00, -4.5048442e+00]], dtype=float32)

    >>> part_.PointData["RTData"]
    VTKArray([220.84135, 220.84135, 223.80856, 233.50835, 217.5993 ,
              ...
              213.36838, 210.3635 , 210.3635 , 213.36838, 232.34589],
             dtype=float32)

    >>> part_.Cells
    VTKArray([ 3,  2,  8,  0,  3,  8, 14,  0,  3, 14, 20,  0,  3, 20, 26,  0,
              ...
               3, 47, 48,  6,  3, 47,  6,  5,  3, 48, 49,  7,  3, 48,  7,  6],
             dtype=int64)

**ensight-reader library**

::

    >>> import ensightreader

    >>> case = ensightreader.read_case("data/sphere/sphere.case")
    >>> geofile = case.get_geometry_model()
    >>> part_ids = geofile.get_part_ids()
    >>> part = geofile.get_part_by_id(part_ids[0])

    >>> with open(geofile.file_path, "rb") as fp_geo:
    ...     nodes = part.read_nodes(fp_geo)
    >>> nodes
    array([[ 0.0000000e+00,  0.0000000e+00,  5.0000000e+00],
           ...
           [ 1.5340106e+00, -1.5340106e+00, -4.5048442e+00]], dtype=float32)

    >>> with open(geofile.file_path, "rb") as fp_geo:
    ...     block = part.element_blocks[0]
    ...     connectivity = block.read_connectivity(fp_geo)
    >>> connectivity
    array([[ 3,  9,  1],
           ...
           [49,  8,  7]])

    >>> variable = case.get_variable("RTData")
    >>> with open(variable.file_path, "rb") as fp_var:
    ...     variable_data = variable.read_node_data(fp_var, part.part_id)
    >>> variable_data
    array([220.84135, 220.84135, 223.80856, 233.50835, 217.5993 , 217.5993 ,
           ...
           213.36838, 232.34589], dtype=float32)
