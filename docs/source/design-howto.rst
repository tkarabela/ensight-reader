Library design & HOWTO
======================

Memory and file management
--------------------------

The library is designed to give you control over memory and file management:
you have to explicitly open the data files and pass these opened files to the
library to get back NumPy arrays with the data. This adds some bookkeeping
overhead for you, but it enables you to choose between:

  1. traditional I/O and getting arrays which own a copy of the data, or
  2. memory-mapped I/O and getting arrays that are views into
     the binary EnSight files on disk.

Depending on your use-case, using **memory-mapped I/O** can be **very
efficient**. It also allows for **limited editing of EnSight files**
(eg. transforming coordinates or variable values in-place). Regular
I/O can be used for **appending new parts and variable data**.

The following code example demonstrates this feature:

::

    import ensightreader

    case = ensightreader.read_case("data/sphere/sphere.case")
    geofile = case.get_geometry_model()
    part_ids = geofile.get_part_ids()
    part = geofile.get_part_by_id(part_ids[0])
    variable = case.get_variable("RTData")

    # (1.1) read-only traditional I/O
    with geofile.open() as fp_geo:
        nodes = part.read_nodes(fp_geo)  # owned buffer

    # (1.2) read-write traditional I/O
    with variable.open_writable() as fp_var:
        variable.ensure_data_for_part(fp_var, part.part_id, 0.0)  # append array with default value if needed

    # (2.1) read-only memory-mapped I/O
    with geofile.mmap() as mm_geo:
        nodes = part.read_nodes(mm_geo)  # buffer backed by read-only mmap

    # (2.2) write-through memory-mapped I/O
    with geofile.mmap_writable() as mm_geo:
        nodes = part.read_nodes(mm_geo)  # buffer backed by write-through mmap
        nodes[:, 0] = 0.0                # set X coordinate to zero for part nodes

.. seealso::
    The ``ensight_transform.py`` and ``ensight2vtk.py`` scripts for examples
    how to use memory-mapped I/O with the library. It's done by simply passing
    ``mmap`` objects wherever the library expects opened files.
    Make sure to keep the ``mmap`` objects opened for as long as you need
    the arrays that are read from them!

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

    >>> with geofile.open() as fp_geo:
    ...     nodes = part.read_nodes(fp_geo)
    >>> nodes
    array([[ 0.0000000e+00,  0.0000000e+00,  5.0000000e+00],
           ...
           [ 1.5340106e+00, -1.5340106e+00, -4.5048442e+00]], dtype=float32)

    >>> with geofile.open() as fp_geo:
    ...     block = part.element_blocks[0]
    ...     connectivity = block.read_connectivity(fp_geo)
    >>> connectivity
    array([[ 3,  9,  1],
           ...
           [49,  8,  7]])

    >>> variable = case.get_variable("RTData")
    >>> with variable.open() as fp_var:
    ...     variable_data = variable.read_node_data(fp_var, part.part_id)
    >>> variable_data
    array([220.84135, 220.84135, 223.80856, 233.50835, 217.5993 , 217.5993 ,
           ...
           213.36838, 232.34589], dtype=float32)

Writing data using ``ensight-reader``
-------------------------------------

Despite its name, ``ensight-reader`` can be used to modify existing EnSight Gold cases
or even create cases from scratch. Due to the unopinionated, low-level nature of the library,
this can be tricky -- especially if you're not already familiar with inner workings
of the EnSight Gold format.

Here are some examples to get you started:

Modifying variable values
~~~~~~~~~~~~~~~~~~~~~~~~~

::

    >>> import ensightreader

    >>> case = ensightreader.read_case("data/sphere/sphere.case")
    >>> geofile = case.get_geometry_model()
    >>> part_ids = geofile.get_part_ids()

    >>> if case.variables["RTData"].timeset is None:
    ...     timesteps = [0]
    ... else:
    ...     timesteps = list(range(len(case.get_time_values()))

    >>> for timestep in timesteps:
    ...     variable = case.get_variable("RTData", timestep)
    ...     with variable.mmap_writable() as mm_var:
    ...         for part_id in part_ids:
    ...             arr = variable.read_node_data(fp_var, part_id)
    ...             arr[:] *= 2

Defining new per-case constant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    >>> import ensightreader

    >>> case = ensightreader.read_case("data/sphere/sphere.case")
    >>> constant = ensightreader.EnsightConstantVariable(timeset=None, variable_name="my_variable", values=[42.0])
    >>> case.constant_variables[constant.variable_name] = constant
    >>> case.to_file("data/sphere/sphere-with-constant.case")


Defining new variable
~~~~~~~~~~~~~~~~~~~~~

::

    >>> from ensightreader import read_case, VariableLocation, VariableType

    >>> case = read_case("data/sphere/sphere.case")
    >>> my_variable = case.define_variable(VariableLocation.PER_NODE, VariableType.VECTOR, "my_variable", "my_variable.bin")

    >>> with my_variable.open_writeable() as fp:
    ...     my_variable.ensure_data_for_all_parts(fp, default_value=0.0)

    >>> with my_variable.mmap_writeable() as mm:
    ...     part_id = 1
    ...     arr = my_variable.read_node_data(mm, part_id)
    ...     arr[0] = 123  # now we can modify the data array in-place to set variable values


Changing node coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~

::

    >>> import ensightreader

    >>> case = ensightreader.read_case("data/sphere/sphere.case")
    >>> geofile = case.get_geometry_model()
    >>> part_ids = geofile.get_part_ids()

    >>> with geofile.mmap_writable() as mm_geo:
    ...     for part_id in geofile.get_part_ids():
    ...         part = geofile.get_part_by_id(part_id)
    ...         arr = part.read_nodes(mm_geo)
    ...         arr[:, 0] += 1.0  # increment X coordiante


Creating geometry file from scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please see ``tests/test_write_geometry.py``, essentially you will need to do:

::

    >>> from ensightreader import EnsightGeometryFile, GeometryPart, GeometryPart

    >>> with open(output_geofile_path, "wb") as fp:
    ...     EnsightGeometryFile.write_header(fp)
    ...     GeometryPart.write_part_header(fp, part_id=1, part_name="TestElementTypes", node_coordinates=node_coordinates)
    ...     UnstructuredElementBlock.write_element_block(fp, element_type=et, connectivity=connectivity)


Copying data from other case
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are convenience methods to append part geometry and variable data from a different case:

::

    >>> import ensightreader

    >>> source_case = ensightreader.read_case("source.case")
    >>> dest_case = ensightreader.read_case("dest.case")

    >>> source_geo = source_case.get_geometry_model()
    >>> dest_geo = dest_case.get_geometry_model()

    >>> source_part = source_geo.get_part_by_name("my_part")
    >>> dest_case.append_part_geometry(source_case, [source_part])
    >>> dest_case.copy_part_variables(source_case, [source_part], ["velocity", "pressure"])


Code examples
-------------

The library comes with scripts that use it to convert EnSight Gold into
other data formats. These are mostly useful for reference as they
output data in text format, making them unsuitable for production use
with large models.

.. tip::
    See the ``tests/`` directory in the repository for more code examples.

.. automodule:: ensight2obj
    :members:

.. automodule:: ensight2vtk
    :members:

.. automodule:: ensight_transform
    :members:
