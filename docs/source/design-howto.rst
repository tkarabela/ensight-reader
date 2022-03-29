Library design & HOWTO
======================

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
    with the library.

Comparison with VTK library
---------------------------

xxx
