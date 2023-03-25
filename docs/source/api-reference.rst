API Reference
=============

.. automodule:: ensightreader


Case file
---------

.. autofunction:: read_case

.. autoclass:: EnsightCaseFile
    :members:
.. autoclass:: EnsightGeometryFileSet
    :members:
.. autoclass:: EnsightVariableFileSet
    :members:
.. autoclass:: Timeset
    :members:


Geometry file
-------------

.. autoclass:: EnsightGeometryFile
    :members:
.. autoclass:: GeometryPart
    :members:
.. autoclass:: UnstructuredElementBlock
    :members:


Variable file
-------------

.. autoclass:: EnsightVariableFile
    :members:


Enum types
----------

.. autoenum:: ElementType
    :members:
.. autoenum:: VariableType
    :members:
.. autoenum:: VariableLocation
    :members:
.. autoenum:: IdHandling
    :members:
.. autoenum:: ChangingGeometry
    :members:


Exceptions and warnings
-----------------------

.. autoclass:: EnsightReaderError
    :members:

.. autoclass:: EnsightReaderWarning
    :members:
