Changelog
=========

0.11.2 (2024-06-31)
-------------------

- Added support for .case file with quoted filenames (`Issue #6 <https://github.com/tkarabela/ensight-reader/issues/6>`_).
- Added more context to exceptions when geometry file parsing fails.
- Added support for NumPy 2.0+

0.11.1 (2023-03-25)
-------------------

- Added a warning when .case file contains transient model or variables that do not have timeset ID defined.
  ``ensightreader.read_case()`` will issue ``EnsightReaderWarning`` with suggested fix.

0.11.0 (2022-12-21)
-------------------

- Fixed incorrect reading of timesteps defined via ``filename start number`` in .case file.
- Added support for ``constant per case`` variables.
- Added methods for writing geometry file (``EnsightGeometryFile.write_header()``, ``UnstructuredElementBlock.write_element_block()``, etc.).
- Added method for writing .case file (``EnsightCaseFile.to_file()``).
- Added ``ensightreader.__version__``.

0.10.0 (2022-05-11)
-------------------

- Added ``ensight_transform`` CLI script.
- Added convenience ``open()``, ``mmap()`` and ``mmap_writable()`` methods to ``EnSightGeometryFile`` and ``EnSightVariableFile``.
- Improved type annotations for data arrays (``npt.NDArray[np.float32]`` and ``npt.NDArray[np.int32]`` instead of ``np.ndarray``).
- Updated requirements to Python 3.9+ and NumPy 1.21+.

0.9.0 (2022-04-03)
------------------

- Initial release together with `Blender EnSight reader plug-in <https://github.com/tkarabela/blender-ensight-reader>`_.
