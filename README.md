![PyPI - Version](https://img.shields.io/pypi/v/ensight-reader.svg?style=flat-square)
![PyPI - Status](https://img.shields.io/pypi/status/ensight-reader.svg?style=flat-square)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ensight-reader.svg?style=flat-square)
![License](https://img.shields.io/pypi/l/ensight-reader.svg?style=flat-square)

# ensight-reader

This library provides a pure Python reader for the EnSight Gold data format,
a common format for results of computational fluid dynamics (CFD) simulations.

It's designed for efficient, selective, memory-mapped access to data from EnSight Gold case --
something that would be useful when importing the data into other systems.

If you're looking for a more "batteries included" solution, look at
[`vtkEnSightGoldBinaryReader`](https://vtk.org/doc/nightly/html/classvtkEnSightGoldBinaryReader.html)
from the VTK library.

### Requirements

- Python 3.7+
- NumPy

### Installation

```sh
pip install ensight-reader
```

### Example

```python
import ensightreader

case = ensightreader.read_case("example.case")
geofile = case.get_geometry_model()

part_names = geofile.get_part_names()                # ["internalMesh", ...]
part = geofile.get_part_by_name(part_names[0])
N = part.number_of_nodes

with open(geofile.file_path, "rb") as fp_geo:
  node_coordinates = part.read_coordinates(fp_geo)  # np.ndarray((N, 3), dtype=np.float32)
```

To learn more, please [see the documentation](https://ensight-reader.readthedocs.io).
