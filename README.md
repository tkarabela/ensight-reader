[![CI - build](https://img.shields.io/github/actions/workflow/status/tkarabela/ensight-reader/main.yml?branch=master)](https://github.com/tkarabela/ensight-reader/actions)
[![CI - coverage](https://img.shields.io/codecov/c/github/tkarabela/ensight-reader)](https://app.codecov.io/github/tkarabela/ensight-reader)
[![MyPy & Ruffle checked](https://img.shields.io/badge/MyPy%20%26%20Ruffle-checked-blue?style=flat)](https://github.com/tkarabela/pysubs2/actions)
![PyPI - Version](https://img.shields.io/pypi/v/ensight-reader.svg?style=flat)
![PyPI - Status](https://img.shields.io/pypi/status/ensight-reader.svg?style=flat)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ensight-reader.svg?style=flat)
![License](https://img.shields.io/pypi/l/ensight-reader.svg?style=flat)

# ensight-reader

This library provides a pure Python reader (with some writing capability) for the EnSight Gold data format,
a common format for results of computational fluid dynamics (CFD) simulations.
It also comes with a few CLI tools, notably `ensight_transform` which
allows you to perform in-place scaling/translation/etc. of the geometry in your case.

The library designed for efficient, selective, memory-mapped access to data from EnSight Gold case –
something that would be useful when importing the data into other systems. If you're looking for a more "batteries included" solution, look at
[`vtkEnSightGoldBinaryReader`](https://vtk.org/doc/nightly/html/classvtkEnSightGoldBinaryReader.html)
from the VTK library ([see docs for comparison](https://ensight-reader.readthedocs.io/en/latest/design-howto.html#comparison-with-vtk-library)).

### Requirements

- Python 3.9+
- NumPy 1.21+

### Installation

```sh
pip install ensight-reader
```

### Example – Python API

```python
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
```

### Example – CLI

```sh
# increment X coordinate
ensight_transform --translate 1 0 0 sphere.case

# scale by 1000 (eg. m -> mm conversion)
ensight_transform --scale 1e3 1e3 1e3 sphere.case

# rotation matrix
ensight_transform --matrix \
    0 -1  0  0 \
    1  0  0  0 \
    0  0  1  0 \
    0  0  0  1 \
    sphere.case

# transform only "internalMesh" part
ensight_transform --translate 1 0 0 --only-parts internalMesh motorbike.case
```

To learn more, please [see the documentation](https://ensight-reader.readthedocs.io).
