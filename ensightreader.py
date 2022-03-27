# Copyright (c) 2022 Tomas Karabela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import io
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import re
import os.path as op


class EnsightReaderError(Exception):
    def __init__(self, msg: str, fp: Optional[io.BufferedIOBase] = None, lineno: Optional[int] = None):
        self.file_path = getattr(fp, "name", None)
        self.file_offset = fp.tell() if fp else None
        self.file_lineno = lineno
        if lineno is not None:
            message = f"{msg} (path={self.file_path}, line={self.file_lineno})"
        else:
            message = f"{msg} (path={self.file_path}, offset={self.file_offset})"
        super(EnsightReaderError, self).__init__(message)


class IdHandling(Enum):
    OFF = "off"
    GIVEN = "given"
    ASSIGN = "assign"
    IGNORE = "ignore"

    @property
    def ids_present(self) -> bool:
        return self == self.GIVEN or self == self.IGNORE


class VariableLocation(Enum):
    PER_ELEMENT = "element"
    PER_NODE = "node"


class VariableType(Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    TENSOR_SYMM = "tensor symm"
    TENSOR_ASYM = "tensor asym"
    # COMPLEX_SCALAR = "complex scalar"
    # COMPLEX_VECTOR = "complex vector"

VALUES_FOR_VARIABLE_TYPE = {
    VariableType.SCALAR: 1,
    VariableType.VECTOR: 3,
    VariableType.TENSOR_SYMM: 6,
    VariableType.TENSOR_ASYM: 9,
}

class ElementType(Enum):
    POINT = "point"
    BAR2 = "bar2"
    BAR3 = "bar3"
    TRIA3 = "tria3"
    TRIA6 = "tria6"
    QUAD4 = "quad4"
    QUAD8 = "quad8"
    TETRA4 = "tetra4"
    TETRA10 = "tetra10"
    PYRAMID5 = "pyramid5"
    PYRAMID13 = "pyramid13"
    PENTA6 = "penta6"
    PENTA15 = "penta15"
    HEXA8 = "hexa8"
    HEXA20 = "hexa20"
    NSIDED = "nsided"
    NFACED = "nfaced"
    # G_POINT = "g_point"
    # G_BAR2 = "g_bar2"
    # G_BAR3 = "g_bar3"
    # G_TRIA3 = "g_tria3"
    # G_TRIA6 = "g_tria6"
    # G_QUAD4 = "g_quad4"
    # G_QUAD8 = "g_quad8"
    # G_TETRA4 = "g_tetra4"
    # G_TETRA10 = "g_tetra10"
    # G_PYRAMID5 = "g_pyramid5"
    # G_PYRAMID13 = "g_pyramid13"
    # G_PENTA6 = "g_penta6"
    # G_PENTA15 = "g_penta15"
    # G_HEXA8 = "g_hexa8"
    # G_HEXA20 = "g_hexa20"
    # G_NSIDED = "g_nsided"
    # G_NFACED = "g_nfaced"

    @classmethod
    def parse_from_line(cls, element_type_line: str) -> "ElementType":
        m = re.match(r"[a-z0-9_]+", element_type_line)
        if not m:
            raise ValueError(f"Unexpected element type line {element_type_line!r}")
        element_name = m.group(0)
        element_type = cls(element_name)
        return element_type

    @property
    def dimension(self) -> int:
        return DIMENSION_PER_ELEMENT[self]

NODES_PER_ELEMENT = {
    ElementType.POINT: 1,
    ElementType.BAR2: 2,
    ElementType.BAR3: 3,
    ElementType.TRIA3: 3,
    ElementType.TRIA6: 6,
    ElementType.QUAD4: 4,
    ElementType.QUAD8: 8,
    ElementType.TETRA4: 4,
    ElementType.TETRA10: 10,
    ElementType.PYRAMID5: 5,
    ElementType.PYRAMID13: 13,
    ElementType.PENTA6: 6,
    ElementType.PENTA15: 15,
    ElementType.HEXA8: 8,
    ElementType.HEXA20: 20,
}

DIMENSION_PER_ELEMENT = {
    ElementType.POINT: 0,
    ElementType.BAR2: 1,
    ElementType.BAR3: 1,
    ElementType.TRIA3: 2,
    ElementType.TRIA6: 2,
    ElementType.QUAD4: 2,
    ElementType.QUAD8: 2,
    ElementType.TETRA4: 3,
    ElementType.TETRA10: 3,
    ElementType.PYRAMID5: 3,
    ElementType.PYRAMID13: 3,
    ElementType.PENTA6: 3,
    ElementType.PENTA15: 3,
    ElementType.HEXA8: 3,
    ElementType.HEXA20: 3,
    ElementType.NSIDED: 2,
    ElementType.NFACED: 3,
}

SIZE_INT = SIZE_FLOAT = 4

@dataclass
class Timeset:
    timeset_id: int
    description: Optional[str]
    number_of_steps: int
    filename_numbers: List[int]
    time_values: List[float]


@dataclass
class UnstructuredElementBlock:
    offset: int  # offset to 'element type' line in file (eg. 'tria3')
    number_of_elements: int
    element_type: ElementType
    element_id_handling: IdHandling
    part_id: int

    def read_connectivity(self, fp: io.BufferedIOBase) -> np.ndarray:
        if self.element_type not in NODES_PER_ELEMENT:
            raise ValueError("Please use other methods for nsided/nfaced")

        fp.seek(self.offset)

        assert read_line(fp).startswith(self.element_type.value)
        assert read_int(fp) == self.number_of_elements
        if self.element_id_handling.ids_present:
            fp.seek(self.number_of_elements * SIZE_INT, os.SEEK_CUR)

        nodes_per_element = NODES_PER_ELEMENT[self.element_type]

        arr = read_ints(fp, self.number_of_elements * nodes_per_element)
        return arr.reshape((self.number_of_elements, nodes_per_element), order="C")

    def read_connectivity_nsided(self, fp: io.BufferedIOBase) -> Tuple[np.ndarray, np.ndarray]:
        if self.element_type != ElementType.NSIDED:
            raise ValueError("Please use other methods for not nsided")

        fp.seek(self.offset)

        assert read_line(fp).startswith(self.element_type.value)
        assert read_int(fp) == self.number_of_elements
        if self.element_id_handling.ids_present:
            fp.seek(self.number_of_elements * SIZE_INT, os.SEEK_CUR)

        polygon_node_counts = read_ints(fp, self.number_of_elements)
        polygon_connectivity = read_ints(fp, polygon_node_counts.sum())
        return polygon_node_counts, polygon_connectivity

    def read_connectivity_nfaced(self, fp: io.BufferedIOBase) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.element_type != ElementType.NFACED:
            raise ValueError("Please use other methods for not nfaced")

        fp.seek(self.offset)

        assert read_line(fp).startswith(self.element_type.value)
        assert read_int(fp) == self.number_of_elements
        if self.element_id_handling.ids_present:
            fp.seek(self.number_of_elements * SIZE_INT, os.SEEK_CUR)

        polyhedra_face_counts = read_ints(fp, self.number_of_elements)
        face_node_counts = read_ints(fp, polyhedra_face_counts.sum())
        face_connectivity = read_ints(fp, face_node_counts.sum())
        return polyhedra_face_counts, face_node_counts, face_connectivity

    @classmethod
    def from_file(cls, fp: io.BufferedIOBase, element_id_handling: IdHandling, part_id: int):
        offset = fp.tell()

        element_type_line = read_line(fp)
        try:
            element_type = ElementType.parse_from_line(element_type_line)
        except ValueError as e:
            raise EnsightReaderError(f"Unexpected element type", fp) from e

        number_of_elements = read_int(fp)

        # skip element IDs
        if element_id_handling.ids_present:
            fp.seek(number_of_elements*SIZE_INT, os.SEEK_CUR)

        if element_type in NODES_PER_ELEMENT:
            nodes_per_element = NODES_PER_ELEMENT[element_type]
            fp.seek(nodes_per_element * number_of_elements * SIZE_INT, os.SEEK_CUR)  # skip connectivity
        elif element_type == ElementType.NSIDED:
            polygon_node_counts = read_ints(fp, number_of_elements)
            fp.seek(polygon_node_counts.sum() * SIZE_INT, os.SEEK_CUR)
        elif element_type == ElementType.NFACED:
            polyhedra_face_counts = read_ints(fp, number_of_elements)
            face_node_counts = read_ints(fp, polyhedra_face_counts.sum())
            fp.seek(face_node_counts.sum() * SIZE_INT, os.SEEK_CUR)  # skip connectivity
        else:
            raise EnsightReaderError(f"Unsupported element type: {element_type}", fp)

        return cls(
            offset=offset,
            number_of_elements=number_of_elements,
            element_type=element_type,
            element_id_handling=element_id_handling,
            part_id=part_id,
        )


@dataclass
class GeometryPart:
    offset: int  # offset to 'part' line in file
    part_id: int
    part_name: str  # 'description' line
    number_of_nodes: int
    element_blocks: List[UnstructuredElementBlock]
    node_id_handling: IdHandling
    element_id_handling: IdHandling

    def read_nodes(self, fp: io.BufferedIOBase) -> np.ndarray:
        fp.seek(self.offset)

        assert read_line(fp).startswith("part")
        assert read_int(fp) == self.part_id
        assert read_line(fp).startswith(self.part_name)
        assert read_line(fp).startswith("coordinates")
        assert read_int(fp) == self.number_of_nodes
        if self.node_id_handling.ids_present:
            fp.seek(self.number_of_nodes * SIZE_INT, os.SEEK_CUR)

        arr = read_floats(fp, 3*self.number_of_nodes)
        return arr.reshape((self.number_of_nodes, 3), order="F")

    def is_volume(self) -> bool:
        return any(block.element_type.dimension == 3 for block in self.element_blocks)

    def is_surface(self) -> bool:
        return any(block.element_type.dimension == 2 for block in self.element_blocks)

    @property
    def number_of_elements(self) -> int:
        return sum(block.number_of_elements for block in self.element_blocks)

    def get_number_of_elements_of_type(self, element_type: ElementType) -> int:
        return sum(block.number_of_elements for block in self.element_blocks if block.element_type == element_type)

    @classmethod
    def from_file(cls, fp: io.BufferedIOBase, node_id_handling: IdHandling, element_id_handling: IdHandling):
        offset = fp.tell()
        fp.seek(0, os.SEEK_END)
        file_len = fp.tell()
        fp.seek(offset)

        element_blocks = []

        part_line = read_line(fp)
        if not part_line.startswith("part"):
            raise EnsightReaderError("Expected 'part' line", fp)
        part_id = read_int(fp)
        part_name = read_line(fp).rstrip("\x00 ")
        coordinates_line = read_line(fp)
        if not coordinates_line.startswith("coordinates"):
            raise EnsightReaderError("Expected 'coordinates' line (other part types not implemented)", fp)
        number_of_nodes = read_int(fp)

        # skip node IDs
        if node_id_handling.ids_present:
            fp.seek(number_of_nodes * SIZE_INT, os.SEEK_CUR)

        # skip node coordinates
        fp.seek(3 * number_of_nodes * SIZE_FLOAT, os.SEEK_CUR)

        # read element blocks
        while fp.tell() != file_len:
            element_type_line = peek_line(fp)
            if element_type_line.startswith("part"):
                break  # end of this part, stop
            else:
                element_block = UnstructuredElementBlock.from_file(fp,
                                                                   element_id_handling=element_id_handling,
                                                                   part_id=part_id)
                element_blocks.append(element_block)

        return cls(
                offset=offset,
                part_id=part_id,
                part_name=part_name,
                number_of_nodes=number_of_nodes,
                element_blocks=element_blocks,
                node_id_handling=node_id_handling,
                element_id_handling=element_id_handling,
            )


def read_array(fp: io.BufferedIOBase, count: int, dtype: np.dtype) -> np.ndarray:
    arr = np.empty((count,), dtype=dtype)
    bytes_to_read = arr.data.nbytes
    n = fp.readinto(arr.data)
    if n != bytes_to_read:
        raise EnsightReaderError(f"Only read {n} bytes, expected {bytes_to_read} bytes", fp)
    return arr


def read_ints(fp: io.BufferedIOBase, count: int) -> np.ndarray:
    return read_array(fp, count, np.int32)


def read_int(fp: io.BufferedIOBase) -> int:
    return int(read_ints(fp, 1)[0])


def read_floats(fp: io.BufferedIOBase, count: int) -> np.ndarray:
    return read_array(fp, count, np.float32)


def read_string(fp: io.BufferedIOBase, count: int) -> str:
    data = fp.read(count)
    if len(data) != count:
        raise EnsightReaderError(f"Only read {len(data)} bytes, expected {count} bytes", fp)
    return data.decode("ascii", "replace")


def read_line(fp: io.BufferedIOBase) -> str:
    return read_string(fp, 80)


def peek_line(fp: io.BufferedIOBase) -> str:
    s = read_string(fp, 80)
    fp.seek(-len(s), os.SEEK_CUR)
    return s


@dataclass
class EnsightGeometryFile:
    file_path: str
    description_line1: str
    description_line2: str
    node_id_handling: IdHandling
    element_id_handling: IdHandling
    extents: Optional[np.ndarray]
    parts: Dict[int, GeometryPart]  # part ID -> GeometryPart

    def read_nodes(self, part_id: int) -> np.ndarray:
        part = self.parts[part_id]
        with open(self.file_path, "rb") as fp:
            return part.read_nodes(fp)

    def get_part_names(self) -> List[str]:
        return [part.part_name for part in self.parts.values()]

    def get_part_by_name(self, name: str) -> Optional[GeometryPart]:
        for part in self.parts.values():
            if part.part_name == name:
                return part
        return None

    @classmethod
    def from_file_path(cls, file_path: str) -> "EnsightGeometryFile":
        extents = None
        parts = {}

        with open(file_path, "rb") as fp:
            fp.seek(0, os.SEEK_END)
            file_len = fp.tell()
            fp.seek(0)

            first_line = read_line(fp)
            if not first_line.startswith("C Binary"):
                raise EnsightReaderError("Only 'C Binary' files are supported", fp)

            description_line1 = read_line(fp)
            description_line2 = read_line(fp)

            node_id_line = read_line(fp)
            m = re.match(r"node id ([a-z]+)", node_id_line)
            if m:
                node_id_handling = IdHandling(m.group(1))
            else:
                raise EnsightReaderError("Unexpected 'node id' line", fp)

            element_id_line = read_line(fp)
            m = re.match(r"element id ([a-z]+)", element_id_line)
            if m:
                element_id_handling = IdHandling(m.group(1))
            else:
                raise EnsightReaderError("Unexpected 'element id' line", fp)

            tmp = peek_line(fp)
            if tmp.startswith("extents"):
                _ = read_line(fp)  # 'extents' line
                extents = read_floats(fp, 6)
            elif tmp.startswith("part"):
                pass  # expected, we can start reading parts
            else:
                raise EnsightReaderError("Expected 'extents' or 'part' line", fp)

            # read parts
            while fp.tell() != file_len:
                part = GeometryPart.from_file(fp,
                                              node_id_handling=node_id_handling,
                                              element_id_handling=element_id_handling)
                if part.part_id in parts:
                    raise EnsightReaderError(f"Duplicate part id: {part.part_id}", fp)
                parts[part.part_id] = part

        return cls(
            file_path=file_path,
            description_line1=description_line1,
            description_line2=description_line2,
            node_id_handling=node_id_handling,
            element_id_handling=element_id_handling,
            extents=extents,
            parts=parts,
        )


@dataclass
class EnsightVariableFile:
    file_path: str
    description_line: str
    variable_location: VariableLocation
    variable_type: VariableType
    part_offsets: Dict[int, int]  # part ID -> offset to 'part' line in file
    part_element_offsets: Optional[Dict[Tuple[int, ElementType], int]]  # (part ID, element type) -> offset to
                                                                        # 'element type' line (for per-node variables
                                                                        # part_element_offsets is None)
    geometry_file: EnsightGeometryFile

    def is_defined_for_part_id(self, part_id: int) -> bool:
        return part_id in self.part_offsets

    def read_node_data(self, fp: io.BufferedIOBase, part_id: int) -> Optional[np.ndarray]:
        part = self.geometry_file.parts[part_id]

        if not self.variable_location == VariableLocation.PER_NODE:
            raise ValueError("Variable is not per node")

        offset = self.part_offsets.get(part_id)
        if offset is None:
            return None

        fp.seek(offset)
        assert read_line(fp).startswith("part")
        assert read_int(fp) == part_id
        assert read_line(fp).startswith("coordinates")

        n = part.number_of_nodes
        k = VALUES_FOR_VARIABLE_TYPE[self.variable_type]
        arr = read_floats(fp, n*k)
        if k > 1:
            arr = arr.reshape((n, k), order="F")
        return arr

    def read_element_data(self, fp: io.BufferedIOBase, part_id: int, element_type: ElementType) -> Optional[np.ndarray]:
        part = self.geometry_file.parts[part_id]

        if not self.variable_location == VariableLocation.PER_ELEMENT:
            raise ValueError("Variable is not per element")

        offset = self.part_element_offsets.get((part_id, element_type))
        if offset is None:
            return None

        fp.seek(offset)
        assert read_line(fp).startswith(element_type.value)

        n = part.get_number_of_elements_of_type(element_type)
        k = VALUES_FOR_VARIABLE_TYPE[self.variable_type]
        arr = read_floats(fp, n*k)
        if k > 1:
            arr = arr.reshape((n, k), order="F")
        return arr

    @classmethod
    def from_file_path(cls, file_path: str, variable_location: VariableLocation, variable_type: VariableType,
                       geofile: EnsightGeometryFile) -> "EnsightVariableFile":
        part_offsets = {}
        part_element_offsets = {} if variable_location == VariableLocation.PER_ELEMENT else None

        with open(file_path, "rb") as fp:
            fp.seek(0, os.SEEK_END)
            file_len = fp.tell()
            fp.seek(0)

            description_line = read_line(fp)

            # read parts
            while fp.tell() != file_len:
                part_offset = fp.tell()

                part_line = read_line(fp)
                if not part_line.startswith("part"):
                    raise EnsightReaderError(f"Expected 'part' line, got: {part_line!r}", fp)

                part_id = read_int(fp)
                if part_id not in geofile.parts:
                    raise EnsightReaderError(f"Variable file has data for part id {part_id}, "
                                             f"but this part is not in geofile {geofile.file_path}", fp)
                if part_id in part_offsets:
                    raise EnsightReaderError(f"Duplicate definition of part id {part_id}", fp)

                part = geofile.parts[part_id]
                part_offsets[part_id] = part_offset

                if variable_location == VariableLocation.PER_NODE:
                    coordinates_line = read_line(fp)
                    if not coordinates_line.startswith("coordinates"):
                        raise EnsightReaderError(f"Expected 'coordinates' line, got: {coordinates_line!r}", fp)

                    part_offsets[part_id] = part_offset

                    # skip data
                    n = part.number_of_nodes
                    k = VALUES_FOR_VARIABLE_TYPE[variable_type]
                    fp.seek(n * k * SIZE_FLOAT, os.SEEK_CUR)
                elif variable_location == VariableLocation.PER_ELEMENT:
                    while fp.tell() != file_len:
                        part_element_offset = fp.tell()
                        element_type_line = peek_line(fp)
                        if element_type_line.startswith("part"):
                            break  # new part starts

                        _ = read_line(fp)  # 'element type' line

                        try:
                            element_type = ElementType.parse_from_line(element_type_line)
                        except ValueError as e:
                            raise EnsightReaderError(f"Bad element type", fp) from e

                        blocks = [block for block in part.element_blocks if block.element_type == element_type]
                        if not blocks:
                            raise EnsightReaderError(f"Variable file has data for part id {part_id}, "
                                                     f"element type {element_type}, but this element type "
                                                     f"is not in geofile {geofile.file_path}", fp)
                        if len(blocks) > 1:
                            raise EnsightReaderError(f"Variable file has data for part id {part_id}, "
                                                     f"element type {element_type}, but there are multiple "
                                                     f"element blocks of this type in geofile {geofile.file_path} "
                                                     f"(handling of this is not implemented, please have only one block "
                                                     f"of each type)", fp)

                        part_element_offsets[part_id, element_type] = part_element_offset

                        # skip data
                        n = blocks[0].number_of_elements
                        k = VALUES_FOR_VARIABLE_TYPE[variable_type]
                        fp.seek(n * k * SIZE_FLOAT, os.SEEK_CUR)
                else:
                    raise EnsightReaderError(f"Bad variable location {variable_location}", fp)

        return cls(
            file_path=file_path,
            description_line=description_line,
            variable_location=variable_location,
            variable_type=variable_type,
            part_offsets=part_offsets,
            part_element_offsets=part_element_offsets,
            geometry_file=geofile,
        )


def fill_wildcard(filename: str, value: int) -> str:
    return re.sub(r"\*+", lambda m: str(value).zfill(len(m.group(0))), filename, count=1)


@dataclass
class EnsightGeometryFileSet:
    casefile_dir_path: str
    timeset: Optional[Timeset]
    filename: str
    # change_coords_only: bool = False

    def get_file(self, timestep: int = 0) -> EnsightGeometryFile:
        if self.timeset is None:
            timestep_filename = self.filename
        else:
            timestep_filename = fill_wildcard(self.filename, self.timeset.filename_numbers[timestep])

        path = op.join(self.casefile_dir_path, timestep_filename)
        return EnsightGeometryFile.from_file_path(path)


@dataclass
class EnsightVariableFileSet:
    casefile_dir_path: str
    timeset: Optional[Timeset]
    variable_location: VariableLocation
    variable_type: VariableType
    variable_name: str  # 'description' field in casefile
    filename: str

    def get_file(self, geofile: EnsightGeometryFile, timestep: int = 0) -> EnsightVariableFile:
        # XXX be sure to match geofile and timestep!
        if self.timeset is None:
            timestep_filename = self.filename
        else:
            timestep_filename = fill_wildcard(self.filename, self.timeset.filename_numbers[timestep])

        path = op.join(self.casefile_dir_path, timestep_filename)
        return EnsightVariableFile.from_file_path(path, variable_location=self.variable_location,
                                                  variable_type=self.variable_type, geofile=geofile)


def read_numbers_from_text_file(path: str, type_: type) -> List:
    output = []
    with open(path) as fp:
        for line in fp:
            values = line.split()
            output.extend(map(type_, values))
    return output


@dataclass
class EnsightCaseFile:
    casefile_path: str
    geometry_model: EnsightGeometryFileSet
    variables: Dict[str, EnsightVariableFileSet]  # variable name -> EnsightVariableFileSet
    timesets: Dict[int, Timeset]  # time set ID -> TimeSet
    _geometry_file_cache: Dict[int, EnsightGeometryFile] = field(default_factory=dict, repr=False)
    _variable_file_cache: Dict[Tuple[int, str], EnsightVariableFile] = field(default_factory=dict, repr=False)

    def get_geometry_model(self, timestep: int = 0) -> EnsightGeometryFile:
        if timestep not in self._geometry_file_cache:
            self._geometry_file_cache[timestep] = self.geometry_model.get_file(timestep)
        return self._geometry_file_cache[timestep]

    def get_variable(self, name: str, timestep: int = 0) -> EnsightVariableFile:
        cache_key = (name, timestep)
        if cache_key not in self._variable_file_cache:
            variable_fileset = self.variables.get(name)
            if variable_fileset is None:
                raise KeyError(f"No variable named {name!r} (present variables: {list(self.variables.keys())})")

            geofile_timeset = self.geometry_model.timeset
            variable_timeset = variable_fileset.timeset
            if geofile_timeset is not None or variable_timeset is not None:
                if geofile_timeset.timeset_id != variable_timeset.timeset_id:
                    raise NotImplementedError(f"Geometry and variable {name!r} use different timesets, "
                                              f"this is currently not handled (if you want this, please call "
                                              f"EnsightVariableFileSet.get_file() with the correct geofile)")

            geofile = self.get_geometry_model(timestep)
            self._geometry_file_cache[cache_key] = self.variables[name].get_file(geofile=geofile, timestep=timestep)
        return self._geometry_file_cache[cache_key]

    def is_transient(self) -> bool:
        return len(self.timesets) > 0

    def get_time_values(self, timeset_id: Optional[int] = None) -> Optional[List[float]]:
        if not self.is_transient():
            return None

        timeset_ids = list(self.timesets.keys())
        if timeset_id is None:
            if len(timeset_ids) == 1:
                timeset_id = timeset_ids[0]
            else:
                raise ValueError("Multiple time sets in case, please specify timeset_id explicitly")

        if timeset_id not in self.timesets:
            raise KeyError(f"No timeset with id {timeset_id} (present timeset ids: {timeset_ids})")

        return self.timesets[timeset_id].time_values

    def get_variables(self) -> List[str]:
        return list(self.variables.keys())

    @classmethod
    def from_file(cls, casefile_path: str) -> "EnsightCaseFile":
        casefile_dir_path = op.dirname(casefile_path)
        geometry_model = None
        geometry_model_ts = None
        variables = {}
        variables_ts = {}
        timesets = {}

        with open(casefile_path) as fp:
            current_section = None
            current_timeset: Optional[Timeset] = None
            current_timeset_file_start_number = None
            current_timeset_filename_increment = None

            for lineno, line in enumerate(fp, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if line.isupper():
                    current_section = line
                    continue

                if ":" in line:
                    key, values = line.split(":", maxsplit=1)
                    values = values.split()
                else:
                    key = None
                    values = line.split()

                if current_section == "FORMAT":
                    if key == "type" and values != ["ensight", "gold"]:
                        raise EnsightReaderError("Expected 'ensight gold' in type line", fp, lineno)
                elif current_section == "GEOMETRY":
                    if key == "model":
                        if values[-1] == "change_coords_only":
                            raise EnsightReaderError("Model with 'change_coords_only' is not supported", fp, lineno)
                        if len(values) == 1:
                            filename, = values
                            geometry_model = EnsightGeometryFileSet(casefile_dir_path,
                                                                    timeset=None,
                                                                    filename=filename)
                        elif len(values) == 2:
                            ts, filename = values
                            geometry_model_ts = int(ts)
                            geometry_model = EnsightGeometryFileSet(casefile_dir_path,
                                                                    timeset=None,  # timeset will be added later
                                                                    filename=filename)
                        else:
                            raise EnsightReaderError("Unsupported model definition (note: fs is not supported)",
                                                     fp, lineno)
                    # note: measured, match, boundary geometries are not supported
                elif current_section == "VARIABLE":
                    try:
                        variable_type_, variable_location_ = key.split(" per ")
                        variable_type = VariableType(variable_type_)
                        variable_location = VariableLocation(variable_location_)
                    except ValueError:
                        print(f"Warning: unsupported variable line ({casefile_path}:{lineno}), skipping")
                        continue

                    filename = values[-1]
                    description = values[-2]
                    ts = None
                    if len(values) == 3:
                        ts = int(values[0])
                    elif len(values) > 3:
                        print(f"Warning: unsupported variable line ({casefile_path}:{lineno}), skipping")
                        continue

                    variables[description] = EnsightVariableFileSet(casefile_dir_path,
                                                                    timeset=None,  # timeset will be added later
                                                                    variable_location=variable_location,
                                                                    variable_type=variable_type,
                                                                    variable_name=description,
                                                                    filename=filename)
                    variables_ts[description] = ts

                elif current_section == "TIME":
                    if key == "time set":
                        ts = int(values[0])
                        description = None
                        if len(values) == 2:
                            description = values[1]
                        timesets[ts] = current_timeset = Timeset(ts, description, -1, [], [])
                        current_timeset_file_start_number = None
                        current_timeset_filename_increment = None
                    elif key == "number of steps":
                        current_timeset.number_of_steps = int(values[0])
                    elif key == "filename start number":
                        current_timeset_file_start_number = int(values[0])
                        if current_timeset_file_start_number is not None and current_timeset_filename_increment is not None:
                            current_timeset.filename_numbers = list(range(current_timeset_file_start_number,
                                                                          current_timeset.number_of_steps,
                                                                          current_timeset_filename_increment))
                    elif key == "filename increment":
                        current_timeset_filename_increment = int(values[0])
                        if current_timeset_file_start_number is not None and current_timeset_filename_increment is not None:
                            current_timeset.filename_numbers = list(range(current_timeset_file_start_number,
                                                                          current_timeset.number_of_steps,
                                                                          current_timeset_filename_increment))
                    elif key == "time values":
                        current_timeset.time_values.extend(map(float, values))
                    elif key == "filename numbers":
                        raise EnsightReaderError("Unsupported timeset definition ('filename numbers' is not supported)",
                                                 fp, lineno)
                    elif key == "filename numbers file":
                        path = op.join(casefile_dir_path, values[0])
                        current_timeset.filename_numbers = read_numbers_from_text_file(path, int)
                    elif key == "time values file":
                        path = op.join(casefile_dir_path, values[0])
                        current_timeset.time_values = read_numbers_from_text_file(path, float)
                    elif key is None:
                        # we expect that this is continuation of 'time values'
                        current_timeset.time_values.extend(map(float, values))
                    else:
                        print(f"Warning: unsupported variable line ({casefile_path}:{lineno}), skipping")
                        continue
                else:
                    print(f"Warning: unsupported section ({casefile_path}:{lineno}), skipping")
                    continue

            if geometry_model is None:
                raise EnsightReaderError("No model defined in casefile", fp)

        # propagate timesets to geometry and variables
        if geometry_model_ts is not None:
            geometry_model.timeset = timesets[geometry_model_ts]

        for variable_name, variable_ts in variables_ts.items():
            if variable_ts is not None:
                variables[variable_name].timeset = timesets[variable_ts]

        return cls(
            casefile_path=casefile_path,
            geometry_model=geometry_model,
            variables=variables,
            timesets=timesets,
        )
