# Copyright (c) 2022-2024 Tomas Karabela
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


import mmap as _mmap
import os
import os.path as op
import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import BinaryIO, Dict, Generator, List, Optional, TextIO, Tuple, Type, TypeVar, Union, Iterator

import numpy as np
import numpy.typing as npt

T = TypeVar('T')
TNum = TypeVar('TNum', np.int32, np.float32)
SeekableBufferedReader = Union[BinaryIO, _mmap.mmap]
SeekableBufferedWriter = Union[BinaryIO, _mmap.mmap]
Float32NDArray = npt.NDArray[np.float32]
Int32NDArray = npt.NDArray[np.int32]

__version__ = "0.11.2"


def add_exception_note(e: Exception, note: str) -> None:
    if hasattr(e, "add_note"):  # Python 3.11+
        e.add_note(note)


@contextmanager
def add_exception_note_block(note: str) -> Iterator[None]:
    try:
        yield
    except Exception as e:
        add_exception_note(e, note)
        raise


class EnsightReaderError(Exception):
    """
    Error raised when parsing EnSight Gold binary files

    Attributes:
        file_path (str): path to file where the error was encountered
        file_offset (int): approximate seek position of the error (this may be a bit past the place where
            the error is - it's the seek position when this exception was raised)
        file_lineno (int): line number of the error (this only applies to errors in ``*.case`` file,
            as other files are binary)

    """
    def __init__(self, msg: str, fp: Optional[Union[TextIO, SeekableBufferedReader]] = None, lineno: Optional[int] = None):
        self.file_path = getattr(fp, "name", None)
        try:
            self.file_offset = fp.tell() if fp else None
        except OSError:
            self.file_offset = None
        self.file_lineno = lineno
        if lineno is not None:
            message = f"{msg} (path={self.file_path}, line={self.file_lineno})"
        else:
            message = f"{msg} (path={self.file_path}, offset={self.file_offset})"
        super(EnsightReaderError, self).__init__(message)


class EnsightReaderWarning(Warning):
    """
    Warning raised when parsing EnSight Gold binary files

    Attributes:
        file_path (str): path to file where the error was encountered
        file_offset (int): approximate seek position of the error (this may be a bit past the place where
            the error is - it's the seek position when this exception was raised)
        file_lineno (int): line number of the error (this only applies to errors in ``*.case`` file,
            as other files are binary)
    """
    def __init__(self, msg: str, fp: Optional[Union[TextIO, SeekableBufferedReader]] = None, lineno: Optional[int] = None):
        self.file_path = getattr(fp, "name", None)
        try:
            self.file_offset = fp.tell() if fp else None
        except OSError:
            self.file_offset = None
        self.file_lineno = lineno
        if lineno is not None:
            message = f"{msg} (path={self.file_path}, line={self.file_lineno})"
        else:
            message = f"{msg} (path={self.file_path}, offset={self.file_offset})"
        super(EnsightReaderWarning, self).__init__(message)


class IdHandling(Enum):
    """
    Handling of node/element IDs in EnSight Gold geometry file.

    This is defined in geometry file header and describes whether
    IDs are present in the file or not.

    """
    OFF = "off"
    GIVEN = "given"
    ASSIGN = "assign"
    IGNORE = "ignore"

    @property
    def ids_present(self) -> bool:
        """Return True if IDs are present in geometry file, otherwise False"""
        return self in (self.GIVEN, self.IGNORE)  # type: ignore[comparison-overlap]

    def __str__(self) -> str:
        return self.value


class ChangingGeometry(Enum):
    """
    Additional information about transient geometry

    """
    NO_CHANGE = "no_change"
    COORD_CHANGE = "coord_change"
    CONN_CHANGE = "conn_change"

    def __str__(self) -> str:
        return self.value


class VariableLocation(Enum):
    """
    Location of variable in EnSight Gold case

    Whether the variable is defined for cells or nodes.
    """
    PER_ELEMENT = "element"
    PER_NODE = "node"

    def __str__(self) -> str:
        return self.value


class VariableType(Enum):
    """
    Type of variable in EnSight Gold case

    .. Note::
        Complex variables and "per measured" variables are not supported.

    """

    SCALAR = "scalar"
    VECTOR = "vector"
    TENSOR_SYMM = "tensor symm"
    TENSOR_ASYM = "tensor asym"
    # COMPLEX_SCALAR = "complex scalar"
    # COMPLEX_VECTOR = "complex vector"

    def __str__(self) -> str:
        return self.value

VALUES_FOR_VARIABLE_TYPE = {
    VariableType.SCALAR: 1,
    VariableType.VECTOR: 3,
    VariableType.TENSOR_SYMM: 6,
    VariableType.TENSOR_ASYM: 9,
}


class ElementType(Enum):
    """
    Element type in EnSight Gold geometry file

    .. Note::
        Ghost cell variants ``g_*`` are not supported.

    """

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
        """
        Return dimension of element

        Returns 3 for volume elements, 2 for surface elements, 1 for line elements
        and 0 for point elements.
        """
        return DIMENSION_PER_ELEMENT[self]

    @property
    def nodes_per_element(self) -> int:
        """
        Return number nodes defining the element

        This only makes sense for elements consisting of constant number of nodes.
        For NSIDED and NFACED element type, this raises and exception.
        """
        return NODES_PER_ELEMENT[self]

    def has_constant_number_of_nodes_per_element(self) -> bool:
        """
        Return True if element type has constant number of nodes defining each element, else False

        This is True for all element types except NSIDED and NFACED.
        """
        return self in NODES_PER_ELEMENT

    def __str__(self) -> str:
        return self.value


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
    """
    Description of time set in EnSight Gold case

    This means a non-decreasing sequence of times
    for which geometry and/or variable values are known.

    Attributes:
        timeset_id: ID of the time set
        description: label of the time set, or None
        number_of_steps: number of timesteps
        filename_numbers: list of numbers for filenames (to be filled in place of ``*`` wildcards)
        time_values: list of time values (ie. seconds, or something else)

    """
    timeset_id: int
    description: Optional[str]
    number_of_steps: int
    filename_numbers: List[int]
    time_values: List[float]

    @staticmethod
    def filename_numbers_from_arithmetic_sequence(file_start_number: int, number_of_steps: int, filename_increment: int) -> List[int]:
        assert filename_increment >= 0
        assert number_of_steps >= 0
        assert file_start_number >= 0

        return [file_start_number + i*filename_increment for i in range(number_of_steps)]


@dataclass
class UnstructuredElementBlock:
    """
    A block of elements of the same type in a part in EnSight Gold binary geometry file

    To use it:

        >>> from ensightreader import read_case, ElementType
        >>> case = read_case("example.case")
        >>> geofile = case.get_geometry_model()
        >>> part_names = geofile.get_part_names()
        >>> part = geofile.get_part_by_name(part_names[0])
        >>> with open(geofile.file_path, "rb") as fp_geo:
        ...     for block in part.element_blocks:
        ...         if block.element_type == ElementType.NFACED:
        ...             polyhedra_face_counts, face_node_counts, face_connectivity = block.read_connectivity_nfaced(fp_geo)
        ...         elif block.element_type == ElementType.NSIDED:
        ...             polygon_node_counts, polygon_connectivity = block.read_connectivity_nsided(fp_geo)
        ...         else:
        ...             connectivity = block.read_connectivity(fp_geo)

    Attributes:
        offset: offset to 'element type' line in file (eg. 'tria3')
        number_of_elements: number of elements in this block
        element_type: type of elements in this block
        element_id_handling: element ID presence
        part_id: part number
    """
    offset: int
    number_of_elements: int
    element_type: ElementType
    element_id_handling: IdHandling
    part_id: int

    def read_element_ids(self, fp: SeekableBufferedReader) -> Optional[Int32NDArray]:
        """
        Read element IDs for this element block, if present

        .. note::
            This method simply returns the element ID array as present
            in the file; it does not differentiate between ``element id given``
            and ``element id ignore``, etc.

        Args:
            fp: opened geometry file object in ``"rb"`` mode

        Returns:
            1D array of int32 with element IDs, or None if element IDs are not
            present in the file
        """
        if not self.element_id_handling.ids_present:
            return None

        fp.seek(self.offset)

        assert read_line(fp).startswith(self.element_type.value)
        assert read_int(fp) == self.number_of_elements
        arr = read_ints(fp, self.number_of_elements)
        return arr

    def read_connectivity(self, fp: SeekableBufferedReader) -> Int32NDArray:
        """
        Read connectivity (for elements other than NSIDED/NFACED)

        Use this for elements which have constant number of nodes
        per element (ie. any element type except polygons and polyhedra).

        Args:
            fp: opened geometry file object in ``"rb"`` mode

        Returns:
            2D ``(n, k)`` array of int32 with node indices (numbered from 1), where
            ``n`` is the number of elements and
            ``k`` is the number of nodes defining each element
        """
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

    def read_connectivity_nsided(self, fp: SeekableBufferedReader) -> Tuple[Int32NDArray, Int32NDArray]:
        """
        Read connectivity (for NSIDED elements)

        Args:
            fp: opened geometry file object in ``"rb"`` mode

        Returns:
            tuple ``(polygon_node_counts, polygon_connectivity)`` where
            ``polygon_node_counts`` is 1D array of type int32
            giving number of nodes for each polygon and
            ``polygon_connectivity`` is 1D array of type int32
            giving node indices (numbered from 1) for every polygon
        """
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

    def read_connectivity_nfaced(self, fp: SeekableBufferedReader) -> Tuple[Int32NDArray, Int32NDArray, Int32NDArray]:
        """
        Read connectivity (for NFACED elements)

        Args:
            fp: opened geometry file object in ``"rb"`` mode

        Returns:
            tuple ``(polyhedra_face_counts, face_node_counts, face_connectivity)`` where
            ``polyhedra_face_counts`` is 1D array of type int32
            giving number of faces for each polygon,
            ``face_node_counts`` is 1D array of type int32
            giving number of nodes for each face for each polygon (ordered as
            1st polygon 1st face, 1st polygon 2nd face, ..., 2nd polygon 1st face, ...),
            ``face_connectivity`` is 1D array of type int32 giving node indices
            (numbered from 1)
        """
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
    def from_file(cls, fp: SeekableBufferedReader, element_id_handling: IdHandling, part_id: int) -> "UnstructuredElementBlock":
        """Used internally by `GeometryPart.from_file()`"""
        offset = fp.tell()

        element_type_line = read_line(fp)
        try:
            element_type = ElementType.parse_from_line(element_type_line)
        except ValueError as e:
            raise EnsightReaderError("Unexpected element type", fp) from e

        with add_exception_note_block(f"element_type = {element_type}"):
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

    @staticmethod
    def write_element_block(fp: SeekableBufferedWriter, element_type: ElementType, connectivity: Int32NDArray,
                            element_ids: Optional[Int32NDArray] = None) -> None:
        """
        Write element block (not NSIDED/NFACED) to given opened file

        See `UnstructuredElementBlock.read_connectivity()`.

        """
        assert element_type not in (ElementType.NFACED, ElementType.NFACED)
        number_of_elements = connectivity.shape[0]
        assert connectivity.shape[1] == element_type.nodes_per_element
        if element_ids is not None:
            assert element_ids.shape == (number_of_elements,)

        write_line(fp, f"{element_type}")
        write_int(fp, number_of_elements)
        if element_ids is not None:
            write_ints(fp, element_ids)
        write_ints(fp, connectivity.flatten("C"))

    @staticmethod
    def write_element_block_nsided(fp: SeekableBufferedWriter, polygon_node_counts: Int32NDArray,
                                   polygon_connectivity: Int32NDArray, element_ids: Optional[Int32NDArray] = None) -> None:
        """
        Write NSIDED element block to given opened file

        See `UnstructuredElementBlock.read_connectivity_nsided()`.
        """
        number_of_elements, = polygon_node_counts.shape
        assert polygon_connectivity.shape == (polygon_node_counts.sum(),)
        if element_ids is not None:
            assert element_ids.shape == (number_of_elements,)

        write_line(fp, f"{ElementType.NSIDED}")
        write_int(fp, number_of_elements)
        if element_ids is not None:
            write_ints(fp, element_ids)
        write_ints(fp, polygon_node_counts)
        write_ints(fp, polygon_connectivity)

    @staticmethod
    def write_element_block_nfaced(fp: SeekableBufferedWriter, polyhedra_face_counts: Int32NDArray,
                                   face_node_counts: Int32NDArray, face_connectivity: Int32NDArray,
                                   element_ids: Optional[Int32NDArray] = None) -> None:
        """
        Write NFACED element block to given opened file

        See `UnstructuredElementBlock.read_connectivity_nfaced()`.
        """
        number_of_elements, = polyhedra_face_counts.shape
        assert face_node_counts.shape == (polyhedra_face_counts.sum(),)
        assert face_connectivity.shape == (face_node_counts.sum(),)
        if element_ids is not None:
            assert element_ids.shape == (number_of_elements,)

        write_line(fp, f"{ElementType.NFACED}")
        write_int(fp, number_of_elements)
        if element_ids is not None:
            write_ints(fp, element_ids)
        write_ints(fp, polyhedra_face_counts)
        write_ints(fp, face_node_counts)
        write_ints(fp, face_connectivity)


@dataclass
class GeometryPart:
    """
    A part in EnSight Gold geometry file

    To use it:

        >>> import ensightreader
        >>> case = ensightreader.read_case("example.case")
        >>> geofile = case.get_geometry_model()
        >>> part_names = geofile.get_part_names()
        >>> part = geofile.get_part_by_name(part_names[0])
        >>> print(part.is_volume())
        >>> print(part.number_of_nodes)

    Each geometry part has its own set of nodes not shared
    with other parts. Elements are defined in blocks, where
    each block may have different type (tetrahedra, wedge, etc.)
    but all elements in the block have the same type.

    Attributes:
        offset: offset to 'part' line in the geometry file
        part_id: part number
        part_name: part name ('description' line)
        number_of_nodes: number of nodes for this part
        element_blocks: list of element blocks, in order of definition in the file
        node_id_handling: node ID presence
        element_id_handling: element ID presence
        changing_geometry: type of transient changes (coordinate/connectivity/none)
    """
    offset: int
    part_id: int
    part_name: str
    number_of_nodes: int
    element_blocks: List[UnstructuredElementBlock]
    node_id_handling: IdHandling
    element_id_handling: IdHandling
    changing_geometry: Optional[ChangingGeometry] = None

    def read_nodes(self, fp: SeekableBufferedReader) -> Float32NDArray:
        """
        Read node coordinates for this part

        Args:
            fp: opened geometry file object in ``"rb"`` mode

        Returns:
            2D ``(n, 3)`` array of float32 with node coordinates
        """
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

    def read_node_ids(self, fp: SeekableBufferedReader) -> Optional[Int32NDArray]:
        """
        Read node IDs for this part, if present

        .. note::
            This method simply returns the node ID array as present
            in the file; it does not differentiate between ``node id given``
            and ``node id ignore``, etc.

        Args:
            fp: opened geometry file object in ``"rb"`` mode

        Returns:
            1D array of int32 with node IDs, or None if node IDs are not
            present in the file
        """
        if not self.node_id_handling.ids_present:
            return None

        fp.seek(self.offset)

        assert read_line(fp).startswith("part")
        assert read_int(fp) == self.part_id
        assert read_line(fp).startswith(self.part_name)
        assert read_line(fp).startswith("coordinates")
        assert read_int(fp) == self.number_of_nodes
        arr = read_ints(fp, self.number_of_nodes)
        return arr

    def is_volume(self) -> bool:
        """Return True if part contains volume elements"""
        return any(block.element_type.dimension == 3 for block in self.element_blocks)

    def is_surface(self) -> bool:
        """Return True if part contains surface elements"""
        return any(block.element_type.dimension == 2 for block in self.element_blocks)

    @property
    def number_of_elements(self) -> int:
        """Return number of elements (of all types)"""
        return sum(block.number_of_elements for block in self.element_blocks)

    def get_number_of_elements_of_type(self, element_type: ElementType) -> int:
        """Return number of elements (of given type)"""
        return sum(block.number_of_elements for block in self.element_blocks if block.element_type == element_type)

    @classmethod
    def from_file(cls, fp: SeekableBufferedReader, node_id_handling: IdHandling,
                  element_id_handling: IdHandling, changing_geometry_per_part: bool) -> "GeometryPart":
        """Used internally by `EnsightGeometryFile.from_file_path()`"""
        offset = fp.tell()
        fp.seek(0, os.SEEK_END)
        file_len = fp.tell()
        fp.seek(offset)

        element_blocks = []
        changing_geometry = None

        part_line = read_line(fp)
        if not part_line.startswith("part"):
            raise EnsightReaderError("Expected 'part' line", fp)
        if changing_geometry_per_part:
            for changing_geometry_ in ChangingGeometry:
                if changing_geometry_.value in part_line:
                    changing_geometry = changing_geometry_
                    break
            else:
                raise EnsightReaderError("Expected no_change/coord_change/conn_change in 'part' line", fp)
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
                with add_exception_note_block(f"part_id = {part_id} ({part_name})"):
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
                changing_geometry=changing_geometry,
            )

    @staticmethod
    def write_part_header(fp: SeekableBufferedWriter, part_id: int, part_name: str,
                          node_coordinates: Float32NDArray, node_ids: Optional[Int32NDArray] = None) -> None:
        """
        Write part header to given opened file

        ``node_coordiantes`` are supposed to be (N, 3)-shaped float32 array.
        """
        number_of_nodes = node_coordinates.shape[0]
        assert node_coordinates.shape[1] == 3
        if node_ids is not None:
            assert node_ids.shape == (number_of_nodes,)

        write_line(fp, "part")
        write_int(fp, part_id)
        write_line(fp, part_name)
        write_line(fp, "coordinates")
        write_int(fp, number_of_nodes)
        if node_ids is not None:
            write_ints(fp, node_ids)
        write_floats(fp, node_coordinates.flatten("F"))


def read_array(fp: SeekableBufferedReader, count: int, dtype: Type[TNum]) -> npt.NDArray[TNum]:
    if isinstance(fp, _mmap.mmap):
        offset = fp.tell()
        bytes_to_read = count * dtype().itemsize
        buffer = fp.read(bytes_to_read)
        if len(buffer) != bytes_to_read:
            raise EnsightReaderError(f"Only read {len(buffer)} bytes, expected {bytes_to_read} bytes", fp)

        # TODO figure out a sane way to ask `mmap.mmap` about its access mode
        if "ACCESS_WRITE" in str(fp):
            # For write-through memory-mapped access, simply create an `ndarray` backed by the
            # mmap, which results in writable `ndarray`; the `fp.read()` call is only used to advance
            # the file pointer in this case.
            arr: npt.NDArray[TNum] = np.ndarray((count,), dtype=dtype, buffer=memoryview(fp), offset=offset)
            return arr
        else:
            # For read-only memory-mapped access, we want to wrap `bytes` as returned from the mmap;
            # this results in non-writeable `ndarray`.
            arr = np.frombuffer(buffer, dtype=dtype)  # type: ignore[no-untyped-call]
            return arr
    else:
        # For regular file access, we allocate buffer first and then read into it,
        # this results in writeable `ndarray`.
        arr = np.empty((count,), dtype=dtype)
        bytes_to_read = arr.data.nbytes
        n = fp.readinto(arr.data)  # type: ignore[attr-defined]
        if n != bytes_to_read:
            raise EnsightReaderError(f"Only read {n} bytes, expected {bytes_to_read} bytes", fp)
        return arr


def write_array(fp: SeekableBufferedWriter, data: npt.NDArray[TNum]) -> None:
    fp.write(data.data)


def read_ints(fp: SeekableBufferedReader, count: int) -> Int32NDArray:
    return read_array(fp, count, np.int32)


def write_ints(fp: SeekableBufferedWriter, data: Int32NDArray) -> None:
    assert np.issubdtype(data.dtype, np.int32)
    write_array(fp, data)


def read_int(fp: SeekableBufferedReader) -> int:
    return int(read_ints(fp, 1)[0])


def write_int(fp: SeekableBufferedReader, value: int) -> None:
    write_ints(fp, np.asarray([value], dtype=np.int32))


def read_floats(fp: SeekableBufferedReader, count: int) -> Float32NDArray:
    return read_array(fp, count, np.float32)


def write_floats(fp: SeekableBufferedWriter, data: Float32NDArray) -> None:
    assert np.issubdtype(data.dtype, np.float32)
    write_array(fp, data)


def read_float(fp: SeekableBufferedReader) -> float:
    return float(read_floats(fp, 1)[0])


def write_float(fp: SeekableBufferedReader, value: float) -> None:
    write_floats(fp, np.asarray([value], dtype=np.float32))


def read_string(fp: SeekableBufferedReader, count: int) -> str:
    data = fp.read(count)
    if len(data) != count:
        raise EnsightReaderError(f"Only read {len(data)} bytes, expected {count} bytes", fp)
    return data.decode("ascii", "replace")


def write_string(fp: SeekableBufferedWriter, s: Union[str, bytes]) -> None:
    if isinstance(s, str):
        data = s.encode("ascii", "replace")
    else:
        data = s

    fp.write(data)


def read_line(fp: SeekableBufferedReader) -> str:
    return read_string(fp, 80)


def peek_line(fp: SeekableBufferedReader) -> str:
    s = read_string(fp, 80)
    fp.seek(-len(s), os.SEEK_CUR)
    return s


def write_line(fp: SeekableBufferedWriter, s: Union[str, bytes]) -> None:
    assert len(s) <= 80

    if isinstance(s, str):
        data = s.encode("ascii", "replace")
    else:
        data = s

    data = data + b"\x00"*(80 - len(data))
    write_string(fp, data)


@dataclass
class EnsightGeometryFile:
    """
    EnSight Gold binary geometry file

    To use it:

        >>> import ensightreader
        >>> case = ensightreader.read_case("example.case")
        >>> geofile = case.get_geometry_model()
        >>> part_names = geofile.get_part_names()
        >>> part = geofile.get_part_by_name(part_names[0])

    This objects contains metadata parsed from EnSight Gold
    geometry file; it does not hold any node coordinates,
    connectivity, etc., but after it's been created it can
    read any such data on demand.

    Attributes:
        file_path: path to the actual geometry file (no wildcards)
        description_line1: first line in header
        description_line2: second line in header
        node_id_handling: node ID presence
        element_id_handling: element ID presence
        extents: optional extents given in header (xmin, xmax, ymin, ymax, zmin, zmax)
        parts: dictonary mapping part IDs to `GeometryPart` objects
        changing_geometry_per_part: whether parts contain information about type of transient changes
    """
    file_path: str
    description_line1: str
    description_line2: str
    node_id_handling: IdHandling
    element_id_handling: IdHandling
    extents: Optional[Float32NDArray]
    parts: Dict[int, GeometryPart]  # part ID -> GeometryPart
    changing_geometry_per_part: bool

    def get_part_names(self) -> List[str]:
        """Return list of part names"""
        return [part.part_name for part in self.parts.values()]

    def get_part_ids(self) -> List[int]:
        """Return list of part IDs"""
        return [part.part_id for part in self.parts.values()]

    def get_part_by_name(self, name: str) -> Optional[GeometryPart]:
        """Return part with given name, or None"""
        for part in self.parts.values():
            if part.part_name == name:
                return part
        return None

    def get_part_by_id(self, part_id: int) -> Optional[GeometryPart]:
        """Return part with given part ID, or None"""
        for part in self.parts.values():
            if part.part_id == part_id:
                return part
        return None

    @classmethod
    def from_file_path(cls, file_path: str, changing_geometry_per_part: bool) -> "EnsightGeometryFile":
        """Parse EnSight Gold geometry file"""
        extents = None
        parts = {}

        with open(file_path, "rb") as fp:
            fp.seek(0, os.SEEK_END)
            file_len = fp.tell()
            fp.seek(0)

            first_line = read_line(fp)
            if not first_line.lower().startswith("c binary"):
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
                                              element_id_handling=element_id_handling,
                                              changing_geometry_per_part=changing_geometry_per_part)
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
            changing_geometry_per_part=changing_geometry_per_part,
        )

    @staticmethod
    def write_header(fp: SeekableBufferedWriter, description_line1: str = "Generated by ensightreader",
                     description_line2: str = "", node_id_handling: IdHandling = IdHandling.OFF,
                     element_id_handling: IdHandling = IdHandling.OFF, extents: Optional[Float32NDArray] = None) -> None:
        """
        Writes geometry file header to given opened file

        Make sure that ``fp`` seek position is at the beginning of the file.

        """
        write_line(fp, "C binary")
        write_line(fp, description_line1)
        write_line(fp, description_line2)
        write_line(fp, f"node id {node_id_handling}")
        write_line(fp, f"element id {element_id_handling}")
        if extents is not None:
            write_line(fp, "extents")
            write_floats(fp, extents)

    def open(self) -> BinaryIO:
        """
        Return the opened file in read-only binary mode (convenience method)

        Use this in ``with`` block and pass the resulting object to `GeometryPart` methods.

        Note:
            This is a simpler alternative to the similar ``mmap()`` method - it doesn't
            require you to reason about lifetime of the opened file since you will be
            getting arrays which own their data (as opposed to views into the memory-mapped file).

        Usage::

            with geometry_file.open() as fp_geo:
                nodes = part.read_nodes(fp_geo)

        Equivalent code::

            with open(geometry_file.file_path, "rb") as fp_geo:
                nodes = part.read_nodes(fp_geo)
        """
        return open(self.file_path, "rb")

    @contextmanager
    def mmap(self) -> Generator[_mmap.mmap, None, None]:
        """
        Return read-only memorymap of the file (convenience method)

        Use this in ``with`` block and pass the resulting object to `GeometryPart` methods.

        Note:
            This is preferred over the similar ``open()`` method if you don't need a copy of the data
            since the returned arrays will be backed by the memorymap. Be careful to keep the memorymap
            around as long as you need the arrays.

        Usage::

            with geometry_file.mmap() as fp_geo:
                nodes = part.read_nodes(fp_geo)

        Equivalent code::

            with open(geometry_file.file_path, "rb") as fp_geo, _mmap.mmap(fp_geo.fileno(), 0, access=mmap.ACCESS_READ) as mm_geo:
                nodes = part.read_nodes(mm_geo)
        """
        with open(self.file_path, "rb") as fp, _mmap.mmap(fp.fileno(), 0, access=_mmap.ACCESS_READ) as mm:
            yield mm

    @contextmanager
    def mmap_writable(self) -> Generator[_mmap.mmap, None, None]:
        """
        Return writable memorymap of the file (convenience method)

        Use this in ``with`` block and pass the resulting object to `GeometryPart` methods.

        Note:
            This special version of the ``mmap()`` method can be used if you wish
            to modify the underlying file. Use carefully.

        Usage::

            with geometry_file.mmap_writable() as fp_geo:
                nodes = part.read_nodes(fp_geo)
                nodes[:, 0] = 0.0  # set first coordinate to zero for part nodes

        Equivalent code::

            with open(geometry_file.file_path, "r+b") as fp_geo, _mmap.mmap(fp_geo.fileno(), 0, access=mmap.ACCESS_WRITE) as mm_geo:
                nodes = part.read_nodes(mm_geo)
                nodes[:, 0] = 0.0  # set X coordinate to zero for part nodes
        """
        with open(self.file_path, "r+b") as fp, _mmap.mmap(fp.fileno(), 0, access=_mmap.ACCESS_WRITE) as mm:
            yield mm


@dataclass
class EnsightVariableFile:
    """
    EnSight Gold binary variable file

    To use it:

        >>> import ensightreader
        >>> case = ensightreader.read_case("example.case")
        >>> geofile = case.get_geometry_model()
        >>> velocity_variable = case.get_variable("U")
        >>> part_names = geofile.get_part_names()
        >>> part = geofile.get_part_by_name(part_names[0])
        >>> with open(velocity_variable.file_path, "rb") as fp_var:
        ...     part_velocity = velocity_variable.read_node_data(fp_var, part.part_id)

    .. note::
        - there are some limitations for per-element variable files, see `EnsightVariableFile.read_element_data()`
        - ``coordinates partial`` is not supported

    This objects contains metadata parsed from EnSight Gold
    variable file; it does not hold any variable data arrays,
    but after it's been created it can read any such data on demand.

    Attributes:
        file_path: path to the actual variable file (no wildcards)
        description_line: line in header
        variable_name: name of the variable in case file
        variable_location: where the variable is defined (elements or nodes)
        variable_type: type of the variable (scalar, ...)
        part_offsets: dictionary mapping part IDs to offset to 'part' line in file
        part_element_offsets: for per-element variables, this holds a dictionary
            mapping ``(part ID, element type)`` tuples to offset to 'element type' line in file
        part_per_node_undefined_values: for per-node variables, this holds a dictionary
            mapping part IDs to value that should be considered as undefined
            (``coordinates undef``)
        part_per_element_undefined_values: for per-element variables, this holds a dictionary
            mapping ``(part ID, element type)`` tuples to value that should be considered as undefined
            (``element_type undef``)
    """
    file_path: str
    description_line: str
    variable_name: str
    variable_location: VariableLocation
    variable_type: VariableType
    part_offsets: Dict[int, int]
    part_element_offsets: Dict[Tuple[int, ElementType], int]
    geometry_file: EnsightGeometryFile
    part_per_node_undefined_values: Dict[int, float]
    part_per_element_undefined_values: Dict[Tuple[int, ElementType], float]

    def is_defined_for_part_id(self, part_id: int) -> bool:
        """Return True if variable is defined for given part, else False"""
        return part_id in self.part_offsets

    def read_node_data(self, fp: SeekableBufferedReader, part_id: int) -> Optional[Float32NDArray]:
        """
        Read per-node variable data for given part

        .. note::
            Variable is always either per-node or per-element;
            be sure to call the right method for your data.

        Args:
            fp: opened variable file object in "rb" mode
            part_id: part number for which to read data

        Returns:
            If the variable is not defined for the part, None is returned.
            If the variable is defined and is a scalar, 1D array of float32 is returned.
            Otherwise the returned value is 2D ``(n, k)`` array of float32
            where ``n`` is number of nodes and ``k`` is number of values
            based on variable type (vector, tensor).
        """
        part = self.geometry_file.parts[part_id]
        undefined_value = self.part_per_node_undefined_values.get(part_id)

        if not self.variable_location == VariableLocation.PER_NODE:
            raise ValueError("Variable is not per node")

        offset = self.part_offsets.get(part_id)
        if offset is None:
            return None

        fp.seek(offset)
        assert read_line(fp).startswith("part")
        assert read_int(fp) == part_id
        assert read_line(fp).startswith("coordinates")
        if undefined_value is not None:
            assert read_float(fp) == undefined_value

        n = part.number_of_nodes
        k = VALUES_FOR_VARIABLE_TYPE[self.variable_type]
        arr = read_floats(fp, n*k)
        if k > 1:
            arr = arr.reshape((n, k), order="F")
        return arr

    def read_element_data(self, fp: SeekableBufferedReader, part_id: int, element_type: ElementType) -> Optional[Float32NDArray]:
        """
        Read per-element variable data for given part and element type

        Due to how EnSight Gold format works, variable file mirrors the geometry file
        in that per-element values are (alas) defined separately for each element block.
        This introduces edge cases (missing element blocks so that the variable could only be
        partially defined for the part; multiple element blocks of the same element type; etc.)
        which may be tedious to handle. This implementation relies on the assumption that there
        are no repeated blocks of the same type.

        .. note::
            Variable is always either per-node or per-element;
            be sure to call the right method for your data.

        Args:
            fp: opened variable file object in "rb" mode
            part_id: part number for which to read data
            element_type: element type for which to read data (typically,
                you want to iterate over element blocks in the part and
                retrieve data for their respective element types)

        Returns:
            If the variable is not defined for the part, None is returned.
            If the variable is defined and is a scalar, 1D array of float32 is returned.
            Otherwise the returned value is 2D ``(n, k)`` array of float32
            where ``n`` is number of nodes and ``k`` is number of values
            based on variable type (vector, tensor).
        """
        part = self.geometry_file.parts[part_id]
        undefined_value = self.part_per_element_undefined_values.get((part_id, element_type))

        if not self.variable_location == VariableLocation.PER_ELEMENT:
            raise ValueError("Variable is not per element")

        offset = self.part_element_offsets.get((part_id, element_type))
        if offset is None:
            return None

        fp.seek(offset)
        assert read_line(fp).startswith(element_type.value)
        if undefined_value is not None:
            assert read_float(fp) == undefined_value

        n = part.get_number_of_elements_of_type(element_type)
        k = VALUES_FOR_VARIABLE_TYPE[self.variable_type]
        arr = read_floats(fp, n*k)
        if k > 1:
            arr = arr.reshape((n, k), order="F")
        return arr

    @classmethod
    def from_file_path(cls, file_path: str, variable_name: str, variable_location: VariableLocation,
                       variable_type: VariableType, geofile: EnsightGeometryFile) -> "EnsightVariableFile":
        """Used internally by `EnsightVariableFileSet.get_file()`"""
        part_offsets = {}

        part_element_offsets: Dict[Tuple[int, ElementType], int] = {}
        part_per_node_undefined_values: Dict[int, float] = {}
        part_per_element_undefined_values: Dict[Tuple[int, ElementType], float] = {}

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
                    if "undef" in coordinates_line:
                        undefined_value = read_float(fp)
                        part_per_node_undefined_values[part_id] = undefined_value
                    elif "partial" in coordinates_line:
                        raise EnsightReaderError(f"'coordinates partial' is not supported (part id {part_id})", fp)

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
                            raise EnsightReaderError("Bad element type", fp) from e

                        if "undef" in element_type_line:
                            undefined_value = read_float(fp)
                            part_per_element_undefined_values[part_id, element_type] = undefined_value
                        elif "partial" in element_type_line:
                            raise EnsightReaderError(f"'element_type partial' is not supported (part id {part_id})", fp)

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
            variable_name=variable_name,
            variable_location=variable_location,
            variable_type=variable_type,
            part_offsets=part_offsets,
            part_element_offsets=part_element_offsets,
            geometry_file=geofile,
            part_per_node_undefined_values=part_per_node_undefined_values,
            part_per_element_undefined_values=part_per_element_undefined_values,
        )

    def open(self) -> BinaryIO:
        """
        Return the opened file in read-only binary mode (convenience method)

        Use this in ``with`` block and pass the resulting object to ``read_node_data()`` and
        ``read_element_data()`` methods.

        Note:
            This is a simpler alternative to the similar ``mmap()`` method - it doesn't
            require you to reason about lifetime of the opened file since you will be
            getting arrays which own their data (as opposed to views into the memory-mapped file).

        Usage::

            with variable_file.open() as fp_var:
                variable_data = variable.read_node_data(fp_var, part.part_id)

        Equivalent code::

            with open(variable_file.file_path, "rb") as fp_var:
                variable_data = variable.read_node_data(fp_var, part.part_id)
        """
        return open(self.file_path, "rb")

    @contextmanager
    def mmap(self) -> Generator[_mmap.mmap, None, None]:
        """
        Return read-only memorymap of the file (convenience method)

        Use this in ``with`` block and pass the resulting object to ``read_node_data()`` and
        ``read_element_data()`` methods.

        Note:
            This is preferred over the similar ``open()`` method if you don't need a copy of the data
            since the returned arrays will be backed by the memorymap. Be careful to keep the memorymap
            around as long as you need the arrays.

        Usage::

            with variable_file.mmap() as mm_var:
                variable_data = variable.read_node_data(mm_var, part.part_id)

        Equivalent code::

            with open(variable_file.file_path, "rb") as fp_var, _mmap.mmap(fp_var.fileno(), 0, access=mmap.ACCESS_READ) as mm_var:
                variable_data = variable.read_node_data(mm_var, part.part_id)
        """
        with open(self.file_path, "rb") as fp, _mmap.mmap(fp.fileno(), 0, access=_mmap.ACCESS_READ) as mm:
            yield mm

    @contextmanager
    def mmap_writable(self) -> Generator[_mmap.mmap, None, None]:
        """
        Return writable memorymap of the file (convenience method)

        Use this in ``with`` block and pass the resulting object to ``read_node_data()`` and
        ``read_element_data()`` methods.

        Note:
            This special version of the ``mmap()`` method can be used if you wish
            to modify the underlying file. Use carefully.

        Usage::

            with geometry_file.mmap_writable() as fp_geo:
                nodes = part.read_nodes(fp_geo)
                nodes[:, 0] = 0.0  # set X coordinate to zero for part nodes

        Equivalent code::

            with open(geometry_file.file_path, "r+b") as fp_geo, _mmap.mmap(fp_geo.fileno(), 0, access=mmap.ACCESS_WRITE) as mm_geo:
                nodes = part.read_nodes(mm_geo)

        Usage::

            with variable_file.mmap_writable() as mm_var:
                variable_data = variable.read_node_data(mm_var, part.part_id)
                variable_data[:] = np.sqrt(variable_data)  # apply square root function to the data

        Equivalent code::

            with open(variable_file.file_path, "r+b") as fp_var, _mmap.mmap(fp_var.fileno(), 0, access=mmap.ACCESS_WRITE) as mm_var:
                variable_data = variable.read_node_data(mm_var, part.part_id)
                variable_data[:] = np.sqrt(variable_data)  # apply square root function to the data
        """
        with open(self.file_path, "r+b") as fp, _mmap.mmap(fp.fileno(), 0, access=_mmap.ACCESS_WRITE) as mm:
            yield mm


def fill_wildcard(filename: str, value: int) -> str:
    return re.sub(r"\*+", lambda m: str(value).zfill(len(m.group(0))), filename, count=1)


def strip_quotes(filename: str) -> str:
    return re.sub('"(.+)"', r"\1", filename)


@dataclass
class EnsightGeometryFileSet:
    """
    Helper object for loading geometry files

    This is used by `EnsightCaseFile.get_geometry_model()` to give you `EnsightGeometryFile`.

    Attributes:
        casefile_dir_path: path to casefile directory (root for relative paths in casefile)
        timeset: time set in which the geometry is defined, or None if it's not transient
        filename: path to the data file(s), including ``*`` wildcards if transient
        changing_geometry_per_part: whether parts contain information about type of transient changes
    """
    casefile_dir_path: str
    timeset: Optional[Timeset]
    filename: str
    changing_geometry_per_part: bool = False

    def get_file(self, timestep: int = 0) -> EnsightGeometryFile:
        """Return geometry for given timestep (use 0 if not transient)"""
        if self.timeset is None:
            timestep_filename = self.filename
        else:
            timestep_filename = fill_wildcard(self.filename, self.timeset.filename_numbers[timestep])

        path = op.join(self.casefile_dir_path, timestep_filename)
        return EnsightGeometryFile.from_file_path(path, changing_geometry_per_part=self.changing_geometry_per_part)


@dataclass
class EnsightVariableFileSet:
    """
    Helper object for loading variable files

    This is used by `EnsightCaseFile.get_variable()` to give you `EnsightVariableFile`.

    Attributes:
        casefile_dir_path: path to casefile directory (root for relative paths in casefile)
        timeset: time set in which the variable is defined, or None if it's not transient
        variable_location: where the variable is defined (elements or nodes)
        variable_type: type of the variable (scalar, ...)
        variable_name: name of the variable ('description' field in casefile)
        filename: path to the data file(s), including ``*`` wildcards if transient
    """
    casefile_dir_path: str
    timeset: Optional[Timeset]
    variable_location: VariableLocation
    variable_type: VariableType
    variable_name: str
    filename: str

    def get_file(self, geofile: EnsightGeometryFile, timestep: int = 0) -> EnsightVariableFile:
        """
        Return variable for given timestep (use 0 if not transient)

        .. note::
            Due to how EnSight Gold format works, you need to have matching geofile
            already parsed before you attempt to read the variable file. The variable
            file itself (alas) does not give lengths of the arrays it contains.

            For transient geometry combined with transient variable, pay extra care because
            they have to match - different timesteps can have different number of elements,
            nodes, etc.

        """
        # XXX be sure to match geofile and timestep!
        if self.timeset is None:
            timestep_filename = self.filename
        else:
            timestep_filename = fill_wildcard(self.filename, self.timeset.filename_numbers[timestep])

        path = op.join(self.casefile_dir_path, timestep_filename)
        return EnsightVariableFile.from_file_path(path, variable_name=self.variable_name,
                                                  variable_location=self.variable_location,
                                                  variable_type=self.variable_type, geofile=geofile)


@dataclass
class EnsightConstantVariable:
    """
    Constant per case variable

    Represents ``constant per case`` or ``constant per case file`` variable.

    Attributes:
        timeset: time set in which the variable is defined, or None if it's not transient
        variable_name: name of the variable ('description' field in casefile)
        values: list of values (it has length 1 for non-transient variables; for transient
            variable its length corresponds to number of timesteps)
    """
    timeset: Optional[Timeset]
    variable_name: str
    values: List[float]

    def get_value(self, timestep: int = 0) -> float:
        """Return constant value at given timestep (for non-transient constants, use timestep 0)"""
        return self.values[timestep]

    @classmethod
    def from_casefile_line(cls, key: str, values: List[str], casefile_dir_path: str) -> Tuple["EnsightConstantVariable",
                                                                                              Optional[int]]:
        if "file" in key:
            # constant per case file
            if len(values) == 2:
                ts = None
                variable_name = values[0]
                cvfilename = values[1]

                return cls(
                    timeset=None,
                    variable_name=variable_name,
                    values=read_numbers_from_text_file(op.join(casefile_dir_path, cvfilename), float)
                ), ts
            elif len(values) == 3:
                ts = int(values[0])
                variable_name = values[1]
                cvfilename = values[2]

                return cls(
                    timeset=None,
                    variable_name=variable_name,
                    values=read_numbers_from_text_file(op.join(casefile_dir_path, cvfilename), float)
                ), ts
            else:
                raise ValueError("Unsupported constant variable line")
        else:
            # constant per case
            if len(values) == 2:
                ts = None
                variable_name = values[0]
                variable_value = float(values[1])
                return cls(
                    timeset=None,
                    variable_name=variable_name,
                    values=[variable_value]
                ), ts
            elif len(values) < 2:
                raise ValueError("Unsupported constant variable line")
            else:
                ts = int(values[0])
                variable_name = values[1]
                variable_values = [float(x) for x in values[2:]]
                return cls(
                    timeset=None,
                    variable_name=variable_name,
                    values=variable_values
                ), ts


def read_numbers_from_text_file(path: str, type_: Type[T]) -> List[T]:
    output: List[T] = []
    with open(path) as fp:
        for line in fp:
            values = line.split()
            output.extend(map(type_, values))
    return output


@dataclass
class EnsightCaseFile:
    """
    EnSight Gold case file

    To load a case, use:

        >>> import ensightreader
        >>> case = ensightreader.read_case("example.case")
        >>> geofile = case.get_geometry_model()
        >>> velocity_variable = case.get_variable("U")

    EnSight Gold casefile is text file with description of the case and links to
    data files where the actual geometry and variables are stored.

    Since the case may be transient and therefore have multiple
    data files for the same variable, etc., there is an indirection:
    the actual data is accessed using `EnsightGeometryFile` and
    `EnsightVariableFile` objects, which are constructed using
    `EnsightGeometryFileSet`, `EnsightVariableFileSet` helper objects
    inside `EnsightCaseFile`.

    .. note::
        - for geometry, only ``model`` is supported (no ``measured``,
          ``match`` or ``boundary``)
        - only unstructured grid (``coordinates``) geometry is supported;
          structured parts are not supported
        - filesets and single-file cases are not supported; only ``C Binary``
          files are supported
        - ``change_coords_only`` is not supported
        - ``changing_geometry_per_part`` is only read and stored, not handled
          in reading geometry data
        - some variable types are unsupported, see `VariableType`
        - ghost cells are not supported
        - cases with more than one time set are supported,
          but you may not be able to use `EnsightCaseFile.get_variable()`
          since the library expects geometry and variable time sets to match;
          if you know which geofile belongs to which variable file you
          can read the variable files using `EnsightVariableFileSet.get_file()`.

    Attributes:
        casefile_path: path to the ``*.case`` file
        geometry_model: accessor to the ``model`` geometry data
        variables: dictionary mapping variable names keys to variable data accessors
        constant_variables: dictionary mapping constant variable names keys to `EnsightConstantVariable`
        timesets: dictionary mapping time set IDs to time set description objects

    """

    casefile_path: str
    geometry_model: EnsightGeometryFileSet
    variables: Dict[str, EnsightVariableFileSet]  # variable name -> EnsightVariableFileSet
    constant_variables: Dict[str, EnsightConstantVariable]  # variable name -> EnsightConstantVariable
    timesets: Dict[int, Timeset]  # time set ID -> TimeSet
    _geometry_file_cache: Dict[int, EnsightGeometryFile] = field(default_factory=dict, repr=False)
    _variable_file_cache: Dict[Tuple[int, str], EnsightVariableFile] = field(default_factory=dict, repr=False)

    def get_geometry_model(self, timestep: int = 0) -> EnsightGeometryFile:
        """
        Get geometry for given timestep

        .. note::
            The returned `EnsightGeometryFile` is cached, so it will not
            parse the file again when you request the geometry again.

        Args:
            timestep: number of timestep, starting from zero (for non-transient
                geometry, use 0)
        """
        if timestep not in self._geometry_file_cache:
            self._geometry_file_cache[timestep] = self.geometry_model.get_file(timestep)
        return self._geometry_file_cache[timestep]

    def get_variable(self, name: str, timestep: int = 0) -> EnsightVariableFile:
        """
        Get variable for given timestep

        .. note::
            The returned `EnsightVariableFile` is cached, so it will not
            parse the file again when you request the variable again.

        Args:
            name: name of the variable
            timestep: number of timestep, starting from zero (for non-transient
                variable, use 0)
        """
        cache_key = (timestep, name)
        if cache_key not in self._variable_file_cache:
            variable_fileset = self.variables.get(name)
            if variable_fileset is None:
                raise KeyError(f"No variable named {name!r} (present variables: {list(self.variables.keys())})")

            geofile_timeset = self.geometry_model.timeset
            variable_timeset = variable_fileset.timeset
            if geofile_timeset is not None and variable_timeset is not None:
                if geofile_timeset.timeset_id != variable_timeset.timeset_id:
                    raise NotImplementedError(f"Geometry and variable {name!r} use different timesets, "
                                              f"this is currently not handled (if you want this, please call "
                                              f"EnsightVariableFileSet.get_file() with the correct geofile)")

            geofile = self.get_geometry_model(timestep)
            self._variable_file_cache[cache_key] = self.variables[name].get_file(geofile=geofile, timestep=timestep)
        return self._variable_file_cache[cache_key]

    def get_constant_variable_value(self, name: str, timestep: int = 0) -> float:
        """
        Get value of constant per-case variable for given timestep

        Args:
            name: name of the constant variable
            timestep: number of timestep, starting from zero (for non-transient
                variable, use 0)

        """
        return self.constant_variables[name].get_value(timestep)

    def is_transient(self) -> bool:
        """Return True if the case is transient (has at least one time set defined)"""
        return len(self.timesets) > 0

    def get_node_variables(self) -> List[str]:
        """Return names of variables defined per-node"""
        return [name for name, variable in self.variables.items()
                if variable.variable_location == VariableLocation.PER_NODE]

    def get_element_variables(self) -> List[str]:
        """Return names of variables defined per-element"""
        return [name for name, variable in self.variables.items()
                if variable.variable_location == VariableLocation.PER_ELEMENT]

    def get_constant_variables(self) -> List[str]:
        """Return names of constant variables defined per-case"""
        return list(self.constant_variables.keys())

    def get_time_values(self, timeset_id: Optional[int] = None) -> Optional[List[float]]:
        """
        Return time values for given time set

        This will return time values as specified in the case file (this could be
        physical time in seconds, or something else). Timesteps ``0``, ``1``, ... occur
        at times ``case.get_time_values()[0]``, ``case.get_time_values()[1]``, ...

        Args:
            timeset_id: time set ID for which to return time values - if there are
                less than two time sets, you can omit this parameter

        Returns:
            List of time values, or None if there are no time sets defined

        """
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
        """
        Return list of variable names

        Note that this does not include per-case constants.

        """
        return list(self.variables.keys())

    @classmethod
    def from_file(cls, casefile_path: str) -> "EnsightCaseFile":
        """
        Read EnSight Gold case

        .. note::
            This only reads the casefile, not any data files.
            Any geometry or variable data that you want to read
            must be read explicitly.

        Args:
            casefile_path: path to the ``*.case`` file

        """
        casefile_dir_path = op.dirname(casefile_path)
        geometry_model = None
        geometry_model_ts: Optional[int] = None
        variables: Dict[str, EnsightVariableFileSet] = {}
        constant_variables: Dict[str, EnsightConstantVariable] = {}
        variables_ts: Dict[str, Optional[int]] = {}
        timesets: Dict[int, Timeset] = {}

        with open(casefile_path) as fp:
            current_section = None
            current_timeset: Optional[Timeset] = None
            current_timeset_file_start_number = None
            current_timeset_filename_increment = None
            changing_geometry_per_part = False
            last_key = None

            for lineno, line in enumerate(fp, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if line.isupper():
                    current_section = line
                    continue

                if ":" in line:
                    key, values_ = line.split(":", maxsplit=1)
                    values: List[str] = values_.split()
                    last_key = key
                else:
                    key = None
                    values = line.split()

                if current_section == "FORMAT":
                    if key == "type" and values != ["ensight", "gold"]:
                        raise EnsightReaderError("Expected 'ensight gold' in type line", fp, lineno)
                elif current_section == "GEOMETRY":
                    if key == "model":
                        if "change_coords_only" in values:
                            raise EnsightReaderError("Model with 'change_coords_only' is not supported", fp, lineno)
                        if values[-1] == "changing_geometry_per_part":
                            changing_geometry_per_part = True
                            values.pop()
                        if len(values) == 1:
                            filename, = values
                            filename = strip_quotes(filename)

                            if "*" in filename:
                                corrected_line = f"{key}: 1 {' '.join(values)}"
                                warnings.warn(EnsightReaderWarning(
                                    f"Geometry model looks transient, but no timeset is given (did you mean: '{corrected_line}'?)",
                                    fp, lineno
                                ))

                            geometry_model = EnsightGeometryFileSet(
                                casefile_dir_path,
                                timeset=None,
                                filename=filename,
                                changing_geometry_per_part=changing_geometry_per_part)
                        elif len(values) == 2:
                            ts_, filename = values
                            filename = strip_quotes(filename)
                            geometry_model_ts = int(ts_)
                            geometry_model = EnsightGeometryFileSet(
                                casefile_dir_path,
                                timeset=None,  # timeset will be added later
                                filename=filename,
                                changing_geometry_per_part=changing_geometry_per_part)
                        else:
                            raise EnsightReaderError("Unsupported model definition (note: fs is not supported)",
                                                     fp, lineno)
                    # note: measured, match, boundary geometries are not supported
                elif current_section == "VARIABLE":
                    try:
                        variable_type_, variable_location_ = key.split(" per ")  # type: ignore[union-attr]

                        if variable_type_ == "constant" and variable_location_.startswith("case"):
                            constant_variable, ts = EnsightConstantVariable.from_casefile_line(key, values, casefile_dir_path)  # type: ignore[arg-type]
                            constant_variables[constant_variable.variable_name] = constant_variable
                            variables_ts[constant_variable.variable_name] = ts
                            continue

                        variable_type = VariableType(variable_type_)
                        variable_location = VariableLocation(variable_location_)
                    except ValueError:
                        print(f"Warning: unsupported variable line ({casefile_path}:{lineno}), skipping")
                        continue

                    filename = values[-1]
                    filename = strip_quotes(filename)
                    description = values[-2]
                    ts = None
                    if len(values) == 3:
                        ts = int(values[0])
                    elif len(values) > 3:
                        print(f"Warning: unsupported variable line ({casefile_path}:{lineno}), skipping")
                        continue

                    if "*" in filename and ts is None:
                        corrected_line = f"{key}: 1 {' '.join(values)}"
                        warnings.warn(EnsightReaderWarning(
                            f"Variable {description} looks transient, but no timeset is given (did you mean: '{corrected_line}'?)",
                            fp, lineno
                        ))

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
                        timesets[ts] = current_timeset = Timeset(
                            timeset_id=ts,
                            description=description,
                            number_of_steps=-1,
                            filename_numbers=[],
                            time_values=[])
                        current_timeset_file_start_number = None
                        current_timeset_filename_increment = None
                    elif key == "number of steps":
                        current_timeset.number_of_steps = int(values[0])  # type: ignore[union-attr]
                    elif key == "filename start number":
                        current_timeset_file_start_number = int(values[0])
                        if current_timeset_file_start_number is not None and current_timeset_filename_increment is not None:
                            current_timeset.filename_numbers = Timeset.filename_numbers_from_arithmetic_sequence(
                                file_start_number=current_timeset_file_start_number,
                                number_of_steps=current_timeset.number_of_steps,
                                filename_increment=current_timeset_filename_increment
                            )
                    elif key == "filename increment":
                        current_timeset_filename_increment = int(values[0])
                        if current_timeset_file_start_number is not None and current_timeset_filename_increment is not None:
                            current_timeset.filename_numbers = Timeset.filename_numbers_from_arithmetic_sequence(  # type: ignore[union-attr]
                                file_start_number=current_timeset_file_start_number,
                                number_of_steps=current_timeset.number_of_steps,  # type: ignore[union-attr]
                                filename_increment=current_timeset_filename_increment
                            )
                    elif key == "time values":
                        current_timeset.time_values.extend(map(float, values))  # type: ignore[union-attr]
                    elif key == "filename numbers":
                        current_timeset.filename_numbers.extend(map(int, values))  # type: ignore[union-attr]
                    elif key == "filename numbers file":
                        path = op.join(casefile_dir_path, values[0])
                        path = strip_quotes(path)
                        current_timeset.filename_numbers = read_numbers_from_text_file(path, int)  # type: ignore[union-attr]
                    elif key == "time values file":
                        path = op.join(casefile_dir_path, values[0])
                        path = strip_quotes(path)
                        current_timeset.time_values = read_numbers_from_text_file(path, float)  # type: ignore[union-attr]
                    elif key is None:
                        if last_key == "time values":
                            current_timeset.time_values.extend(map(float, values))  # type: ignore[union-attr]
                        elif last_key == "filename numbers":
                            current_timeset.filename_numbers.extend(map(int, values))  # type: ignore[union-attr]
                        else:
                            print(f"Warning: unsupported time line ({casefile_path}:{lineno}), skipping")
                            continue
                    else:
                        print(f"Warning: unsupported time line ({casefile_path}:{lineno}), skipping")
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
                if variable_name in variables:
                    variables[variable_name].timeset = timesets[variable_ts]
                elif variable_name in constant_variables:
                    constant_variables[variable_name].timeset = timesets[variable_ts]

        return cls(
            casefile_path=casefile_path,
            geometry_model=geometry_model,
            variables=variables,
            constant_variables=constant_variables,
            timesets=timesets,
        )

    def to_string(self) -> str:
        """
        Return .case file contents as a string

        This method is useful if you wish to modify the case,
        eg. add/remove variables, apply offset to time values, etc.

        .. note::
            This method works by serializing the internal representation,
            meaning that any lines in the original .case file that were
            skipped due to missing support from the library (as well as comments)
            will not appear in the output at all.

        """
        case_lines = [
            "FORMAT",
            "type: ensight gold",
            "",
        ]

        # model
        case_lines.append("GEOMETRY")
        model_line = ["model:"]
        if self.geometry_model.timeset:
            model_line.append(str(self.geometry_model.timeset.timeset_id))
        model_line.append(self.geometry_model.filename)
        if self.geometry_model.changing_geometry_per_part:
            model_line.append("changing_geometry_per_part")
        case_lines.append(" ".join(model_line))
        case_lines.append("")

        # variables
        if self.variables:
            case_lines.append("VARIABLE")
            for constant_variable in self.constant_variables.values():
                variable_line = ["constant per case:"]
                if constant_variable.timeset:
                    variable_line.append(str(constant_variable.timeset.timeset_id))
                variable_line.append(constant_variable.variable_name)
                variable_line.append(" ".join(f"{x:g}" for x in constant_variable.values))
                case_lines.append(" ".join(variable_line))

            for variable in self.variables.values():
                variable_line = [f"{variable.variable_type} per {variable.variable_location}:"]
                if variable.timeset:
                    variable_line.append(str(variable.timeset.timeset_id))
                variable_line.append(variable.variable_name)
                variable_line.append(variable.filename)
                case_lines.append(" ".join(variable_line))

            case_lines.append("")

        # time
        if self.timesets:
            case_lines.append("TIME")

            for timeset in self.timesets.values():
                case_lines.append(f"time set:              {timeset.timeset_id}{' '+timeset.description if timeset.description else ''}")
                case_lines.append(f"number of steps:       {timeset.number_of_steps}")

                if len(timeset.filename_numbers) >= 2:
                    timeset_file_start_number = timeset.filename_numbers[0]
                    timeset_filename_increment = timeset.filename_numbers[1] - timeset.filename_numbers[0]
                else:
                    timeset_file_start_number = 0
                    timeset_filename_increment = 1

                if timeset.filename_numbers == Timeset.filename_numbers_from_arithmetic_sequence(
                                                   file_start_number=timeset_file_start_number,
                                                   number_of_steps=timeset.number_of_steps,
                                                   filename_increment=timeset_filename_increment
                                               ):
                    case_lines.append(f"filename start number: {timeset_file_start_number}")
                    case_lines.append(f"filename increment:    {timeset_filename_increment}")
                else:
                    case_lines.append("filename numbers:")
                    for i in range(0, len(timeset.filename_numbers), 6):
                        case_lines.append(" ".join(f"{x}" for x in timeset.filename_numbers[i:i+6]))

                case_lines.append("time values:")
                for i in range(0, len(timeset.time_values), 6):
                    case_lines.append(" ".join(f"{x:g}" for x in timeset.time_values[i:i+6]))

        return "\n".join(case_lines)

    def to_file(self, casefile_path: str) -> None:
        """
        Write EnSight Gold case to file

        See `EnsightCaseFile.to_string()` for more.

        Args:
            casefile_path: path to the ``*.case`` file
        """
        text = self.to_string()
        with open(casefile_path, "w") as fp:
            fp.write(text)


def read_case(path: str) -> EnsightCaseFile:
    """
    Read EnSight Gold case

    This will only parse the case file, not any data files.
    Use the returned object to load whichever data you need.

    Args:
        path: Path to the ``*.case`` file

    Returns:
        `EnsightCaseFile` object

    """
    return EnsightCaseFile.from_file(path)
