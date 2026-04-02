#!/usr/bin/env python3

"""
obj2ensight script
==================

This script converts OBJ (text) geometry file into EnSight Gold case.

For commandline usage, run the script with ``--help``.

"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from ensightreader import (
    EnsightCaseFile,
    GeometryPart,
    ElementType,
    UnstructuredElementBlock,
    VariableLocation,
    VariableType
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_obj", metavar="*.obj", help="input OBJ file (text)", type=Path)
    parser.add_argument("ensight_case", metavar="*.case", help="output EnSight Gold case (C Binary)", type=Path)
    parser.add_argument("--add-random-variables", action="store_true", help="add dummy variables of different types")

    args = parser.parse_args()
    ensight_case_path = args.ensight_case
    input_obj_path = args.input_obj
    add_random_variables = bool(args.add_random_variables)

    return obj2ensight(ensight_case_path, input_obj_path, add_random_variables)


def obj2ensight(ensight_case_path: Path, input_obj_path: Path, add_random_variables: bool = False) -> int:
    """Main function of obj2ensight.py"""

    nodes: list[float] = []
    faces: dict[int, list[int]] = {}

    print("Reading input OBJ", input_obj_path)
    object_id = 0
    with input_obj_path.open() as fp:
        for line in fp:
            tag, *data = line.split()
            if tag == "v":
                nodes.append(float(data[0]))
                nodes.append(float(data[1]))
                nodes.append(float(data[2]))
            elif tag == "f":
                indices = [object_id] + [int(x.split("/")[0]) for x in data]
                faces.setdefault(len(indices)-1, []).append(indices)
            elif tag == "o":
                object_id += 1

    case = EnsightCaseFile.create_empty_case(ensight_case_path)
    geofile = case.get_geometry_model()
    object_id_variable = case.define_variable(
        VariableLocation.PER_ELEMENT,
        VariableType.SCALAR,
        "object_id",
        str(ensight_case_path.with_suffix(".object_id").name)
    )
    part_id = 1
    arr_per_element = {}

    with geofile.open_writeable() as fp:
        fp.seek(0, os.SEEK_END)
        node_arr = np.asarray(nodes, dtype=np.float32).reshape((-1, 3))
        GeometryPart.write_part_header(fp, part_id, "obj", node_arr)

        if triangles := faces.pop(3, []):
            arr = np.asarray(triangles, dtype=np.int32).reshape((-1, 4))
            UnstructuredElementBlock.write_element_block(fp, ElementType.TRIA3, arr[:, 1:])
            arr_per_element[ElementType.TRIA3] = arr[:, 0].flatten().astype(np.float32)

        if quads := faces.pop(4, []):
            arr = np.asarray(quads, dtype=np.int32).reshape((-1, 5))
            UnstructuredElementBlock.write_element_block(fp, ElementType.QUAD4, arr[:, 1:])
            arr_per_element[ElementType.QUAD4] = arr[:, 0].flatten().astype(np.float32)

        polygon_object_ids = []
        polygon_node_counts = []
        polygon_connectivity = []
        for k, data in faces.items():
            for indices in data:
                polygon_node_counts.append(k)
                polygon_object_ids.append(indices[0])
                polygon_connectivity.extend(indices[1:])

        polygon_object_ids_arr = np.asarray(polygon_object_ids, dtype=np.float32)
        polygon_node_counts_arr = np.asarray(polygon_node_counts, dtype=np.int32)
        polygon_connectivity_arr = np.asarray(polygon_connectivity, dtype=np.int32)
        UnstructuredElementBlock.write_element_block_nsided(fp, polygon_node_counts_arr, polygon_connectivity_arr)
        arr_per_element[ElementType.NSIDED] = polygon_object_ids_arr

    geofile.reload_from_file()
    part = geofile.get_part_by_id(part_id)

    with object_id_variable.open_writeable() as fp:
        fp.seek(0, os.SEEK_END)
        object_id_variable.write_element_data(fp, part_id, arr_per_element)

    if add_random_variables:
        random_element_scalar_variable = case.define_variable(
            VariableLocation.PER_ELEMENT,
            VariableType.SCALAR,
            "random_element_scalar",
            str(ensight_case_path.with_suffix(".random_element_scalar").name)
        )
        with random_element_scalar_variable.open_writeable() as fp:
            random_element_scalar_variable.ensure_data_for_all_parts(fp)
        with random_element_scalar_variable.mmap_writable() as mm:
            for block in part.element_blocks:
                arr = random_element_scalar_variable.read_element_data(mm, part_id, block.element_type)
                arr[:] = np.random.random(arr.shape)

        random_element_vector_variable = case.define_variable(
            VariableLocation.PER_ELEMENT,
            VariableType.VECTOR,
            "random_element_vector",
            str(ensight_case_path.with_suffix(".random_element_vector").name)
        )
        with random_element_vector_variable.open_writeable() as fp:
            random_element_vector_variable.ensure_data_for_all_parts(fp)
        with random_element_vector_variable.mmap_writable() as mm:
            for block in part.element_blocks:
                arr = random_element_vector_variable.read_element_data(mm, part_id, block.element_type)
                arr[:] = np.random.random(arr.shape)

        random_node_scalar_variable = case.define_variable(
            VariableLocation.PER_NODE,
            VariableType.SCALAR,
            "random_node_scalar",
            str(ensight_case_path.with_suffix(".random_node_scalar").name)
        )
        with random_node_scalar_variable.open_writeable() as fp:
            random_node_scalar_variable.ensure_data_for_all_parts(fp)
        with random_node_scalar_variable.mmap_writable() as mm:
            arr = random_node_scalar_variable.read_node_data(mm, part_id)
            arr[:] = np.random.random(arr.shape)

        random_node_vector_variable = case.define_variable(
            VariableLocation.PER_NODE,
            VariableType.VECTOR,
            "random_node_vector",
            str(ensight_case_path.with_suffix(".random_node_vector").name)
        )
        with random_node_vector_variable.open_writeable() as fp:
            random_node_vector_variable.ensure_data_for_all_parts(fp)
        with random_node_vector_variable.mmap_writable() as mm:
            arr = random_node_vector_variable.read_node_data(mm, part_id)
            arr[:] = np.random.random(arr.shape)

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
