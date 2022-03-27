#!/usr/bin/env python

import re
import sys
from ensightreader import EnsightCaseFile, ElementType, VariableType
import argparse
import os.path as op


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ensight_case", metavar="*.case", help="input EnSight Gold case (C Binary)")
    parser.add_argument("output_vtk", metavar="*.vtk", help="output VTK file (text)")
    parser.add_argument("--only-parts", metavar="regex", help="only export parts matching given "
                                                              "regular expression (Python re.search)")

    args = parser.parse_args()
    ensight_case_path = args.ensight_case
    output_vtk_path_given = args.output_vtk
    part_name_regex = args.only_parts

    output_vtk_prefix, _ = op.splitext(output_vtk_path_given)

    print("Reading input EnSight case", ensight_case_path)
    case = EnsightCaseFile.from_file(ensight_case_path)
    geofile = case.get_geometry_model()

    print("I see", len(geofile.get_part_names()), "parts in case")
    part_ids = []
    for part_id, part in geofile.parts.items():
        if part_name_regex and not re.search(part_name_regex, part.part_name):
            print("Skipping part", part.part_name, "(name doesn't match)")
        else:
            part_ids.append(part_id)

    for part_id in part_ids:
        part = geofile.parts[part_id]
        part_name = part.part_name

        vtk_output_path = f"{output_vtk_prefix}_{part_name}.vtk"
        print("Writing part", part_name)
        write_vtk_part(case, part_id, vtk_output_path)

    print("\nAll done.")
    return 0


VTK_ELEMENT_TYPES = {
    ElementType.POINT:       1,  # VTK_VERTEX
    ElementType.BAR2:        2,  # VTK_LINE
    ElementType.TRIA3:       5,  # VTK_TRIANGLE
    ElementType.QUAD4:       9,  # VTK_QUAD
    ElementType.TETRA4:     10,  # VTK_TETRA
    ElementType.PYRAMID5:   14,  # VTK_PYRAMID
    ElementType.PENTA6:     13,  # VTK_WEDGE
    ElementType.HEXA8:      12,  # VTK_HEXAHEDRON
    ElementType.NSIDED:      7,  # VTK_POLYGON
}


def write_vtk_part(case: EnsightCaseFile, part_id: int, vtk_output_path: str):
    geofile = case.get_geometry_model()
    part = geofile.parts[part_id]

    with open(vtk_output_path, "w") as fp_vtk, open(geofile.file_path, "rb") as fp_geo:
        print("# vtk DataFile Version 2.0", file=fp_vtk)
        print("ensight2vtk output file", file=fp_vtk)
        print("ASCII", file=fp_vtk)

        print("DATASET UNSTRUCTURED_GRID", file=fp_vtk)
        print("POINTS", part.number_of_nodes, "float", file=fp_vtk)
        nodes = geofile.read_nodes(part_id)
        for i in range(len(nodes)):
            print(*nodes[i], file=fp_vtk)

        cell_list_size = 0
        for block in part.element_blocks:
            if block.element_type not in VTK_ELEMENT_TYPES:
                raise NotImplementedError(f"Exporting element type {block.element_type} is not supported")
            if block.element_type != ElementType.NSIDED:
                cell_list_size += block.number_of_elements * (1 + block.element_type.nodes_per_element)
            else:
                polygon_node_counts, _ = block.read_connectivity_nsided(fp_geo)
                cell_list_size += block.number_of_elements + polygon_node_counts.sum()

        print("CELLS", part.number_of_elements, cell_list_size, file=fp_vtk)
        for block in part.element_blocks:
            if block.element_type != ElementType.NSIDED:
                connectivity = block.read_connectivity(fp_geo)
                connectivity -= 1  # VTK numbers from 0, EnSight numbers from 1
                n, k = connectivity.shape
                for i in range(n):
                    print(k, *connectivity[i], file=fp_vtk)
            else:
                polygon_node_counts, polygon_connectivity = block.read_connectivity_nsided(fp_geo)
                polygon_connectivity -= 1
                n = polygon_node_counts.shape[0]
                k = 0
                for i in range(n):
                    node_count = polygon_node_counts[i]
                    print(node_count, *(polygon_connectivity[k:k + node_count]), file=fp_vtk)
                    k += node_count

        print("CELL_TYPES", part.number_of_elements, file=fp_vtk)
        for block in part.element_blocks:
            vtk_element_type = VTK_ELEMENT_TYPES[block.element_type]
            for i in range(block.number_of_elements):
                print(vtk_element_type, file=fp_vtk)

        point_variables = case.get_node_variables()
        cell_variables = case.get_element_variables()

        if point_variables:
            print("POINT_DATA", part.number_of_nodes, file=fp_vtk)
            for variable_name in point_variables:
                variable = case.get_variable(variable_name)
                if not variable.is_defined_for_part_id(part_id):
                    continue
                with open(variable.file_path, "rb") as fp:
                    data = variable.read_node_data(fp, part_id)

                    if variable.variable_type == VariableType.SCALAR:
                        print("SCALARS", variable_name.replace(" ", "_"), "float 1", file=fp_vtk)
                        print("LOOKUP_TABLE default", file=fp_vtk)
                        for i in range(part.number_of_nodes):
                            print(data[i], file=fp_vtk)
                    elif variable.variable_type == VariableType.VECTOR:
                        print("VECTORS", variable_name.replace(" ", "_"), "float", file=fp_vtk)
                        for i in range(part.number_of_nodes):
                            print(*data[i], file=fp_vtk)
                    else:
                        raise NotImplementedError(f"Exporting variable type {variable.variable_type} is not supported")

        if cell_variables:
            print("CELL_DATA", part.number_of_elements, file=fp_vtk)
            for variable_name in cell_variables:
                variable = case.get_variable(variable_name)
                if not variable.is_defined_for_part_id(part_id):
                    continue

                with open(variable.file_path, "rb") as fp:
                    all_data = []
                    for block in part.element_blocks:
                        data = variable.read_element_data(fp, part_id, block.element_type)
                        if data is None:
                            raise RuntimeError("Variable must be either undefined or defined for all elements")
                        all_data.append(data)

                    if variable.variable_type == VariableType.SCALAR:
                        print("SCALARS", variable_name.replace(" ", "_"), "float 1", file=fp_vtk)
                        print("LOOKUP_TABLE default", file=fp_vtk)
                        for data in all_data:
                            for i in range(data.shape[0]):
                                print(data[i], file=fp_vtk)
                    elif variable.variable_type == VariableType.VECTOR:
                        print("VECTORS", variable_name.replace(" ", "_"), "float", file=fp_vtk)
                        for data in all_data:
                            for i in range(data.shape[0]):
                                print(*data[i], file=fp_vtk)
                    else:
                        raise NotImplementedError(f"Exporting variable type {variable.variable_type} is not supported")


if __name__ == "__main__":
    sys.exit(main())
