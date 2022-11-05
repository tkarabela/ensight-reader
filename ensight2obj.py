#!/usr/bin/env python

"""
ensight2obj script
==================

This script converts surface elements of EnSight Gold parts
into OBJ format (text). EnSight parts are represented as OBJ groups.

Demonstrates reading steady-state geometry, node coordinates,
connectivity using traditional I/O.

For commandline usage, run the script with ``--help``.

"""

import argparse
import re
import sys
from typing import Optional

import numpy as np

import ensightreader


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ensight_case", metavar="*.case", help="input EnSight Gold case (C Binary)")
    parser.add_argument("output_obj", metavar="*.obj", help="output OBJ file (text)")
    parser.add_argument("--only-parts", metavar="regex", help="only export parts matching given "
                                                              "regular expression (Python re.search)")

    args = parser.parse_args()
    ensight_case_path = args.ensight_case
    output_obj_path = args.output_obj
    part_name_regex = args.only_parts

    return ensight2obj(ensight_case_path, output_obj_path, part_name_regex)


def ensight2obj(ensight_case_path: str, output_obj_path: str, part_name_regex: Optional[str] = None) -> int:
    """Main function of ensight2obj.py"""

    print("Reading input EnSight case", ensight_case_path)
    case = ensightreader.read_case(ensight_case_path)
    geofile = case.get_geometry_model()

    print("I see", len(geofile.get_part_names()), "parts in case")
    parts = []
    for part_id, part in geofile.parts.items():
        if not part.is_surface():
            print("Skipping part", part.part_name, "(not a surface part)")
        elif part_name_regex and not re.search(part_name_regex, part.part_name):
            print("Skipping part", part.part_name, "(name doesn't match)")
        else:
            parts.append(part)

    print("Reading nodes...")
    node_arrays = []

    with geofile.open() as fp_geo:
        for part in parts:
            node_array = part.read_nodes(fp_geo)
            node_arrays.append(node_array)

    all_nodes = np.vstack(node_arrays)
    number_of_nodes = all_nodes.shape[0]

    print("Writing output OBJ", output_obj_path)

    # OBJ uses uses global vertex numbering, starting from 1.
    # EnSight uses per-part vertex numbering, starting from 1.
    # To accommodate this, we need to increment the IDs for subsequent EnSight parts.
    node_id_offset = 0

    with open(output_obj_path, "w") as fp_obj, geofile.open() as fp_geo:
        print(f"Writing {number_of_nodes} nodes...", flush=True)
        for i in range(number_of_nodes):
            print("v", *all_nodes[i], file=fp_obj)

        for i, part in enumerate(parts):
            print(f"Writing part {part.part_name}...", flush=True)
            print("g", part.part_name, file=fp_obj)  # translate EnSight parts to OBJ groups

            for block in part.element_blocks:
                if block.element_type.dimension != 2:
                    print(f"\tSkipping {block.number_of_elements} {block.element_type.value} elements", flush=True)
                    continue

                print(f"\tWriting {block.number_of_elements} {block.element_type.value} elements", flush=True)
                print("#", block.element_type.value, file=fp_obj)

                if block.element_type == block.element_type.NSIDED:
                    polygon_node_counts, polygon_connectivity = block.read_connectivity_nsided(fp_geo)
                    polygon_connectivity += node_id_offset
                    k = 0
                    for j in range(len(polygon_node_counts)):
                        node_count = polygon_node_counts[j]
                        print("f", *(polygon_connectivity[k:k + node_count]), file=fp_obj)
                        k += node_count
                else:
                    connectivity = block.read_connectivity(fp_geo)
                    connectivity += node_id_offset
                    for j in range(connectivity.shape[0]):
                        print("f", *(connectivity[j]), file=fp_obj)

            node_id_offset += part.number_of_nodes

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
