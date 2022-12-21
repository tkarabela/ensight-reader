#!/usr/bin/env python

r"""
ensight_transform script
========================

This script does **in-place** transformation of node coordinates
in given EnSight Gold case. Your original geofile will be modified!

Examples:

::

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


For commandline usage, run the script with ``--help``.

"""

import argparse
import re
import sys
from typing import Optional

import numpy as np
import numpy.typing as npt

import ensightreader

Float32NDArray = npt.NDArray[np.float32]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ensight_case", metavar="*.case", help="EnSight Gold case (C Binary)")
    parser.add_argument("--only-parts", metavar="regex", help="only export parts matching given "
                                                              "regular expression (Python re.search)")

    action = parser.add_mutually_exclusive_group()
    action.add_argument("--translate", nargs=3, type=float, metavar=tuple("dX dY dZ".split()),
                        help="translate nodes by given dX, dY, dZ values")
    action.add_argument("--scale", nargs=3, type=float, metavar=tuple("sX sY sZ".split()),
                        help="scale nodes by given sX, sY, sZ values")
    action.add_argument("--matrix", nargs=16, type=float, metavar=tuple("a11 a12 a13 a14 a21 a22 a23 a24 a31 a32 a33 a34 a41 a42 a43 a44".split()),
                        help="do affine transformation of nodes by multiplying via the (a_ij) matrix")

    args = parser.parse_args()
    ensight_case_path = args.ensight_case
    part_name_regex = args.only_parts
    translate = args.translate
    scale = args.scale
    matrix = args.matrix

    if translate is not None:
        translate = np.asarray(translate)
    if scale is not None:
        scale = np.asarray(scale)
    if matrix is not None:
        matrix = np.asarray(matrix).reshape((4, 4))

    return ensight_transform(ensight_case_path=ensight_case_path,
                             translate=translate,
                             scale=scale,
                             matrix=matrix,
                             part_name_regex=part_name_regex)


def ensight_transform(ensight_case_path: str,
                      translate: Optional[Float32NDArray] = None,
                      scale: Optional[Float32NDArray] = None,
                      matrix: Optional[Float32NDArray] = None,
                      part_name_regex: Optional[str] = None) -> int:
    """Main function of ensight_transform.py"""

    print("Reading input EnSight case", ensight_case_path)
    case = ensightreader.read_case(ensight_case_path)
    geofile = case.get_geometry_model()

    print("I see", len(geofile.get_part_names()), "parts in case")
    parts = []
    for part_id, part in geofile.parts.items():
        if part_name_regex and not re.search(part_name_regex, part.part_name):
            print("Skipping part", part.part_name, "(name doesn't match)")
        else:
            parts.append(part)

    print("Transforming nodes...")
    if translate is not None:
        print("Translate by", translate)
    if scale is not None:
        print("Scale by", scale)
    if matrix is not None:
        print("Affine transformation", matrix, sep="\n")

    with geofile.mmap_writable() as mm_geo:
        for part in parts:
            node_array = part.read_nodes(mm_geo)
            N = node_array.shape[0]

            if translate is not None:
                node_array[:, 0] += translate[0]
                node_array[:, 1] += translate[1]
                node_array[:, 2] += translate[2]

            if scale is not None:
                node_array[:, 0] *= scale[0]
                node_array[:, 1] *= scale[1]
                node_array[:, 2] *= scale[2]

            if matrix is not None:
                tmp = np.empty((N, 4), dtype=np.float32)
                tmp[:, :3] = node_array
                tmp[:, 3] = 1.0

                tmp = tmp.dot(matrix)
                node_array[:, :3] = tmp[:, :3]

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
