from ensightreader import EnsightGeometryFile, GeometryPart, UnstructuredElementBlock, ElementType, read_case
import numpy as np
import tempfile
import os.path as op


ENSIGHT_CASE_PATH = "./data/cell_types/cell_types.case"
REFERENCE_GEOFILE_PATH = "./data/cell_types/cell_types.geo"


def test_create_geometry_with_every_non_ghost_cell_type():
    element_types_to_nodes = {
        ElementType.POINT: [
            [0, 0, 0],
        ],
        ElementType.BAR2: [
            [0, 0, 0],
            [1, 0, 0],
        ],
        ElementType.BAR3: [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
        ],
        ElementType.TRIA3: [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ],
        ElementType.TRIA6: [
            [0, 0, 0],
            [2, 0, 0],
            [0, 2, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ],
        ElementType.QUAD4: [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ],
        ElementType.QUAD8: [
            [0, 0, 0],
            [2, 0, 0],
            [2, 2, 0],
            [0, 2, 0],
            [1, 0, 0],
            [2, 1, 0],
            [1, 2, 0],
            [0, 1, 0],
        ],
        ElementType.TETRA4: [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        ElementType.TETRA10: [
            [0, 0, 0],  # 1
            [2, 0, 0],  # 2
            [0, 2, 0],  # 3
            [0, 0, 2],  # 4
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
        ],
        ElementType.PYRAMID5: [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        ElementType.PYRAMID13: [
            [0, 0, 0],  # 1
            [2, 0, 0],  # 2
            [2, 2, 0],  # 3
            [0, 2, 0],  # 4
            [0, 0, 2],  # 5
            [1, 0, 0],  # 6
            [2, 1, 0],  # 7
            [1, 2, 0],  # 8
            [0, 1, 0],  # 9
            [0, 0, 1],  # 10
            [1, 0, 1],  # 11
            [1, 1, 1],  # 12
            [0, 1, 1],  # 13
        ],
        ElementType.PENTA6: [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
        ],
        ElementType.PENTA15: [
            [0, 0, 0],  # 1
            [2, 0, 0],  # 2
            [0, 2, 0],  # 3
            [0, 0, 2],  # 4
            [2, 0, 2],  # 5
            [0, 2, 2],  # 6
            [1, 0, 0],  # 7
            [1, 1, 0],  # 8
            [0, 1, 0],  # 9
            [1, 0, 2],  # 10
            [1, 1, 2],  # 11
            [0, 1, 2],  # 12
            [0, 0, 1],  # 13
            [2, 0, 1],  # 14
            [0, 2, 1],  # 15
        ],
        ElementType.HEXA8: [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        ElementType.HEXA20: [
            [0, 0, 0],  # 1
            [2, 0, 0],  # 2
            [2, 2, 0],  # 3
            [0, 2, 0],  # 4
            [0, 0, 2],  # 5
            [2, 0, 2],  # 6
            [2, 2, 2],  # 7
            [0, 2, 2],  # 8
            [1, 0, 0],  # 9
            [2, 1, 0],  # 10
            [1, 2, 0],  # 11
            [0, 1, 0],  # 12
            [1, 0, 2],  # 13
            [2, 1, 2],  # 14
            [1, 2, 2],  # 15
            [0, 1, 2],  # 16
            [0, 0, 1],  # 17
            [2, 0, 1],  # 18
            [2, 2, 1],  # 19
            [0, 2, 1],  # 20
        ],
        ElementType.NSIDED: [  # hexagon
            [1, 0, 0],
            [2, 0.5, 0],
            [2, 1.5, 0],
            [1, 2, 0],
            [0, 1.5, 0],
            [0, 0.5, 0],
        ],
        ElementType.NFACED: [  # asymmetric polyhedra
            [-1.000000, 0.000000, -1.000000],
            [1.000000, 0.000000, 1.000000],
            [-0.686936, 0.686936, 1.000000],
            [-0.686936, 1.000000, 0.686936],
            [-1.000000, 0.686936, 0.686936],
            [0.686936, 1.000000, 0.686936],
            [1.000000, 0.686936, 1.000000],
            [-0.686936, 1.000000, -0.686936],
            [-1.000000, 0.686936, -1.000000],
            [0.686936, 1.000000, -0.686936],
            [1.000000, 0.686936, -0.686936],
            [0.686936, 0.686936, -1.000000],
            [0.686936, 0.000000, -1.000000],
            [1.000000, 0.000000, -0.686936],
            [-0.686936, 0.000000, 1.000000],
            [-1.000000, 0.000000, 0.686936],
        ],
    }

    element_types = list(element_types_to_nodes.keys())
    tmp = 0
    element_types_to_node_offset = {}
    for et in element_types:
        element_types_to_node_offset[et] = tmp
        tmp += len(element_types_to_nodes[et])

    polygon_node_counts = [len(element_types_to_nodes[ElementType.NSIDED])]
    polygon_connectivity = list(range(1, polygon_node_counts[0] + 1))
    _polyhedra_connectivity = [
        [1, 16, 5, 9],
        [4, 6, 10, 8],
        [13, 1, 9, 12],
        [2, 14, 11, 7],
        [15, 16, 1, 13, 14, 2],
        [3, 4, 5],
        [10, 11, 12],
        [3, 7, 6, 4],
        [9, 5, 4, 8],
        [12, 9, 8, 10],
        [13, 12, 11, 14],
        [5, 16, 15, 3],
        [7, 11, 10, 6],
        [15, 2, 7, 3],
    ]
    polyhedra_face_counts = [len(_polyhedra_connectivity)]
    face_node_counts = [len(x) for x in _polyhedra_connectivity]
    face_connectivity = []
    for x in _polyhedra_connectivity:
        face_connectivity.extend(x)

    node_block_offset = np.asarray([3, 0, 0], dtype=np.float32)

    node_coordinates_ = []
    for i, et in enumerate(element_types):
        arr = np.asarray(element_types_to_nodes[et], dtype=np.float32)
        arr += i * node_block_offset
        node_coordinates_.append(arr)

    node_coordinates = np.vstack(node_coordinates_)

    with tempfile.TemporaryDirectory("ensightreader") as temp_dir:
        output_geofile_path = op.join(temp_dir, "cell_types.geo")

        with open(output_geofile_path, "wb") as fp:
            EnsightGeometryFile.write_header(fp)
            GeometryPart.write_part_header(fp, part_id=1, part_name="TestElementTypes", node_coordinates=node_coordinates)
            for et in element_types:
                if et == ElementType.NSIDED:
                    UnstructuredElementBlock.write_element_block_nsided(
                        fp,
                        polygon_node_counts=np.asarray(polygon_node_counts, dtype=np.int32),
                        polygon_connectivity=np.asarray(polygon_connectivity, dtype=np.int32) +
                                             element_types_to_node_offset[et]
                    )
                elif et == ElementType.NFACED:
                    UnstructuredElementBlock.write_element_block_nfaced(
                        fp,
                        polyhedra_face_counts=np.asarray(polyhedra_face_counts, dtype=np.int32),
                        face_node_counts=np.asarray(face_node_counts, dtype=np.int32),
                        face_connectivity=np.asarray(face_connectivity, dtype=np.int32) + element_types_to_node_offset[et]
                    )
                else:
                    UnstructuredElementBlock.write_element_block(
                        fp,
                        element_type=et,
                        connectivity=
                        (np.asarray(range(et.nodes_per_element), dtype=np.int32) + 1 + element_types_to_node_offset[et])[
                            np.newaxis]
                    )

        # check binary
        with open(output_geofile_path, "rb") as fp:
            geofile_data = fp.read()

        with open(REFERENCE_GEOFILE_PATH, "rb") as fp:
            ref_geofile_data = fp.read()

        assert geofile_data == ref_geofile_data

        # check contents - this exercises reading all non-ghost element types
        geofile = EnsightGeometryFile.from_file_path(output_geofile_path, changing_geometry_per_part=False)
        pids = geofile.get_part_ids()
        assert len(pids) == 1
        part = geofile.get_part_by_id(pids[0])
        assert len(part.element_blocks) == len(element_types_to_nodes)
        assert part.is_surface()
        assert part.is_volume()

        with geofile.open() as fp:
            for block in part.element_blocks:
                assert block.number_of_elements == 1
                if block.element_type.has_constant_number_of_nodes_per_element():
                    connectivity = block.read_connectivity(fp)
                    assert connectivity.shape == (1, block.element_type.nodes_per_element)
                    assert connectivity.tolist() == list(sorted(connectivity.tolist()))
                    if len(connectivity) > 1:
                        assert set(list(np.diff(connectivity[0]))) == {1}
                elif block.element_type == ElementType.NSIDED:
                    polygon_node_counts_, polygon_connectivity_ = block.read_connectivity_nsided(fp)
                    assert (polygon_node_counts_ == polygon_node_counts).all()
                    assert (polygon_connectivity_ == np.add(polygon_connectivity, element_types_to_node_offset[ElementType.NSIDED])).all()
                elif block.element_type == ElementType.NFACED:
                    polyhedra_face_counts_, face_node_counts_, face_connectivity_ = block.read_connectivity_nfaced(fp)
                    assert (polyhedra_face_counts_ == polyhedra_face_counts).all()
                    assert (face_node_counts_ == face_node_counts).all()
                    assert (face_connectivity_ == np.add(face_connectivity, element_types_to_node_offset[ElementType.NFACED])).all()


def test_write_cell_types_case():
    case = read_case(ENSIGHT_CASE_PATH)
    text = case.to_string()
    assert text.strip() == """
FORMAT
type: ensight gold

GEOMETRY
model: cell_types.geo
""".strip()
