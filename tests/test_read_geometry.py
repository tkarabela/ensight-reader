import numpy as np

from ensightreader import EnsightGeometryFile, GeometryPart, UnstructuredElementBlock, ElementType


def test_read_geometry(tmp_path):
    output_geofile_path = tmp_path / "surface_geometry.geo"

    node_coordinates = np.random.uniform(size=(10, 3)).astype(np.float32)
    triangles_connectivity = np.asarray([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.int32)
    polygon_node_counts = np.asarray([3, 3, 3, 5, 4], dtype=np.int32)
    polygon_connectivity = np.asarray([
        0, 1, 2,
        1, 2, 3,
        2, 3, 4,
        5, 6, 7, 8, 9,
        1, 3, 5, 7,
    ], dtype=np.int32)

    with open(output_geofile_path, "wb") as fp:
        EnsightGeometryFile.write_header(fp)
        GeometryPart.write_part_header(fp, part_id=1, part_name="TestSurfaceGeometry", node_coordinates=node_coordinates)
        UnstructuredElementBlock.write_element_block(fp, ElementType.TRIA3, triangles_connectivity)
        UnstructuredElementBlock.write_element_block_nsided(fp, polygon_node_counts, polygon_connectivity)

    geofile = EnsightGeometryFile.from_file_path(output_geofile_path, False)
    with geofile.mmap() as mm:
        part = geofile.get_part_by_id(1)

        tria_block = part.get_element_block(ElementType.TRIA3)

        tmp = tria_block.read_connectivity(mm)
        assert np.array_equal(tmp, triangles_connectivity)

        tmp = tria_block.read_connectivity(mm, np.asarray([0, 1, 2], dtype=np.int32))
        assert np.array_equal(tmp, triangles_connectivity)

        tmp = tria_block.read_connectivity(mm, np.asarray([0, 1], dtype=np.int32))
        assert np.array_equal(tmp, np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int32))

        tmp = tria_block.read_connectivity(mm, np.asarray([2, 0], dtype=np.int32))
        assert np.array_equal(tmp, np.asarray([[2, 3, 4], [0, 1, 2]], dtype=np.int32))

        poly_block = part.get_element_block(ElementType.NSIDED)

        tmp_counts, tmp_conn = poly_block.read_connectivity_nsided(mm)
        assert np.array_equal(tmp_counts, polygon_node_counts)
        assert np.array_equal(tmp_conn, polygon_connectivity)

        tmp_counts, tmp_conn = poly_block.read_connectivity_nsided(mm, np.asarray([0, 1, 2, 3, 4], dtype=np.int32))
        assert np.array_equal(tmp_counts, polygon_node_counts)
        assert np.array_equal(tmp_conn, polygon_connectivity)

        tmp_counts, tmp_conn = poly_block.read_connectivity_nsided(mm, np.asarray([3, 2, 0], dtype=np.int32))
        assert np.array_equal(tmp_counts, np.asarray([5, 3, 3], dtype=np.int32))
        assert np.array_equal(tmp_conn, np.asarray([5, 6, 7, 8, 9, 2, 3, 4, 0, 1, 2], dtype=np.int32))
