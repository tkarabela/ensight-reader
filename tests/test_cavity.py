import numpy as np
from ensightreader import read_case, EnsightGeometryFile, GeometryPart, IdHandling, ElementType, VariableLocation, \
    VariableType


def test_read_cavity_case():
    path = "./data/cavity/cavity.case"

    # check casefile
    case = read_case(path)
    VARIABLE_NAMES = ["U", "p"]
    TIME_VALUES = [0.00000e+00, 1.00000e-01, 2.00000e-01, 3.00000e-01, 4.00000e-01, 5.00000e-01]
    TIMESTEPS = list(range(len(TIME_VALUES)))

    assert case.get_variables() == VARIABLE_NAMES
    assert case.get_node_variables() == []
    assert case.get_element_variables() == VARIABLE_NAMES

    assert case.is_transient()
    assert case.get_time_values() == TIME_VALUES

    # check geofile
    geofile = case.get_geometry_model()
    assert isinstance(geofile, EnsightGeometryFile)

    assert geofile.description_line1.startswith("Ensight Geometry File")
    assert geofile.description_line2.startswith("Written by OpenFOAM 2012")
    assert geofile.node_id_handling == IdHandling.ASSIGN
    assert geofile.element_id_handling == IdHandling.ASSIGN
    assert geofile.extents is None

    assert geofile.get_part_names() == ["internalMesh", "movingWall", "fixedWalls"]

    part_internalMesh = geofile.get_part_by_name("internalMesh")
    part_movingWall = geofile.get_part_by_name("movingWall")
    part_fixedWalls = geofile.get_part_by_name("fixedWalls")
    parts = [part_internalMesh, part_movingWall, part_fixedWalls]

    assert part_internalMesh.part_id == 1
    assert part_movingWall.part_id == 2
    assert part_fixedWalls.part_id == 3

    assert part_internalMesh.number_of_nodes == 882
    assert part_movingWall.number_of_nodes == 42
    assert part_fixedWalls.number_of_nodes == 122

    assert part_internalMesh.number_of_elements == 400
    assert part_movingWall.number_of_elements == 20
    assert part_fixedWalls.number_of_elements == 60

    assert len(part_internalMesh.element_blocks) == 1
    assert len(part_movingWall.element_blocks) == 1
    assert len(part_fixedWalls.element_blocks) == 1

    assert part_internalMesh.element_blocks[0].element_type == ElementType.HEXA8
    assert part_movingWall.element_blocks[0].element_type == ElementType.QUAD4
    assert part_fixedWalls.element_blocks[0].element_type == ElementType.QUAD4

    with open(geofile.file_path, "rb") as fp_geo:
        # TODO check node and connectivity values
        for part in parts:
            nodes = part.read_nodes(fp_geo)
            assert nodes.shape == (part.number_of_nodes, 3)
            assert part.read_node_ids(fp_geo) is None

            for block in part.element_blocks:
                connectivity = block.read_connectivity(fp_geo)
                assert connectivity.shape == (part.number_of_elements, block.element_type.nodes_per_element)

    # check variables
    assert case.variables["p"].variable_type == VariableType.SCALAR
    assert case.variables["U"].variable_type == VariableType.VECTOR

    for variable_name in VARIABLE_NAMES:
        for timestep in TIMESTEPS:
            variable = case.get_variable(variable_name, timestep)
            assert variable.variable_location == VariableLocation.PER_ELEMENT

            with open(variable.file_path, "rb") as fp_var:
                # TODO check variable values
                for part in parts:
                    for block in part.element_blocks:
                        variable_data = variable.read_element_data(fp_var, part.part_id, block.element_type)
                        assert variable_data.shape[0] == block.number_of_elements
