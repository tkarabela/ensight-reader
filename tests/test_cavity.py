import os.path as op
import tempfile
import pytest

from ensight2obj import ensight2obj
from ensight2vtk import ensight2vtk
from ensightreader import (ElementType, EnsightGeometryFile, IdHandling, VariableLocation,
                           VariableType, read_case)

REPO_ROOT = op.dirname(op.dirname(__file__))
ENSIGHT_CASE_PATH = op.join(REPO_ROOT, "./data/cavity/cavity.case")
ENSIGHT_CASE_WITH_QUOTES_PATH = op.join(REPO_ROOT, "./data/cavity/cavity_with_quotes_in_filenames.case")


@pytest.mark.parametrize(
    "case_path",
    [
        ENSIGHT_CASE_PATH,
        ENSIGHT_CASE_WITH_QUOTES_PATH,
    ]
)
def test_read_cavity_case(case_path, recwarn):
    # check casefile
    case = read_case(case_path)
    assert len(recwarn) == 0

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

    with geofile.open() as fp_geo:
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

            with variable.open() as fp_var:
                # TODO check variable values
                for part in parts:
                    for block in part.element_blocks:
                        variable_data = variable.read_element_data(fp_var, part.part_id, block.element_type)
                        assert variable_data.shape[0] == block.number_of_elements


def test_write_cavity_case():
    case = read_case(ENSIGHT_CASE_PATH)
    text = case.to_string()
    assert text.strip() == """
FORMAT
type: ensight gold

GEOMETRY
model: geometry

VARIABLE
vector per element: 1 U data/********/U
scalar per element: 1 p data/********/p

TIME
time set:              1
number of steps:       6
filename start number: 0
filename increment:    20
time values:
0 0.1 0.2 0.3 0.4 0.5
""".strip()

    # check that writing to file results in the same text
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_case_path = op.join(temp_dir, "cavity.case")
        case.to_file(temp_case_path)

        with open(temp_case_path) as fp:
            assert fp.read().strip() == text.strip()


def test_cavity_case_ensight2obj():
    with tempfile.TemporaryDirectory() as temp_dir:
        # TODO check output file
        assert 0 == ensight2obj(
            ensight_case_path=ENSIGHT_CASE_PATH,
            output_obj_path=op.join(temp_dir, "cavity.obj")
        )


def test_cavity_case_ensight2vtk():
    with tempfile.TemporaryDirectory() as temp_dir:
        # TODO check output file
        assert 0 == ensight2vtk(
            ensight_case_path=ENSIGHT_CASE_PATH,
            output_vtk_path_given=op.join(temp_dir, "cavity.vtk")
        )
