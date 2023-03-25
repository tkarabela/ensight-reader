import pytest

from ensightreader import read_case, VariableLocation, VariableType, EnsightReaderWarning

ENSIGHT_CASE_PATH = "./data/dummy_case_file/dummy_case_file.case"
ENSIGHT_CASE_PATH_WITH_UNMARKED_TIMESETS = "./data/dummy_case_file/dummy_case_file_with_unmarked_timesets.case"


def test_read_dummy_case_file():
    case = read_case(ENSIGHT_CASE_PATH)

    # geometry
    assert case.geometry_model.filename == "*****.geo"
    assert case.geometry_model.timeset.timeset_id == 1
    assert case.geometry_model.changing_geometry_per_part

    # variables
    assert case.get_constant_variable_value("my_steady_constant") == 1.234
    assert case.get_constant_variable_value("my_transient_constant", 0) == 2.0
    assert case.get_constant_variable_value("my_transient_constant", 1) == 2.1
    assert case.get_constant_variable_value("my_transient_constant", 2) == 2.2
    assert case.get_constant_variable_value("my_transient_constant_in_file", 0) == 3.0
    assert case.get_constant_variable_value("my_transient_constant_in_file", 1) == 3.1
    assert case.get_constant_variable_value("my_transient_constant_in_file", 2) == 3.2

    assert case.variables["my_transient_scalar_per_element"].variable_location == VariableLocation.PER_ELEMENT
    assert case.variables["my_transient_scalar_per_element"].variable_type == VariableType.SCALAR
    assert case.variables["my_transient_scalar_per_element"].timeset.timeset_id == 1
    assert case.variables["my_transient_vector_per_element"].variable_location == VariableLocation.PER_ELEMENT
    assert case.variables["my_transient_vector_per_element"].variable_type == VariableType.VECTOR
    assert case.variables["my_transient_vector_per_element"].timeset.timeset_id == 1
    assert case.variables["my_transient_tensorsymm_per_element"].variable_location == VariableLocation.PER_ELEMENT
    assert case.variables["my_transient_tensorsymm_per_element"].variable_type == VariableType.TENSOR_SYMM
    assert case.variables["my_transient_tensorsymm_per_element"].timeset.timeset_id == 1
    assert case.variables["my_transient_tensorasym_per_element"].variable_location == VariableLocation.PER_ELEMENT
    assert case.variables["my_transient_tensorasym_per_element"].variable_type == VariableType.TENSOR_ASYM
    assert case.variables["my_transient_tensorasym_per_element"].timeset.timeset_id == 1

    assert case.variables["my_steady_scalar_per_node"].variable_location == VariableLocation.PER_NODE
    assert case.variables["my_steady_scalar_per_node"].variable_type == VariableType.SCALAR
    assert case.variables["my_steady_scalar_per_node"].timeset is None
    assert case.variables["my_steady_vector_per_node"].variable_location == VariableLocation.PER_NODE
    assert case.variables["my_steady_vector_per_node"].variable_type == VariableType.VECTOR
    assert case.variables["my_steady_vector_per_node"].timeset is None
    assert case.variables["my_steady_tensorsymm_per_node"].variable_location == VariableLocation.PER_NODE
    assert case.variables["my_steady_tensorsymm_per_node"].variable_type == VariableType.TENSOR_SYMM
    assert case.variables["my_steady_tensorsymm_per_node"].timeset is None
    assert case.variables["my_steady_tensorasym_per_node"].variable_location == VariableLocation.PER_NODE
    assert case.variables["my_steady_tensorasym_per_node"].variable_type == VariableType.TENSOR_ASYM
    assert case.variables["my_steady_tensorasym_per_node"].timeset is None

    # timesets
    ts1 = case.timesets[1]
    ts2 = case.timesets[2]
    ts3 = case.timesets[3]
    ts4 = case.timesets[4]

    assert case.get_time_values(1) == [100.0, 100.1, 100.2]
    assert case.get_time_values(2) == [100.0, 100.1, 100.2]
    assert case.get_time_values(3) == [100.0, 100.1, 100.2]
    assert case.get_time_values(4) == [100.0, 100.1, 100.2]

    assert ts1.filename_numbers == [100, 200, 300]
    assert ts2.filename_numbers == [100, 200, 300]
    assert ts3.filename_numbers == [100, 200, 300]
    assert ts4.filename_numbers == [100, 220, 300]

    assert ts4.description == "non_arithmetic_progression_timeset"

def test_write_dummy_case_file():
    case = read_case(ENSIGHT_CASE_PATH)
    text = case.to_string()
    print(text)
    assert text.strip() == """
FORMAT
type: ensight gold

GEOMETRY
model: 1 *****.geo changing_geometry_per_part

VARIABLE
constant per case: my_steady_constant 1.234
constant per case: 1 my_transient_constant 2 2.1 2.2
constant per case: 1 my_transient_constant_in_file 3 3.1 3.2
scalar per element: 1 my_transient_scalar_per_element *****.my_transient_scalar_per_element
vector per element: 1 my_transient_vector_per_element *****.my_transient_vector_per_element
tensor symm per element: 1 my_transient_tensorsymm_per_element *****.my_transient_tensorsymm_per_element
tensor asym per element: 1 my_transient_tensorasym_per_element *****.my_transient_tensorasym_per_element
scalar per node: my_steady_scalar_per_node 00000.my_steady_scalar_per_node
vector per node: my_steady_vector_per_node 00000.my_steady_vector_per_node
tensor symm per node: my_steady_tensorsymm_per_node 00000.my_steady_tensorsymm_per_node
tensor asym per node: my_steady_tensorasym_per_node 00000.my_steady_tensorasym_per_node

TIME
time set:              1
number of steps:       3
filename start number: 100
filename increment:    100
time values:
100 100.1 100.2
time set:              2
number of steps:       3
filename start number: 100
filename increment:    100
time values:
100 100.1 100.2
time set:              3
number of steps:       3
filename start number: 100
filename increment:    100
time values:
100 100.1 100.2
time set:              4 non_arithmetic_progression_timeset
number of steps:       3
filename numbers:
100 220 300
time values:
100 100.1 100.2
""".strip()


def test_read_dummy_case_file_with_unmarked_timesets():
    with pytest.warns(EnsightReaderWarning) as records:
        read_case(ENSIGHT_CASE_PATH_WITH_UNMARKED_TIMESETS)

    assert len(records) == 5
    assert str(records[0].message) == "Geometry model looks transient, but no timeset is given (did you mean: 'model: 1 *****.geo'?) (path=./data/dummy_case_file/dummy_case_file_with_unmarked_timesets.case, line=7)"
    assert str(records[1].message) == "Variable my_transient_scalar_per_element looks transient, but no timeset is given (did you mean: 'scalar per element: 1 my_transient_scalar_per_element *****.my_transient_scalar_per_element'?) (path=./data/dummy_case_file/dummy_case_file_with_unmarked_timesets.case, line=11)"
    assert str(records[2].message) == "Variable my_transient_vector_per_element looks transient, but no timeset is given (did you mean: 'vector per element: 1 my_transient_vector_per_element *****.my_transient_vector_per_element'?) (path=./data/dummy_case_file/dummy_case_file_with_unmarked_timesets.case, line=12)"
    assert str(records[3].message) == "Variable my_transient_tensorsymm_per_element looks transient, but no timeset is given (did you mean: 'tensor symm per element: 1 my_transient_tensorsymm_per_element *****.my_transient_tensorsymm_per_element'?) (path=./data/dummy_case_file/dummy_case_file_with_unmarked_timesets.case, line=13)"
    assert str(records[4].message) == "Variable my_transient_tensorasym_per_element looks transient, but no timeset is given (did you mean: 'tensor asym per element: 1 my_transient_tensorasym_per_element *****.my_transient_tensorasym_per_element'?) (path=./data/dummy_case_file/dummy_case_file_with_unmarked_timesets.case, line=14)"
