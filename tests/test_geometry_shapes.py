from pathlib import Path

import obj2ensight


OBJ_CASE_PATH = Path("./data/geometry_shapes/obj/geometry_shapes.obj")
REFERENCE_ENSIGHT_DIR = Path("./data/geometry_shapes/case")


def test_obj2ensight(tmp_path):
    output_case_path = tmp_path.joinpath("geometry_shapes.case")
    assert 0 == obj2ensight.obj2ensight(output_case_path, OBJ_CASE_PATH)

    for reference_file in REFERENCE_ENSIGHT_DIR.glob("*"):
        generated_file = tmp_path.joinpath(reference_file.name)
        assert generated_file.is_file()
        assert generated_file.read_bytes() == reference_file.read_bytes()
