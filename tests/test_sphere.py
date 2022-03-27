from ensightreader import EnsightCaseFile, EnsightGeometryFile, GeometryPart, IdHandling


def test_read_sphere_case():
    path = "./data/sphere/sphere.case"

    # check casefile
    case = EnsightCaseFile.from_file(path)

    assert case.get_variables() == ["RTData"]
    assert case.get_node_variables() == ["RTData"]
    assert case.get_element_variables() == []

    assert not case.is_transient()
    assert case.get_time_values() is None

    # check geofile
    geofile = case.get_geometry_model()
    assert isinstance(geofile, EnsightGeometryFile)

    assert geofile.description_line1.startswith("Written by VTK EnSight Writer")
    assert geofile.description_line2.startswith("No Title was Specified")
    assert geofile.node_id_handling == IdHandling.GIVEN
    assert geofile.element_id_handling == IdHandling.GIVEN
    assert geofile.extents is None

    assert geofile.get_part_names() == ["VTK Part"]
    part = geofile.get_part_by_name("VTK Part")
    assert isinstance(part, GeometryPart)
    assert part.part_name == "VTK Part"
    assert geofile.parts[part.part_id] is part
    assert part.part_id == 1

    # TODO nodes
    # TODO elements
    # TODO variables
