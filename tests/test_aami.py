from src.data.aami import AAMI_CLASSES, map_to_aami


def test_normal_class_mapping():
    assert map_to_aami("N") == "N"
    assert map_to_aami("L") == "N"
    assert map_to_aami("R") == "N"
    assert map_to_aami("e") == "N"
    assert map_to_aami("j") == "N"


def test_supraventricular_class_mapping():
    assert map_to_aami("A") == "S"
    assert map_to_aami("a") == "S"
    assert map_to_aami("J") == "S"
    assert map_to_aami("S") == "S"


def test_ventricular_class_mapping():
    assert map_to_aami("V") == "V"
    assert map_to_aami("E") == "V"


def test_fusion_class_mapping():
    assert map_to_aami("F") == "F"


def test_unknown_class_mapping():
    assert map_to_aami("/") == "Q"
    assert map_to_aami("f") == "Q"
    assert map_to_aami("Q") == "Q"


def test_unmapped_annotation_returns_none():
    assert map_to_aami("+") is None


def test_expected_aami_classes():
    assert AAMI_CLASSES == ("N", "S", "V", "F", "Q")