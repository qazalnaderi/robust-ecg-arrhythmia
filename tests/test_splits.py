from src.data.splits import (
    DS1_RECORDS,
    DS2_RECORDS,
    PACED_EXCLUDED_RECORDS,
    get_dataset_split,
)


def test_expected_split_sizes():
    assert len(DS1_RECORDS) == 23
    assert len(DS2_RECORDS) == 21

def test_ds1_and_ds2_do_not_overlap():
    assert set(DS1_RECORDS).isdisjoint(DS2_RECORDS)


def test_paced_records_are_not_in_core_splits():
    core_records = set(DS1_RECORDS) | set(DS2_RECORDS)

    assert core_records.isdisjoint(
        PACED_EXCLUDED_RECORDS
    )


def test_core_split_contains_44_records():
    core_records = set(DS1_RECORDS) | set(DS2_RECORDS)

    assert len(core_records) == 44


def test_record_split_lookup():
    assert get_dataset_split("101") == "DS1"
    assert get_dataset_split("100") == "DS2"
    assert get_dataset_split("102") == "excluded"

def test_same_subject_records_stay_in_same_split():
    assert get_dataset_split("201") == "DS1"
    assert get_dataset_split("202") == "DS1"