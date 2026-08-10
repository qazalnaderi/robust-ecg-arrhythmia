"""Patient-wise split definitions for MIT-BIH."""

DS1_RECORDS = (
    "101", "106", "108", "109", "112", "114",
    "115", "116", "118", "119", "122", "124",
    "201", "202", "203", "205", "207", "208",
    "209", "215", "220", "223", "230",
)

DS2_RECORDS = (
    "100", "103", "105", "111", "113", "117",
    "121", "123", "200", "210", "212", "213",
    "214", "219", "221", "222", "228", "231",
    "232", "233", "234",
)

PACED_EXCLUDED_RECORDS = (
    "102", "104", "107", "217",
)


def get_dataset_split(record_id: str) -> str:
    """Return the patient-level dataset split for one MIT-BIH record."""

    if record_id in DS1_RECORDS:
        return "DS1"

    if record_id in DS2_RECORDS:
        return "DS2"

    if record_id in PACED_EXCLUDED_RECORDS:
        return "excluded"

    raise ValueError(
        f"Unknown MIT-BIH record: {record_id}"
    )