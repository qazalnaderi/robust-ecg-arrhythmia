"""Patient-wise split definitions for the MIT-BIH Arrhythmia Database."""


DS1_RECORDS = (
    "101",
    "106",
    "108",
    "109",
    "112",
    "114",
    "115",
    "116",
    "118",
    "119",
    "122",
    "124",
    "201",
    "202",
    "203",
    "205",
    "207",
    "208",
    "209",
    "215",
    "220",
    "223",
    "230",
)

DS2_RECORDS = (
    "100",
    "103",
    "105",
    "111",
    "113",
    "117",
    "121",
    "123",
    "200",
    "210",
    "212",
    "213",
    "214",
    "219",
    "221",
    "222",
    "228",
    "231",
    "232",
    "233",
    "234",
)

PACED_EXCLUDED_RECORDS = (
    "102",
    "104",
    "107",
    "217",
)



VALIDATION_RECORDS = (
    "108",
    "114",
    "205",
    "207",
    "223",
)

TRAIN_RECORDS = tuple(
    record_id
    for record_id in DS1_RECORDS
    if record_id not in VALIDATION_RECORDS
)


def get_dataset_split(record_id: str) -> str:
    """
    Return the top-level dataset split for a MIT-BIH record.

    Parameters
    ----------
    record_id:
        MIT-BIH record identifier, for example "101".

    Returns
    -------
    str
        "DS1" for development records,
        "DS2" for final-test records,
        or "excluded" for paced-heavy records.

    Raises
    ------
    ValueError
        If the record ID is not part of the known MIT-BIH protocol.
    """

    if record_id in DS1_RECORDS:
        return "DS1"

    if record_id in DS2_RECORDS:
        return "DS2"

    if record_id in PACED_EXCLUDED_RECORDS:
        return "excluded"

    raise ValueError(
        f"Unknown MIT-BIH record: {record_id}"
    )


def get_development_split(record_id: str) -> str:
    """
    Return the train/validation assignment for a DS1 record.

    Parameters
    ----------
    record_id:
        MIT-BIH record identifier belonging to DS1.

    Returns
    -------
    str
        "train" or "validation".

    Raises
    ------
    ValueError
        If the record does not belong to the DS1 development set.
    """

    if record_id in TRAIN_RECORDS:
        return "train"

    if record_id in VALIDATION_RECORDS:
        return "validation"

    raise ValueError(
        f"Record {record_id} is not part of DS1 development data."
    )