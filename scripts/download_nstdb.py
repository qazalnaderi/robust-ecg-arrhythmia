from pathlib import Path

import wfdb


DATA_DIR = Path("data/raw/nstdb")

NOISE_RECORDS = ["bw", "ma", "em"]


def main() -> None:
    """Download the three raw NSTDB noise recordings."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading NSTDB noise recordings...")
    print(f"Records: {NOISE_RECORDS}")
    print(f"Destination: {DATA_DIR.resolve()}")

    wfdb.dl_database(
        db_dir="nstdb",
        dl_dir=str(DATA_DIR),
        records=NOISE_RECORDS,
        annotators=None,
        keep_subdirs=True,
        overwrite=False,
    )

    print("\nDownload complete.")


if __name__ == "__main__":
    main()