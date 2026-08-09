from pathlib import Path

import wfdb


DATA_DIR = Path("data/raw/mitdb")


def main() -> None:
    """Download the MIT-BIH Arrhythmia Database from PhysioNet."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading MIT-BIH Arrhythmia Database...")
    print(f"Destination: {DATA_DIR.resolve()}")

    wfdb.dl_database(
        db_dir="mitdb",
        dl_dir=str(DATA_DIR),
        records="all",
        annotators=["atr"],
        keep_subdirs=True,
        overwrite=False,
    )

    print("\nDownload complete.")


if __name__ == "__main__":
    main()