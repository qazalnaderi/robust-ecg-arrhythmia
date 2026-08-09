from pathlib import Path

import wfdb


DATA_DIR = Path("data/raw/mitdb")
TEST_RECORD = "100"


def main() -> None:
    record_path = DATA_DIR / TEST_RECORD

    # Read ECG signal and header
    record = wfdb.rdrecord(str(record_path))

    # Read expert beat annotations
    annotation = wfdb.rdann(
        str(record_path),
        extension="atr",
    )

    print("=" * 60)
    print("MIT-BIH SMOKE TEST")
    print("=" * 60)

    print(f"Record: {TEST_RECORD}")
    print(f"Sampling frequency: {record.fs} Hz")
    print(f"Number of channels: {record.n_sig}")
    print(f"Signal names: {record.sig_name}")
    print(f"Signal shape: {record.p_signal.shape}")
    print(f"Signal length: {record.sig_len}")

    print("-" * 60)

    print(f"Number of annotations: {len(annotation.sample)}")
    print(
        "First 20 annotation symbols:",
        annotation.symbol[:20],
    )

    print(
        "First 20 annotation samples:",
        annotation.sample[:20],
    )

    print("=" * 60)


if __name__ == "__main__":
    main()