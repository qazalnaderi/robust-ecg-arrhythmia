from src.noise.nstdb import (
    VALID_NOISE_TYPES,
    load_noise_record,
)


def main() -> None:
    print("=" * 60)
    print("NSTDB NOISE CHECK")
    print("=" * 60)

    for noise_type in VALID_NOISE_TYPES:
        signal, fs = load_noise_record(noise_type)

        print(f"\nNoise type: {noise_type}")
        print(f"Sampling frequency: {fs} Hz")
        print(f"Signal shape: {signal.shape}")
        print(f"Number of channels: {signal.shape[1]}")

        print(
            f"Finite values: "
            f"{signal.size == (signal == signal).sum()}"
        )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()