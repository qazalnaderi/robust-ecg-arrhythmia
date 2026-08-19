"""Epoch sampler for fair clean-vs-noise-augmented training.

Dataset layout is assumed to be:

    [clean beat 0 ... clean beat N-1]
    [noisy beat 0 ... noisy beat N-1]

For every epoch, exactly one version of each underlying heartbeat
is selected.

Therefore:
    - epoch size stays identical to clean baseline: N
    - every heartbeat appears exactly once per epoch
    - exactly 50% of selected samples are clean
    - exactly 50% are noisy
"""

import torch
from torch.utils.data import Sampler


class PairedCleanNoisyEpochSampler(
    Sampler[int]
):
    """Select one member of every clean/noisy pair per epoch."""

    def __init__(
        self,
        clean_count: int,
        noisy_count: int,
        seed: int = 42,
    ) -> None:

        if clean_count <= 0:
            raise ValueError(
                "clean_count must be positive."
            )

        if clean_count != noisy_count:
            raise ValueError(
                "Clean and noisy counts must be identical."
            )

        if clean_count % 2 != 0:
            raise ValueError(
                "clean_count must be even to obtain "
                "an exact 50/50 epoch."
            )

        self.clean_count = int(
            clean_count
        )

        self.noisy_count = int(
            noisy_count
        )

        self.seed = int(
            seed
        )

        self.epoch = 0

    def set_epoch(
        self,
        epoch: int,
    ) -> None:
        """Set epoch so sampling changes reproducibly."""

        self.epoch = int(
            epoch
        )

    def __len__(
        self,
    ) -> int:

        return self.clean_count

    def __iter__(
        self,
    ):

        generator = torch.Generator()

        generator.manual_seed(
            self.seed
            + self.epoch
        )

        # Random permutation of underlying heartbeat pairs.
        pair_order = torch.randperm(
            self.clean_count,
            generator=generator,
        )

        half = (
            self.clean_count
            // 2
        )

        # Half of underlying beats are represented by
        # their noisy counterpart this epoch.
        noisy_pairs = pair_order[
            :half
        ]

        # The other half remain clean.
        clean_pairs = pair_order[
            half:
        ]

        # Noisy half begins after all clean samples.
        selected_indices = torch.cat(
            (
                clean_pairs,
                noisy_pairs
                + self.clean_count,
            )
        )

        if len(
            selected_indices
        ) != self.clean_count:

            raise RuntimeError(
                "Unexpected epoch sample count."
            )

        # Shuffle the final mixture so clean/noisy examples
        # are not presented in blocks.
        final_order = torch.randperm(
            self.clean_count,
            generator=generator,
        )

        selected_indices = (
            selected_indices[
                final_order
            ]
        )

        return iter(
            selected_indices.tolist()
        )