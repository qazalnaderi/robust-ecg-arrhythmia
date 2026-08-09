"""AAMI heartbeat class mapping for the MIT-BIH Arrhythmia Database."""

from typing import Optional


# MIT-BIH beat annotations grouped according to the commonly used
# AAMI heartbeat classification scheme.
AAMI_CLASS_MAP: dict[str, str] = {
    # N: Normal / non-ectopic beats
    "N": "N",  # Normal beat
    "L": "N",  # Left bundle branch block beat
    "R": "N",  # Right bundle branch block beat
    "e": "N",  # Atrial escape beat
    "j": "N",  # Nodal (junctional) escape beat

    # S: Supraventricular ectopic beats
    "A": "S",  # Atrial premature beat
    "a": "S",  # Aberrated atrial premature beat
    "J": "S",  # Nodal (junctional) premature beat
    "S": "S",  # Supraventricular premature/ectopic beat

    # V: Ventricular ectopic beats
    "V": "V",  # Premature ventricular contraction
    "E": "V",  # Ventricular escape beat

    # F: Fusion beats
    "F": "F",  # Fusion of ventricular and normal beat

    # Q: Unknown / paced-related beats
    "/": "Q",  # Paced beat
    "f": "Q",  # Fusion of paced and normal beat
    "Q": "Q",  # Unclassifiable beat
}


AAMI_CLASSES: tuple[str, ...] = ("N", "S", "V", "F", "Q")


def map_to_aami(symbol: str) -> Optional[str]:
    """
    Convert a MIT-BIH beat annotation symbol to its AAMI class.

    Parameters
    ----------
    symbol:
        Original MIT-BIH annotation symbol.

    Returns
    -------
    str | None
        One of N, S, V, F, Q for supported beat annotations.
        Returns None for annotations that are not part of the
        heartbeat classification mapping.
    """
    return AAMI_CLASS_MAP.get(symbol)