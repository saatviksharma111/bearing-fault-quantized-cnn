"""
Maps CWRU .mat filenames to fault labels.

Fault classes (4-class version):
    0 = Normal (no fault)
    1 = Inner Race fault
    2 = Outer Race fault
    3 = Ball fault

This mapping is filled in based on the 16 files downloaded from the CWRU
Bearing Data Center: Normal baseline (97-100), Inner Race 0.007" (105-108),
Outer Race 0.007" centered (130-133), and Ball 0.007" (118-121) - each
across all four motor loads (0-3 HP).
"""

FILENAME_TO_LABEL = {
    # --- Normal baseline (0 HP, 1 HP, 2 HP, 3 HP) ---
    "97.mat": 0,
    "98.mat": 0,
    "99.mat": 0,
    "100.mat": 0,

    # --- Inner race faults, 0.007" (0 HP, 1 HP, 2 HP, 3 HP) ---
    "105.mat": 1,
    "106.mat": 1,
    "107.mat": 1,
    "108.mat": 1,

    # --- Outer race faults, 0.007" centered @6:00 (0 HP, 1 HP, 2 HP, 3 HP) ---
    "130.mat": 2,
    "131.mat": 2,
    "132.mat": 2,
    "133.mat": 2,

    # --- Ball faults, 0.007" (0 HP, 1 HP, 2 HP, 3 HP) ---
    "118.mat": 3,
    "119.mat": 3,
    "120.mat": 3,
    "121.mat": 3,
}

LABEL_NAMES = {
    0: "Normal",
    1: "Inner Race Fault",
    2: "Outer Race Fault",
    3: "Ball Fault",
}

# CWRU .mat files store the signal under a key like 'X097_DE_time' where the
# number matches the filename. This helper finds that key automatically
# instead of hardcoding it per file.
def find_de_signal_key(mat_dict):
    """Return the key in a loaded .mat dict corresponding to the Drive-End
    (DE) vibration channel, e.g. 'X097_DE_time'."""
    candidates = [k for k in mat_dict.keys() if k.endswith("DE_time")]
    if not candidates:
        raise KeyError(
            f"No '*_DE_time' key found in mat file. Keys present: {list(mat_dict.keys())}"
        )
    return candidates[0]
