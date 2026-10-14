"""
Maps CWRU .mat filenames to fault labels.

CWRU's file naming isn't self-describing, so you need to fill this in from
the table on the official download page:
https://engineering.case.edu/bearingdatacenter/download-data-file

Fault classes (4-class version, most common in papers):
    0 = Normal (no fault)
    1 = Inner Race fault
    2 = Outer Race fault
    3 = Ball fault

Fill in FILENAME_TO_LABEL below once you've downloaded your files and noted
which fault each one corresponds to. Example entries are given — replace/
extend with the actual files you download. It's fine to start with a subset
(e.g., just the 12k DE, 0 HP load files) to keep the class-balance simple,
then expand later.
"""

FILENAME_TO_LABEL = {
    # --- Normal baseline ---
    # "97.mat": 0,
    # "98.mat": 0,
    # "99.mat": 0,
    # "100.mat": 0,

    # --- Inner race faults (0.007", 0.014", 0.021") ---
    # "105.mat": 1,
    # "106.mat": 1,
    # "107.mat": 1,
    # "108.mat": 1,

    # --- Outer race faults ---
    # "130.mat": 2,
    # "131.mat": 2,
    # "132.mat": 2,
    # "133.mat": 2,

    # --- Ball faults ---
    # "118.mat": 3,
    # "119.mat": 3,
    # "120.mat": 3,
    # "121.mat": 3,
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
