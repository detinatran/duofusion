"""SynthSeg (FreeSurfer LUT) label groups, hemispheres and left-right homologues."""
import numpy as np

# left label -> right label
LR_PAIRS = {2: 41, 3: 42, 4: 43, 5: 44, 7: 46, 8: 47, 10: 49, 11: 50, 12: 51, 13: 52,
            17: 53, 18: 54, 26: 58, 28: 60}
LEFT = set(LR_PAIRS)
RIGHT = set(LR_PAIRS.values())

CORTEX = {3, 42}
WM = {2, 41}
DEEP = {10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51, 52, 53, 54, 58, 60}
CEREBELLUM = {7, 8, 46, 47}
BRAINSTEM = {16}
CSF = {4, 5, 14, 15, 24, 43, 44}

# tissue classes used for references and analysis (CSF excluded)
TISSUES = {"cortex": CORTEX, "wm": WM, "deep": DEEP, "cerebellum": CEREBELLUM, "brainstem": BRAINSTEM}
TISSUE_CODE = {name: i + 1 for i, name in enumerate(TISSUES)}


def tissue_map(seg):
    """Integer tissue code per voxel (0 = background or CSF), see TISSUE_CODE."""
    seg = np.asarray(seg).astype(np.int32)
    out = np.zeros(seg.shape, np.uint8)
    for name, labs in TISSUES.items():
        out[np.isin(seg, list(labs))] = TISSUE_CODE[name]
    return out


def hemisphere_map(seg):
    """1 = left, 2 = right, 0 = midline structures or background."""
    seg = np.asarray(seg).astype(np.int32)
    out = np.zeros(seg.shape, np.uint8)
    out[np.isin(seg, list(LEFT))] = 1
    out[np.isin(seg, list(RIGHT))] = 2
    return out


def side_free_label(seg):
    """Right-hemisphere labels replaced by their left homologue, so that a label
    identifies a structure regardless of side."""
    seg = np.asarray(seg).astype(np.int32).copy()
    for left, right in LR_PAIRS.items():
        seg[seg == right] = left
    return seg
