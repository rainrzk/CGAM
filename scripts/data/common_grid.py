from __future__ import annotations
from typing import Tuple
import numpy as np
from ..utils.tools import AffChart
from ..utils.grid_quantizer import build_grid, quantize_labels
from ..utils.config_utils import GRID_LABEL_TOLERANCE_MS

def build_ticks_and_raw_labels(chart: AffChart, tol_ms: float | None=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """returns: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]"""
    ticks_ms, frac_index, timing_index = build_grid(chart)
    if tol_ms is None:
        tol_ms = float(GRID_LABEL_TOLERANCE_MS)
    labels_raw = quantize_labels(chart, ticks_ms, frac_index, timing_index, tol_ms=tol_ms).astype(np.float32)
    return (ticks_ms, frac_index, timing_index, labels_raw)

def compute_onset_type_from_raw_labels(labels_raw: np.ndarray) -> np.ndarray:
    """returns: np.ndarray"""
    raw = np.asarray(labels_raw, dtype=np.float32)
    T = int(raw.shape[0])
    is_tap = raw[:, 1:2]
    is_hold = raw[:, 2:3]
    is_arcfalse0 = raw[:, 3:4]
    is_arcfalse1 = raw[:, 4:5]
    is_arctrue = raw[:, 5:6]
    is_arctap = raw[:, 6:7]
    onset_type = np.zeros((T, 6), dtype=np.float32)

    def rising_edge(arr: np.ndarray) -> np.ndarray:
        """returns: np.ndarray"""
        prev = np.vstack([np.zeros((1, 1), dtype=arr.dtype), arr[:-1]])
        return (arr > 0.5) & (prev <= 0.5)
    if T > 0:
        r_tap = rising_edge(is_tap)
        r_hold = rising_edge(is_hold)
        r_arcfalse0 = rising_edge(is_arcfalse0)
        r_arcfalse1 = rising_edge(is_arcfalse1)
        r_arctrue = rising_edge(is_arctrue)
        r_arctap = rising_edge(is_arctap)
        onset_type[r_tap[:, 0], 0] = 1.0
        onset_type[r_hold[:, 0], 1] = 1.0
        onset_type[r_arcfalse0[:, 0], 2] = 1.0
        onset_type[r_arcfalse1[:, 0], 3] = 1.0
        onset_type[r_arctrue[:, 0], 4] = 1.0
        onset_type[r_arctap[:, 0], 5] = 1.0
    return onset_type
__all__ = ['build_ticks_and_raw_labels', 'compute_onset_type_from_raw_labels']
