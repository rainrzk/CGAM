from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from .tools import AffChart
from .config_utils import GRID_LABEL_TOLERANCE_MS, LEGAL_FRAC_24
ALLOWED_FRAC_24 = LEGAL_FRAC_24

def _estimate_chart_end_time(chart: AffChart, margin_ms: int=3000, end_time_ms: Optional[float]=None) -> int:
    """returns: int"""
    if end_time_ms is not None and end_time_ms > 0:
        return int(end_time_ms)
    times: List[int] = []
    times.extend([n.time for n in chart.taps])
    times.extend([n.end for n in chart.holds])
    times.extend([a.end for a in chart.arcs])
    if not times:
        if chart.timings:
            t0 = chart.timings[0].time
            bpm0 = chart.timings[0].bpm
            beat_len = 60000.0 / max(1e-06, bpm0)
            return int(t0 + beat_len * 16)
        return 60000
    return int(max(times) + margin_ms)

def build_grid(chart: AffChart, end_time_ms: Optional[float]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """returns: Tuple[np.ndarray, np.ndarray, np.ndarray]"""
    timings = sorted(chart.timings, key=lambda t: t.time)
    end_time = _estimate_chart_end_time(chart, end_time_ms=end_time_ms)
    ticks_ms: List[int] = []
    frac_idx: List[int] = []
    timing_idx: List[int] = []
    for idx, seg in enumerate(timings):
        seg_start = float(seg.time)
        bpm = float(seg.bpm)
        if bpm <= 0:
            continue
        beat_len = 60000.0 / max(1e-06, bpm)
        if idx + 1 < len(timings):
            seg_end = min(float(timings[idx + 1].time), float(end_time))
        else:
            seg_end = float(end_time)
        if seg_start >= seg_end:
            continue
        n_beats = int(np.ceil((seg_end - seg_start) / beat_len)) + 1
        for b in range(n_beats):
            base_time = seg_start + b * beat_len
            if base_time >= seg_end:
                break
            for f in ALLOWED_FRAC_24:
                frac = f / 24.0
                t = base_time + frac * beat_len
                if t >= seg_end:
                    continue
                ticks_ms.append(int(round(t)))
                frac_idx.append(int(f))
                timing_idx.append(idx)
    ticks_arr = np.array(ticks_ms, dtype=np.int32)
    frac_arr = np.array(frac_idx, dtype=np.int32)
    timing_arr = np.array(timing_idx, dtype=np.int32)
    return (ticks_arr, frac_arr, timing_arr)

def _build_segment_indices(timing_index: np.ndarray, n_segments: int) -> List[np.ndarray]:
    """returns: List[np.ndarray]"""
    seg_indices: List[np.ndarray] = []
    for i in range(n_segments):
        seg_indices.append(np.where(timing_index == i)[0])
    return seg_indices

def _find_timing_segment(timings, t_ms: float) -> int:
    """returns: int"""
    idx = 0
    while idx + 1 < len(timings) and t_ms >= float(timings[idx + 1].time):
        idx += 1
    return idx

def _quantize_time_to_tick(t_ms: float, seg_id: int, ticks_ms: np.ndarray, seg_tick_indices: List[np.ndarray], tol_ms: float) -> int | None:
    """returns: int | None"""
    idxs = seg_tick_indices[seg_id]
    if idxs.size == 0:
        return None
    seg_ticks = ticks_ms[idxs]
    pos = int(np.searchsorted(seg_ticks, t_ms))
    candidates = []
    if pos > 0:
        candidates.append(pos - 1)
    if pos < seg_ticks.size:
        candidates.append(pos)
    if not candidates:
        return None
    best_local = min(candidates, key=lambda k: abs(seg_ticks[k] - t_ms))
    best_time = float(seg_ticks[best_local])
    if abs(best_time - t_ms) < tol_ms:
        return int(idxs[best_local])
    return None

def quantize_labels(chart: AffChart, ticks_ms: np.ndarray, frac_index: np.ndarray, timing_index: np.ndarray, tol_ms: float | None=None) -> np.ndarray:
    """returns: np.ndarray"""
    if tol_ms is None:
        tol_ms = float(GRID_LABEL_TOLERANCE_MS)
    T = int(ticks_ms.shape[0])
    if T == 0:
        return np.zeros((0, 6), dtype=np.float32)
    timings = sorted(chart.timings, key=lambda t: t.time)
    n_segments = len(timings)
    is_note = np.zeros(T, dtype=np.float32)
    is_tap = np.zeros(T, dtype=np.float32)
    is_hold = np.zeros(T, dtype=np.float32)
    is_arcfalse0 = np.zeros(T, dtype=np.float32)
    is_arcfalse1 = np.zeros(T, dtype=np.float32)
    is_arctrue = np.zeros(T, dtype=np.float32)
    is_arctap = np.zeros(T, dtype=np.float32)
    seg_tick_indices = _build_segment_indices(timing_index, n_segments)
    for n in chart.taps:
        t = float(n.time)
        seg_id = _find_timing_segment(timings, t)
        j = _quantize_time_to_tick(t, seg_id, ticks_ms, seg_tick_indices, tol_ms)
        if j is None:
            continue
        is_note[j] = 1.0
        is_tap[j] = 1.0
    for n in chart.holds:
        start = float(n.start)
        end = float(n.end)
        if end < start:
            start, end = (end, start)
        seg_start = _find_timing_segment(timings, start)
        seg_end = _find_timing_segment(timings, end)
        if seg_start != seg_end:
            continue
        j_start = _quantize_time_to_tick(start, seg_start, ticks_ms, seg_tick_indices, tol_ms)
        j_end = _quantize_time_to_tick(end, seg_end, ticks_ms, seg_tick_indices, tol_ms)
        if j_start is None or j_end is None:
            continue
        a, b = sorted([j_start, j_end])
        is_note[a:b + 1] = 1.0
        is_hold[a:b + 1] = 1.0
    for a in chart.arcs:
        start = float(a.start)
        end = float(a.end)
        if end < start:
            start, end = (end, start)
        seg_start = _find_timing_segment(timings, start)
        seg_end = _find_timing_segment(timings, end)
        if seg_start != seg_end:
            continue
        j_start = _quantize_time_to_tick(start, seg_start, ticks_ms, seg_tick_indices, tol_ms)
        j_end = _quantize_time_to_tick(end, seg_end, ticks_ms, seg_tick_indices, tol_ms)
        if j_start is None or j_end is None:
            continue
        a_idx, b_idx = sorted([j_start, j_end])
        is_note[a_idx:b_idx + 1] = 1.0
        if a.is_trace:
            is_arctrue[a_idx:b_idx + 1] = 1.0
        else:
            hand = int(getattr(a, 'hand', 0))
            if hand == 1:
                is_arcfalse1[a_idx:b_idx + 1] = 1.0
            else:
                is_arcfalse0[a_idx:b_idx + 1] = 1.0
        if a.arctaps:
            arc_tick_indices = np.arange(a_idx, b_idx + 1, dtype=np.int32)
            arc_tick_times = ticks_ms[arc_tick_indices].astype(np.float32)
            for at in a.arctaps:
                t = float(at.time)
                if t < start - tol_ms or t > end + tol_ms:
                    continue
                if arc_tick_times.size == 0:
                    continue
                j_local = int(np.argmin(np.abs(arc_tick_times - t)))
                j = int(arc_tick_indices[j_local])
                if abs(float(arc_tick_times[j_local]) - t) >= tol_ms:
                    continue
                is_note[j] = 1.0
                is_arctap[j] = 1.0
    labels = np.stack([is_note, is_tap, is_hold, is_arcfalse0, is_arcfalse1, is_arctrue, is_arctap], axis=1).astype(np.float32)
    return labels
__all__ = ['ALLOWED_FRAC_24', 'build_grid', 'quantize_labels']
