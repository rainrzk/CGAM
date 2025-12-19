from __future__ import annotations
from typing import List, Optional, Sequence
import numpy as np
from .tools import AffChart

def _estimate_chart_end_time(chart: AffChart, margin_ms: int=3000) -> int:
    """returns: int"""
    times: List[int] = []
    times.extend([n.time for n in chart.taps])
    times.extend([n.end for n in chart.holds])
    times.extend([a.end for a in chart.arcs])
    if not times:
        if chart.timings:
            t0 = chart.timings[0].time
            bpm0 = chart.timings[0].bpm
            beat_len = 60000.0 / bpm0
            return int(t0 + beat_len * 16)
        return 60000
    return int(max(times) + margin_ms)

def build_tick_grid(chart: AffChart, divisor: int=4, end_time: Optional[int]=None) -> List[int]:
    """returns: List[int]"""
    assert divisor > 0
    if not chart.timings:
        raise ValueError('AffChart.timings 为空，至少需要一段 timing 才能生成 tick 网格。')
    timings = sorted(chart.timings, key=lambda t: t.time)
    if end_time is None:
        end_time = _estimate_chart_end_time(chart)
    ticks: List[int] = []
    for idx, seg in enumerate(timings):
        seg_start = seg.time
        bpm = seg.bpm
        if bpm <= 0:
            continue
        beat_len = 60000.0 / bpm
        tick_len = beat_len / divisor
        if idx + 1 < len(timings):
            seg_end = min(timings[idx + 1].time, end_time)
        else:
            seg_end = end_time
        if seg_start >= seg_end:
            continue
        t = float(seg_start)
        if ticks:
            t = max(t, ticks[-1] + 1.0)
        while t <= seg_end:
            ticks.append(int(round(t)))
            t += tick_len
    ticks = sorted(set(ticks))
    return ticks

def build_tick_divisor_features(ticks: List[int], chart: AffChart, divisor: int=4) -> np.ndarray:
    """returns: np.ndarray"""
    timings = sorted(chart.timings, key=lambda t: t.time)
    bpm_per_tick: List[float] = []
    ticklen_per_tick: List[float] = []
    j = 0
    for t in ticks:
        while j + 1 < len(timings) and t >= timings[j + 1].time:
            j += 1
        bpm = max(1e-06, float(timings[j].bpm))
        beat_len = 60000.0 / bpm
        tick_len = beat_len / divisor
        bpm_per_tick.append(bpm)
        ticklen_per_tick.append(tick_len)
    extra = np.stack([np.array(bpm_per_tick, dtype=np.float32), np.array(ticklen_per_tick, dtype=np.float32)], axis=0)
    return extra

def group_ticks_by_measure(ticks_ms: np.ndarray, timings: Sequence, timing_index: np.ndarray, *, T_limit: Optional[int]=None, is_note_start: Optional[np.ndarray]=None) -> List[np.ndarray]:
    """returns: List[np.ndarray]"""
    ticks_arr = np.asarray(ticks_ms, dtype=np.float32)
    timing_index_arr = np.asarray(timing_index, dtype=np.int32)
    if T_limit is not None:
        T = int(T_limit)
        ticks_arr = ticks_arr[:T]
        timing_index_arr = timing_index_arr[:T]
        is_note_arr = None if is_note_start is None else np.asarray(is_note_start, dtype=np.float32)[:T]
    else:
        is_note_arr = None if is_note_start is None else np.asarray(is_note_start, dtype=np.float32)
    n_segments = len(timings)
    seg_ticks_idx: List[np.ndarray] = []
    for seg_id in range(n_segments):
        seg_ticks_idx.append(np.where(timing_index_arr == seg_id)[0])
    groups: List[np.ndarray] = []
    for seg_id, seg in enumerate(timings):
        idxs = seg_ticks_idx[seg_id]
        if idxs.size == 0:
            continue
        bpm = float(seg.bpm)
        beats = float(seg.beats)
        if bpm <= 0 or beats <= 0:
            continue
        beat_len = 60000.0 / max(1e-06, bpm)
        measure_len = beat_len * beats
        local_times = ticks_arr[idxs]
        if local_times.size == 0:
            continue
        seg_start_time = float(seg.time)
        rel_times = local_times - seg_start_time
        measure_ids = np.floor(rel_times / measure_len).astype(np.int32)
        min_m = int(measure_ids.min())
        max_m = int(measure_ids.max())
        for m_id in range(min_m, max_m + 1):
            mask = measure_ids == m_id
            if not np.any(mask):
                continue
            meas_idx = idxs[mask]
            if is_note_arr is not None:
                onset_mask = is_note_arr[meas_idx] > 0.5
                meas_idx = meas_idx[onset_mask]
                if meas_idx.size == 0:
                    continue
            groups.append(meas_idx)
    return groups
__all__ = ['build_tick_grid', 'build_tick_divisor_features', 'group_ticks_by_measure']
