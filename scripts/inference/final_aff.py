from __future__ import annotations
import os
from typing import List, Tuple, Optional
import numpy as np
from ..utils.tools import read_aff_file, write_aff_file, AffChart, TapNote, HoldNote, ArcNote, Arctap, get_aff_directory
from ..utils.grid_quantizer import build_grid
from ..models.gan import load_generator, GROUP_SIZE, PARAM_DIM
from ..utils.config_utils import ROOT_DIR, PATHS
from ..utils.timing_tools import group_ticks_by_measure
SCRIPTS_DIR = os.path.dirname(__file__)
OUTPUTS_DIR = os.path.join(ROOT_DIR, PATHS['outputs_dir'])

def _quantize_lane(raw_lane: float) -> int:
    """returns: int"""
    try:
        v = float(raw_lane)
    except Exception:
        return 1
    if not np.isfinite(v):
        return 2
    v = v + 2.5
    frac, integ = np.modf(v)
    if abs(frac - 0.5) < 0.001:
        lower = int(np.floor(v))
        upper = int(np.ceil(v))
        if np.random.rand() < 0.5:
            lane = lower
        else:
            lane = upper
    else:
        lane = int(np.round(v))
    lane = int(np.clip(lane, 1, 4))
    return lane

def _curve_code_to_type(code: float) -> str:
    """returns: str"""
    vocab = ['s', 'b', 'si', 'so', 'sisi', 'soso', 'siso', 'sosi']
    idx = int(np.round(code))
    if idx < 0 or idx >= len(vocab):
        return 's'
    return vocab[idx]

def _normalize_arc_coords(x_start: float, x_end: float, y_start: float, y_end: float) -> Tuple[float, float, float, float]:
    """returns: Tuple[float, float, float, float]"""

    def _safe_float(v: float) -> float:
        """returns: float"""
        try:
            fv = float(v)
        except Exception:
            return 0.0
        if not np.isfinite(fv):
            return 0.0
        return fv
    xs_raw = _safe_float(x_start)
    xe_raw = _safe_float(x_end)
    ys_raw = _safe_float(y_start)
    ye_raw = _safe_float(y_end)
    xs = float(np.tanh(xs_raw) + 0.5)
    xe = float(np.tanh(xe_raw) + 0.5)
    ys = float(1.0 / (1.0 + np.exp(-ys_raw)))
    ye = float(1.0 / (1.0 + np.exp(-ye_raw)))

    def _clamp_to_trapezoid(x: float, y: float) -> Tuple[float, float]:
        """returns: Tuple[float, float]"""
        x = float(np.clip(x, -0.5, 1.5))
        y = float(np.clip(y, 0.0, 1.0))
        if x < 0.0:
            y_max = 2.0 * (x + 0.5)
        elif x > 1.0:
            y_max = -2.0 * (x - 1.0) + 1.0
        else:
            y_max = 1.0
        if y > y_max:
            y = y_max
        return (x, y)
    xs, ys = _clamp_to_trapezoid(xs, ys)
    xe, ye = _clamp_to_trapezoid(xe, ye)

    def _snap(v: float, step: float=0.25) -> float:
        """returns: float"""
        return float(np.round(v / step) * step)
    xs = _snap(xs)
    xe = _snap(xe)
    ys = _snap(ys)
    ye = _snap(ye)
    xs, ys = _clamp_to_trapezoid(xs, ys)
    xe, ye = _clamp_to_trapezoid(xe, ye)
    return (xs, xe, ys, ye)

def _infer_audio_duration_ms(aff_path: str, audio_name: str='base.ogg') -> Optional[float]:
    """returns: Optional[float]"""
    try:
        import librosa
    except Exception:
        return None
    aff_dir = get_aff_directory(aff_path)
    candidates = []
    if os.path.isabs(audio_name):
        candidates.append(audio_name)
    else:
        candidates.append(os.path.join(aff_dir, audio_name))
        candidates.append(os.path.join(ROOT_DIR, audio_name))
    for p in candidates:
        if not os.path.isfile(p):
            continue
        try:
            dur_sec = librosa.get_duration(path=p)
            if dur_sec and dur_sec > 0:
                return float(dur_sec) * 1000.0
        except Exception:
            continue
    return None


def _sample_gan_params_if_available(chart: AffChart, ticks: np.ndarray, is_note_start: Optional[np.ndarray]=None, end_time_ms: Optional[float]=None) -> Optional[np.ndarray]:
    """returns: Optional[np.ndarray]"""
    try:
        generator = load_generator()
    except Exception as e:
        print(f'[final_aff] GAN generator not available, skip GAN geometry: {e}')
        return None
    T = len(ticks)
    if T == 0:
        return None
    ticks_ms, frac_index, timing_index = build_grid(chart, end_time_ms=end_time_ms)
    if len(ticks_ms) < T:
        T = len(ticks_ms)
        ticks = ticks[:T]
    if is_note_start is not None:
        is_note_arr = np.asarray(is_note_start, dtype=np.float32)[:T]
    else:
        is_note_arr = None
    timings = sorted(chart.timings, key=lambda t: t.time)
    if not timings:
        return None
    timing_index_arr = np.asarray(timing_index[:T], dtype=np.int32)
    measure_tick_indices = group_ticks_by_measure(ticks_ms[:T], timings, timing_index_arr, is_note_start=is_note_arr)
    if not measure_tick_indices:
        return None
    n_groups = len(measure_tick_indices)
    latent_dim = generator.input_shape[1]
    z = np.random.normal(0.0, 1.0, (n_groups, latent_dim)).astype(np.float32)
    gen_out = generator.predict(z, verbose=0)
    if gen_out.shape[1] == GROUP_SIZE and gen_out.shape[2] == 34:
        old = gen_out
        tap = old[:, :, 0:4]
        hold = old[:, :, 4:8]
        af0 = old[:, :, 8:13]
        af1 = old[:, :, 14:19]
        at0 = old[:, :, 20:27]
        at1 = old[:, :, 27:34]
        gen_out = np.concatenate([tap, hold, af0, af1, at0, at1], axis=-1)
    if gen_out.shape[1] != GROUP_SIZE or gen_out.shape[2] != PARAM_DIM:
        raise ValueError(f'[final_aff] unexpected GAN output shape {gen_out.shape}, expected (_, {GROUP_SIZE}, {PARAM_DIM})')
    gan_params = np.zeros((T, PARAM_DIM), dtype=np.float32)
    for g, meas_idx in enumerate(measure_tick_indices):
        L_meas = len(meas_idx)
        if L_meas == 0:
            continue
        L_use = min(L_meas, GROUP_SIZE)
        idx_arr = np.asarray(meas_idx, dtype=np.int32)
        gan_params[idx_arr[:L_use], :] = gen_out[g, :L_use, :]
    return gan_params

def _build_tap_and_hold_from_labels_full(ticks: np.ndarray, labels_full: np.ndarray, gan_params: Optional[np.ndarray], *, tap_temp: float=1.0, tap_noise_std: float=0.05) -> Tuple[List[TapNote], List[HoldNote]]:
    """returns: Tuple[List[TapNote], List[HoldNote]]"""
    taps: List[TapNote] = []
    holds: List[HoldNote] = []
    T = len(ticks)
    if gan_params is None:
        is_note_start = labels_full[:, 0]
        tap_count = labels_full[:, 1]
        hold_count = labels_full[:, 2]
        in_hold = labels_full[:, 8]
        for i in range(T):
            t_time = int(ticks[i])
            if tap_count[i] > 0.5:
                taps.append(TapNote(time=t_time, lane=1))
        i = 0
        while i < T:
            if hold_count[i] <= 0.5:
                i += 1
                continue
            start_idx = i
            j = i
            while j + 1 < T and in_hold[j + 1] > 0.5:
                j += 1
            end_idx = j
            holds.append(HoldNote(start=int(ticks[start_idx]), end=int(ticks[end_idx]), lane=1))
            i = j + 1
        return (taps, holds)
    tap_count = labels_full[:, 1:2]
    hold_count = labels_full[:, 2:3]
    in_hold0 = labels_full[:, 9] > 0.5
    in_hold1 = labels_full[:, 10] > 0.5
    tap_scores = gan_params[:, 0:4]
    hold_scores = gan_params[:, 4:8]
    LANE_COUNT = 4
    for i in range(T):
        t_time = int(ticks[i])
        k = int(np.clip(np.round(tap_count[i, 0]), 0, LANE_COUNT))
        if k <= 0:
            continue
        scores = tap_scores[i]
        if tap_noise_std > 0:
            scores = scores + np.random.normal(0.0, tap_noise_std, size=scores.shape)
        probs = np.exp(scores / max(1e-6, tap_temp))
        probs = np.where(np.isfinite(probs), probs, 0.0)
        denom = float(np.sum(probs))
        if denom <= 0.0:
            probs = np.ones_like(scores, dtype=np.float32) / float(len(scores))
        else:
            probs = probs / denom
        k_use = max(0, min(int(k), len(scores)))
        if k_use == 0:
            continue
        chosen = np.random.choice(len(scores), size=k_use, replace=False, p=probs)
        for lane_idx in chosen:
            lane_aff = int(np.clip(int(lane_idx) + 1, 1, 4))
            taps.append(TapNote(time=t_time, lane=lane_aff))

    def _append_hold_segments(mask: np.ndarray) -> None:
        """returns: None"""
        i = 0
        while i < T:
            if not mask[i]:
                i += 1
                continue
            start_idx = i
            j = i
            while j + 1 < T and mask[j + 1]:
                j += 1
            end_idx = j
            scores = hold_scores[start_idx]
            if tap_noise_std > 0:
                scores = scores + np.random.normal(0.0, tap_noise_std, size=scores.shape)
            probs = np.exp(scores / max(1e-6, tap_temp))
            probs = np.where(np.isfinite(probs), probs, 0.0)
            denom = probs.sum()
            if denom <= 0.0:
                probs = np.ones_like(scores, dtype=np.float32) / float(len(scores))
            else:
                probs = probs / denom
            lane_idx = int(np.random.choice(len(scores), p=probs))
            lane_aff = int(np.clip(lane_idx + 1, 1, 4))
            holds.append(HoldNote(start=int(ticks[start_idx]), end=int(ticks[end_idx]), lane=lane_aff))
            i = j + 1
    _append_hold_segments(in_hold0)
    _append_hold_segments(in_hold1)
    return (taps, holds)

def _build_arcs_from_labels_full(ticks: np.ndarray, labels_full: np.ndarray, gan_params: Optional[np.ndarray]) -> List[ArcNote]:
    """returns: List[ArcNote]"""
    arcs: List[ArcNote] = []
    if gan_params is None:
        return arcs
    T = len(ticks)
    af0_geom = gan_params[:, 8:13]
    af1_geom = gan_params[:, 13:18]
    at0_geom = gan_params[:, 18:25]
    at1_geom = gan_params[:, 25:32]
    in_arcfalse0 = labels_full[:, 7] > 0.5
    in_arcfalse1 = labels_full[:, 8] > 0.5

    def _append_af_segment(mask: np.ndarray, geom: np.ndarray, hand: int) -> None:
        """returns: None"""
        i = 0
        while i < T:
            if not mask[i]:
                i += 1
                continue
            start_idx = i
            j = i
            while j + 1 < T and mask[j + 1]:
                j += 1
            end_idx = j
            base = geom[start_idx]
            if not np.isfinite(base).any():
                i = j + 1
                continue
            x_start, x_end, y_start, y_end = _normalize_arc_coords(base[0], base[1], base[2], base[3])
            curve_type = _curve_code_to_type(base[4])
            arcs.append(ArcNote(start=int(ticks[start_idx]), end=int(ticks[end_idx]), x_start=float(x_start), x_end=float(x_end), curve_type=curve_type, y_start=float(y_start), y_end=float(y_end), hand=hand, extra='none', is_trace=False, arctaps=[]))
            i = j + 1
    _append_af_segment(in_arcfalse0, af0_geom, hand=0)
    _append_af_segment(in_arcfalse1, af1_geom, hand=1)
    in_arctrue0 = labels_full[:, 11] > 0.5
    in_arctrue1 = labels_full[:, 12] > 0.5

    def _build_trace_from_slot(slot_geom: np.ndarray, mask: np.ndarray):
        """returns: None"""
        active = mask
        i = 0
        while i < T:
            if not active[i]:
                i += 1
                continue
            start_idx = i
            j = i
            while j + 1 < T and active[j + 1]:
                j += 1
            end_idx = j
            base = slot_geom[start_idx]
            if not np.isfinite(base).any():
                i = j + 1
                continue
            x_start, x_end, y_start, y_end = _normalize_arc_coords(base[0], base[1], base[2], base[3])
            curve_type = _curve_code_to_type(base[4])
            hand = int(np.clip(np.round(base[5]), 0, 1)) if np.isfinite(base[5]) else 0
            arcs.append(ArcNote(start=int(ticks[start_idx]), end=int(ticks[end_idx]), x_start=float(x_start), x_end=float(x_end), curve_type=curve_type, y_start=float(y_start), y_end=float(y_end), hand=hand, extra='none', is_trace=True, arctaps=[]))
            i = j + 1
    _build_trace_from_slot(at0_geom, in_arctrue0)
    _build_trace_from_slot(at1_geom, in_arctrue1)
    return arcs

def _attach_arctaps(arcs: List[ArcNote], ticks: np.ndarray, labels: np.ndarray) -> None:
    """returns: None"""
    is_arctap = labels[:, 5] > 0.5
    T = len(ticks)
    for i in range(T):
        if not is_arctap[i]:
            continue
        t = int(ticks[i])
        for arc in arcs:
            if not arc.is_trace:
                continue
            if arc.start <= t <= arc.end:
                arc.arctaps.append(Arctap(time=t))

def build_aff_from_predictions(skeleton_aff_path: str, rhythm_npz_path: str, tap_temp: float=1.0, tap_noise_std: float=0.05) -> str:
    """returns: str"""
    chart = read_aff_file(skeleton_aff_path)
    audio_duration_ms = _infer_audio_duration_ms(skeleton_aff_path, audio_name='base.ogg')
    with np.load(rhythm_npz_path) as data_r:
        ticks = data_r['timestamps']
        labels_full = data_r['labels']
    T = len(ticks)
    ticks = ticks[:T]
    lf = np.asarray(labels_full, dtype=np.float32)[:T]
    if lf.shape[1] < 13:
        raise ValueError(f'labels dimension is too small: {lf.shape}')
    is_note_start = lf[:, 0]
    gan_params = _sample_gan_params_if_available(chart, ticks, is_note_start=is_note_start, end_time_ms=audio_duration_ms)
    taps, holds = _build_tap_and_hold_from_labels_full(ticks, lf, gan_params, tap_temp=tap_temp, tap_noise_std=tap_noise_std)
    arcs = _build_arcs_from_labels_full(ticks, lf, gan_params)
    labels6 = np.zeros((T, 6), dtype=np.float32)
    labels6[:, 0] = (is_note_start > 0.5).astype(np.float32)
    labels6[:, 5] = (lf[:, 6] > 0.5).astype(np.float32)
    _attach_arctaps(arcs, ticks, labels6)
    cleaned_arcs_initial: List[ArcNote] = []
    for a in arcs:
        if a.is_trace and int(a.start) == int(a.end):
            has_arctap_at_start = any((int(at.time) == int(a.start) for at in a.arctaps))
            if has_arctap_at_start:
                a.end = int(a.start) + 1
                a.x_end = a.x_start
                a.y_end = a.y_start
                cleaned_arcs_initial.append(a)
            continue
        cleaned_arcs_initial.append(a)
    arcs = cleaned_arcs_initial
    MIN_SEGMENT_MS = 17
    holds = [h for h in holds if int(h.end) - int(h.start) >= MIN_SEGMENT_MS]
    arcs = [a for a in arcs if a.is_trace or int(a.end) - int(a.start) >= MIN_SEGMENT_MS]
    taps.sort(key=lambda n: (int(n.time), int(n.lane)))
    seen_tap = set()
    dedup_taps: List[TapNote] = []
    for t in taps:
        key = (int(t.time), int(t.lane))
        if key in seen_tap:
            continue
        seen_tap.add(key)
        dedup_taps.append(t)
    taps = dedup_taps
    holds.sort(key=lambda h: (int(h.lane), int(h.start), int(h.end)))
    merged_holds: List[HoldNote] = []
    current_by_lane: dict[int, HoldNote] = {}
    for h in holds:
        lane = int(h.lane)
        cur = current_by_lane.get(lane)
        if cur is None or int(h.start) >= int(cur.end):
            merged_holds.append(h)
            current_by_lane[lane] = h
        else:
            continue
    holds = merged_holds
    holds_by_lane: dict[int, List[HoldNote]] = {lane: [] for lane in range(1, 5)}
    for h in holds:
        holds_by_lane.setdefault(int(h.lane), []).append(h)
    for lane in holds_by_lane:
        holds_by_lane[lane].sort(key=lambda h: (int(h.start), int(h.end)))
    filtered_taps: List[TapNote] = []
    for t in taps:
        lane = int(t.lane)
        t_time = int(t.time)
        lane_holds = holds_by_lane.get(lane, [])
        in_any_hold = False
        for h in lane_holds:
            if int(h.start) <= t_time <= int(h.end):
                in_any_hold = True
                break
            if t_time < int(h.start):
                break
        if not in_any_hold:
            filtered_taps.append(t)
    taps = filtered_taps

    def _arc_group_key(a: ArcNote) -> tuple[bool, int]:
        """returns: tuple[bool, int]"""
        if a.is_trace:
            return (True, 0)
        return (False, int(a.hand))
    arcs.sort(key=lambda a: (_arc_group_key(a)[0], _arc_group_key(a)[1], int(a.start), int(a.end)))
    cleaned_arcs: List[ArcNote] = []
    last_by_group: dict[tuple[bool, int], ArcNote] = {}
    for a in arcs:
        group = _arc_group_key(a)
        last = last_by_group.get(group)
        if last is None or int(a.start) >= int(last.end):
            cleaned_arcs.append(a)
            last_by_group[group] = a
        else:
            continue
    arcs = cleaned_arcs
    for arc in arcs:
        if not arc.arctaps:
            continue
        seen_times = set()
        dedup_arctaps: List[Arctap] = []
        for at in sorted(arc.arctaps, key=lambda x: int(x.time)):
            t_time = int(at.time)
            if t_time in seen_times:
                continue
            seen_times.add(t_time)
            dedup_arctaps.append(at)
        arc.arctaps = dedup_arctaps
    taps.sort(key=lambda n: (int(n.time), int(n.lane)))
    holds.sort(key=lambda h: (int(h.start), int(h.lane), int(h.end)))
    arcs.sort(key=lambda a: (int(a.start), int(a.end), bool(a.is_trace)))
    new_chart = AffChart(audio_offset=chart.audio_offset, timings=list(chart.timings), taps=taps, holds=holds, arcs=arcs)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(skeleton_aff_path))[0]
    out_path = os.path.join(OUTPUTS_DIR, f'{base_name}_generated.aff')
    write_aff_file(new_chart, out_path)
    print(f'generated aff written to: {out_path}')
    return out_path
__all__ = ['build_aff_from_predictions']
