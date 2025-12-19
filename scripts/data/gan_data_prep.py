from __future__ import annotations
import os
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from ..utils.tools import read_aff_file, ArcNote
from ..models.gan import GROUP_SIZE, PARAM_DIM
from ..utils.config_utils import PATHS, ROOT_DIR, GRID_LABEL_TOLERANCE_MS
from .common_grid import build_ticks_and_raw_labels, compute_onset_type_from_raw_labels
from ..utils.grid_quantizer import _build_segment_indices, _find_timing_segment, _quantize_time_to_tick
from ..utils.timing_tools import group_ticks_by_measure
SCRIPTS_DIR = os.path.dirname(__file__)
MAPLIST_FILE = os.path.join(ROOT_DIR, PATHS['maplist_path'])
GAN_DATA_DIR = os.path.join(ROOT_DIR, PATHS['gan_data_dir'])
GAN_REAL_DATA_PATH = os.path.join(ROOT_DIR, PATHS['gan_real_data'])
GAN_VIS_DIR = os.path.join(ROOT_DIR, PATHS.get('gan_vis_dir', os.path.join('data', 'gan_vis')))
_CURVE_TYPE_VOCAB: Dict[str, int] = {'s': 0, 'b': 1, 'si': 2, 'so': 3, 'sisi': 4, 'soso': 5, 'siso': 6, 'sosi': 7}

def _curve_type_to_code(curve_type: str) -> int:
    """returns: int"""
    ct = (curve_type or 's').strip().lower()
    return _CURVE_TYPE_VOCAB.get(ct, 0)
LANE_COUNT = 4
TRUE_SLOTS = 2

def _build_tap_hold_onehot(chart, ticks_ms: List[float], timing_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """returns: Tuple[np.ndarray, np.ndarray]"""
    T = len(ticks_ms)
    tap_onehot = np.zeros((T, LANE_COUNT), dtype=np.float32)
    hold_onehot = np.zeros((T, LANE_COUNT), dtype=np.float32)
    ticks_arr = np.asarray(ticks_ms, dtype=np.int32)
    timings = sorted(chart.timings, key=lambda t: t.time)
    timing_index_arr = np.asarray(timing_index, dtype=np.int32)
    n_segments = len(timings)
    seg_tick_indices = _build_segment_indices(timing_index_arr, n_segments)
    for n in chart.taps:
        lane_raw = int(getattr(n, 'lane', 0))
        lane = lane_raw - 1
        if lane < 0 or lane >= LANE_COUNT:
            continue
        t = float(n.time)
        seg_id = _find_timing_segment(timings, t)
        j = _quantize_time_to_tick(t, seg_id, ticks_arr, seg_tick_indices, float(GRID_LABEL_TOLERANCE_MS))
        if j is None:
            continue
        tap_onehot[int(j), lane] = 1.0
    for h in chart.holds:
        lane_raw = int(getattr(h, 'lane', 0))
        lane = lane_raw - 1
        if lane < 0 or lane >= LANE_COUNT:
            continue
        start = float(h.start)
        end = float(h.end)
        if end < start:
            start, end = (end, start)
        seg_start = _find_timing_segment(timings, start)
        seg_end = _find_timing_segment(timings, end)
        if seg_start != seg_end:
            continue
        j_start = _quantize_time_to_tick(start, seg_start, ticks_arr, seg_tick_indices, float(GRID_LABEL_TOLERANCE_MS))
        if j_start is None:
            continue
        hold_onehot[int(j_start), lane] = 1.0
    return (tap_onehot, hold_onehot)

def _build_arc_params_with_slots(chart, ticks_ms: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """returns: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]"""
    T = len(ticks_ms)
    ticks_arr = np.asarray(ticks_ms, dtype=np.float32)
    tol = 3.0
    af0_geom = np.full((T, 5), np.nan, dtype=np.float32)
    af1_geom = np.full((T, 5), np.nan, dtype=np.float32)
    at0_geom = np.full((T, 7), np.nan, dtype=np.float32)
    at1_geom = np.full((T, 7), np.nan, dtype=np.float32)
    arcs = [a for a in chart.arcs if isinstance(a, ArcNote)]
    for t_idx in range(T):
        t_ms = float(ticks_arr[t_idx])
        best_af0 = None
        best_af0_start = None
        best_af1 = None
        best_af1_start = None
        active_true_arcs: List[ArcNote] = []
        for a in arcs:
            t_s = float(a.start)
            t_e = float(a.end)
            if t_e < t_s:
                t_s, t_e = (t_e, t_s)
            if t_ms < t_s - tol or t_ms > t_e + tol:
                continue
            if a.is_trace:
                active_true_arcs.append(a)
            else:
                hand = int(getattr(a, 'hand', 0))
                if hand == 1:
                    if best_af1_start is None or t_s >= best_af1_start:
                        best_af1 = a
                        best_af1_start = t_s
                elif best_af0_start is None or t_s >= best_af0_start:
                    best_af0 = a
                    best_af0_start = t_s

        def _write_af_geom(dst: np.ndarray, a: ArcNote, code: float):
            """returns: None"""
            dst[t_idx, 0] = float(a.x_start)
            dst[t_idx, 1] = float(a.x_end)
            dst[t_idx, 2] = float(a.y_start)
            dst[t_idx, 3] = float(a.y_end)
            dst[t_idx, 4] = code
        if best_af0 is not None:
            code = float(_curve_type_to_code(best_af0.curve_type))
            _write_af_geom(af0_geom, best_af0, code)
        if best_af1 is not None:
            code = float(_curve_type_to_code(best_af1.curve_type))
            _write_af_geom(af1_geom, best_af1, code)
        if active_true_arcs:
            active_true_arcs.sort(key=lambda a: float(a.start))
            slots = active_true_arcs[:TRUE_SLOTS]
            for slot_idx, a in enumerate(slots):
                code = float(_curve_type_to_code(a.curve_type))
                hand = int(getattr(a, 'hand', 0))
                has_at = 0.0
                if getattr(a, 'arctaps', None):
                    for at in a.arctaps:
                        t_at = float(at.time)
                        if abs(t_at - t_ms) <= tol:
                            has_at = 1.0
                            break
                dst = at0_geom if slot_idx == 0 else at1_geom
                dst[t_idx, 0] = float(a.x_start)
                dst[t_idx, 1] = float(a.x_end)
                dst[t_idx, 2] = float(a.y_start)
                dst[t_idx, 3] = float(a.y_end)
                dst[t_idx, 4] = code
                dst[t_idx, 5] = float(hand)
                dst[t_idx, 6] = has_at
    return (af0_geom, af1_geom, at0_geom, at1_geom)

def build_gan_real_data() -> str:
    """returns: str"""
    if not os.path.isfile(MAPLIST_FILE):
        raise FileNotFoundError(f'Cannot find maplist file: {MAPLIST_FILE}')
    with open(MAPLIST_FILE, encoding='utf-8') as fp:
        entries = [ln.strip() for ln in fp.readlines() if ln.strip()]
    segments: List[np.ndarray] = []
    segments_mask: List[np.ndarray] = []
    os.makedirs(GAN_VIS_DIR, exist_ok=True)
    for fn in os.listdir(GAN_VIS_DIR):
        if fn.endswith('.tsv'):
            try:
                os.remove(os.path.join(GAN_VIS_DIR, fn))
            except OSError:
                pass
    vis_idx = 0
    for entry in entries:
        base_path = entry
        if not os.path.isabs(base_path):
            base_path = os.path.abspath(os.path.join(ROOT_DIR, base_path))
        if os.path.isdir(base_path):
            song_dir = base_path
            aff_files = [os.path.join(song_dir, fn) for fn in os.listdir(song_dir) if fn.lower().endswith('.aff')]
        elif os.path.isfile(base_path) and base_path.lower().endswith('.aff'):
            song_dir = os.path.dirname(base_path)
            aff_files = [base_path]
        else:
            print(f'[gan_data_prep] skip invalid entry: {entry}')
            continue
        for aff_path in aff_files:
            try:
                chart = read_aff_file(aff_path)
                ticks_ms, frac_index, timing_index, labels = build_ticks_and_raw_labels(chart, tol_ms=GRID_LABEL_TOLERANCE_MS)
                if len(ticks_ms) == 0:
                    continue
                tap_onehot, hold_onehot = _build_tap_hold_onehot(chart, ticks_ms, np.asarray(timing_index, dtype=np.int32))
                af0_geom, af1_geom, at0_geom, at1_geom = _build_arc_params_with_slots(chart, ticks_ms)
                if np.random.rand() < 0.05:
                    tap_onehot = tap_onehot[:, ::-1]
                    hold_onehot = hold_onehot[:, ::-1]

                    def _flip_x_in_geom(geom: np.ndarray) -> None:
                        """returns: None"""
                        if geom.size == 0:
                            return
                        geom[:, 0] = 1.0 - geom[:, 0]
                        geom[:, 1] = 1.0 - geom[:, 1]
                    _flip_x_in_geom(af0_geom)
                    _flip_x_in_geom(af1_geom)
                    _flip_x_in_geom(at0_geom)
                    _flip_x_in_geom(at1_geom)
                T = len(ticks_ms)
                ticks_arr = np.asarray(ticks_ms, dtype=np.float32)
                labels_arr = np.asarray(labels, dtype=np.float32)
                onset_type = compute_onset_type_from_raw_labels(labels_arr)
                is_note_start = np.sum(onset_type[:, [0, 1, 2, 3, 5]], axis=1).astype(np.float32)
                T = len(ticks_ms)
                params = np.zeros((T, PARAM_DIM), dtype=np.float32)
                params[:, 0:4] = tap_onehot
                params[:, 4:8] = hold_onehot
                params[:, 8:13] = af0_geom
                params[:, 13:18] = af1_geom
                params[:, 18:25] = at0_geom
                params[:, 25:32] = at1_geom
                assert params.shape[1] == PARAM_DIM, f'PARAM_DIM mismatch: expected {PARAM_DIM}, got {params.shape[1]}'
                vis_path = os.path.join(GAN_VIS_DIR, f'{vis_idx}.tsv')
                vis_idx += 1
                with open(vis_path, 'w', encoding='utf-8') as vf:
                    vf.write('tick_idx\tglobal_time_ms\ttap_l0\ttap_l1\ttap_l2\ttap_l3\thold_l0\thold_l1\thold_l2\thold_l3\taf0_x_start\taf0_x_end\taf0_y_start\taf0_y_end\taf0_curve_code\taf1_x_start\taf1_x_end\taf1_y_start\taf1_y_end\taf1_curve_code\tat0_x_start\tat0_x_end\tat0_y_start\tat0_y_end\tat0_curve_code\tat0_hand\tat0_has_arctap\tat1_x_start\tat1_x_end\tat1_y_start\tat1_y_end\tat1_curve_code\tat1_hand\tat1_has_arctap\tis_note\tis_tap\tis_hold\tis_arcfalse0\tis_arcfalse1\tis_arctrue\tis_arctap\n')
                    for t_i in range(T):
                        is_note = int(labels[t_i, 0] > 0.5)
                        is_tap = int(labels[t_i, 1] > 0.5)
                        is_hold = int(labels[t_i, 2] > 0.5)
                        is_arcfalse0 = int(labels[t_i, 3] > 0.5)
                        is_arcfalse1 = int(labels[t_i, 4] > 0.5)
                        is_arctrue = int(labels[t_i, 5] > 0.5)
                        is_arctap = int(labels[t_i, 6] > 0.5)
                        has_label = bool(is_note or is_tap or is_hold or is_arcfalse0 or is_arcfalse1 or is_arctrue or is_arctap)
                        geom_eps = 0.0001
                        has_geom = np.isfinite(af0_geom[t_i]).any() or np.isfinite(af1_geom[t_i]).any() or np.isfinite(at0_geom[t_i]).any() or np.isfinite(at1_geom[t_i]).any()
                        if not (has_label or has_geom):
                            continue
                        vf.write(f'{t_i}\t{int(ticks_ms[t_i])}\t{tap_onehot[t_i, 0]:.0f}\t{tap_onehot[t_i, 1]:.0f}\t{tap_onehot[t_i, 2]:.0f}\t{tap_onehot[t_i, 3]:.0f}\t{hold_onehot[t_i, 0]:.0f}\t{hold_onehot[t_i, 1]:.0f}\t{hold_onehot[t_i, 2]:.0f}\t{hold_onehot[t_i, 3]:.0f}\t{af0_geom[t_i, 0]:.4f}\t{af0_geom[t_i, 1]:.4f}\t{af0_geom[t_i, 2]:.4f}\t{af0_geom[t_i, 3]:.4f}\t{af0_geom[t_i, 4]:.0f}\t{af1_geom[t_i, 0]:.4f}\t{af1_geom[t_i, 1]:.4f}\t{af1_geom[t_i, 2]:.4f}\t{af1_geom[t_i, 3]:.4f}\t{af1_geom[t_i, 4]:.0f}\t{at0_geom[t_i, 0]:.4f}\t{at0_geom[t_i, 1]:.4f}\t{at0_geom[t_i, 2]:.4f}\t{at0_geom[t_i, 3]:.4f}\t{at0_geom[t_i, 4]:.0f}\t{at0_geom[t_i, 5]:.0f}\t{at0_geom[t_i, 6]:.0f}\t{at1_geom[t_i, 0]:.4f}\t{at1_geom[t_i, 1]:.4f}\t{at1_geom[t_i, 2]:.4f}\t{at1_geom[t_i, 3]:.4f}\t{at1_geom[t_i, 4]:.0f}\t{at1_geom[t_i, 5]:.0f}\t{at1_geom[t_i, 6]:.0f}\t{is_note}\t{is_tap}\t{is_hold}\t{is_arcfalse0}\t{is_arcfalse1}\t{is_arctrue}\t{is_arctap}\n')
                timings = sorted(chart.timings, key=lambda t: t.time)
                if not timings:
                    continue
                timing_index_arr = np.asarray(timing_index, dtype=np.int32)
                measure_groups = group_ticks_by_measure(ticks_arr, timings, timing_index_arr, is_note_start=is_note_start)
                for meas_idx in measure_groups:
                    meas_params = params[meas_idx]
                    meas_mask = np.isfinite(meas_params).astype(np.float32)
                    L_meas = meas_params.shape[0]
                    if L_meas >= GROUP_SIZE:
                        seg_mat = meas_params[:GROUP_SIZE]
                        seg_mask = meas_mask[:GROUP_SIZE]
                    else:
                        pad = np.zeros((GROUP_SIZE - L_meas, PARAM_DIM), dtype=meas_params.dtype)
                        pad_mask = np.zeros_like(pad, dtype=np.float32)
                        seg_mat = np.concatenate([meas_params, pad], axis=0)
                        seg_mask = np.concatenate([meas_mask, pad_mask], axis=0)
                    segments.append(seg_mat)
                    segments_mask.append(seg_mask)
            except Exception as e:
                print(f'[gan_data_prep] error on {aff_path}: {e}')
    if not segments:
        raise RuntimeError('Failed to build any GAN training segments from maplist_aff.txt.')
    real_data = np.stack(segments, axis=0)
    real_mask = np.stack(segments_mask, axis=0).astype(np.float32)
    os.makedirs(GAN_DATA_DIR, exist_ok=True)
    np.savez_compressed(GAN_REAL_DATA_PATH, real_data=real_data, real_mask=real_mask)
    print(f'GAN real_data saved at: {GAN_REAL_DATA_PATH}, shape = {real_data.shape}, mask_shape = {real_mask.shape}')
    return GAN_REAL_DATA_PATH
__all__ = ['GAN_REAL_DATA_PATH', 'build_gan_real_data']
