from __future__ import annotations
import os
from typing import Tuple, Optional
import numpy as np
try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:
    keras = None
from ..models.classifier import load_classifier, DEFAULT_MODEL_PATH
from ..utils.device_utils import choose_tf_device
from ..utils.config_utils import ROOT_DIR, PATHS, DIVISION_CODE_TO_NAME, DIVISION_PENALTY_TABLE, NOTE_TYPE_THRESHOLDS, NOTE_IN_ARC_THRESHOLD, NOTE_IN_HOLD_THRESHOLD, LEGAL_FRAC_24, frac24_to_division_code
from ..utils.tools import read_aff_file, get_aff_directory
from ..utils.grid_quantizer import build_grid
from ..utils.timing_tools import group_ticks_by_measure
SCRIPTS_DIR = os.path.dirname(__file__)
OUTPUTS_DIR = os.path.join(ROOT_DIR, PATHS['outputs_dir'])
MAPTHIS_DIR = os.path.join(ROOT_DIR, PATHS.get('mapthis_dir', os.path.join('data', 'mapthis')))
RHYTHM_DIR = os.path.join(ROOT_DIR, PATHS.get('rhythm_dir', os.path.join('data', 'rhythm')))
RHYTHM_VIS_DIR = os.path.join(ROOT_DIR, PATHS.get('rhythm_vis_dir', os.path.join('data', 'rhythm_vis')))

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

def load_mapthis_npz(fn: str | None=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """returns: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]"""
    if fn is None:
        fn = os.path.join(MAPTHIS_DIR, 'mapthis.npz')
    if not os.path.isfile(fn):
        raise FileNotFoundError(f'Cannot find mapthis.npz: {fn}. Please run scripts.data.newmap_prep.prepare_new_map() first.')
    with np.load(fn) as data:
        ticks = data['ticks']
        timestamps = data['timestamps']
        wav = data['wav']
        extra = data['extra']
    return (ticks, timestamps, wav, extra)

def predict_notes(model_path: str=DEFAULT_MODEL_PATH, npz_path: str | None=None, note_density: float=0.24, device: str='auto', aff_path: str | None=None, ticks_in: np.ndarray | None=None, timestamps_in: np.ndarray | None=None, wav_in: np.ndarray | None=None, extra_in: np.ndarray | None=None, max_finger_count: int=2, overlap_ticks: int=24) -> str:
    """returns: str"""
    if keras is None:
        raise RuntimeError('TensorFlow/Keras is not installed in the current environment; inference cannot be performed.')
    if ticks_in is not None and timestamps_in is not None and (wav_in is not None) and (extra_in is not None):
        ticks = np.asarray(ticks_in)
        timestamps = np.asarray(timestamps_in)
        wav = np.asarray(wav_in)
        extra = np.asarray(extra_in)
    else:
        ticks, timestamps, wav, extra = load_mapthis_npz(npz_path)
    dev_str = choose_tf_device(device)
    from ..models.classifier import SEQ_LEN, INPUT_SEQ_LEN, CONTEXT_TICKS
    if int(overlap_ticks) != int(CONTEXT_TICKS):
        raise ValueError(f'overlap_ticks({overlap_ticks}) 必须等于模型 CONTEXT_TICKS({CONTEXT_TICKS})，否则推理窗口会错位。')
    T = min(len(ticks), len(timestamps), wav.shape[0])
    ticks = ticks[:T]
    timestamps = timestamps[:T]
    wav = wav[:T]
    division_code = None
    use_measure_groups = False
    idx_groups = None
    audio_duration_ms = None
    frac_index_24 = None
    if aff_path is not None:
        aff_full = aff_path
        if not os.path.isabs(aff_full):
            aff_full = os.path.abspath(os.path.join(ROOT_DIR, aff_full))
        chart = read_aff_file(aff_full)
        audio_duration_ms = _infer_audio_duration_ms(aff_full, audio_name='base.ogg')
        ticks_ms, frac_index, timing_index = build_grid(chart, end_time_ms=audio_duration_ms)
        if len(ticks_ms) < T:
            T = len(ticks_ms)
            ticks = ticks[:T]
            timestamps = timestamps[:T]
            wav = wav[:T]
        ticks_ms_arr = np.asarray(ticks_ms[:T], dtype=np.float32)
        timing_index_arr = np.asarray(timing_index[:T], dtype=np.int32)
        frac_index = frac_index[:T]
        frac_index_24 = np.asarray(frac_index, dtype=np.int32)
        division_code = np.array([frac24_to_division_code(int(f)) for f in frac_index], dtype=np.int32)
        timings = sorted(chart.timings, key=lambda t: t.time)
        if timings:
            measure_groups = group_ticks_by_measure(ticks_ms_arr, timings, timing_index_arr, T_limit=T, is_note_start=None)
            if measure_groups:
                idx_groups = [np.asarray(g, dtype=np.int32) for g in measure_groups]
                use_measure_groups = True
    if use_measure_groups and idx_groups is not None:
        seq_list = []
        ext_groups: list[np.ndarray] = []
        for g, idxs in enumerate(idx_groups):
            prev_idxs = idx_groups[g - 1] if g - 1 >= 0 else np.zeros((0,), dtype=np.int32)
            prev_tail = prev_idxs[-overlap_ticks:] if overlap_ticks > 0 else np.zeros((0,), dtype=np.int32)
            center = idxs[:SEQ_LEN].astype(np.int32)

            pad_front = int(overlap_ticks) - int(prev_tail.shape[0])
            if pad_front < 0:
                pad_front = 0
            pad_end = int(SEQ_LEN) - int(center.shape[0])
            if pad_end < 0:
                pad_end = 0

            x_prev = wav[prev_tail] if prev_tail.size > 0 else np.zeros((0, *wav.shape[1:]), dtype=wav.dtype)
            x_center = wav[center] if center.size > 0 else np.zeros((0, *wav.shape[1:]), dtype=wav.dtype)
            x_in = np.concatenate(
                [
                    np.zeros((pad_front, *wav.shape[1:]), dtype=wav.dtype),
                    x_prev,
                    x_center,
                    np.zeros((pad_end, *wav.shape[1:]), dtype=wav.dtype),
                ],
                axis=0,
            )
            if x_in.shape[0] != INPUT_SEQ_LEN:
                x_in = x_in[:INPUT_SEQ_LEN] if x_in.shape[0] > INPUT_SEQ_LEN else np.concatenate([x_in, np.zeros((INPUT_SEQ_LEN - x_in.shape[0], *wav.shape[1:]), dtype=wav.dtype)], axis=0)
            seq_list.append(x_in)
            ext_groups.append(center)
        wav_in = np.stack(seq_list, axis=0)
        with tf.device(dev_str):
            model = load_classifier(model_path)
            preds_seq = model.predict(wav_in, verbose=0)
        if preds_seq.ndim == 2:
            preds_seq = preds_seq.reshape(preds_seq.shape[0], -1, preds_seq.shape[-1])
        if preds_seq.ndim == 4:
            preds_seq = preds_seq.reshape(preds_seq.shape[0], preds_seq.shape[1], preds_seq.shape[-1])
        if preds_seq.ndim != 3:
            flat = preds_seq.reshape(-1, preds_seq.shape[-1])
            T_eff = min(T, flat.shape[0])
            preds = flat[:T_eff]
            ticks = ticks[:T_eff]
            timestamps = timestamps[:T_eff]
            wav = wav[:T_eff]
            T = T_eff
        else:
            n_groups, seq_len_pred, D = preds_seq.shape
            preds = np.zeros((T, D), dtype=preds_seq.dtype)
            for g, center_idx in enumerate(ext_groups):
                if g >= n_groups:
                    break
                out_g = preds_seq[g]
                out96 = out_g[-SEQ_LEN:] if out_g.shape[0] >= SEQ_LEN else out_g
                L_use = min(center_idx.shape[0], out96.shape[0], SEQ_LEN)
                for pos in range(L_use):
                    gi = int(center_idx[pos])
                    if 0 <= gi < T:
                        preds[gi] = out96[pos]
    else:
        T_total = wav.shape[0]
        preds = None
        with tf.device(dev_str):
            model = load_classifier(model_path)
        n_seq = int(np.ceil(T_total / SEQ_LEN))
        dummy = np.zeros((1, INPUT_SEQ_LEN, *wav.shape[1:]), dtype=wav.dtype)
        d_out = model.predict(dummy, verbose=0)
        if d_out.ndim == 2:
            d_out = d_out.reshape(1, -1, d_out.shape[-1])
        if d_out.ndim == 4:
            d_out = d_out.reshape(1, d_out.shape[1], d_out.shape[-1])
        D = int(d_out.shape[-1])
        preds = np.zeros((T_total, D), dtype=d_out.dtype)
        for k in range(n_seq):
            center_start = k * SEQ_LEN
            center_end = min(T_total, center_start + SEQ_LEN)
            prev_start = max(0, center_start - overlap_ticks)
            prev_tail_idx = np.arange(prev_start, center_start, dtype=np.int32)
            center_idx = np.arange(center_start, center_end, dtype=np.int32)
            pad_front = int(overlap_ticks) - int(prev_tail_idx.shape[0])
            if pad_front < 0:
                pad_front = 0
            pad_end = int(SEQ_LEN) - int(center_idx.shape[0])
            if pad_end < 0:
                pad_end = 0
            x_prev = wav[prev_tail_idx] if prev_tail_idx.size > 0 else np.zeros((0, *wav.shape[1:]), dtype=wav.dtype)
            x_center = wav[center_idx] if center_idx.size > 0 else np.zeros((0, *wav.shape[1:]), dtype=wav.dtype)
            x_in = np.concatenate(
                [
                    np.zeros((pad_front, *wav.shape[1:]), dtype=wav.dtype),
                    x_prev,
                    x_center,
                    np.zeros((pad_end, *wav.shape[1:]), dtype=wav.dtype),
                ],
                axis=0,
            )
            if x_in.shape[0] != INPUT_SEQ_LEN:
                x_in = x_in[:INPUT_SEQ_LEN] if x_in.shape[0] > INPUT_SEQ_LEN else np.concatenate([x_in, np.zeros((INPUT_SEQ_LEN - x_in.shape[0], *wav.shape[1:]), dtype=wav.dtype)], axis=0)
            out = model.predict(x_in.reshape(1, INPUT_SEQ_LEN, *wav.shape[1:]), verbose=0)
            if out.ndim == 2:
                out = out.reshape(1, -1, out.shape[-1])
            if out.ndim == 4:
                out = out.reshape(1, out.shape[1], out.shape[-1])
            out120 = out[0]
            out96 = out120[-SEQ_LEN:] if out120.shape[0] >= SEQ_LEN else out120
            L_use = min(center_end - center_start, out96.shape[0], SEQ_LEN)
            if L_use > 0:
                preds[center_start:center_start + L_use] = out96[:L_use]
        T_eff = min(len(ticks), preds.shape[0])
        ticks = ticks[:T_eff]
        timestamps = timestamps[:T_eff]
        wav = wav[:T_eff]
        preds = preds[:T_eff]
        T = T_eff
    if preds.shape[1] < 13:
        raise ValueError(f'Classifier output dimension too small ({preds.shape[1]}), at least 13 dimensions are required to represent [is_note_start, counts(12)].')
    logits = preds
    logit_note = logits[:, 0:1]
    counts_pred = logits[:, 1:]
    if NOTE_IN_ARC_THRESHOLD is not None:
        counts_pred[:, 6] = np.where(counts_pred[:, 6] >= float(NOTE_IN_ARC_THRESHOLD), counts_pred[:, 6], 0.0)
        counts_pred[:, 7] = np.where(counts_pred[:, 7] >= float(NOTE_IN_ARC_THRESHOLD), counts_pred[:, 7], 0.0)
    if NOTE_IN_HOLD_THRESHOLD is not None:
        counts_pred[:, 8] = np.where(counts_pred[:, 8] >= float(NOTE_IN_HOLD_THRESHOLD), counts_pred[:, 8], 0.0)
        counts_pred[:, 9] = np.where(counts_pred[:, 9] >= float(NOTE_IN_HOLD_THRESHOLD), counts_pred[:, 9], 0.0)
    if NOTE_IN_ARC_THRESHOLD is not None:
        counts_pred[:, 10] = np.where(counts_pred[:, 10] >= float(NOTE_IN_ARC_THRESHOLD), counts_pred[:, 10], 0.0)
        counts_pred[:, 11] = np.where(counts_pred[:, 11] >= float(NOTE_IN_ARC_THRESHOLD), counts_pred[:, 11], 0.0)
    is_note_scores = 1.0 / (1.0 + np.exp(-logit_note[:, 0]))
    T_total = is_note_scores.shape[0]
    is_note = np.zeros(T_total, dtype=np.int32)
    legal_mask = None
    if frac_index_24 is not None:
        legal_mask = np.isin(frac_index_24[:T_total], LEGAL_FRAC_24)
    if division_code is not None:
        codes = np.asarray(division_code, dtype=np.int32)
        max_code = DIVISION_PENALTY_TABLE.shape[0] - 1
        codes = np.clip(codes, 0, max_code)
        penalties = DIVISION_PENALTY_TABLE[codes].astype(np.float32)
        penalties = np.clip(penalties, 0.0, 1.0)
        allow_mask = penalties < 1.0 - 1e-06
        if legal_mask is not None:
            allow_mask = allow_mask & legal_mask
        if np.any(allow_mask):
            decay = 1.0 - penalties
            scores_eff = is_note_scores * decay
            scores_allowed = scores_eff[allow_mask]
            target_count = int(np.round(note_density * scores_allowed.shape[0]))
            target_count = max(1, min(target_count, scores_allowed.shape[0]))
            sorted_scores = np.sort(scores_allowed)
            borderline = sorted_scores[-target_count]
            select_mask = allow_mask & (scores_eff >= borderline)
            is_note[select_mask] = 1
        else:
            scores_eff = is_note_scores
            target_count = int(np.round(note_density * scores_eff.shape[0]))
            target_count = max(1, min(target_count, scores_eff.shape[0]))
            sorted_scores = np.sort(scores_eff)
            borderline = sorted_scores[-target_count]
            is_note = (scores_eff >= borderline).astype(np.int32)
    else:
        scores_eff = is_note_scores
        if legal_mask is not None:
            scores_eff = np.where(legal_mask, scores_eff, -1e9)
        target_count = int(np.round(note_density * scores_eff.shape[0]))
        target_count = max(1, min(target_count, scores_eff.shape[0]))
        sorted_scores = np.sort(scores_eff)
        borderline = sorted_scores[-target_count]
        is_note = (scores_eff >= borderline).astype(np.int32)

    if legal_mask is not None:
        is_note = np.where(legal_mask, is_note, 0).astype(np.int32)
    counts_clipped = np.clip(np.round(counts_pred), 0.0, 8.0)
    tap_count = counts_clipped[:, 0:1]
    hold_count = counts_clipped[:, 1:2]
    arcfalse0_count = counts_clipped[:, 2:3]
    arcfalse1_count = counts_clipped[:, 3:4]
    arctrue_count = counts_clipped[:, 4:5]
    arctap_count = counts_clipped[:, 5:6]
    if NOTE_IN_ARC_THRESHOLD is not None:
        thr_arc = float(NOTE_IN_ARC_THRESHOLD)
        in_arcfalse0_count = (counts_pred[:, 6:7] >= thr_arc).astype(np.float32)
        in_arcfalse1_count = (counts_pred[:, 7:8] >= thr_arc).astype(np.float32)
        in_arctrue0_count = (counts_pred[:, 10:11] >= thr_arc).astype(np.float32)
        in_arctrue1_count = (counts_pred[:, 11:12] >= thr_arc).astype(np.float32)
    else:
        in_arcfalse0_count = (counts_pred[:, 6:7] > 0.0).astype(np.float32)
        in_arcfalse1_count = (counts_pred[:, 7:8] > 0.0).astype(np.float32)
        in_arctrue0_count = (counts_pred[:, 10:11] > 0.0).astype(np.float32)
        in_arctrue1_count = (counts_pred[:, 11:12] > 0.0).astype(np.float32)
    if NOTE_IN_HOLD_THRESHOLD is not None:
        thr_hold = float(NOTE_IN_HOLD_THRESHOLD)
        in_hold0_count = (counts_pred[:, 8:9] >= thr_hold).astype(np.float32)
        in_hold1_count = (counts_pred[:, 9:10] >= thr_hold).astype(np.float32)
    else:
        in_hold0_count = (counts_pred[:, 8:9] > 0.0).astype(np.float32)
        in_hold1_count = (counts_pred[:, 9:10] > 0.0).astype(np.float32)
    in_arcfalse0 = (in_arcfalse0_count[:, 0] > 0.5).astype(np.int32)
    in_arcfalse1 = (in_arcfalse1_count[:, 0] > 0.5).astype(np.int32)
    in_arc = (in_arcfalse0 + in_arcfalse1 > 0).astype(np.int32)
    in_hold0 = (in_hold0_count[:, 0] > 0.5).astype(np.int32)
    in_hold1 = (in_hold1_count[:, 0] > 0.5).astype(np.int32)
    in_hold = (in_hold0 + in_hold1 > 0).astype(np.int32)
    T_eff = len(is_note)
    type_kinds = ['tap', 'hold', 'arcfalse0', 'arcfalse1', 'arctrue', 'arctap']
    kind_priority = {name: idx for idx, name in enumerate(type_kinds + ['arc_cont', 'hold_cont'])}
    for i in range(T_eff):
        if not bool(is_note[i]):
            tap_count[i, 0] = 0.0
            hold_count[i, 0] = 0.0
            arcfalse0_count[i, 0] = 0.0
            arcfalse1_count[i, 0] = 0.0
            arctap_count[i, 0] = 0.0
            in_arcfalse0_count[i, 0] = 0.0
            in_arcfalse1_count[i, 0] = 0.0
            in_hold0_count[i, 0] = 0.0
            in_hold1_count[i, 0] = 0.0
            in_arctrue0_count[i, 0] = 0.0
            in_arctrue1_count[i, 0] = 0.0
            continue
        candidates: list[tuple[float, str, str]] = []
        if NOTE_TYPE_THRESHOLDS is not None and NOTE_TYPE_THRESHOLDS.shape[0] >= 6:
            for k, kind in enumerate(type_kinds):
                if kind == 'arctrue':
                    continue
                v = float(counts_pred[i, k])
                thr_k = float(NOTE_TYPE_THRESHOLDS[k])
                if thr_k <= 0.0 or v <= thr_k:
                    continue
                desire = v / thr_k
                n_inst = int(np.floor(desire + 0.5))
                if n_inst <= 0:
                    n_inst = 1
                score = float(desire)
                for _ in range(n_inst):
                    candidates.append((score, kind, ''))
        else:
            if tap_count[i, 0] > 0.5:
                candidates.append((1.0, 'tap', ''))
            if hold_count[i, 0] > 0.5:
                candidates.append((1.0, 'hold', ''))
            if arcfalse0_count[i, 0] > 0.5:
                candidates.append((1.0, 'arcfalse0', ''))
            if arcfalse1_count[i, 0] > 0.5:
                candidates.append((1.0, 'arcfalse1', ''))
            if arctap_count[i, 0] > 0.5:
                candidates.append((1.0, 'arctap', ''))
        thr_arc = float(NOTE_IN_ARC_THRESHOLD) if NOTE_IN_ARC_THRESHOLD is not None else 0.0

        def _maybe_add_arc_cont(slot_name: str, slot_in_count: float, slot_onset_count: float, slot_pred_val: float):
            """returns: None"""
            if slot_in_count <= 0.5 or slot_onset_count >= 0.5:
                return
            base_val = float(slot_pred_val)
            if thr_arc > 0.0 and base_val > thr_arc:
                score = base_val / thr_arc
                candidates.append((float(score), 'arc_cont', slot_name))
            elif thr_arc <= 0.0 and base_val > 0.0:
                candidates.append((float(base_val), 'arc_cont', slot_name))
        _maybe_add_arc_cont('arcfalse0', in_arcfalse0_count[i, 0], arcfalse0_count[i, 0], float(counts_pred[i, 6]))
        _maybe_add_arc_cont('arcfalse1', in_arcfalse1_count[i, 0], arcfalse1_count[i, 0], float(counts_pred[i, 7]))
        if hold_count[i, 0] < 0.5:
            thr_hold = float(NOTE_IN_HOLD_THRESHOLD) if NOTE_IN_HOLD_THRESHOLD is not None else 0.0

            def _maybe_add_hold_cont(slot_idx: int, slot_val: float):
                """returns: None"""
                base_val = float(slot_val)
                if thr_hold > 0.0 and base_val > thr_hold:
                    score = base_val / thr_hold
                    candidates.append((float(score), 'hold_cont', str(slot_idx)))
                elif thr_hold <= 0.0 and base_val > 0.0:
                    candidates.append((float(base_val), 'hold_cont', str(slot_idx)))
            if in_hold0_count[i, 0] > 0.5:
                _maybe_add_hold_cont(0, counts_pred[i, 8])
            if in_hold1_count[i, 0] > 0.5:
                _maybe_add_hold_cont(1, counts_pred[i, 9])
        if not candidates:
            tap_count[i, 0] = 0.0
            hold_count[i, 0] = 0.0
            arcfalse0_count[i, 0] = 0.0
            arcfalse1_count[i, 0] = 0.0
            arctrue_count[i, 0] = 0.0
            arctap_count[i, 0] = 0.0
            in_arcfalse0_count[i, 0] = 0.0
            in_arcfalse1_count[i, 0] = 0.0
            in_hold0_count[i, 0] = 0.0
            in_hold1_count[i, 0] = 0.0
            in_arctrue0_count[i, 0] = 0.0
            in_arctrue1_count[i, 0] = 0.0
            continue
        max_f = int(max_finger_count) if max_finger_count is not None else 0
        candidates.sort(key=lambda x: (-x[0], kind_priority.get(x[1], 999)))
        selected = candidates[:max(0, max_f) or 0]
        tap_count[i, 0] = 0.0
        hold_count[i, 0] = 0.0
        arcfalse0_count[i, 0] = 0.0
        arcfalse1_count[i, 0] = 0.0
        arctap_count[i, 0] = 0.0
        arcfalse0_cont_selected = False
        arcfalse1_cont_selected = False
        hold0_cont_selected = False
        hold1_cont_selected = False
        for _, kind, slot_info in selected:
            if kind == 'tap':
                tap_count[i, 0] += 1.0
            elif kind == 'hold':
                hold_count[i, 0] += 1.0
            elif kind == 'arcfalse0':
                arcfalse0_count[i, 0] += 1.0
            elif kind == 'arcfalse1':
                arcfalse1_count[i, 0] += 1.0
            elif kind == 'arctap':
                arctap_count[i, 0] += 1.0
            elif kind == 'arc_cont':
                if slot_info == 'arcfalse0':
                    arcfalse0_cont_selected = True
                elif slot_info == 'arcfalse1':
                    arcfalse1_cont_selected = True
            elif kind == 'hold_cont':
                if slot_info == '0':
                    hold0_cont_selected = True
                elif slot_info == '1':
                    hold1_cont_selected = True
        if not arcfalse0_cont_selected:
            in_arcfalse0_count[i, 0] = 0.0
        if not arcfalse1_cont_selected:
            in_arcfalse1_count[i, 0] = 0.0
        if not hold0_cont_selected:
            in_hold0_count[i, 0] = 0.0
        if not hold1_cont_selected:
            in_hold1_count[i, 0] = 0.0
    is_note_start = (tap_count + hold_count + arcfalse0_count + arcfalse1_count + arctap_count).astype(np.float32)
    in_arcfalse0 = (in_arcfalse0_count[:, 0] > 0.5).astype(np.int32)
    in_arcfalse1 = (in_arcfalse1_count[:, 0] > 0.5).astype(np.int32)
    in_arc = (in_arcfalse0 + in_arcfalse1 > 0).astype(np.int32)
    in_hold0 = (in_hold0_count[:, 0] > 0.5).astype(np.int32)
    in_hold1 = (in_hold1_count[:, 0] > 0.5).astype(np.int32)
    in_hold = (in_hold0 + in_hold1 > 0).astype(np.int32)
    labels = np.concatenate([is_note_start, tap_count, hold_count, arcfalse0_count, arcfalse1_count, arctrue_count, arctap_count, in_arcfalse0_count, in_arcfalse1_count, in_hold0_count, in_hold1_count, in_arctrue0_count, in_arctrue1_count], axis=1)
    os.makedirs(RHYTHM_DIR, exist_ok=True)
    os.makedirs(RHYTHM_VIS_DIR, exist_ok=True)
    for fn in os.listdir(RHYTHM_VIS_DIR):
        if fn.endswith('.tsv'):
            try:
                os.remove(os.path.join(RHYTHM_VIS_DIR, fn))
            except OSError:
                pass
    out_path = os.path.join(RHYTHM_DIR, 'rhythm.npz')
    np.savez_compressed(out_path, ticks=ticks, timestamps=timestamps, division_code=division_code if division_code is not None else None, is_note=is_note, labels=labels, in_arc=in_arc, in_hold=in_hold, raw_preds=preds, extra=extra, p_is_note=is_note_scores)
    vis_path = os.path.join(RHYTHM_VIS_DIR, '0.tsv')
    with open(vis_path, 'w', encoding='utf-8') as vf:
        header = 'tick_idx\ttimestamp_ms\tdivision\tis_note_start\ttap_count\thold_count\tarcfalse0_count\tarcfalse1_count\tarctrue_count\tarctap_count\tin_arcfalse0\tin_arcfalse1\tin_hold\n'
        vf.write(header)
        T_out = len(ticks)
        for i in range(T_out):
            tick_idx = i
            timestamp_ms = int(timestamps[i])
            if division_code is not None:
                code = int(division_code[i])
                div_name = DIVISION_CODE_TO_NAME.get(code, 'unknown')
            else:
                div_name = 'unknown'
            in_hold_vis = int(in_hold0_count[i, 0] > 0.5 or in_hold1_count[i, 0] > 0.5)
            vf.write(f'{tick_idx}\t{timestamp_ms}\t{div_name}\t{int(is_note_start[i, 0])}\t{int(tap_count[i, 0])}\t{int(hold_count[i, 0])}\t{int(arcfalse0_count[i, 0])}\t{int(arcfalse1_count[i, 0])}\t{int(arctrue_count[i, 0])}\t{int(arctap_count[i, 0])}\t{int(in_arcfalse0[i])}\t{int(in_arcfalse1[i])}\t{in_hold_vis}\n')
    all_pred_path = os.path.join(RHYTHM_VIS_DIR, '0_pred.tsv')
    with open(all_pred_path, 'w', encoding='utf-8') as pf:
        pf.write('tick_idx\ttimestamp_ms\tnote_total\ttap_count_pred\thold_count_pred\tarcfalse0_count_pred\tarcfalse1_count_pred\tarctrue_count_pred\tarctap_count_pred\tin_arcfalse0_count_pred\tin_arcfalse1_count_pred\tin_hold0_count_pred\tin_hold1_count_pred\tin_arctrue0_count_pred\tin_arctrue1_count_pred\n')
        T_out = len(ticks)
        for i in range(T_out):
            tick_idx = i
            timestamp_ms = int(timestamps[i])
            c = counts_pred[i]
            note_total = tap_count[i, 0] + hold_count[i, 0] + arcfalse0_count[i, 0] + arcfalse1_count[i, 0] + arctrue_count[i, 0] + arctap_count[i, 0]
            pf.write(f'{tick_idx}\t{timestamp_ms}\t{int(note_total)}\t{float(c[0]):.6f}\t{float(c[1]):.6f}\t{float(c[2]):.6f}\t{float(c[3]):.6f}\t{float(c[4]):.6f}\t{float(c[5]):.6f}\t{float(c[6]):.6f}\t{float(c[7]):.6f}\t{float(c[8]):.6f}\t{float(c[9]):.6f}\t{float(c[10]):.6f}\t{float(c[11]):.6f}\n')
    print(f'saved rhythm predictions to: {out_path}')
    print(f'saved rhythm visualization to: {vis_path}')
    return out_path
if __name__ == '__main__':
    predict_notes()
