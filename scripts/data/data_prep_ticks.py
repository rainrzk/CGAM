from __future__ import annotations
import os
import time
import numpy as np
from ..utils.tools import read_aff_file, get_aff_directory
from ..utils.audio_tools import read_wav_data
from .common_grid import build_ticks_and_raw_labels, compute_onset_type_from_raw_labels
from ..utils.grid_quantizer import _build_segment_indices, _find_timing_segment, _quantize_time_to_tick
from ..utils.config_utils import GRID_LABEL_TOLERANCE_MS
from ..utils.config_utils import ROOT_DIR, PATHS, DIVISION_CODE_TO_NAME, frac24_to_division_code
SCRIPTS_DIR = os.path.dirname(__file__)
MAPDATA_TICKS_PATH = os.path.join(ROOT_DIR, PATHS['ticks_dir'])
MAPDATA_TICKS_VIS_PATH = os.path.join(ROOT_DIR, PATHS.get('ticks_vis_dir', os.path.join('data', 'mapdata_ticks_vis')))
MAPLIST_FILE = os.path.join(ROOT_DIR, PATHS['maplist_path'])
LANE_COUNT = 4
MAX_ARCTRUE_PER_TICK = 4
MAX_ARCTAP_PER_TICK = 4
MAX_HOLD_SLOTS = 2
MAX_ARCTRUE_SLOTS = 2

def _rising_edge(arr: np.ndarray) -> np.ndarray:
    """returns: np.ndarray"""
    prev = np.vstack([np.zeros((1, 1), dtype=arr.dtype), arr[:-1]])
    return (arr > 0.5) & (prev <= 0.5)

def load_maps_ticks(audio_name: str='base.ogg') -> None:
    """returns: None"""
    os.makedirs(MAPDATA_TICKS_PATH, exist_ok=True)
    os.makedirs(MAPDATA_TICKS_VIS_PATH, exist_ok=True)
    if not os.path.isfile(MAPLIST_FILE):
        raise FileNotFoundError(f'Cannot find maplist file: {MAPLIST_FILE}')
    with open(MAPLIST_FILE, encoding='utf-8') as fp:
        lines = [ln.strip() for ln in fp.readlines() if ln.strip()]
    print(f'entries in maplist_aff.txt: {len(lines)}')
    for file in os.listdir(MAPDATA_TICKS_PATH):
        if file.endswith('.npz'):
            os.remove(os.path.join(MAPDATA_TICKS_PATH, file))
    for file in os.listdir(MAPDATA_TICKS_VIS_PATH):
        if file.endswith('.txt') or file.endswith('.tsv'):
            os.remove(os.path.join(MAPDATA_TICKS_VIS_PATH, file))
    idx = 0
    all_ticks = []
    all_timestamps = []
    all_div_codes = []
    all_wav = []
    all_timing = []
    all_labels = []
    all_is_tap_lanes = []
    all_is_hold_lanes = []
    all_in_hold_lanes = []
    all_arctrue_ids_per_tick = []
    all_arctap_arctrue_ids_per_tick = []
    for entry in lines:
        try:
            start = time.time()
            base_path = entry
            if not os.path.isabs(base_path):
                base_path = os.path.abspath(os.path.join(ROOT_DIR, base_path))
            if os.path.isdir(base_path):
                song_dir = base_path
                wav_path = os.path.join(song_dir, audio_name)
                if not os.path.isfile(wav_path):
                    raise FileNotFoundError(f'Audio file not found in song directory: {wav_path}')
                aff_files = [f for f in os.listdir(song_dir) if f.lower().endswith('.aff')]
                if not aff_files:
                    raise FileNotFoundError(f'曲目目录中找不到任何 .aff 文件: {song_dir}')
                for aff_name in aff_files:
                    aff_path = os.path.join(song_dir, aff_name)
                    res = _process_single_chart(aff_path, wav_path, idx)
                    if res is not None:
                        ticks_arr, timestamps_arr, div_codes, wav_data, timing_arr, labels_new, is_tap_lanes, is_hold_lanes, in_hold_lanes, arctrue_ids_per_tick, arctap_arctrue_ids_per_tick = res
                        all_ticks.append(ticks_arr)
                        all_timestamps.append(timestamps_arr)
                        all_div_codes.append(div_codes)
                        all_wav.append(wav_data)
                        all_timing.append(timing_arr)
                        all_labels.append(labels_new)
                        all_is_tap_lanes.append(is_tap_lanes)
                        all_is_hold_lanes.append(is_hold_lanes)
                        all_in_hold_lanes.append(in_hold_lanes)
                        all_arctrue_ids_per_tick.append(arctrue_ids_per_tick)
                        all_arctap_arctrue_ids_per_tick.append(arctap_arctrue_ids_per_tick)
                        idx += 1
            elif os.path.isfile(base_path) and base_path.lower().endswith('.aff'):
                aff_path = base_path
                chart = read_aff_file(aff_path)
                aff_dir = get_aff_directory(aff_path)
                wav_path = os.path.join(aff_dir, audio_name)
                if not os.path.isfile(wav_path):
                    raise FileNotFoundError(f'Audio file not found: {wav_path}')
                res = _process_single_chart(aff_path, wav_path, idx)
                if res is not None:
                    ticks_arr, timestamps_arr, div_codes, wav_data, timing_arr, labels_new, is_tap_lanes, is_hold_lanes, in_hold_lanes, arctrue_ids_per_tick, arctap_arctrue_ids_per_tick = res
                    all_ticks.append(ticks_arr)
                    all_timestamps.append(timestamps_arr)
                    all_div_codes.append(div_codes)
                    all_wav.append(wav_data)
                    all_timing.append(timing_arr)
                    all_labels.append(labels_new)
                    all_is_tap_lanes.append(is_tap_lanes)
                    all_is_hold_lanes.append(is_hold_lanes)
                    all_in_hold_lanes.append(in_hold_lanes)
                    all_arctrue_ids_per_tick.append(arctrue_ids_per_tick)
                    all_arctap_arctrue_ids_per_tick.append(arctap_arctrue_ids_per_tick)
                    idx += 1
            else:
                raise FileNotFoundError(f'Entry in maplist_aff.txt is neither a directory nor an .aff file: {base_path}')
            end = time.time()
            print(f'tick data upto #{idx} processed in {end - start:.2f} s for entry {entry}')
        except Exception as e:
            print(f"error on entry '{entry}', error = {e}")
    if not all_ticks:
        raise RuntimeError('load_maps_ticks did not generate any tick data; please check maplist_aff.txt and the audio/chart files.')
    ticks_all = np.concatenate(all_ticks, axis=0)
    timestamps_all = np.concatenate(all_timestamps, axis=0)
    div_all = np.concatenate(all_div_codes, axis=0)
    wav_all = np.concatenate(all_wav, axis=0)
    timing_all = np.concatenate(all_timing, axis=1)
    labels_all = np.concatenate(all_labels, axis=0)
    is_tap_lanes_all = np.concatenate(all_is_tap_lanes, axis=0)
    is_hold_lanes_all = np.concatenate(all_is_hold_lanes, axis=0)
    in_hold_lanes_all = np.concatenate(all_in_hold_lanes, axis=0)
    arctrue_ids_all = np.concatenate(all_arctrue_ids_per_tick, axis=0)
    arctap_arctrue_ids_all = np.concatenate(all_arctap_arctrue_ids_per_tick, axis=0)
    all_path = os.path.join(MAPDATA_TICKS_PATH, 'mapdata.npz')
    np.savez_compressed(all_path, ticks=ticks_all, timestamps=timestamps_all, division_code=div_all, wav=wav_all, timing=timing_all, labels=labels_all, is_tap_lanes=is_tap_lanes_all, is_hold_lanes=is_hold_lanes_all, in_hold_lanes=in_hold_lanes_all, arctrue_ids_per_tick=arctrue_ids_all, arctap_arctrue_ids_per_tick=arctap_arctrue_ids_all)
    print(f'mapdata.npz saved at: {all_path}, total ticks = {ticks_all.shape[0]}')

def _process_single_chart(aff_path: str, wav_path: str, idx: int):
    """returns: tuple | None"""
    chart = read_aff_file(aff_path)
    ticks_ms, frac_index, timing_index, labels_raw = build_ticks_and_raw_labels(chart, tol_ms=GRID_LABEL_TOLERANCE_MS)
    timestamps_model = ticks_ms
    timings = sorted(chart.timings, key=lambda t: t.time)
    timing_time: list[float] = []
    timing_bpm: list[float] = []
    timing_beats: list[float] = []
    for seg_idx in timing_index:
        seg = timings[int(seg_idx)]
        timing_time.append(float(seg.time))
        timing_bpm.append(float(seg.bpm))
        timing_beats.append(float(seg.beats))
    timing_arr = np.stack([np.array(timing_time, dtype=np.float32), np.array(timing_bpm, dtype=np.float32), np.array(timing_beats, dtype=np.float32)], axis=0)
    timestamps_audio = [t + chart.audio_offset for t in ticks_ms]
    wav_data = read_wav_data(timestamps_audio, wav_path, fft_size=128)
    wav_data = np.swapaxes(wav_data, 0, 1)
    div_codes = np.array([frac24_to_division_code(int(f)) for f in frac_index], dtype=np.int32)
    timestamps_arr = np.array(timestamps_model, dtype=np.float32)
    T = len(timestamps_arr)
    raw = labels_raw.astype(np.float32)
    is_arcfalse0 = raw[:, 3:4]
    is_arcfalse1 = raw[:, 4:5]
    is_arctrue = raw[:, 5:6]
    is_arctap = raw[:, 6:7]
    onset_type = compute_onset_type_from_raw_labels(labels_raw)
    ticks_arr = np.asarray(ticks_ms, dtype=np.int32)
    timing_index_arr = np.asarray(timing_index, dtype=np.int32)
    n_segments = len(timings)
    seg_tick_indices = _build_segment_indices(timing_index_arr, n_segments)
    is_tap_lanes = np.zeros((T, LANE_COUNT), dtype=np.float32)
    hold_occ = np.zeros((T, LANE_COUNT), dtype=np.float32)
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
        is_tap_lanes[int(j), lane] = 1.0
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
        j_end = _quantize_time_to_tick(end, seg_end, ticks_arr, seg_tick_indices, float(GRID_LABEL_TOLERANCE_MS))
        if j_start is None or j_end is None:
            continue
        a_idx, b_idx = sorted([int(j_start), int(j_end)])
        hold_occ[a_idx:b_idx + 1, lane] = 1.0
    in_hold_lanes = hold_occ.astype(np.float32)
    is_hold_lanes = np.zeros_like(in_hold_lanes, dtype=np.float32)
    if T > 0:
        for lane in range(LANE_COUNT):
            col = in_hold_lanes[:, lane:lane + 1]
            r = _rising_edge(col)
            is_hold_lanes[r[:, 0] > 0.5, lane] = 1.0
    trace_arcs = [a for a in chart.arcs if getattr(a, 'is_trace', False)]
    arctrue_ids_per_tick = np.full((T, MAX_ARCTRUE_PER_TICK), -1, dtype=np.int32)
    arctap_arctrue_ids_per_tick = np.full((T, MAX_ARCTAP_PER_TICK), -1, dtype=np.int32)
    arctap_count_arr = np.zeros((T, 1), dtype=np.float32)
    arctrue_start_count_arr = np.zeros((T, 1), dtype=np.float32)
    for arc_id, a in enumerate(trace_arcs):
        start = float(a.start)
        end = float(a.end)
        if end < start:
            start, end = (end, start)
        seg_start = _find_timing_segment(timings, start)
        seg_end = _find_timing_segment(timings, end)
        if seg_start != seg_end:
            continue
        j_start = _quantize_time_to_tick(start, seg_start, ticks_arr, seg_tick_indices, float(GRID_LABEL_TOLERANCE_MS))
        j_end = _quantize_time_to_tick(end, seg_end, ticks_arr, seg_tick_indices, float(GRID_LABEL_TOLERANCE_MS))
        if j_start is None or j_end is None:
            continue
        a_idx, b_idx = sorted([int(j_start), int(j_end)])
        arctrue_start_count_arr[int(j_start), 0] += 1.0
        for t_i in range(a_idx, b_idx + 1):
            slots = arctrue_ids_per_tick[t_i]
            if (slots == arc_id).any():
                continue
            free = np.where(slots == -1)[0]
            if free.size == 0:
                continue
            slots[free[0]] = arc_id
        if not getattr(a, 'arctaps', None):
            continue
        arc_tick_indices = np.arange(a_idx, b_idx + 1, dtype=np.int32)
        arc_tick_times = ticks_arr[arc_tick_indices].astype(np.float32)
        for at in a.arctaps:
            t = float(at.time)
            if t < start - float(GRID_LABEL_TOLERANCE_MS) or t > end + float(GRID_LABEL_TOLERANCE_MS):
                continue
            if arc_tick_times.size == 0:
                continue
            j_local = int(np.argmin(np.abs(arc_tick_times - t)))
            j = int(arc_tick_indices[j_local])
            if abs(float(arc_tick_times[j_local]) - t) >= float(GRID_LABEL_TOLERANCE_MS):
                continue
            slots = arctap_arctrue_ids_per_tick[j]
            free = np.where(slots == -1)[0]
            if free.size == 0:
                continue
            slots[free[0]] = arc_id
            arctap_count_arr[j, 0] += 1.0
    tap_count = np.sum(is_tap_lanes, axis=1, keepdims=True).astype(np.float32)
    hold_count = np.sum(is_hold_lanes, axis=1, keepdims=True).astype(np.float32)
    arcfalse0_count = onset_type[:, 2:3].astype(np.float32)
    arcfalse1_count = onset_type[:, 3:4].astype(np.float32)
    arctrue_count = arctrue_start_count_arr.astype(np.float32)
    arctap_count = arctap_count_arr.astype(np.float32)
    is_note_start = tap_count + hold_count + arcfalse0_count + arcfalse1_count + arctap_count
    in_arcfalse0 = is_arcfalse0.astype(np.float32)
    in_arcfalse1 = is_arcfalse1.astype(np.float32)
    in_hold_slots = np.zeros((T, MAX_HOLD_SLOTS), dtype=np.float32)
    for t_i in range(T):
        active_lanes = np.where(in_hold_lanes[t_i] > 0.5)[0]
        for slot_idx, _lane in enumerate(active_lanes[:MAX_HOLD_SLOTS]):
            in_hold_slots[t_i, slot_idx] = 1.0
    in_hold0 = in_hold_slots[:, 0:1]
    in_hold1 = in_hold_slots[:, 1:2] if MAX_HOLD_SLOTS > 1 else np.zeros_like(in_hold0)
    in_arctrue_slots = (arctrue_ids_per_tick[:, :MAX_ARCTRUE_SLOTS] >= 0).astype(np.float32)
    if in_arctrue_slots.shape[1] < MAX_ARCTRUE_SLOTS:
        pad = np.zeros((T, MAX_ARCTRUE_SLOTS - in_arctrue_slots.shape[1]), dtype=np.float32)
        in_arctrue_slots = np.concatenate([in_arctrue_slots, pad], axis=1)
    in_arctrue0 = in_arctrue_slots[:, 0:1]
    in_arctrue1 = in_arctrue_slots[:, 1:2]
    labels_new = np.concatenate([is_note_start.astype(np.float32), tap_count, hold_count, arcfalse0_count, arcfalse1_count, arctrue_count, arctap_count, in_arcfalse0, in_arcfalse1, in_hold0, in_hold1, in_arctrue0, in_arctrue1], axis=1)
    vis_path = os.path.join(MAPDATA_TICKS_VIS_PATH, f'{idx}.tsv')
    with open(vis_path, 'w', encoding='utf-8') as vf:
        vf.write('tick_idx\ttimestamp_ms\tdivision\tis_note_start\ttap_count\thold_count\tarcfalse0_count\tarcfalse1_count\tarctue_count\tarctap_count\tin_arcfalse0\tin_arcfalse1\tin_hold0\tin_hold1\tin_arctrue0\tin_arctrue1\n')
        for t_i in range(T):
            if labels_new[t_i, 0] <= 0.0:
                continue
            div_name = DIVISION_CODE_TO_NAME.get(int(div_codes[t_i]), 'unknown')
            vf.write(f'{t_i}\t{int(ticks_ms[t_i])}\t{div_name}\t{int(labels_new[t_i, 0])}\t{int(labels_new[t_i, 1])}\t{int(labels_new[t_i, 2])}\t{int(labels_new[t_i, 3])}\t{int(labels_new[t_i, 4])}\t{int(labels_new[t_i, 5])}\t{int(labels_new[t_i, 6])}\t{int(labels_new[t_i, 7])}\t{int(labels_new[t_i, 8])}\t{int(labels_new[t_i, 9])}\t{int(labels_new[t_i, 10])}\t{int(labels_new[t_i, 11])}\t{int(labels_new[t_i, 12])}\n')
    ticks_arr = np.arange(len(ticks_ms), dtype=np.int32)
    return (ticks_arr, np.array(timestamps_model, dtype=np.int32), div_codes, wav_data, timing_arr, labels_new, is_tap_lanes, is_hold_lanes, in_hold_lanes, arctrue_ids_per_tick, arctap_arctrue_ids_per_tick)
if __name__ == '__main__':
    load_maps_ticks()
