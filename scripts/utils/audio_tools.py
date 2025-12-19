from __future__ import annotations
import os
from typing import List, Tuple, Dict, Any
import numpy as np
import librosa
from .tools import read_aff_file, get_aff_directory
from .timing_tools import build_tick_divisor_features
from .grid_quantizer import build_grid
from .config_utils import ROOT_DIR, PATHS

def get_freqs(sig: np.ndarray, fft_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """returns: Tuple[np.ndarray, np.ndarray]"""
    Lf = np.fft.fft(sig, fft_size)
    Lc = Lf[0:fft_size // 2]
    La = np.abs(Lc[0:fft_size // 2])
    Lg = np.angle(Lc[0:fft_size // 2])
    return (La, Lg)

def slice_wave_at(ms: float, sig: np.ndarray, samplerate: int, size: int) -> np.ndarray:
    """returns: np.ndarray"""
    ind = ms / 1000.0 * samplerate // 1
    ind = int(ind)
    return sig[max(0, ind - size // 2):int(ind + size - size // 2)]

def get_wav_data_at(ms: float, sig: np.ndarray, samplerate: int, fft_size: int=1024, freq_low: int=0, freq_high: int=-1) -> Tuple[np.ndarray, np.ndarray]:
    """returns: Tuple[np.ndarray, np.ndarray]"""
    if freq_high == -1:
        freq_high = samplerate // 2
    waveslice = slice_wave_at(ms, sig, samplerate, fft_size)
    La, Lg = get_freqs(waveslice, fft_size)
    La = La[fft_size * freq_low // samplerate:fft_size * freq_high // samplerate]
    Lg = Lg[fft_size * freq_low // samplerate:fft_size * freq_high // samplerate]
    return (La, Lg)

def read_wav_data(timestamps: List[float], wavfile: str, snapint: List[float] | None=None, fft_size: int=1024) -> np.ndarray:
    """returns: np.ndarray"""
    if snapint is None:
        snapint = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]
    sig, samplerate = librosa.load(wavfile, sr=None, mono=True)
    sig = sig / np.max(np.abs(sig))
    tmpts = np.array(timestamps)
    if len(tmpts) <= 1:
        timestamp_interval = np.array([500.0])
    else:
        timestamp_interval = tmpts[1:] - tmpts[:-1]
        timestamp_interval = np.append(timestamp_interval, timestamp_interval[-1])
    data = []
    for sz in snapint:
        data_r = np.array([get_wav_data_at(max(0.0, min(len(sig) - fft_size, coord + timestamp_interval[i] * sz)), sig, samplerate, fft_size=fft_size, freq_high=samplerate // 4) for i, coord in enumerate(timestamps)])
        data.append(data_r)
    raw_data = np.array(data)
    norm_data = np.tile(np.expand_dims(np.mean(raw_data, axis=1), 1), (1, raw_data.shape[1], 1, 1))
    std_data = np.tile(np.expand_dims(np.std(raw_data, axis=1), 1), (1, raw_data.shape[1], 1, 1))
    return (raw_data - norm_data) / std_data

def _build_event_list(chart: 'AffChart') -> Tuple[List[int], List[List[float]]]:
    """returns: Tuple[List[int], List[List[float]]]"""
    rows: List[List[float]] = []
    for n in chart.taps:
        rows.append([float(n.time), 1.0, float(n.lane)])
    for n in chart.holds:
        rows.append([float(n.start), 2.0, float(n.lane)])
        rows.append([float(n.end), 3.0, float(n.lane)])
    for a in chart.arcs:
        rows.append([float(a.start), 4.0, a.x_start, a.x_end, a.y_start, a.y_end, float(a.hand)])
        for t in a.arctaps:
            rows.append([float(t.time), 5.0, a.x_start])
    rows.sort(key=lambda r: r[0])
    timestamps = [int(r[0]) for r in rows]
    return (timestamps, rows)

def read_and_save_aff_file(aff_path: str, filename: str='saved', audio_name: str='base.ogg') -> None:
    """returns: None"""
    chart = read_aff_file(aff_path)
    aff_dir = get_aff_directory(aff_path)
    if os.path.isabs(audio_name):
        wav_path = audio_name
    else:
        wav_path_candidate = os.path.join(aff_dir, audio_name)
        if os.path.isfile(wav_path_candidate):
            wav_path = wav_path_candidate
        else:
            wav_path_candidate = os.path.join(ROOT_DIR, audio_name)
            if os.path.isfile(wav_path_candidate):
                wav_path = wav_path_candidate
    timestamps, lst_rows = _build_event_list(chart)
    wav_data = read_wav_data(timestamps, wav_path, fft_size=128)
    wav_data = np.swapaxes(wav_data, 0, 1)
    lst = np.array(lst_rows, dtype=np.float32)
    flow = np.zeros((0,), dtype=np.float32)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez_compressed(filename, lst=lst, wav=wav_data, flow=flow)

def build_aff_tester_arrays(aff_path: str, audio_name: str='base.ogg') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """returns: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]"""
    chart = read_aff_file(aff_path)
    aff_dir = get_aff_directory(aff_path)
    if os.path.isabs(audio_name):
        wav_path = audio_name
    else:
        wav_path_candidate = os.path.join(aff_dir, audio_name)
        if os.path.isfile(wav_path_candidate):
            wav_path = wav_path_candidate
        else:
            wav_path_candidate = os.path.join(ROOT_DIR, audio_name)
            if os.path.isfile(wav_path_candidate):
                wav_path = wav_path_candidate
    audio_duration_ms = None
    try:
        audio_duration_sec = librosa.get_duration(path=wav_path)
        audio_duration_ms = audio_duration_sec * 1000.0
    except Exception:
        audio_duration_ms = None
    ticks_ms, frac_index, timing_index = build_grid(chart, end_time_ms=audio_duration_ms)
    timestamps_model = ticks_ms
    timestamps_audio = [t + chart.audio_offset for t in ticks_ms]
    extra = build_tick_divisor_features(ticks_ms.tolist(), chart, divisor=4)
    wav_data = read_wav_data(timestamps_audio, wav_path, fft_size=128)
    wav_data = np.swapaxes(wav_data, 0, 1)
    ticks_arr = np.array(list(range(len(ticks_ms))), dtype=np.int32)
    timestamps_arr = np.array(timestamps_model, dtype=np.int32)
    return (ticks_arr, timestamps_arr, wav_data, extra)

def read_and_save_aff_tester_file(aff_path: str, filename: str='mapthis', audio_name: str='base.ogg') -> None:
    """returns: None"""
    ticks, timestamps_model, wav_data, extra = build_aff_tester_arrays(aff_path, audio_name=audio_name)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez_compressed(filename, ticks=ticks, timestamps=timestamps_model, wav=wav_data, extra=extra)
    mapthis_vis_rel = PATHS.get('mapthis_vis_dir')
    if mapthis_vis_rel:
        mapthis_vis_dir = os.path.join(ROOT_DIR, mapthis_vis_rel)
        os.makedirs(mapthis_vis_dir, exist_ok=True)
        for fn in os.listdir(mapthis_vis_dir):
            if fn.endswith('.tsv'):
                try:
                    os.remove(os.path.join(mapthis_vis_dir, fn))
                except OSError:
                    pass
        vis_path = os.path.join(mapthis_vis_dir, '0.tsv')
        with open(vis_path, 'w', encoding='utf-8') as vf:
            vf.write('tick_idx\ttimestamp_ms\tbpm\ttick_length_ms\n')
            T = len(timestamps_model)
            bpm_arr = extra[0] if extra.shape[0] >= 1 else np.zeros(T, dtype=np.float32)
            ticklen_arr = extra[1] if extra.shape[0] >= 2 else np.zeros(T, dtype=np.float32)
            for i in range(T):
                vf.write(f'{i}\t{int(timestamps_model[i])}\t{float(bpm_arr[i]):.4f}\t{float(ticklen_arr[i]):.4f}\n')
__all__ = ['read_and_save_aff_file', 'read_and_save_aff_tester_file', 'build_aff_tester_arrays', 'read_wav_data']
