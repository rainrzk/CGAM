from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterable
import os
import re
from .config_utils import ROOT_DIR, PATHS

@dataclass
class Timing:
    time: int
    bpm: float
    beats: float

@dataclass
class TapNote:
    time: int
    lane: int

@dataclass
class HoldNote:
    start: int
    end: int
    lane: int

@dataclass
class Arctap:
    time: int

@dataclass
class ArcNote:
    start: int
    end: int
    x_start: float
    x_end: float
    curve_type: str
    y_start: float
    y_end: float
    hand: int
    extra: str
    is_trace: bool
    arctaps: List[Arctap] = field(default_factory=list)

@dataclass
class AffChart:
    audio_offset: int = 0
    timings: List[Timing] = field(default_factory=list)
    taps: List[TapNote] = field(default_factory=list)
    holds: List[HoldNote] = field(default_factory=list)
    arcs: List[ArcNote] = field(default_factory=list)
TIMING_RE = re.compile('timing\\(([^)]+)\\);')
HOLD_RE = re.compile('hold\\(([^)]+)\\);')
ARC_RE = re.compile('arc\\(([^)]+)\\)')
ARCTAP_BLOCK_RE = re.compile('\\[(.*?)\\]')
ARCTAP_RE = re.compile('arctap\\(([^)]+)\\)')
SHORT_TAP_RE = re.compile('^\\(([^,]+),([^,]+)\\);$')

def _parse_csv_numbers(payload: str) -> List[str]:
    """returns: List[str]"""
    return [p.strip() for p in payload.split(',') if p.strip() != '']

def read_aff_file(path: str) -> AffChart:
    """returns: AffChart"""
    chart = AffChart()
    abs_path = os.path.abspath(path)
    dataset_dir = os.path.join(ROOT_DIR, PATHS.get('dataset_dir', 'dataset'))
    dataset_dir = os.path.abspath(dataset_dir)
    is_dataset_chart = abs_path.startswith(dataset_dir + os.sep)
    in_timinggroup = False
    with open(path, encoding='utf-8') as fp:
        for raw_line in fp:
            if is_dataset_chart:
                stripped = raw_line.strip()
                if not in_timinggroup and stripped.startswith('timinggroup') and ('{' in stripped):
                    in_timinggroup = True
                    continue
                if in_timinggroup:
                    if stripped == '};':
                        in_timinggroup = False
                    continue
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('AudioOffset:'):
                try:
                    chart.audio_offset = int(line.split(':', 1)[1].strip())
                except ValueError:
                    chart.audio_offset = 0
                continue
            if line == '-':
                continue
            m = TIMING_RE.search(line)
            if m:
                vals = _parse_csv_numbers(m.group(1))
                if len(vals) >= 3:
                    t = int(float(vals[0]))
                    bpm = float(vals[1])
                    beats = float(vals[2])
                    chart.timings.append(Timing(time=t, bpm=bpm, beats=beats))
                continue
            m = SHORT_TAP_RE.match(line)
            if m:
                t = int(float(m.group(1)))
                lane = int(float(m.group(2)))
                chart.taps.append(TapNote(time=t, lane=lane))
                continue
            m = HOLD_RE.search(line)
            if m:
                vals = _parse_csv_numbers(m.group(1))
                if len(vals) >= 3:
                    t_start = int(float(vals[0]))
                    t_end = int(float(vals[1]))
                    lane = int(vals[2])
                    chart.holds.append(HoldNote(start=t_start, end=t_end, lane=lane))
                continue
            m = ARC_RE.search(line)
            if m:
                arc_payload = m.group(1)
                vals = _parse_csv_numbers(arc_payload)
                if len(vals) >= 10:
                    t_start = int(float(vals[0]))
                    t_end = int(float(vals[1]))
                    x1 = float(vals[2])
                    x2 = float(vals[3])
                    curve_type = vals[4]
                    y1 = float(vals[5])
                    y2 = float(vals[6])
                    hand = int(vals[7])
                    extra = vals[8]
                    is_trace = vals[9].lower() == 'true'
                    arc_note = ArcNote(start=t_start, end=t_end, x_start=x1, x_end=x2, curve_type=curve_type, y_start=y1, y_end=y2, hand=hand, extra=extra, is_trace=is_trace)
                    block_match = ARCTAP_BLOCK_RE.search(line)
                    if block_match:
                        block = block_match.group(1)
                        for m2 in ARCTAP_RE.finditer(block):
                            t_str = m2.group(1).strip()
                            if not t_str:
                                continue
                            t_val = int(float(t_str))
                            arc_note.arctaps.append(Arctap(time=t_val))
                    chart.arcs.append(arc_note)
                continue
    chart.timings.sort(key=lambda t: t.time)
    chart.taps.sort(key=lambda n: n.time)
    chart.holds.sort(key=lambda n: (n.start, n.end))
    chart.arcs.sort(key=lambda a: (a.start, a.end))
    return chart

def _format_timing(t: Timing) -> str:
    """returns: str"""
    return f'timing({t.time},{t.bpm:.2f},{t.beats:.2f});'

def _format_tap(n: TapNote) -> str:
    """returns: str"""
    return f'({n.time},{n.lane});'

def _format_hold(n: HoldNote) -> str:
    """returns: str"""
    return f'hold({n.start},{n.end},{n.lane});'

def _format_arc(a: ArcNote) -> str:
    """returns: str"""
    base = f'arc({a.start},{a.end},{a.x_start:.2f},{a.x_end:.2f},{a.curve_type},{a.y_start:.2f},{a.y_end:.2f},{a.hand},{a.extra},{str(a.is_trace).lower()})'
    if a.arctaps:
        taps = ','.join((f'arctap({t.time})' for t in sorted(a.arctaps, key=lambda x: x.time)))
        return f'{base}[{taps}];'
    return base + ';'

def write_aff_file(chart: AffChart, path: str) -> None:
    """returns: None"""
    lines: List[str] = []
    lines.append(f'AudioOffset:{chart.audio_offset}')
    lines.append('-')
    for t in chart.timings:
        lines.append(_format_timing(t))
    events = []
    for n in chart.taps:
        events.append((int(n.time), 0, _format_tap(n)))
    for n in chart.holds:
        events.append((int(n.start), 1, _format_hold(n)))
    for a in chart.arcs:
        events.append((int(a.start), 2, _format_arc(a)))
    events.sort(key=lambda x: (x[0], x[1]))
    for _, _, line in events:
        lines.append(line)
    with open(path, 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(lines) + '\n')

def get_aff_directory(path: str) -> str:
    """returns: str"""
    return os.path.dirname(os.path.abspath(path))

def generate_maplist_from_dataset(root_dir: Optional[str]=None, dataset_subdir: str='dataset', maplist_name: str='maplist_aff.txt') -> str:
    """returns: str"""
    if root_dir is None:
        scripts_dir = os.path.dirname(__file__)
        root_dir = os.path.dirname(scripts_dir)
    dataset_dir = os.path.join(root_dir, dataset_subdir)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f'dataset 目录不存在: {dataset_dir}')
    subdirs = [d for d in sorted(os.listdir(dataset_dir)) if os.path.isdir(os.path.join(dataset_dir, d))]
    rel_paths: List[str] = []
    for d in subdirs:
        song_dir = os.path.join(dataset_dir, d)
        base_ogg = os.path.join(song_dir, 'base.ogg')
        has_aff = any((fn.lower().endswith('.aff') for fn in os.listdir(song_dir)))
        if not os.path.isfile(base_ogg) or not has_aff:
            continue
        rel_paths.append(os.path.join(dataset_subdir, d))
    maplist_path = os.path.join(root_dir, maplist_name)
    with open(maplist_path, 'w', encoding='utf-8') as fp:
        for p in rel_paths:
            fp.write(p + '\n')
    print(f'generated maplist at: {maplist_path} with {len(rel_paths)} entries')
    return maplist_path
__all__ = ['Timing', 'TapNote', 'HoldNote', 'Arctap', 'ArcNote', 'AffChart', 'read_aff_file', 'write_aff_file', 'get_aff_directory', 'generate_maplist_from_dataset']
