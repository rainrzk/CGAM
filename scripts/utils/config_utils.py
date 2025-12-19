from __future__ import annotations
import os
from typing import Any, Dict, Optional
import numpy as np
try:
    import yaml
except Exception:
    yaml = None

def _find_root_dir(start_path: str) -> str:
    """returns: str"""
    cur = os.path.abspath(start_path)
    for _ in range(5):
        cfg_path = os.path.join(cur, 'config.yaml')
        if os.path.isfile(cfg_path):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(start_path)
ROOT_DIR = _find_root_dir(os.path.dirname(__file__))

def _load_config() -> Dict[str, Any]:
    """returns: Dict[str, Any]"""
    cfg_path = os.path.join(ROOT_DIR, 'config.yaml')
    if yaml is None or not os.path.isfile(cfg_path):
        return {}
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
_CFG = _load_config()

def _section(name: str, default: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    """returns: Dict[str, Any]"""
    val = _CFG.get(name, {}) if isinstance(_CFG, dict) else {}
    if not isinstance(val, dict):
        val = {}
    if default:
        d = default.copy()
        d.update(val)
        return d
    return val
PATHS: Dict[str, str] = _section('paths', {})

def _abs_path(rel: str) -> str:
    """returns: str"""
    return os.path.join(ROOT_DIR, rel)
_MODELS_CFG = _section('models', {})
MODEL_PATHS: Dict[str, str] = {}
for name, filename in _MODELS_CFG.items():
    if not isinstance(filename, str):
        continue
    MODEL_PATHS[name] = _abs_path(os.path.join(PATHS.get('model_dir', 'models'), filename))
TRAIN_CFG: Dict[str, Any] = _section('train', {})
INFER_CFG: Dict[str, Any] = _section('inference', {})
_NOTE_INFER_CFG = INFER_CFG.get('note_classifier', {}) if isinstance(INFER_CFG, dict) else {}
NOTE_TYPE_THRESHOLD: float = float(_NOTE_INFER_CFG.get('type_threshold', 0.5))
NOTE_IN_ARC_THRESHOLD: float = float(_NOTE_INFER_CFG.get('in_arc_threshold', NOTE_TYPE_THRESHOLD))
NOTE_IN_HOLD_THRESHOLD: float = float(_NOTE_INFER_CFG.get('in_hold_threshold', NOTE_TYPE_THRESHOLD))
_TYPE_THRESHOLDS_CFG = _NOTE_INFER_CFG.get('type_thresholds', None)
if isinstance(_TYPE_THRESHOLDS_CFG, dict) and _TYPE_THRESHOLDS_CFG:
    NOTE_TYPE_THRESHOLDS = np.array([float(_TYPE_THRESHOLDS_CFG.get('tap', NOTE_TYPE_THRESHOLD)), float(_TYPE_THRESHOLDS_CFG.get('hold', NOTE_TYPE_THRESHOLD)), float(_TYPE_THRESHOLDS_CFG.get('arcfalse0', NOTE_TYPE_THRESHOLD)), float(_TYPE_THRESHOLDS_CFG.get('arcfalse1', NOTE_TYPE_THRESHOLD)), float(_TYPE_THRESHOLDS_CFG.get('arctrue', NOTE_TYPE_THRESHOLD)), float(_TYPE_THRESHOLDS_CFG.get('arctap', NOTE_TYPE_THRESHOLD))], dtype=np.float32)
else:
    NOTE_TYPE_THRESHOLDS = np.full((6,), float(NOTE_TYPE_THRESHOLD), dtype=np.float32)
_GRID_CFG = _section('grid', {})
_divisors = _GRID_CFG.get('divisors', [])
if not _divisors:
    raise ValueError('config.yaml:grid.divisors 未配置或为空')
GRID_DIVISORS = np.array(list(_divisors), dtype=np.int32)
GRID_LABEL_TOLERANCE_MS: float = float(_GRID_CFG.get('label_tolerance_ms', 5.0))
_legal_frac24 = _GRID_CFG.get('legal_frac24', None)
if not isinstance(_legal_frac24, (list, tuple)) or not _legal_frac24:
    raise ValueError('config.yaml:grid.legal_frac24 必须是非空列表')
LEGAL_FRAC_24 = np.array(sorted({int(x) % 24 for x in _legal_frac24}), dtype=np.int32)
_division_codes_cfg = _GRID_CFG.get('division_codes')
if not isinstance(_division_codes_cfg, dict) or not _division_codes_cfg:
    raise ValueError('config.yaml:grid.division_codes 必须是非空字典')
DIVISION_NAME_TO_CODE: Dict[str, int] = {str(k): int(v) for k, v in _division_codes_cfg.items()}
DIVISION_CODE_TO_NAME: Dict[int, str] = {v: k for k, v in DIVISION_NAME_TO_CODE.items()}
_frac24 = _GRID_CFG.get('frac24_indices_16th')
if not isinstance(_frac24, (list, tuple)) or not _frac24:
    raise ValueError('config.yaml:grid.frac24_indices_16th 必须是非空列表')
FRAC24_INDICES_16TH = np.array(list(_frac24), dtype=np.int32)
_pen_cfg = _GRID_CFG.get('division_penalties')
if not isinstance(_pen_cfg, dict) or not _pen_cfg:
    raise ValueError('config.yaml:grid.division_penalties 必须是非空字典')
DIVISION_CODE_TO_RAW_PENALTY: Dict[int, float] = {}
for name, w in _pen_cfg.items():
    if not isinstance(name, str) or name not in DIVISION_NAME_TO_CODE:
        continue
    code = DIVISION_NAME_TO_CODE[name]
    try:
        w_val = float(w)
    except (TypeError, ValueError):
        continue
    if w_val < 0.0:
        w_val = 0.0
    DIVISION_CODE_TO_RAW_PENALTY[code] = w_val
_max_code = max(DIVISION_CODE_TO_RAW_PENALTY.keys())
_RAW_DIVISION_PENALTY_TABLE = np.zeros(_max_code + 1, dtype=np.float32)
for c, w in DIVISION_CODE_TO_RAW_PENALTY.items():
    if 0 <= c < _RAW_DIVISION_PENALTY_TABLE.shape[0]:
        _RAW_DIVISION_PENALTY_TABLE[c] = float(w)
_K = 14.0
_eps = 0.001
DIVISION_PENALTY_INFER_TABLE = 1.0 - np.exp(-_K * _RAW_DIVISION_PENALTY_TABLE)
DIVISION_PENALTY_TRAIN_TABLE = DIVISION_PENALTY_INFER_TABLE / np.maximum(1.0 - DIVISION_PENALTY_INFER_TABLE, _eps)
DIVISION_PENALTY_TABLE = DIVISION_PENALTY_INFER_TABLE
_DIVISOR_TO_DIVISION_NAME = {4: 'quarter', 8: 'eighth', 12: 'twelvth', 16: 'sixteenth', 24: 'twentyfourth', 32: 'thirtysecond'}
_DIVISION_PRIORITY = ['quarter', 'eighth', 'twelvth', 'sixteenth', 'twentyfourth', 'thirtysecond']

def frac24_to_division_code(frac_index: int) -> int:
    """returns: int"""
    frac_index = int(frac_index) % 24
    if frac_index in (0, 12):
        name = 'quarter'
    elif frac_index in (6, 18):
        name = 'sixteenth'
    elif frac_index in (3, 9, 15, 21):
        name = 'thirtysecond'
    elif frac_index in (8, 16):
        name = 'twelvth'
    elif frac_index in (4, 20):
        name = 'twentyfourth'
    else:
        name = 'sixteenth'
    return int(DIVISION_NAME_TO_CODE[name])
__all__ = ['ROOT_DIR', 'PATHS', 'MODEL_PATHS', 'TRAIN_CFG', 'INFER_CFG', 'GRID_DIVISORS', 'GRID_LABEL_TOLERANCE_MS', 'LEGAL_FRAC_24', 'FRAC24_INDICES_16TH', 'DIVISION_NAME_TO_CODE', 'DIVISION_CODE_TO_NAME', 'DIVISION_PENALTY_TABLE', 'NOTE_TYPE_THRESHOLD', 'NOTE_TYPE_THRESHOLDS', 'NOTE_IN_ARC_THRESHOLD', 'NOTE_IN_HOLD_THRESHOLD', 'frac24_to_division_code']
