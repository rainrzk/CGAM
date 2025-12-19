from __future__ import annotations
from typing import Literal
try:
    import tensorflow as tf
except Exception:
    tf = None
DeviceKind = Literal['auto', 'cpu', 'gpu', 'mps']

def choose_tf_device(device: DeviceKind='auto') -> str:
    """returns: str"""
    if tf is None:
        return '/CPU:0'
    dev = device.lower()
    if dev == 'cpu':
        return '/CPU:0'
    gpus = tf.config.list_physical_devices('GPU')
    if dev in ('gpu', 'mps'):
        if gpus:
            return '/GPU:0'
        return '/CPU:0'
    if gpus:
        return '/GPU:0'
    return '/CPU:0'
__all__ = ['choose_tf_device', 'DeviceKind']
