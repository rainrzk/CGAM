from __future__ import annotations
import glob
import os
from typing import List, Tuple
import numpy as np
try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:
    tf = None
    keras = None
from ..utils.config_utils import DIVISION_PENALTY_TRAIN_TABLE, PATHS, MODEL_PATHS, ROOT_DIR, TRAIN_CFG
DATA_DIR_TICKS = os.path.join(ROOT_DIR, PATHS['ticks_dir'])
NOTE_CLASSIFIER_MODEL_PATH = MODEL_PATHS['note_classifier']
DEFAULT_MODEL_PATH = NOTE_CLASSIFIER_MODEL_PATH
SEQ_LEN = 96
CONTEXT_TICKS = 24
INPUT_SEQ_LEN = SEQ_LEN + CONTEXT_TICKS

def _load_training_npz() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """returns: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]"""
    all_path = os.path.join(DATA_DIR_TICKS, 'mapdata.npz')
    if not os.path.isfile(all_path):
        raise RuntimeError(f'Merged mapdata.npz does not exist: {all_path}. Please run data.data_prep_ticks.load_maps_ticks() first.')
    with np.load(all_path) as data:
        wav = data['wav']
        labels = data['labels']
        division_code = data.get('division_code', None)
        timestamps = data.get('timestamps', None)
        timing = data.get('timing', None)
    x_all = wav.astype(np.float32)
    labels = labels.astype(np.float32)
    y_is_note_start = labels[:, 0:1]
    y_counts_all = labels[:, 1:]
    if division_code is not None:
        codes = division_code.astype(np.int32)
        table = DIVISION_PENALTY_TRAIN_TABLE
        max_code = table.shape[0] - 1
        penalties = []
        for c in codes:
            c_int = int(c)
            if c_int < 0:
                c_int = 0
            if c_int > max_code:
                c_int = max_code
            penalties.append([float(table[c_int])])
        division_penalty = np.asarray(penalties, dtype=np.float32)
    else:
        division_penalty = np.zeros_like(y_is_note_start, dtype=np.float32)
    y_all = np.concatenate([y_is_note_start, y_counts_all, division_penalty], axis=1).astype(np.float32)
    if timestamps is None or timing is None:
        raise RuntimeError('timestamps / timing fields are missing in mapdata.npz; cannot split by measures.')
    timestamps_all = timestamps.astype(np.float32).reshape(-1)
    timing_all = timing.astype(np.float32)
    return (x_all, y_all, timestamps_all, timing_all)

def _reshape_to_sequences(x_all: np.ndarray, y_all: np.ndarray, timestamps_all: np.ndarray, timing_all: np.ndarray, seq_len: int=SEQ_LEN, overlap_ticks: int=24) -> Tuple[np.ndarray, np.ndarray]:
    """returns: Tuple[np.ndarray, np.ndarray]"""
    if int(overlap_ticks) != int(CONTEXT_TICKS):
        raise ValueError(f'overlap_ticks({overlap_ticks}) 必须等于 CONTEXT_TICKS({CONTEXT_TICKS})，否则输入/输出切片会错位。')
    T = x_all.shape[0]
    if T == 0:
        raise ValueError('No available tick training data.')
    if timestamps_all.shape[0] != T or timing_all.shape[1] != T:
        raise ValueError(f'timestamps ({timestamps_all.shape}) / timing ({timing_all.shape}) are inconsistent with wav length {T}.')
    seg_time = timing_all[0]
    bpm = timing_all[1]
    beats = timing_all[2]
    beat_len = 60000.0 / np.maximum(bpm, 1e-06)
    measure_len = beat_len * np.maximum(beats, 1e-06)
    rel_times = timestamps_all - seg_time
    measure_ids = np.floor(rel_times / measure_len).astype(np.int32)
    segments_x: list[np.ndarray] = []
    segments_y: list[np.ndarray] = []
    start = 0
    for i in range(1, T):
        new_song = timestamps_all[i] < timestamps_all[i - 1]
        key_changed = seg_time[i] != seg_time[i - 1] or bpm[i] != bpm[i - 1] or beats[i] != beats[i - 1] or (measure_ids[i] != measure_ids[i - 1])
        if new_song or key_changed:
            idxs = np.arange(start, i, dtype=np.int32)
            if idxs.size > 0:
                segments_x.append(x_all[idxs])
                segments_y.append(y_all[idxs])
            start = i
    if start < T:
        idxs = np.arange(start, T, dtype=np.int32)
        segments_x.append(x_all[idxs])
        segments_y.append(y_all[idxs])
    if not segments_x:
        raise RuntimeError('No training segments were obtained after splitting by measures.')
    seq_x_list: list[np.ndarray] = []
    seq_y_list: list[np.ndarray] = []
    for idx, (x_seg, y_seg) in enumerate(zip(segments_x, segments_y)):
        in_len = int(overlap_ticks) + int(seq_len)
        y_dim = int(y_all.shape[1])

        prev_x = segments_x[idx - 1] if idx - 1 >= 0 else np.zeros((0, *x_all.shape[1:]), dtype=x_all.dtype)
        prev_y = segments_y[idx - 1] if idx - 1 >= 0 else np.zeros((0, y_dim), dtype=y_all.dtype)
        prev_tail_x = prev_x[-overlap_ticks:] if overlap_ticks > 0 else np.zeros((0, *x_all.shape[1:]), dtype=x_all.dtype)
        prev_tail_y = prev_y[-overlap_ticks:] if overlap_ticks > 0 else np.zeros((0, y_dim), dtype=y_all.dtype)

        center_x = x_seg[:seq_len]
        center_y = y_seg[:seq_len]

        pad_front = int(overlap_ticks) - int(prev_tail_x.shape[0])
        if pad_front < 0:
            pad_front = 0
        pad_front_x = np.zeros((pad_front, *x_all.shape[1:]), dtype=x_all.dtype)
        pad_front_y = np.zeros((pad_front, y_dim), dtype=y_all.dtype)

        pad_end = int(seq_len) - int(center_x.shape[0])
        if pad_end < 0:
            pad_end = 0
        pad_end_x = np.zeros((pad_end, *x_all.shape[1:]), dtype=x_all.dtype)
        pad_end_y = np.zeros((pad_end, y_dim), dtype=y_all.dtype)

        x_in = np.concatenate([pad_front_x, prev_tail_x, center_x, pad_end_x], axis=0)
        y_in = np.concatenate([pad_front_y, prev_tail_y, center_y, pad_end_y], axis=0)

        if x_in.shape[0] != in_len:
            x_in = x_in[:in_len] if x_in.shape[0] > in_len else np.concatenate([x_in, np.zeros((in_len - x_in.shape[0], *x_all.shape[1:]), dtype=x_all.dtype)], axis=0)
        if y_in.shape[0] != in_len:
            y_in = y_in[:in_len] if y_in.shape[0] > in_len else np.concatenate([y_in, np.zeros((in_len - y_in.shape[0], y_dim), dtype=y_all.dtype)], axis=0)

        seq_x_list.append(x_in)
        seq_y_list.append(y_in)
    x_seq = np.stack(seq_x_list, axis=0)
    y_seq = np.stack(seq_y_list, axis=0)
    return (x_seq, y_seq)

def build_classifier_model(input_shape) -> 'keras.Model':
    """returns: keras.Model"""
    if keras is None:
        raise RuntimeError('TensorFlow / Keras is not installed.')
    inputs = keras.Input(shape=input_shape)
    x = inputs
    x = keras.layers.TimeDistributed(keras.layers.Conv2D(64, (3, 3), padding='same', use_bias=False))(x)
    x = keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x)
    x = keras.layers.TimeDistributed(keras.layers.Activation('relu'))(x)
    x = keras.layers.TimeDistributed(keras.layers.Conv2D(128, (3, 3), padding='same', use_bias=False))(x)
    x = keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x)
    x = keras.layers.TimeDistributed(keras.layers.Activation('relu'))(x)
    x = keras.layers.TimeDistributed(keras.layers.MaxPool2D((2, 2)))(x)
    x = keras.layers.TimeDistributed(keras.layers.Conv2D(256, (3, 3), padding='same', use_bias=False))(x)
    x = keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x)
    x = keras.layers.TimeDistributed(keras.layers.Activation('relu'))(x)
    x = keras.layers.TimeDistributed(keras.layers.Conv2D(256, (3, 3), padding='same', use_bias=False))(x)
    x = keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x)
    x = keras.layers.TimeDistributed(keras.layers.Activation('relu'))(x)
    x = keras.layers.TimeDistributed(keras.layers.MaxPool2D((2, 1)))(x)
    x = keras.layers.TimeDistributed(keras.layers.GlobalAveragePooling2D())(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True))(x)
    x = keras.layers.TimeDistributed(keras.layers.Dense(256, activation='relu'))(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.TimeDistributed(keras.layers.Dense(13, activation='linear'))(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def _make_classifier_loss(lambda_reg: float=0.1):
    """returns: Callable"""
    from tensorflow.keras import backend as K

    def loss_fn(y_true, y_pred):
        """returns: Any"""
        import tensorflow as tf
        y_true_flat = tf.reshape(y_true, (-1, tf.shape(y_true)[-1]))
        y_pred_flat = tf.reshape(y_pred, (-1, tf.shape(y_pred)[-1]))
        y_is_note_start = y_true_flat[:, 0:1]
        y_counts = y_true_flat[:, 1:13]
        penalty = y_true_flat[:, 13:14]
        logit_note = y_pred_flat[:, 0:1]
        counts_pred = y_pred_flat[:, 1:13]
        p_note = keras.activations.sigmoid(logit_note)
        y_is_note_binary = K.cast(y_is_note_start > 0.5, 'float32')
        bce_note = keras.losses.binary_crossentropy(y_is_note_binary, p_note)
        nc_cfg = TRAIN_CFG.get('note_classifier', {}) if isinstance(TRAIN_CFG, dict) else {}
        try:
            nonzero_w = float(nc_cfg.get('nonzero_count_weight', 1.0))
        except (TypeError, ValueError):
            nonzero_w = 1.0
        if nonzero_w < 1.0:
            nonzero_w = 1.0
        mse_per_tick = K.mean(K.square(counts_pred - y_counts), axis=-1)
        sum_counts = K.sum(y_counts, axis=-1)
        nonzero_mask = K.cast(sum_counts > 0.5, 'float32')
        w_counts = 1.0 + (nonzero_w - 1.0) * nonzero_mask
        mse_counts = mse_per_tick * w_counts
        base = bce_note + mse_counts
        reg = lambda_reg * K.mean(penalty * p_note, axis=-1)
        return base + reg
    return loss_fn

def train_classifier(model_path: str=DEFAULT_MODEL_PATH, epochs: int=5, batch_size: int=8, device: str='auto', log_dir: str | None=None, overlap_ticks: int=24) -> None:
    """returns: None"""
    if keras is None:
        raise RuntimeError('TensorFlow / Keras is not installed.')
    from ..utils.device_utils import choose_tf_device
    if int(overlap_ticks) != int(CONTEXT_TICKS):
        raise ValueError(f'overlap_ticks({overlap_ticks}) 必须等于 CONTEXT_TICKS({CONTEXT_TICKS})，否则模型输出与监督会错位。')
    x_all, y_all, timestamps_all, timing_all = _load_training_npz()
    x, y = _reshape_to_sequences(x_all, y_all, timestamps_all, timing_all, seq_len=SEQ_LEN, overlap_ticks=overlap_ticks)
    input_shape = x.shape[1:]
    nc_cfg = TRAIN_CFG.get('note_classifier', {}) if isinstance(TRAIN_CFG, dict) else {}
    lr = float(nc_cfg.get('lr', 0.0005))
    clipnorm = nc_cfg.get('clipnorm', None)
    clipvalue = nc_cfg.get('clipvalue', None)
    opt_kwargs = {'learning_rate': lr}
    if clipnorm is not None:
        opt_kwargs['clipnorm'] = float(clipnorm)
    if clipvalue is not None:
        opt_kwargs['clipvalue'] = float(clipvalue)
    optimizer = keras.optimizers.Adam(**opt_kwargs)
    dev_str = choose_tf_device(device)
    with tf.device(dev_str):
        model = build_classifier_model(input_shape)
        loss_fn = _make_classifier_loss(lambda_reg=0.1)
        model.compile(loss=loss_fn, optimizer=optimizer)
        from datetime import datetime
        if log_dir is None:
            log_root = os.path.join(ROOT_DIR, 'logs', 'classifier')
            os.makedirs(log_root, exist_ok=True)
            run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
            log_dir = os.path.join(log_root, run_id)
        else:
            os.makedirs(log_dir, exist_ok=True)
        tensorboard_cb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=[tensorboard_cb])
        model.save(model_path)
        print(f'TensorBoard logs written to: {log_dir}')

def load_classifier(model_path: str=DEFAULT_MODEL_PATH):
    """returns: keras.Model"""
    if keras is None:
        raise RuntimeError('TensorFlow / Keras is not installed.')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'model not found at {model_path}')
    return keras.models.load_model(model_path, compile=False)
__all__ = ['train_classifier', 'load_classifier', 'NOTE_CLASSIFIER_MODEL_PATH', 'SEQ_LEN', 'CONTEXT_TICKS', 'INPUT_SEQ_LEN']
