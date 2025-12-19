from __future__ import annotations
import os
from typing import Tuple
import numpy as np
try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:
    tf = None
    keras = None
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None
from ..utils.config_utils import MODEL_PATHS
SCRIPTS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
GEN_PATH = MODEL_PATHS['gan_generator']
DIS_PATH = MODEL_PATHS['gan_discriminator']
GROUP_SIZE = 96
PARAM_DIM = 32

def build_generator(latent_dim: int=64) -> 'keras.Model':
    """returns: keras.Model"""
    if keras is None:
        raise RuntimeError('TensorFlow / Keras is not installed.')
    d_model = 128
    num_heads = 4
    num_layers = 2
    z = keras.Input(shape=(latent_dim,), name='z')
    x = keras.layers.Dense(GROUP_SIZE * d_model, activation='relu')(z)
    x = keras.layers.Reshape((GROUP_SIZE, d_model))(x)
    for i in range(num_layers):
        attn_out = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, name=f'gen_mha_{i}')(x, x)
        x = keras.layers.LayerNormalization(epsilon=1e-06)(x + attn_out)
        ffn = keras.layers.Dense(d_model * 4, activation='relu', name=f'gen_ffn_{i}_1')(x)
        ffn = keras.layers.Dense(d_model, name=f'gen_ffn_{i}_2')(ffn)
        x = keras.layers.LayerNormalization(epsilon=1e-06)(x + ffn)
    out = keras.layers.Dense(PARAM_DIM, name='gen_out')(x)
    return keras.Model(z, out, name='arc_gan_generator')

def build_discriminator() -> 'keras.Model':
    """returns: keras.Model"""
    if keras is None:
        raise RuntimeError('TensorFlow / Keras is not installed.')
    d_model = 128
    num_heads = 4
    num_layers = 2
    inp = keras.Input(shape=(GROUP_SIZE, PARAM_DIM), name='tick_group')
    x = keras.layers.Dense(d_model, name='disc_in_proj')(inp)
    for i in range(num_layers):
        attn_out = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, name=f'disc_mha_{i}')(x, x)
        x = keras.layers.LayerNormalization(epsilon=1e-06)(x + attn_out)
        ffn = keras.layers.Dense(d_model * 4, activation='relu', name=f'disc_ffn_{i}_1')(x)
        ffn = keras.layers.Dense(d_model, name=f'disc_ffn_{i}_2')(ffn)
        x = keras.layers.LayerNormalization(epsilon=1e-06)(x + ffn)
    x = keras.layers.GlobalAveragePooling1D(name='disc_pool')(x)
    x = keras.layers.Dense(128, activation='relu', name='disc_dense')(x)
    out = keras.layers.Dense(1, activation='sigmoid', name='disc_out')(x)
    model = keras.Model(inp, out, name='arc_gan_discriminator')
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
    return model

def build_gan(generator: 'keras.Model', discriminator: 'keras.Model') -> 'keras.Model':
    """returns: keras.Model"""
    if keras is None:
        raise RuntimeError('TensorFlow / Keras 未安装。')
    z = keras.Input(shape=(generator.input_shape[1],))
    fake_seq = generator(z)
    validity = discriminator(fake_seq)
    gan = keras.Model(z, validity, name='arc_gan')
    gan.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0002, 0.5))
    return gan

def train_gan(real_data: np.ndarray, epochs: int=1000, batch_size: int=32, latent_dim: int=64, device: str='auto', real_mask: np.ndarray | None=None, log_dir: str | None=None) -> Tuple['keras.Model', 'keras.Model']:
    """returns: Tuple[keras.Model, keras.Model]"""
    if keras is None:
        raise RuntimeError('TensorFlow / Keras is not installed.')
    from ..utils.device_utils import choose_tf_device
    real_data = np.asarray(real_data, dtype=np.float32)
    if real_mask is None:
        real_mask = np.isfinite(real_data).astype(np.float32)
    else:
        real_mask = np.asarray(real_mask, dtype=np.float32)
        if real_mask.shape != real_data.shape:
            raise ValueError(f'real_mask shape {real_mask.shape} does not match real_data {real_data.shape}')
    real_data = np.nan_to_num(real_data, nan=0.0, posinf=0.0, neginf=0.0)
    dev_str = choose_tf_device(device)
    summary_writer = None
    if log_dir is not None:
        summary_writer = tf.summary.create_file_writer(log_dir)
    with tf.device(dev_str):
        generator = build_generator(latent_dim=latent_dim)
        discriminator = build_discriminator()
        gen_optimizer = keras.optimizers.Adam(0.0002, 0.5)
        valid = np.ones((batch_size, 1), dtype=np.float32)
        fake = np.zeros((batch_size, 1), dtype=np.float32)
        epoch_iter = tqdm(range(epochs), desc='GAN training') if tqdm is not None else range(epochs)
        for epoch in epoch_iter:
            idx = np.random.randint(0, real_data.shape[0], batch_size)
            real_batch = real_data[idx]
            mask_batch = real_mask[idx]
            z = np.random.normal(0, 1, (batch_size, latent_dim)).astype(np.float32)
            gen_batch = generator.predict(z, verbose=0)
            real_batch_masked = real_batch * mask_batch
            fake_batch_masked = gen_batch * mask_batch
            d_loss_real = discriminator.train_on_batch(real_batch_masked, valid)
            d_loss_fake = discriminator.train_on_batch(fake_batch_masked, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            z = np.random.normal(0, 1, (batch_size, latent_dim)).astype(np.float32)
            with tf.GradientTape() as tape:
                gen_out = generator(z, training=True)
                gen_out_masked = gen_out * mask_batch
                pred_fake = discriminator(gen_out_masked, training=False)
                g_loss = keras.losses.binary_crossentropy(valid, pred_fake)
                g_loss = tf.reduce_mean(g_loss)
            grads = tape.gradient(g_loss, generator.trainable_variables)
            gen_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
            d_loss_val = d_loss[0] if isinstance(d_loss, (list, tuple, np.ndarray)) else float(d_loss)
            d_acc_val = d_loss[1] if isinstance(d_loss, (list, tuple, np.ndarray)) and len(d_loss) > 1 else 0.0
            if summary_writer is not None:
                with summary_writer.as_default():
                    tf.summary.scalar('d_loss', d_loss_val, step=epoch)
                    tf.summary.scalar('d_acc', d_acc_val, step=epoch)
                    tf.summary.scalar('g_loss', float(g_loss), step=epoch)
            if tqdm is None and epoch % 100 == 0:
                print(f'[epoch {epoch}] d_loss={d_loss_val:.4f}, d_acc={d_acc_val:.4f}, g_loss={float(g_loss):.4f} (device={dev_str})')
        generator.save(GEN_PATH)
        discriminator.save(DIS_PATH)
    return (generator, discriminator)

def load_generator(path: str=GEN_PATH):
    """returns: keras.Model"""
    if keras is None:
        raise RuntimeError('TensorFlow / Keras is not installed.')
    if not os.path.exists(path):
        raise FileNotFoundError(f'generator model not found at {path}')
    return keras.models.load_model(path, compile=False)
__all__ = ['GROUP_SIZE', 'PARAM_DIM', 'train_gan', 'load_generator']
