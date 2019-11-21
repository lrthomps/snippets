from difflib import context_diff
import git
import logging

import numpy as np
import tensorflow as tf


def get_repo_state():
    """
    commit_hash, is_dirty, diffs = get_repo_state()
    print(git_hash)
    if is_dirty:
        for mod_file in diffs:
            for line in mod_file:
                print(line)
    :return:
    """
    repo = git.Repo(search_parent_directories=True)
    commit_hash = repo.head.object.hexsha
    is_dirty = repo.is_dirty()

    diffs = []
    if is_dirty:
        for diff_item in repo.index.diff(None).iter_change_type('M'):
            file_orig = diff_item.a_blob.data_stream.read().decode('utf-8').split('\n')
            with open(diff_item.a_path, 'r') as f:
                file_now = f.read().splitlines()
            diffs.append(context_diff(file_orig, file_now))

    return commit_hash, is_dirty, diffs


class FullModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """ must use FullModelCheckpoint hacked to save optimizer (for when model is not serializable)
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    cb = [FullModelCheckpoint(ckpt, model_dir, monitor=to_monitor, save_best_only=True),
          tf.keras.callbacks.TensorBoard(log_dir=log_dir)]
    """

    def __init__(self, ckpt, model_dir, max_to_keep=1, **kwargs):
        super(FullModelCheckpoint, self).__init__(filepath='', **kwargs)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=max_to_keep)

    def _save_model(self, epoch, logs):
        """Saves the model.

        Arguments:
            epoch: the epoch this iteration is in.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        if isinstance(self.save_freq,
                      int) or self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logging.warning('Can save best model only with %s available, '
                                    'skipping.', self.monitor)
                else:
                    if self.monitor_op(current, self.best):
                        filepath = self.ckpt_manager.save(checkpoint_number=epoch + 1)
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s.' % (epoch + 1, self.monitor, self.best, current, filepath))
                        self.best = current
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                filepath = self.ckpt_manager.save(checkpoint_number=epoch + 1)
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))


def osc_warmup_decay_by_epoch(epoch, d_model, n_warmup, osc=True):
    """ to use in a callback:
    tf.keras.callbacks.LearningRateScheduler(lambda e: osc_warmup_decay_by_epoch(e, d_model, n_warmup, osc))

    :param epoch: training epoch number
    :param warmup: number of warmup _epochs_
    :param d_model: scale of model normalized by num steps per epoch
    :param osc: boolean
    :return:
    """
    ratio = (epoch + 1) / n_warmup
    lr_lo = (d_model * (epoch + 1)) ** -0.5
    lr_warm = ratio * (d_model * n_warmup) ** -0.5
    if not osc:
        return np.min((lr_lo, lr_warm))
    lr_hi = lr_lo ** 0.8
    lr_osc = lr_lo + 0.5 * (lr_hi - lr_lo) * (1 + np.cos(0.05 * 3.1416 * epoch)) ** 2
    print(np.min((lr_osc, lr_warm)))
    return np.min((lr_osc, lr_warm))


def limit_tf_gpu(memory_limit_kB):
    print(tf.__version__)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate ## of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_kB)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

def conv_normd(num_filters, kernel_size, stride=1, dilation=1, name=None, activation='relu'):
    return tf.keras.Sequential([
        tf.keras.layers.Conv1D(num_filters, kernel_size, padding='same', strides=stride, dilation_rate=dilation,
                               activation=activation),
        tf.keras.layers.BatchNormalization(epsilon=1.e-6)
    ], name=name)


# ATTENTION!

def init_vec(seq_len):
    return (np.arange(2 * seq_len) - seq_len + 1)[:, np.newaxis]


def unroll_rel_pos(a, batch_size, seq_len):
    # (2 * seq_len, channels)
    a = tf.tile(tf.expand_dims(a, 0), (seq_len, 1, 1))
    a = tf.pad(a, [[0, 0], [1, 0], [0, 0]])
    a = tf.reshape(a, (2 * seq_len + 1, seq_len, -1))[1:, :, :]
    a = tf.reshape(a, (seq_len, 2 * seq_len, -1))[:, :seq_len, :]
    return tf.tile(tf.expand_dims(a, 0), (batch_size, 1, 1, 1))


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    from 'Attention is all you Need' applied to a time sequence
    there is possibly a transformation of the time sequence in key vs query and value that requires
    upsampling by a dilation_factor
    """
    def __init__(self, d_model, num_heads, seq_len, dilation_factor=1, activation='relu'):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        self.seq_len = seq_len
        self.t_vec = tf.concat([tf.sin(1.5708 / seq_len * init_vec(seq_len)),
                                tf.cos(1.5708 / seq_len * init_vec(seq_len))], axis=1)

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        if dilation_factor > 1:
            self.upsample = tf.keras.layers.UpSampling1D(size=dilation_factor)
        else:
            self.upsample = None

        self.wqs = [tf.keras.layers.Dense(self.depth, activation=activation) for _ in range(num_heads)]
        self.wks = [tf.keras.layers.Dense(self.depth, activation=activation) for _ in range(num_heads)]
        self.wvs = [tf.keras.layers.Dense(self.depth, activation=activation) for _ in range(num_heads)]

        self.av = tf.keras.layers.Dense(self.depth, activation=activation)
        self.ak = tf.keras.layers.Dense(self.depth, activation=activation)

        self.dense = tf.keras.layers.Dense(d_model, activation=activation)

    def call(self, x, k, q, batch_size):
        if self.upsample:
            q = self.upsample(q)
            x = self.upsample(x)

        ak = unroll_rel_pos(self.ak(self.t_vec), batch_size, self.seq_len)
        av = unroll_rel_pos(self.av(self.t_vec), batch_size, self.seq_len)

        concat_attn = []
        for wq, wk, wv in zip(self.wqs, self.wks, self.wvs):
            q_e = wq(q)  # (batch_size, seq_len, d_model)
            k_e = wk(k)  # (batch_size, seq_len, d_model)
            v = wv(x)  # (batch_size, 1, seq_len, d_model)
            concat_attn.append(self.scaled_dot_product_attention(q_e, k_e, v, ak, av, batch_size))
        concat_attn = tf.concat(concat_attn, axis=-1)

        output = self.dense(concat_attn)  # (batch_size, seq_len_q, d_model)
        return output

    def _relative_attention_inner(self, x, y, a, transpose, batch_size):
        """Relative position-aware dot-product attention inner calculation.
        This batches matrix multiply calculations to avoid unnecessary broadcasting.
        Args:
          x: when q: tensor with shape [batch_size, heads, length, depth];
             when attention_weights [batch_size, heads, length, length]
          y: when k: tensor with shape [batch_size, heads, length, depth];
             when v [batch_size, heads, length, depth]
          a: relative encoding tensor with shape [length, length, depth].
          transpose: Whether to transpose inner matrices of y and a. Should be true if
              last dimension of q is depth, not length.
        Returns:
          A Tensor with shape [batch_size, heads, length, length].
        """
        # qk^t: xy_matmul is [batch_size, 1, length, length]
        # wv: xy_matmul is [batch_size, 1, length, depth]
        xy_matmul = tf.matmul(x, y, transpose_b=transpose)

        # q_t/x_t is [batch_size, length, 1, depth]
        # wv_t is [batch_size, length, 1, length]
        x_t = tf.transpose(x, [0, 2, 1, 3])

        # x_t_r is [batch_size * length, 1, depth] / [batch_size * length, 1, length]
        x_t_r = tf.reshape(x_t, [batch_size * self.seq_len, 1, -1])

        a_r = tf.reshape(a, [batch_size * self.seq_len, self.seq_len, -1])  # [batch_size * length, length, depth]

        # q_ta: x_ta_matmul is [batch_size * length, 1, length]
        # wv_ta: [batch_size * length, 1, length]
        x_ta_matmul = tf.matmul(x_t_r, a_r, transpose_b=transpose)

        # x_ta_matmul_r is [batch_size, length, 1, length]
        x_ta_matmul_r = tf.reshape(x_ta_matmul, [batch_size, self.seq_len, 1, -1])
        # x_ta_matmul_r_t is [batch_size, 1, length, length]
        x_ta_matmul_r_t = tf.transpose(x_ta_matmul_r, [0, 2, 1, 3])

        return xy_matmul + x_ta_matmul_r_t

    def scaled_dot_product_attention(self, q, k, v, ak, av, batch_size):
        """Calculate the attention weights.
        q, k, v, ak, av must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
          q: query shape == (..., seq_len_q, depth)
          k: key shape == (..., seq_len_k, depth)
          v: value shape == (..., seq_len_v, depth_v)
          ak: rel position encoding (k) == (seq_len_k, seq_len_k, depth_v)
          av: rel pos encoding (v) == (seq_len_k, seq_len_k, depth_v)
          mask: Float tensor with shape broadcastable
                to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
          output, attention_weights
        """
        q = tf.expand_dims(q, 1)
        k = tf.expand_dims(k, 1)
        v = tf.expand_dims(v, 1)

        # sum over depth:
        # (..., seq_len_q, seq_len_k)
        matmul_qk = self._relative_attention_inner(q, k, ak, transpose=True, batch_size=batch_size)

        attention_weights = tf.nn.softmax(matmul_qk / self.depth, axis=-1)

        # sum over seq_len
        # (..., seq_len_q, depth_v)
        output = self._relative_attention_inner(attention_weights, v, av, transpose=False, batch_size=batch_size)

        return tf.reshape(output, (batch_size, self.seq_len, output.shape[-1]))
