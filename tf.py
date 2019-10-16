import logging

import numpy as np
import tensorflow as tf


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


def mish(x):
    return tf.multiply(x, tf.tanh(tf.math.softplus(x)))


def swish(x, beta=1):
    return tf.multiply(x, tf.sigmoid(beta * x))


default_activation = tf.keras.layers.Activation(mish)


def conv_normd(num_filters, kernel_size, stride=1, dilation=1, name=None, activation=None):
    activation = activation or default_activation
    return tf.keras.Sequential([
        tf.keras.layers.Conv1D(num_filters, kernel_size, padding='same', strides=stride, dilation_rate=dilation,
                               activation=activation),
        tf.keras.layers.BatchNormalization(epsilon=1.e-6)
    ], name=name)


# ATTENTION!

class TwoNeighSelfAttn(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_heads, dilation_factor=1):
        super(TwoNeighSelfAttn, self).__init__()
        self.num_heads = num_heads
        self.num_channels = num_channels

        assert num_channels % self.num_heads == 0

        self.depth = num_channels // self.num_heads

        if dilation_factor > 1:
            self.upsample = tf.keras.layers.UpSampling1D(size=dilation_factor)
        else:
            self.upsample = None

        self.wq = tf.keras.layers.Dense(num_channels)
        self.wk = tf.keras.layers.Dense(num_channels)
        self.wv = tf.keras.layers.Dense(num_channels)

        self.context = tf.keras.layers.Conv2D(num_heads, 1, activation='tanh')

        self.final = tf.keras.layers.Conv1D(num_channels, 1, activation=default_activation)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size * self.num_heads, -1, self.depth))

    def merge_heads(self, x, batch_size, seq_len):
        x = tf.reshape(x, (batch_size, self.num_heads, seq_len, -1))
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size, seq_len, -1))

    def call(self, x, y, t_mat):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        if self.upsample:
            y = self.upsample(y)

        q = self.wq(y)  # (batch_size, seq_len, num_channels)
        k = self.wk(x)  # (batch_size, seq_len, num_channels)
        v = self.wv(y)  # (batch_size, seq_len, num_channels)

        q = self.split_heads(q, batch_size)  # (batch_size * num_heads, seq_len, depth)
        k = self.split_heads(k, batch_size)  # (batch_size * num_heads, seq_len, depth)
        v = self.split_heads(v, batch_size)  # (batch_size * num_heads, seq_len, depth)

        c = self.context(tf.expand_dims(t_mat, 3))  # (batch_size, seq_len, seq_len, num_heads)
        c = tf.transpose(c, perm=[0, 3, 1, 2])  # (batch_size, num_heads, seq_len, seq_len)
        c = tf.reshape(c, (batch_size * self.num_heads, seq_len, seq_len))

        # scaled_attention.shape == (batch_size * num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size * num_heads, seq_len_q, seq_len_k)
        scaled_attention = nn_scaled_dot_product_attention(q, k, v, c)

        concat_attention = self.merge_heads(
            scaled_attention, batch_size, seq_len)  # (batch_size, seq_len_q, num_channels)
        # attention_weights = self.merge_heads(attention_weights, batch_size, seq_len)

        output = self.final(concat_attention)  # (batch_size, seq_len_q, num_channels)
        return output


def nn_scaled_dot_product_attention(q, k, v, c):
    """Calculate the attention weights.
    q, k, v, ak, av must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (batch, num_heads, seq_len_q, depth)
      k: key shape == (batch, num_heads, seq_len_k, depth)
      v: value shape == (batch, num_heads, seq_len_v, depth_v)
      c: rel position encoding == (batch, num_heads, seq_len_k, seq_len_k)

    Returns:
      output, attention_weights
    """
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)

    matmul_qk = tf.matmul(k, q, transpose_b=True) / tf.math.sqrt(dk)  # (batch * num_heads, seq_len, seq_len)
    cv = tf.matmul(c, v, transpose_b=False)  # (batch * num_heads, seq_len, depth)

    output = tf.matmul(matmul_qk, cv, transpose_b=False)  # (batch * num_heads, seq_len, depth)

    return output


def init_vec(seq_len):
    return (np.arange(2 * seq_len) - seq_len + 1)[:, np.newaxis]


def unroll_rel_pos(a, batch_size, seq_len):
    # (2 * seq_len, channels)
    a = tf.tile(tf.expand_dims(a, 0), (seq_len, 1, 1))
    a = tf.pad(a, [[0, 0], [1, 0], [0, 0]])
    a = tf.reshape(a, (2 * seq_len + 1, seq_len, -1))[1:, :, :]
    a = tf.reshape(a, (seq_len, 2 * seq_len, -1))[:, :seq_len, :]
    return tf.tile(tf.expand_dims(a, 0), (batch_size, 1, 1, 1))


class LocalUnetAttn(tf.keras.layers.Layer):
    def __init__(self, num_blocks, in_channels, inter_channels, dilation_factor, attn_kernel):
        super(LocalUnetAttn, self).__init__()

        self.gate_blocks = [AttentionBlock(in_channels, inter_channels, dilation_factor, attn_kernel)
                            for _ in range(num_blocks)]
        self.combine_gates = conv_normd(in_channels, kernel_size=1)

    def call(self, input, gating_signal):
        gates, attns = [], []
        for block in self.gate_blocks:
            gate, attn = block(input, gating_signal)
            gates.append(gate)
            attns.append(attn)

        return self.combine_gates(tf.concat(gates, axis=-1)), tf.concat(attns, axis=-1)


class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, inter_channels, dilation_factor, attn_kernel):
        super(AttentionBlock, self).__init__()

        self.W = conv_normd(in_channels, kernel_size=1)
        self.theta = tf.keras.layers.Conv1D(inter_channels, dilation_factor, padding='same', use_bias=False,
                                            strides=dilation_factor, activation=default_activation)

        self.phi = tf.keras.layers.Conv1D(inter_channels, 1, padding='same', activation=default_activation)

        self.psi = tf.keras.layers.Conv1D(1, attn_kernel, padding='same', activation='sigmoid')

        self.upsample = tf.keras.layers.UpSampling1D(size=dilation_factor)

    def call(self, x, g):
        # (batch size, seq len / 2^n, filters[n]), (batch size, seq len / 2^n+1, filters[n+1])

        theta_x = self.theta(x)  # (batch size, seq len / 2^n+1, inter_channels)
        phi_g = self.phi(g)  # (batch size, seq len / 2^n + 1, inter_channels)

        f = tf.nn.relu(theta_x + phi_g)
        attn = self.psi(f)  # (batch size, seq len / 2^n + 1, 1)
        y = tf.multiply(self.upsample(attn), x)  # (batch size, seq len / 2^n, filters[n])
        gate = self.W(y)  # (batch size, seq len / 2^n, filters[n])

        return gate, attn


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, seq_len, dilation_factor=1):
        super(MultiHeadAttention, self).__init__()
        activation = default_activation

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

        self.wqs = [tf.keras.layers.Dense(self.depth, activation=None) for _ in range(num_heads)]
        self.wks = [tf.keras.layers.Dense(self.depth, activation=None) for _ in range(num_heads)]
        self.wvs = [tf.keras.layers.Dense(self.depth, activation=None) for _ in range(num_heads)]

        self.av = tf.keras.layers.Dense(self.depth, activation=None)
        self.ak = tf.keras.layers.Dense(self.depth, activation=None)

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