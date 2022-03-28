import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.python.keras import backend as K

def NDTModel(config):
    def forward():
        encoder_transform = keras.models.Sequential([
            keras.layers.LSTM(config.NDT_hidden, return_sequences=True, name="LSTM_enc", stateful=True,
                              batch_input_shape=[config.batch_size, 1, 2*config.x_dim],
                              dropout=config.NDT_dropout, recurrent_dropout=config.NDT_dropout),
            keras.layers.Dense(config.NDT_hidden, activation="elu"),
            keras.layers.Dense(config.NDT_last_hidden, activation="elu"),
            keras.layers.Dense(config.x_dim, activation=None),
            constraint])

        enc_out = list()
        channel_out = list()
        enc_in_stable = keras.layers.Input(shape=[config.bptt, config.x_dim])
        enc_split = tf.split(enc_in_stable, num_or_size_splits=config.bptt, axis=1)
        enc_in_feedback = keras.layers.Input(shape=[1, config.x_dim])

        enc_in_0 = tf.concat([enc_split[0], enc_in_feedback], axis=-1)
        for t in range(config.bptt):
            if t == 0:
                enc_out.append(encoder_transform(enc_in_0))
            else:
                enc_in_t = tf.concat([enc_split[t], enc_out[t - 1]], axis=-1)
                # enc_out.append(norm_layer(encoder_transform(enc_in_t)))
                enc_out.append(encoder_transform(enc_in_t))
            channel_out.append(channel(enc_out[t]))

        channel_out = tf.concat(channel_out, axis=1)
        enc_out = tf.concat(enc_out, axis=1)

        encoder = keras.models.Model(inputs=[enc_in_stable, enc_in_feedback], outputs=[enc_out, channel_out])

        return encoder

    def feedback():
        encoder_transform = keras.models.Sequential([
            keras.layers.LSTM(config.NDT_hidden, return_sequences=True, name="LSTM_enc", stateful=True,
                              batch_input_shape=[config.batch_size, 1, 3 * config.x_dim],
                              recurrent_dropout=config.NDT_dropout, dropout=config.NDT_dropout),
            keras.layers.Dense(config.NDT_hidden, activation="elu"),
            keras.layers.Dense(config.NDT_last_hidden, activation="elu"),
            keras.layers.Dense(config.x_dim, activation=None),
            constraint])

        # encoder_transform_tmp = keras.models.Sequential([
        #     keras.layers.LSTM(config.NDT_hidden, return_sequences=True, name="LSTM_enc", stateful=True,
        #                       batch_input_shape=[config.batch_size, 1, 3 * config.x_dim],
        #                       recurrent_dropout=config.NDT_dropout, dropout=config.NDT_dropout),
        #     keras.layers.Dense(config.NDT_hidden, activation="elu"),
        #     keras.layers.Dense(config.NDT_last_hidden, activation="elu"),
        #     keras.layers.Dense(config.x_dim, activation=None)])


        enc_out = list()
        channel_out = list()
        enc_in_stable = keras.layers.Input(shape=[config.bptt, config.x_dim])
        enc_split = tf.split(enc_in_stable, num_or_size_splits=config.bptt, axis=1)
        enc_in_feedback = keras.layers.Input(shape=[1, 2 * config.x_dim])

        enc_in_0 = tf.concat([enc_split[0], enc_in_feedback], axis=-1)
        for t in range(config.bptt):
            if t == 0:
                enc_out.append(encoder_transform(enc_in_0))
            else:
                enc_in_t = tf.concat([enc_split[t], enc_out[t - 1], channel_out[t - 1]], axis=-1)
                # enc_out.append(norm_layer(encoder_transform(enc_in_t)))
                enc_out.append(encoder_transform(enc_in_t))
            channel_out.append(channel(enc_out[t]))

        channel_out = tf.concat(channel_out, axis=1)
        enc_out = tf.concat(enc_out, axis=1)

        encoder = keras.models.Model(inputs=[enc_in_stable, enc_in_feedback], outputs=[enc_out, channel_out])

        return encoder


    channel = ChannelLayer(config)
    constraint = ConstraintLayer(config)

    encoder = feedback() if config.feedback else forward()

    return encoder

def NDTModel_iid(config):
    def build_ndt():
        randN_05 = keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None)
        bias_init = randN_05

        if config.deep_ndt:
            encoder_transform = keras.models.Sequential([
                keras.layers.Dense(config.NDT_hidden / 3, activation="elu", bias_initializer=bias_init),
                keras.layers.Dense(config.NDT_hidden / 3, activation="elu", bias_initializer=bias_init),
                keras.layers.Dense(config.NDT_hidden / 3, activation="elu", bias_initializer=bias_init),
                keras.layers.Dense(config.NDT_hidden / 3, activation="elu", bias_initializer=bias_init),
                keras.layers.Dense(config.NDT_hidden / 3, activation="elu", bias_initializer=bias_init),
                keras.layers.Dense(config.NDT_last_hidden, activation="elu"),
                keras.layers.Dense(config.x_dim, activation=None),
                constraint])
        else:
            encoder_transform = keras.models.Sequential([
                keras.layers.Dense(config.NDT_hidden, activation="elu"),
                keras.layers.Dense(config.NDT_hidden, activation="elu"),
                keras.layers.Dense(config.NDT_last_hidden, activation="elu"),
                keras.layers.Dense(config.x_dim, activation=None),
                constraint])


        enc_in = keras.layers.Input(shape=[config.x_dim])
        enc_out = encoder_transform(enc_in)
        channel_out = channel(enc_out)

        encoder = keras.models.Model(inputs=enc_in, outputs=[enc_out, channel_out])

        return encoder


    channel = ChannelLayerMINE(config)
    constraint = ConstraintLayerMINE(config)

    encoder = build_ndt()

    return encoder

################################
### CHANNEL CLASS - DINE NDT ###
################################

class ChannelLayer(keras.layers.Layer):
    def __init__(self, config):
        super(ChannelLayer, self).__init__()
        self.config = config
        self.channel = config.channel_name
        self.shape = [config.batch_size, 1, config.x_dim]
        if self.channel is not "awgn":
            self.state = tf.Variable(tf.zeros(shape=self.shape, dtype='float32'), trainable=False)
        self.build_channel(config)

    def build_channel(self, config):
        if config.channel_mat == "eye":
            self.Delta = tf.eye(config.x_dim)
        elif config.channel_mat == "tril":
            self.Delta = tf.constant(1 / config.n * np.tril(np.ones(config.n)))
        else:
            raise ValueError("'{}' is an invalid channel matrix name")

        if self.channel == "awgn":
            self.std = config.v_std
            self.channel_fn = self.call_awgn
        elif self.channel == "magn":
            self.v_std = config.v_std
            self.alpha = config.alpha
            self.channel_fn = self.call_magn
        elif self.channel == "argn":
            self.v_std = config.v_std
            self.alpha = config.alpha
            self.channel_fn = self.call_argn
        else:
            raise ValueError("'{}' is an invalid channel name")

    def call_awgn(self, x):
        """
        AWGN channel implementation
        """
        z = tf.random.normal(shape=self.shape, stddev=self.std)
        return tf.einsum('lij,jk->lik', x, self.Delta) + z

    def call_magn(self, x):
        """
        MA-AGN channel implementation, channel state is the previous innovation noise value
        """
        v = tf.random.normal(shape=self.shape, stddev=self.v_std)
        z = self.alpha * self.state + v
        self.state.assign(v)
        return tf.einsum('lij,jk->lik', x, self.Delta) + z

    def call_argn(self, x):
        """
        AR-AGN channel implementation, channel state is the previous innovation noise value
        """
        v = tf.random.normal(shape=self.shape, stddev=self.v_std)
        z = self.alpha * self.state + v
        self.state.assign(z)
        return tf.einsum('lij,jk->lik', x, self.Delta) + z

    def call(self, x, training=None, mask=None):
        return self.channel_fn(x)

    def reset_states(self):
        if self.channel == "awgn":
            pass
        else:
            self.state.assign(tf.zeros(shape=[self.batch_size, 1, 1], dtype='float32'))

class ConstraintLayer(keras.layers.Layer):
    def __init__(self, config):
        super(ConstraintLayer, self).__init__()
        self.P = config.P
        self.constraint = config.constraint
        self.build_constraint(config)

    def call(self, x, training=None, mask=None):
        return self.call_fn(x)

    def build_constraint(self, config):
        if self.constraint == "norm":
            self.P = config.P
            self.call_fn = self.norm_constraint
        elif self.constraint == "amplitude":
            self.call_fn = self.amplitude_constraint
        elif self.constraint == "old_norm":
            self.call_fn = self.old_norm_constraint
        elif self.constraint == "unconstrained":
            self.call_fn = lambda x: x
        else:
            raise ValueError("'{}' is an invalid ndt constraint name")

    def norm_constraint(self, x):
        return x / tf.sqrt(
                tf.linalg.trace((1 / x.shape[0]) * tf.matmul(tf.transpose(tf.squeeze(x, axis=1)), tf.squeeze(x, axis=1)))) * tf.sqrt(self.P)
    # tf.sqrt(tf.linalg.trace((1 / x.shape[0]) * tf.matmul(tf.transpose(x), x))) * tf.sqrt(2.0)
    def old_norm_constraint(self, x):
        return tf.divide(x, tf.sqrt(tf.reduce_mean(tf.square(x)))) * tf.sqrt(self.P)

    def amplitude_constraint(self, x):
        # Should be written in the future
        return x


################################
### CHANNEL CLASS - MINE NDT ###
################################

class ChannelLayerMINE(ChannelLayer):
    def __init__(self, config):
        super(ChannelLayer, self).__init__()
        self.channel = config.channel_name
        self.shape = [config.batch_size, config.x_dim]
        self.build_channel(config)

    def build_channel(self, config):
        if config.channel_mat == "eye":
            self.Delta = tf.eye(config.x_dim)
        elif config.channel_mat == "tril":
            self.Delta = tf.constant(1 / config.n * np.tril(np.ones(config.n)))
        else:
            raise ValueError("'{}' is an invalid channel matrix name")

        if self.channel == "awgn":
            self.std = config.v_std
            self.channel_fn = self.call_awgn
        elif self.channel == "magn":
            self.v_std = config.v_std
            self.alpha = config.alpha
            self.channel_fn = self.call_magn
        elif self.channel == "argn":
            self.v_std = config.v_std
            self.alpha = config.alpha
            self.channel_fn = self.call_argn
        else:
            raise ValueError("'{}' is an invalid channel name")

    def call_awgn(self, x):
        """
        AWGN channel implementation
        """
        z = tf.random.normal(shape=self.shape, stddev=self.std)
        return tf.einsum('ij,jk->ik', x, self.Delta) + z

    def call_magn(self, x):
        """
        MA-AGN channel implementation, channel state is the previous innovation noise value
        """
        v = tf.random.normal(shape=self.shape, stddev=self.v_std)
        z = self.alpha * self.state + v
        self.state.assign(v)
        return tf.einsum('lij,jk->lik', x, self.Delta) + z

    def call_argn(self, x):
        """
        AR-AGN channel implementation, channel state is the previous innovation noise value
        """
        v = tf.random.normal(shape=self.shape, stddev=self.v_std)
        z = self.alpha * self.state + v
        self.state.assign(z)
        return tf.einsum('lij,jk->lik', x, self.Delta) + z

    def call(self, x, training=None, mask=None):
        return self.channel_fn(x)

    def reset_states(self):
        if self.channel == "awgn":
            pass
        else:
            self.state.assign(tf.zeros(shape=[self.batch_size, 1, 1], dtype='float32'))

class ConstraintLayerMINE(keras.layers.Layer):
    def __init__(self, config):
        super(ConstraintLayerMINE, self).__init__()
        self.P = config.P
        self.constraint = config.constraint
        self.build_constraint(config)
        self.batch_size = config.batch_size

    def call(self, x, training=None, mask=None):
        return self.call_fn(x)

    def build_constraint(self, config):
        if self.constraint == "norm":
            self.P = config.P
            self.call_fn = self.norm_constraint
        elif self.constraint == "amplitude":
            self.call_fn = self.amplitude_constraint
        elif self.constraint == "old_norm":
            self.call_fn = self.old_norm_constraint
        elif self.constraint == "unconstrained":
            self.call_fn = lambda x: x
        else:
            raise ValueError("'{}' is an invalid ndt constraint name")

    # tf.sqrt(tf.linalg.trace((1 / x.shape[0]) * tf.matmul(tf.transpose(x), x))) * tf.sqrt(2.0)
    def old_norm_constraint(self, x):
        return tf.divide(x, tf.sqrt(tf.reduce_mean(tf.square(x)))) * tf.sqrt(self.P)

    def amplitude_constraint(self, x):
        # Should be written in the future
        return x

    def norm_constraint(self, x):
        return x / tf.sqrt(tf.linalg.trace((1.0 / self.batch_size) * tf.matmul(tf.transpose(x), x))) * tf.sqrt(self.P)



