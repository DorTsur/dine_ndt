import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import backend as K
from models.layers import ModLSTM


def DVModel(config):
    def build_DV(name, input_shape, config, split_input=False):
        randN_05 = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        bias_init = randN_05

        lstm = ModLSTM(config.DI_hidden, return_sequences=True, name=name, stateful=True,
                            dropout=config.DI_dropout, recurrent_dropout=config.DI_dropout,
                            contrastive_duplicates=config.contrastive_duplicates)
        split = layers.Lambda(tf.split, arguments={'axis': -2, 'num_or_size_splits': [1, config.contrastive_duplicates]})
        dense0 = layers.Dense(config.DI_hidden, bias_initializer=bias_init, kernel_initializer=randN_05, activation="elu")
        dense1 = layers.Dense(config.DI_last_hidden, bias_initializer=bias_init, kernel_initializer=randN_05,
                              activation="elu")
        dense2 = layers.Dense(1, bias_initializer=bias_init, kernel_initializer=randN_05, activation=None)

        in_ = t = layers.Input(batch_shape=input_shape)
        t = lstm(t)
        t_1, t_2 = split(t)
        t_1, t_2 = dense0(t_1), dense0(t_2)
        t_1, t_2 = dense1(t_1), dense1(t_2)
        t_1, t_2 = dense2(t_1), K.exp(dense2(t_2))
        model = keras.models.Model(inputs=in_, outputs=[t_1, t_2])
        return model

    y_in_shape = [config.batch_size, config.bptt, config.x_dim * (1 + config.contrastive_duplicates)]
    xy_in_shape = [config.batch_size, config.bptt, (config.x_dim + config.y_dim) * (1 + config.contrastive_duplicates)]

    h_y_model = build_DV("LSTM_y",
                         y_in_shape,
                         config)
    h_xy_model = build_DV("LSTM_xy",
                          xy_in_shape,
                          config,
                          split_input=True)

    model = {'y': h_y_model,
             'xy': h_xy_model}

    return model

def DVModelMINE(config):
    def build_DV(input_shape, config):
        randN_05 = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        bias_init = randN_05

        dense0 = layers.Dense(config.DV_hidden, bias_initializer=bias_init, kernel_initializer=randN_05, activation="elu")
        dense1 = layers.Dense(config.DV_last_hidden, bias_initializer=bias_init, kernel_initializer=randN_05,
                              activation="elu")
        dense2 = layers.Dense(1, bias_initializer=bias_init, kernel_initializer=randN_05, activation=None)

        in_ = t = layers.Input(batch_shape=input_shape)
        t = dense0(t)
        t = dense1(t)
        t = dense2(t)
        model = keras.models.Model(inputs=in_, outputs=t)
        return model

    in_shape = [config.batch_size, config.x_dim + config.y_dim]

    model = build_DV(in_shape, config)

    return model
