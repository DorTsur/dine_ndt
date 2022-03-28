import tensorflow as tf


def load_data(config):
    if config.data_name == "uniform":
        return load_data_uniform(config)  # in this option we have an encoder which is fed with zeros
    elif config.data_name == "gaussian":
        return load_data_gaussian(config)
    else:
        raise ValueError("'{}' is invalid data name".format(config.data_name))


def load_data_uniform(config):
    """
    Creates an input batch of zeros for the sequential PMF generator as the shape of an input x.
    """
    def data_gen():
        """
        Data Generator - creates the encoder inputs
        """
        noise = tf.random.uniform(shape=[config.batch_size, 1, config.x_dim])
        yield noise

    train = tf.data.Dataset.from_generator(data_gen,
                                           tf.float32,
                                           output_shapes=tf.TensorShape([config.batch_size, 1, config.x_dim]))

    # define a data loader for each case (vary by length)
    data = {'train': lambda: train.take(config.train_epoch_len).repeat(config.batches),
            'eval': lambda: train.take(config.eval_epoch_len).repeat(10*config.batches),
            'long_eval': lambda: train.take(config.long_eval_epoch_len).repeat(100*config.batches)}
    return data

def load_data_gaussian(config):
    """
    Creates an input batch of zeros for the sequential PMF generator as the shape of an input x.
    """
    def data_gen():
        """
        Data Generator - creates the encoder inputs
        """
        noise = tf.random.normal(shape=[config.batch_size, config.bptt, config.x_dim], mean=0, stddev=tf.sqrt(config.P))
        yield noise

    train = tf.data.Dataset.from_generator(data_gen,
                                           tf.float32,
                                           output_shapes=tf.TensorShape([config.batch_size, config.bptt, config.x_dim]))

    # define a data loader for each case (vary by length)
    data = {'train': lambda: train.take(config.train_epoch_len).repeat(config.batches),
            'eval': lambda: train.take(config.eval_epoch_len).repeat(10*config.batches),
            'long_eval': lambda: train.take(config.long_eval_epoch_len).repeat(100*config.batches)}
    return data
