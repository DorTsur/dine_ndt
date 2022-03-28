import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
from losses.dv_loss import DINELoss, DVLoss
from metrics.dine_ndt_metrics import DINE_NDT_Metrics, MINE_NDT_Metrics
from trainers.saver import DINE_NDT_vis, MINE_vis
import os


def build_trainer(model, data, config):
    if config.trainer_name == "dine_ndt":
        trainer = DINE_NDT_trainer(model, data, config)
    elif config.trainer_name == "mine_ndt":
        trainer = MINE_NDT_trainer(model, data, config)
    else:
        raise ValueError("'{}' is an invalid trainer name")
    return trainer


class DINE_NDT_trainer(object):
    def __init__(self, model, data, config):
        self.config = config
        self.data = data
        self.model = model
        self.dine_loss = DINELoss()
        self.ndt_loss = DINELoss(subtract=True)
        self.contrastive_duplicates = config.contrastive_duplicates
        self.lr = config.lr
        self.feedback = config.feedback

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.global_step_enc = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.saver = DINE_NDT_vis(config)


        # determine optimizer:
        if config.optimizer == "adam":
            self.optimizer = {"dv_y": Adam(amsgrad=True, learning_rate=self.lr),
                              "dv_xy": Adam(amsgrad=True, learning_rate=self.lr),
                              "ndt": Adam(amsgrad=True, learning_rate=self.lr/2)}  # the model's optimizers
        else:
            self.optimizer = {"dv_y": SGD(learning_rate=config.lr),
                              "dv_xy": SGD(learning_rate=config.lr),
                              "ndt": SGD(learning_rate=config.lr/4)
                              }  # the model's optimizers

        # determine contrastive noise:
        if config.contrastive_type == "uniform":
            self.contrastive_fn = self.contrastive_uniform
        elif config.contrastive_type == "gauss":
            self.contrastive_fn = self.contrastive_gauss
        else:
            raise ValueError("invalid contrastive noise type")


        self.metrics = {"train": DINE_NDT_Metrics(config.train_writer, name='dv_train'),
                        "eval": DINE_NDT_Metrics(config.test_writer, name='dv_eval')}  # the trainer's metrics

    def train(self):
        self.evaluate(epoch=0)
        print('Training starts...')
        for epoch in range(self.config.num_epochs):
            self.train_epoch(epoch)                                     # perform a training epoch
            if (epoch % self.config.eval_freq == 0) and (epoch > 0):
                self.evaluate(epoch)                                    # perform evaluation step

    def reset_fb(self):
        if self.feedback:
            self.y_feedback = tf.convert_to_tensor(np.zeros([self.config.batch_size, 1, 2*self.config.x_dim]))
        else:
            self.y_feedback = tf.convert_to_tensor(np.zeros([self.config.batch_size, 1, self.config.x_dim]))

    def evaluate(self, epoch, iterator='eval'):
        self.sync_eval_model()
        self.metrics["eval"].reset_states()
        self.reset_model_states()
        self.saver.reset_state()
        self.reset_fb()

        for sample in self.data[iterator]():
            output = self.eval_step(sample)
            self.metrics['eval'].update_state(output[0], output[1])

        self.metrics["eval"].log_metrics(epoch, model_name="Evaluation")

        if iterator == "long_eval":
            print('Saving models')
            self.saver.save_models(models=self.model, path=os.path.join(self.config.tensor_board_dir, "models_weights"))
            print('Models saved.')

    def final_eval(self):
        print('Final evaluation...')
        self.evaluate(epoch=0, iterator="long_eval")

    def eval_step(self, sample):
        X_batch = [sample, self.y_feedback]

        [x, y] = self.model['ndt'](X_batch, training=False)
        [input_y, input_xy] = self.DI_data(x, y)

        t_y = self.model['dv_eval']['y'](input_y, training=False)
        t_xy = self.model['dv_eval']['xy'](input_xy, training=False)

        if self.feedback:
            self.y_feedback = tf.concat([tf.reshape(x[:, -1, :], [self.config.batch_size, 1, self.config.x_dim]),
                                    tf.reshape(y[:, -1, :], [self.config.batch_size, 1, self.config.x_dim])],
                                   axis=-1)
        else:
            self.y_feedback = tf.expand_dims(x[:, -1, :], axis=1)

        return t_y, t_xy

    def train_epoch(self, epoch):
        self.metrics["train"].reset_states()
        self.reset_model_states()
        self.reset_fb()

        if np.random.rand() > 0.3 or epoch < 20:  # train DINE
            training_model = "DINE"
            for sample in self.data["train"]():
                outputs = self.train_step(sample, model="DINE")
                self.metrics["train"].update_state(outputs[0], outputs[1])

        else:  # train Enc.
            training_model = "NDT"
            for sample in self.data["train"]():
                outputs = self.train_step(sample, model="NDT")
                self.metrics["train"].update_state(outputs[0], outputs[1])

        print("Epoch = {}, training model is {}, est_di = {}".format(epoch, training_model, self.metrics["train"].result()[2]))

        # if np.random.rand() > 0.3 or epoch < 10:  # train DINE
        #     model_name = "DV epoch"
        #     for sample in self.data_iterators["train"]():
        #         if sample is None:
        #             self.reset_model_states()  # in case we deal with a dataset
        #             continue
        #         # if i > 0:
        #         if i >= 0:
        #                 sample = self.fb_input
        #         output = self.train_dine_step(sample)  # calculate model outputs and perform a training step
        #         self.metrics["train"].update_state(output[0], output[1], output[2])  # update trainer metrics
        # else:  # train Enc.
        #     model_name = "Encoder epoch"
        #     for sample in self.data_iterators["train"]():
        #         if sample is None:
        #             self.reset_model_states()  # in case we deal with a dataset
        #             continue
        #         output = self.train_enc_step(self.fb_input)  # calculate model outputs and perform a training step
        #         self.metrics["train"].update_state(output[0], output[1], output[2])  # update trainer metrics

    def train_step(self, sample, model):
        if model == "DINE":
            gradients_dv_y, gradients_dv_xy, t_y, t_xy = self.compute_dine_grads(sample)
            self.apply_dv_grads(gradients_dv_y, gradients_dv_xy)
        else:
            gradients_enc, t_y, t_xy = self.compute_ndt_grads(sample)  # calculate gradients
            self.apply_ndt_grads(gradients_enc)
        return [t_y, t_xy]

    def compute_dine_grads(self, sample):
        with tf.GradientTape(persistent=True) as tape:
            X_batch = [sample, self.y_feedback]
            [x, y] = self.model['ndt'](X_batch, training=False)  # obtain samples through model
            [input_y, input_xy] = self.DI_data(x, y)
            t_y = self.model['dv']['y'](input_y, training=True)
            t_xy = self.model['dv']['xy'](input_xy, training=True)
            loss = self.dine_loss(t_y, t_xy)  # calculate loss for each model
            loss_y, loss_xy = tf.split(loss, num_or_size_splits=2, axis=0)

        gradients_dv_y = tape.gradient(loss_y, self.model['dv']['y'].trainable_weights)
        gradients_dv_xy = tape.gradient(loss_xy, self.model['dv']['xy'].trainable_weights)  # calculate gradients

        gradients_dv_y, grad_norm_dv_y = tf.clip_by_global_norm(gradients_dv_y, self.config.clip_grad_norm)
        gradients_dv_xy, grad_norm_dv_xy = tf.clip_by_global_norm(gradients_dv_xy,
                                                                  self.config.clip_grad_norm)  # normalize gradients

        with self.config.train_writer.as_default():  # update trainer metrics
            tf.summary.scalar("grad_norm_dv_y", grad_norm_dv_y, self.global_step)
            tf.summary.scalar("grad_norm_dv_xy", grad_norm_dv_xy, self.global_step)
            tf.summary.scalar("x_mean", tf.math.reduce_mean(x), self.global_step)
            tf.summary.scalar("y_mean", tf.math.reduce_mean(y), self.global_step)
            self.global_step.assign_add(1)

        if self.feedback:
            self.y_feedback = tf.concat([tf.reshape(x[:, -1, :], [self.config.batch_size, 1, self.config.x_dim]),
                                         tf.reshape(y[:, -1, :], [self.config.batch_size, 1, self.config.x_dim])],
                                        axis=-1)
        else:
            self.y_feedback = tf.expand_dims(x[:, -1, :], axis=1)

        return gradients_dv_y, gradients_dv_xy, t_y, t_xy

    def compute_ndt_grads(self, sample):
        with tf.GradientTape(persistent=True) as tape:
            X_batch = [sample, self.y_feedback]
            [x, y] = self.model['ndt'](X_batch, training=True)  # obtain samples through model
            [input_y, input_xy] = self.DI_data(x, y)
            t_y = self.model['dv']['y'](input_y, training=False)
            t_xy = self.model['dv']['xy'](input_xy, training=False)
            loss = self.ndt_loss(t_y, t_xy)  # calculate loss for each model

        gradients_ndt = tape.gradient(loss, self.model['ndt'].trainable_weights)

        gradients_ndt, grad_norm_ndt = tf.clip_by_global_norm(gradients_ndt, self.config.clip_grad_norm)

        with self.config.train_writer.as_default():  # update trainer metrics
            tf.summary.scalar("grad_norm_ndt", grad_norm_ndt, self.global_step)
            tf.summary.scalar("di_training", -loss, self.global_step)
            self.global_step_enc.assign_add(1)

        if self.feedback:
            self.y_feedback = tf.concat([tf.reshape(x[:, -1, :], [self.config.batch_size, 1, self.config.x_dim]),
                                         tf.reshape(y[:, -1, :], [self.config.batch_size, 1, self.config.x_dim])],
                                        axis=-1)
        else:
            self.y_feedback = tf.expand_dims(x[:, -1, :], axis=1)

        return gradients_ndt, t_y, t_xy

    def DI_data(self, x, y):
        y_tilde = []
        xy_tilde = []
        for i in range(self.contrastive_duplicates):
            y_tilde.append(self.contrastive_fn(y))
            xy_tilde.append(tf.concat([x, y_tilde[i]], axis=-1))
        y_tilde = tf.concat(y_tilde, axis=-1)
        input_y = tf.concat([y, y_tilde], axis=-1)
        xy = tf.concat([x, y], axis=-1)
        xy_tilde = tf.concat(xy_tilde, axis=-1)
        input_xy = tf.concat([xy, xy_tilde], axis=-1)
        return [input_y, input_xy]

    def contrastive_uniform(self, y):
        return tf.random.uniform(tf.shape(y), minval=tf.reduce_min(y), maxval=tf.reduce_max(y))

    def contrastive_gauss(self, y):
        std = tf.math.reduce_std(y)
        return tf.random.normal(shape=tf.shape(y), dtype=tf.float64, stddev=std)

    def sync_eval_model(self):
        # sync DV:
        w_y = self.model['dv']['y'].get_weights()
        w_xy = self.model['dv']['xy'].get_weights()
        self.model['dv_eval']['y'].set_weights(w_y)
        self.model['dv_eval']['xy'].set_weights(w_xy)
        # sync enc:
        w_enc = self.model['ndt'].get_weights()  # similarly sync encoder model
        self.model['ndt_eval'].set_weights(w_enc)

    def reset_model_states(self):
        def reset_recursively(models):
            for model in models.values():
                if isinstance(model, dict):
                    reset_recursively(model)
                else:
                    model.reset_states()

        reset_recursively(self.model)
        # THIS MIGHT MEAN I NEED TO RESET CHANNEL BY MYSELF:
        # if (self.config.channel_name == "ising") or (self.config.channel_name == "trapdoor"):
        #     self.model['enc'].layers[3].reset_states()
        #     self.model['enc_eval'].layers[3].reset_states()

    @tf.function
    def apply_dv_grads(self, gradients_dv_y, gradients_dv_xy):
        self.optimizer['dv_y'].apply_gradients(zip(gradients_dv_y, self.model['dv']['y'].trainable_weights))
        self.optimizer['dv_xy'].apply_gradients(zip(gradients_dv_xy, self.model['dv']['xy'].trainable_weights))

    @tf.function
    def apply_ndt_grads(self, gradients_enc):
        self.optimizer['ndt'].apply_gradients(zip(gradients_enc, self.model['ndt'].trainable_weights))


class MINE_NDT_trainer(object):
    def __init__(self, model, data, config):
        self.config = config
        self.data = data
        self.model = model
        self.mine_loss = DVLoss()
        self.ndt_loss = DVLoss()
        self.lr = config.lr
        self.feedback = config.feedback

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.global_step_enc = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.saver = MINE_vis(config)
        self.ndt_eval_dir = os.path.join(config.tensor_board_dir, "NDT_eval")
        os.mkdir(self.ndt_eval_dir)


        # determine optimizer:
        if config.optimizer == "adam":
            self.optimizer = {"dv": Adam(amsgrad=True, learning_rate=self.lr),
                              "ndt": Adam(amsgrad=True, learning_rate=self.lr/2)}  # the model's optimizers
        else:
            self.optimizer = {"dv": SGD(learning_rate=config.lr),
                              "ndt": SGD(learning_rate=config.lr/4)
                              }  # the model's optimizers

        self.metrics = {"train": MINE_NDT_Metrics(config.train_writer, name='dv_train'),
                        "eval": MINE_NDT_Metrics(config.test_writer, name='dv_eval')}  # the trainer's metrics

    def train(self):
        self.evaluate(epoch=0)
        print('Training starts...')
        for epoch in range(self.config.num_epochs):
            self.train_epoch(epoch)                                     # perform a training epoch
            if (epoch % self.config.eval_freq == 0) and (epoch > 0):
                self.evaluate(epoch)   # perform evaluation step
            if (epoch % self.config.ndt_eval_freq == 0):
                self.saver.evaluate_ndt(ndt_model=self.model['ndt'], path=self.ndt_eval_dir, epoch=epoch)

    def evaluate(self, epoch, iterator='eval'):
        self.sync_eval_model()
        self.metrics["eval"].reset_states()
        self.reset_model_states()
        self.saver.reset_state()

        for sample in self.data[iterator]():
            sample = tf.squeeze(sample, axis=1)
            output = self.eval_step(sample)
            self.metrics['eval'].update_state(output[0], output[1])

        self.metrics["eval"].log_metrics(epoch, model_name="Evaluation")

        if iterator == "long_eval":
            print('Saving models')
            self.saver.save_models(models=self.model, path=os.path.join(self.config.tensor_board_dir, "models_weights"))
            print('Models saved.')

    def final_eval(self):
        print('Final evaluation...')
        self.evaluate(epoch=0, iterator="long_eval")

    def eval_step(self, sample):
        [x, y] = self.model['ndt'](sample, training=False)
        [input_xy, input_xy_bar] = self.DV_data(x, y)

        t = self.model['dv_eval'](input_xy, training=False)
        t_ = tf.exp(self.model['dv_eval'](input_xy_bar, training=False))

        return [t, t_]

    def train_epoch(self, epoch):
        self.metrics["train"].reset_states()
        self.reset_model_states()

        if np.random.rand() > 0.3 or epoch < 20:  # train DINE
            training_model = "MINE"
            for sample in self.data["train"]():
                sample = tf.squeeze(sample, axis=1)
                outputs = self.train_step(sample, model=training_model)
                self.metrics["train"].update_state(outputs[0], outputs[1])

        else:  # train Enc.
            training_model = "NDT"
            for sample in self.data["train"]():
                sample = tf.squeeze(sample, axis=1)
                outputs = self.train_step(sample, model=training_model)
                self.metrics["train"].update_state(outputs[0], outputs[1])

        print("Epoch = {}, training model is {}, est_di = {}".format(epoch, training_model,
                                                                     self.metrics["train"].result()[0]))

    def train_step(self, sample, model):
        if model == "MINE":
            gradients, t = self.compute_mine_grads(sample)
            self.apply_dv_grads(gradients)
        else:
            gradients_enc, t = self.compute_ndt_grads(sample)  # calculate gradients
            self.apply_ndt_grads(gradients_enc)
        return t

    def compute_mine_grads(self, sample):
        with tf.GradientTape(persistent=True) as tape:
            [x, y] = self.model['ndt'](sample, training=False)  # obtain samples through model
            [input_xy, input_xy_bar] = self.DV_data(x, y)
            t = self.model['dv'](input_xy, training=True)
            t_ = tf.exp(self.model['dv'](input_xy_bar, training=True))
            loss = self.mine_loss(t, t_)  # calculate loss for each model

        gradients_dv = tape.gradient(loss, self.model['dv'].trainable_weights)

        gradients_dv, grad_norm_dv = tf.clip_by_global_norm(gradients_dv, self.config.clip_grad_norm) # normalize gradients

        with self.config.train_writer.as_default():  # update trainer metrics
            tf.summary.scalar("grad_norm_dv_y", grad_norm_dv, self.global_step)
            tf.summary.scalar("x_mean", tf.math.reduce_mean(x), self.global_step)
            tf.summary.scalar("y_mean", tf.math.reduce_mean(y), self.global_step)
            self.global_step.assign_add(1)


        return gradients_dv, [t, t_]

    def compute_ndt_grads(self, sample):
        with tf.GradientTape(persistent=True) as tape:
            [x, y] = self.model['ndt'](sample, training=True)  # obtain samples through model
            [input_xy, input_xy_bar] = self.DV_data(x, y)
            t = self.model['dv'](input_xy, training=False)
            t_ = tf.exp(self.model['dv'](input_xy_bar, training=False))
            loss = self.ndt_loss(t, t_)  # calculate loss for each model

        gradients_ndt = tape.gradient(loss, self.model['ndt'].trainable_weights)

        gradients_ndt, grad_norm_ndt = tf.clip_by_global_norm(gradients_ndt, self.config.clip_grad_norm)

        with self.config.train_writer.as_default():  # update trainer metrics
            tf.summary.scalar("grad_norm_ndt", grad_norm_ndt, self.global_step)
            tf.summary.scalar("di_training", -loss, self.global_step)
            self.global_step_enc.assign_add(1)

        return gradients_ndt, [t, t_]

    def DV_data(self, x, y):
        y_bar = tf.random.shuffle(y.numpy())
        input = tf.concat([x, y], axis=-1)
        input_bar = tf.concat([x, y_bar], axis=-1)

        return [input, input_bar]

    def sync_eval_model(self):
        # sync DV:
        w_dv = self.model['dv'].get_weights()
        self.model['dv_eval'].set_weights(w_dv)
        # sync enc:
        w_enc = self.model['ndt'].get_weights()  # similarly sync encoder model
        self.model['ndt_eval'].set_weights(w_enc)

    def reset_model_states(self):
        def reset_recursively(models):
            for model in models.values():
                if isinstance(model, dict):
                    reset_recursively(model)
                else:
                    model.reset_states()

        reset_recursively(self.model)

    @tf.function
    def apply_dv_grads(self, gradients_dv):
        self.optimizer['dv'].apply_gradients(zip(gradients_dv, self.model['dv'].trainable_weights))

    @tf.function
    def apply_ndt_grads(self, gradients_enc):
        self.optimizer['ndt'].apply_gradients(zip(gradients_enc, self.model['ndt'].trainable_weights))



