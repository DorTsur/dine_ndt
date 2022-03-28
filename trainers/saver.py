import numpy as np

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
import os
from scipy.io import savemat
from scipy.stats import norm

logger = logging.getLogger("logger")


class Visualizer(object):
    def __init__(self, config):
        self.config = config
        self.save_path = os.path.join(config.tensor_board_dir, 'visual')

    def reset_state(self):
        pass

    def update_state(self, *args):
        pass

    def visualize(self):
        pass

    def save_raw_data(self):
        pass


class DVVisualizer(Visualizer):
    def __init__(self, config):
        # Class for saving DV potentials values
        super().__init__(config)
        self.t_y_list = list()
        self.t_xy_list = list()

    def reset_state(self):
        self.t_y_list = list()
        self.t_xy_list = list()

    def update_state(self, data):
        self.t_y_list.append(data['t_y'])
        self.t_xy_list.append(data['t_xy'])

    def convert_lists_to_np(self):
        t_y = [y[0] for y in self.t_y_list]
        t_y = tf.concat(t_y, axis=1)
        t_y_ = [y[1] for y in self.t_y_list]
        t_y_ = tf.concat(t_y_, axis=1)

        t_xy = [xy[0] for xy in self.t_xy_list]
        t_xy = tf.concat(t_xy, axis=1)
        t_xy_ = [xy[1] for xy in self.t_xy_list]
        t_xy_ = tf.concat(t_xy_, axis=1)
        return t_y, t_y_, t_xy, t_xy_

    def save(self, name=None, save_dv=False):
        save_dict = {}
        if save_dv:
            t_y, t_y_, t_xy, t_xy_ = self.convert_lists_to_np()
            save_dict["t_y"] = t_y.numpy()
            save_dict["t_xy"] = t_xy.numpy()

        file_name = name if name is not None else 'raw_data_latest.mat'
        savemat(os.path.join(self.config.tensor_board_dir, 'visual',
                                 file_name), save_dict)

    def histogram(self, x):
        return


class DINE_NDT_vis(DVVisualizer):
        def __init__(self, config):
            super().__init__(config)
            self.x_list = list()
            self.y_list = list()

        def reset_state(self):
            super().reset_state()
            self.x_list = list()
            self.y_list = list()

        def update_state(self, data):
            super().update_state(data)
            self.x_list.append(data['x'])
            self.y_list.append(['y'])

        def convert_lists_to_np(self):
            t_y, t_y_, t_xy, t_xy_ = super().convert_lists_to_np()
            x_n = [x for x in self.x_list]
            x_np = tf.concat(x_n, axis=1)
            y_n = [y for y in self.y_list]
            y_np = tf.concat(y_n, axis=1)
            return t_y, t_y_, t_xy, t_xy_, x_np, y_np

        def save(self, models=None, path=None, name=None, save_dv=False):
            save_dict = {}
            if save_dv:
                t_y, t_y_, t_xy, t_xy_, x, y = self.convert_lists_to_np()
                save_dict["t_y"] = t_y.numpy()
                save_dict["t_xy"] = t_xy.numpy()
                save_dict["x"] = x.numpy()
                save_dict["y"] = y.numpy()

            file_name = name if name is not None else 'raw_data_latest.mat'
            savemat(os.path.join(self.config.tensor_board_dir, 'visual',
                                 file_name), save_dict)

            self.save_models(models, path)

        def save_models(self, models, path):
            def save_recursively(models, path):
                for model in models:
                    if isinstance(models[model], dict):
                        save_recursively(models[model], path)
                    else:
                        path = os.path.join(path, model, model)
                        # if model == 'ndt':
                        #     models[model].save(filepath=os.path.join(path, "enc_model"))
                        #     # models[model].save_weights(filepath=os.path.join(path, model + "weights_h5.h5"),save_format="h5")
                        models[model].save_weights(filepath=os.path.join(path, model, "weights_tf", "weights"),
                                                   save_format="tf")

            save_recursively(models, path)


class MINE_vis(Visualizer):
    def __init__(self, config):
        # Class for saving DV potentials values
        super().__init__(config)
        self.t_list = list()
        self.config = config

    def reset_state(self):
        self.t_list = list()

    def update_state(self, data):
        self.t_list.append(data['t'])

    def convert_lists_to_np(self):
        t = [y[0] for y in self.t_list]
        t = tf.concat(t, axis=1)
        t_ = [y[1] for y in self.t_list]
        t_ = tf.concat(t_, axis=1)

        return t, t_

    def save(self, models=None, path=None, name=None, save_dv=False):
        save_dict = {}
        if save_dv:
            t, t_ = self.convert_lists_to_np()
            save_dict["t"] = t.numpy()
            save_dict["t_"] = t_.numpy()

        file_name = name if name is not None else 'raw_data_latest.mat'
        savemat(os.path.join(self.config.tensor_board_dir, 'visual',
                                 file_name), save_dict)

        self.save_models(models, path)

    def histogram(self, x):
        return

    def save_models(self, models, path):
        def save_recursively(models, path):
            for model in models:
                if isinstance(models[model], dict):
                    save_recursively(models[model], path)
                else:
                    path = os.path.join(path, model, model)
                    # if model == 'ndt':
                    #     models[model].save(filepath=os.path.join(path, "enc_model"))
                    #     # models[model].save_weights(filepath=os.path.join(path, model + "weights_h5.h5"),save_format="h5")
                    models[model].save_weights(filepath=os.path.join(path, model, "weights_tf", "weights"),
                                               save_format="tf")

        save_recursively(models, path)

    def evaluate_ndt(self, ndt_model, path, epoch):
        self.evaluate_ndt_struct(ndt_model, path, epoch)
        self.evaluate_ndt_hist(ndt_model, path, epoch)

    def evaluate_ndt_struct(self, ndt_model, path, epoch):
        # obtain model input and output (for uniform p)
        p = tf.expand_dims(tf.linspace(start=0., stop=1., num=self.config.batch_size), axis=-1)
        x = ndt_model(p, training=False)

        theo = norm.ppf(p)

        # convert to numpy
        xn, pn = x[0].numpy(), p.numpy()

        data = {"p": pn,
                "x": xn,
                "theo": theo}

        savemat(os.path.join(path, f"NDT_struct_data_epoch_{epoch}"), data)

        # plot the mapping:
        plt.figure()
        plt.plot(pn, xn, 'bo', label="NDT")
        plt.plot(pn, theo, label="Theoretical")
        plt.legend()
        plt.title("NDT mapping vs. Gaussian inverse")
        plt.savefig(os.path.join(path, f"NDT structure for epoch {epoch}"))


        # save p and x

    def evaluate_ndt_hist(self, ndt_model,path, epoch):
        ul = []
        xl = []

        for i in range(self.config.repeat_uniform):
            ul.append(tf.random.uniform(shape=[self.config.batch_size, self.config.x_dim]))
            xl.append(ndt_model(ul[i]))

        u = tf.concat(ul, axis=0)
        x = tf.concat(xl, axis=0)

        un, xn = tf.squeeze(x[0]).numpy(), tf.squeeze(u).numpy()

        data = {"u": un,
                "x": xn}

        savemat(os.path.join(path, f"NDT_hist_data_epoch_{epoch}"), data)

        fig, axs = plt.subplots(2)
        axs[0].set_title('Input Histogram')
        axs[0].hist(un, bins=35)
        axs[1].set_title('Output Histogram')
        axs[1].hist(xn, bins=35)

        plt.savefig(os.path.join(path, f"NDT mapping for epoch {epoch}"))

        fig, axs = plt.subplots(2)
        axs[0].set_title('Input Histogram')
        axs[0].hist(un, density=True, bins=35)
        axs[1].set_title('Output Histogram')
        axs[1].hist(xn, density=True, bins=35)

        plt.savefig(os.path.join(path, f"NDT mapping for epoch {epoch} with density"))






###################################
####### HISTOGRAM OBJECTS #########
###################################

class Figure(object):

    def __init__(self, name='fig', **kwargs):
        self.name = name
        self.fig_data = list()

    def reset_states(self):
        self.fig_data = list()

    def set_data(self, *args, **kwargs):
        pass

    def aggregate_data(self):
        if isinstance(self.fig_data, list):
            return np.concatenate(self.fig_data, axis=0)
        else:
            return self.fig_data

    def update_state(self, data):
        self.fig_data.append(data)

    def plot(self, save=None):
        pass


class Histogram2d(Figure):
    def __init__(self, name, **kwargs):
        super(Histogram2d, self).__init__(name, **kwargs)

    def aggregate_data(self):
        try:
            data = np.concatenate(self.fig_data, axis=1)
        except ValueError:
            return None
        return data
        # return np.reshape(data, [-1, data.shape[-1]])  # - ziv's line

    def plot(self, save=None, save_path="./visual", save_name="fig.png"):

        data = self.aggregate_data()

        if data is None:
            logger.info("no data aggregated at visualizer")
            return

        plt.figure()
        data_hist = np.reshape(data, newshape=[np.prod(data.shape[:-1]),data.shape[-1]])
        d = plt.hist2d(data_hist[100:, 0], data_hist[100:, 1], bins=50)
        plt.title(self.name)
        bins = d[0]
        edges = d[1]
        if save:
            plt.savefig(os.path.join(save_path, save_name))
            savemat(os.path.join(save_path, self.name + '_raw_data.mat'),
                    {"bins": bins,
                     "edges": edges,
                     "data": data})

        plt.close()

class Histogram(Figure):
    def __init__(self, name, **kwargs):
        super(Histogram, self).__init__(name, **kwargs)

    def aggregate_data(self):
        try:
            data = np.concatenate(self.fig_data, axis=0)
        except ValueError:
            return None
        return np.reshape(data, [-1, data.shape[-1]])

    def plot(self, save=None, save_path="./", save_name="fig.png"):

        data = self.aggregate_data()

        if data is None:
            logger.info("no data aggregated at visualizer")
            return

        plt.figure()
        d = plt.hist(data[100:], bins=np.linspace(np.min(data), np.max(data), 200))
        plt.title(self.name)
        bins = d[0]
        edges = d[1]
        if save:
            plt.savefig(os.path.join(save_path, save_name))
            savemat(os.path.join(save_path, self.name + '_raw_data.mat'),
                    {"bins": bins,
                     "edges": edges})
        plt.close()
