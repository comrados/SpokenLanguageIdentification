import dask.array as da  # to avoid using generators
import h5py
import tensorflow as tf
import tensorflow.keras.optimizers as tko
import numpy as np
from . import utils


class AudioLangRecognitionNN:

    def __init__(self, path: str, epochs: int = 50, model: str or tf.keras.models.Sequential = None,
                 optimizer: tko.Optimizer = tko.Adadelta(lr=0.1), threshold: float = None, verbose: bool = False):
        """
        Initialize audio downloader

        :param path: working path
        :param epochs: max training epochs
        :param model: path to existing or keras Sequential
        :param optimizer: keras optimizer
        :param threshold: stop training if threshold exceeds given value
        :param verbose: verbose output
        """

        self.path = path
        self.model = model
        self.optimizer = optimizer
        self.verbose = verbose
        self.data = {}
        self.model = model
        self.threshold = threshold
        self.epochs = epochs

    def train(self, data_path):
        """train model"""
        self._load_data(data_path)
        self._build_model_for_training()
        return self._train()

    def predict(self, data_path, labels=('x', 'y'), model=None, save=None):
        """predict labels"""
        self._load_data(data_path)
        self._load_model_for_prediction(model)
        return self._predict(save, model, labels)

    def _load_data(self, data_path):
        """load data from file(s)"""
        if isinstance(data_path, dict):
            self._load_data_dict(data_path)
        elif isinstance(data_path, str):
            self._load_data_string(data_path)
        else:
            raise Exception("Unsupported pointer to the data, must be string or dict of strings")
        for key in self.data:
            print("SAMPLES IN '" + key + "':", self.data[key].shape)

    def _load_data_string(self, data_path):
        """load data (from path string)"""
        f = h5py.File(data_path)
        # dask allows to avoid using .fit_generator() and writing generators
        if 'x_va' and 'y_va' in f.keys():
            self.data['x_va'] = da.from_array(f['x_va'], chunks='auto')
            self.data['y_va'] = da.from_array(f['y_va'], chunks='auto')
        else:
            print("Validation set doesn't exist")
            self.data['x_va'] = None
            self.data['y_va'] = None

        if 'x_te' and 'y_te' in f.keys():
            self.data['x_te'] = da.from_array(f['x_te'], chunks='auto')
            self.data['y_te'] = da.from_array(f['y_te'], chunks='auto')
        else:
            print("Test set doesn't exist")
            self.data['x_te'] = None
            self.data['y_te'] = None

        if 'w' and 'w_va' and 'w_te' in f.keys():
            self.data['w'] = da.from_array(f['w'], chunks='auto')
            self.data['w_va'] = da.from_array(f['w_va'], chunks='auto')
            self.data['w_te'] = da.from_array(f['w_te'], chunks='auto')
        else:
            print("Weights don't exist")
            self.data['w'] = None
            self.data['w_va'] = None
            self.data['w_te'] = None

        self.data['x'] = da.from_array(f['x'], chunks='auto')
        self.data['y'] = da.from_array(f['y'], chunks='auto')

    def _load_data_dict(self, files_dict):
        """load data (from dict of links)"""
        # dask allows to avoid using .fit_generator() and writing generators
        for key in files_dict:
            self.data[key] = da.from_array(h5py.File(files_dict[key])[key], chunks='auto')

    def _load_model_for_prediction(self, model):
        """
        Load existing model for prediction
        """
        # create default model if not exists, or load/use existing
        if model is None:
            model = self.model
        if isinstance(model, str):
            self.model = tf.keras.models.load_model(model)
        else:
            raise TypeError('Wrong path to the existing model')

    def _build_model_for_training(self):
        """Create new model (or load existing one for training)"""
        # create default model if not exists, or load/use existing
        if not self.model:
            self.model = self._default_model()
        elif isinstance(self.model, str):
            self.model = tf.keras.models.load_model(self.model)
        elif isinstance(self.model, tf.keras.models.Sequential):
            pass
        else:
            raise TypeError('Model must be either None, str or tf.keras.models.Sequential')

        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        if self.verbose:
            self.model.summary()

    def _default_model(self):
        """default model layout"""
        return tf.keras.models.Sequential([

            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=self.data['x'][0].shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(rate=0.3),

            tf.keras.layers.Dense(3, activation='softmax')
        ])

    def _train(self):
        """train nn"""

        print("TRAINING MODEL")

        class StopTrainingCallback(tf.keras.callbacks.Callback):

            def __init__(self, threshold):
                super().__init__()
                self.threshold = threshold

            def on_epoch_end(self, epoch, logs={}):
                if logs.get('val_acc') > self.threshold:
                    print(f"\nReached {self.threshold} validation accuracy so cancelling training!")
                    self.model.stop_training = True

        path = utils.check_path(self.path, "models", "model.hdf5")
        cb_cp = tf.keras.callbacks.ModelCheckpoint(path, monitor='val_acc', verbose=0, save_best_only=True,
                                                   save_weights_only=False, mode='auto', period=1)

        cbs = []

        if self.threshold:
            cb_stop = StopTrainingCallback(self.threshold)
            cbs.append(cb_stop)

        cbs.append(cb_cp)

        if self.data['w'] is not None:
            if self.verbose:
                print("TRAINING WITH WEIGHTS")
            history = self.model.fit(self.data['x'], self.data['y'], epochs=self.epochs, batch_size=100, verbose=1,
                                     validation_data=(self.data['x_va'], self.data['y_va']), callbacks=cbs,
                                     sample_weight=self.data['w'])
        else:
            print("TRAINING WITHOUT WEIGHTS")
            history = self.model.fit(self.data['x'], self.data['y'], epochs=self.epochs, batch_size=100, verbose=1,
                                     validation_data=(self.data['x_va'], self.data['y_va']), callbacks=cbs)

        print("TRAINING FINISHED")

        return history

    def _predict(self, save, model, labels):
        """predicts (if only x given) and evaluate (if y also exists)"""
        ev = None
        if labels[1] in self.data:
            print("EVALUATING MODEL:", model)
            ev = self.model.evaluate(x=self.data[labels[0]], y=self.data[labels[1]], batch_size=100, verbose=0)
            if self.verbose and ev:
                print("LOSS:", ev[0], "ACCURACY:", ev[1])
        print("PREDICTING LABELS:", model)
        pr = self.model.predict(self.data[labels[0]], batch_size=100, verbose=0)
        pr_l = self.n_largest_setarr(pr)
        self._save_predict_res(save, pr, pr_l)
        return pr, pr_l, ev

    def _save_predict_res(self, save, pr, pr_l):
        if save in ['probab', 'both']:
            utils.arr_to_csv(pr, self.path, "prediction_probabilities.csv")
        if save in ['labels', 'both']:
            utils.arr_to_csv(pr_l, self.path, "prediction_labels.csv")

    @staticmethod
    def n_largest_setarr(arr, n=1):
        """n-largest element of each row is set to 1, other elements - 0"""
        out = np.zeros_like(arr)
        out[np.arange(len(arr)), np.argpartition(arr, -n, axis=1)[:, -n]] = 1
        return out
