import numpy as np
import pandas as pd
import os
import dask.array as da
import h5py
from skimage.io import imread as sk_imread
from dask.base import tokenize
import matplotlib.pyplot as plt
import random
from sli import utils


class AudioToH5Converter:

    def __init__(self, path: str, audios: str, val_part: float = 0.25, transpose: bool = False, plotting: bool = False,
                 verbose: bool = False):
        """
        Initialize audio to h5 converter
        :param path: working path
        :param audios: path to file-lang correspondence csv-file
        :param val_part: validation set part
        :param transpose: transpose picture or not
        :param plotting: toggle random samples plotting
        :param verbose: verbose output
        """
        self.path = path
        self.audios = audios
        self.verbose = verbose
        self.val_part = val_part
        self.transpose = transpose
        self.out_files = {}
        self.plotting = plotting

    def convert(self):
        self._get_h5_datasets()
        if self.plotting:
            self._plot(self.out_files)
        return self.out_files

    def _del_previous_data(self, *name):
        """removes data files"""
        if len(name) == 0:
            name = ["temp.h5", "x.h5", "y.h5", "x_va.h5", "y_va.h5"]
        try:
            for n in name:
                full = os.path.join(self.path, n)
                if os.path.exists(full):
                    os.remove(full)
        except Exception as err:
            print(err)

    def _pics_to_h5(self, filenames, target, name):
        """array of pictures to h5"""
        if self.transpose:
            arr = self._imread(filenames, preprocess=np.transpose)
        else:
            arr = self._imread(filenames)
        if len(arr.shape) == 3:
            arr = arr.reshape(arr.shape + (1,))
        arr.to_hdf5(target, name)
        return len(filenames)

    def _get_h5_datasets(self):
        """create h5 file from images"""
        self._del_previous_data()
        df = pd.read_csv(self.audios)
        fl = df['file'].values.tolist()
        print("WRITING", df.shape[0], "FILES INTO H5")

        # labels
        y = df['lang']
        y = pd.get_dummies(y)
        y = y.reindex(sorted(y.columns), axis=1).values  # sort colums, get ndarray from DataFrame
        ya = da.from_array(y, chunks='auto')

        # data
        temp_out_data = os.path.join(self.path, "temp.h5")
        count = self._pics_to_h5(fl, temp_out_data, "temp")
        x = h5py.File(temp_out_data)
        xa = da.from_array(x["temp"], chunks='auto') / 255.

        # calculate test/validation sizes
        va_size = int(self.val_part * count)
        tr_size = count - va_size

        # sampling
        shfl = np.random.permutation(count)
        tr_idx = shfl[:tr_size]
        va_idx = shfl[tr_size:tr_size + va_size]

        print("VALIDATION SET SIZE", va_size)
        print("TRAINING SET SIZE", tr_size)

        xa[tr_idx].astype(np.float16).to_hdf5(os.path.join(self.path, "x.h5"), 'x')
        ya[tr_idx].to_hdf5(os.path.join(self.path, "y.h5"), 'y')
        xa[va_idx].astype(np.float16).to_hdf5(os.path.join(self.path, "x_va.h5"), 'x_va')
        ya[va_idx].to_hdf5(os.path.join(self.path, "y_va.h5"), 'y_va')
        utils.arr_to_csv(ya[tr_idx], self.path, "tr_labels.csv")
        utils.arr_to_csv(ya[va_idx], self.path, "va_labels.csv")
        if self.verbose:
            print("TRAINING DATA SHAPE:", xa[tr_idx].shape)
            print("TRAINING LABELS SHAPE:", y[tr_idx].shape)
            print("VALIDATION DATA SHAPE:", xa[va_idx].shape)
            print("VALIDATION LABELS SHAPE:", y[va_idx].shape)

        x.close()
        self._del_previous_data("temp.h5")
        print()

        self.out_files = {f: os.path.join(self.path, f + ".h5") for f in ["x", "y", "x_va", "y_va"]}

    @staticmethod
    def _imread(filenames, imread=None, preprocess=None):
        """
        modified dask imread method, accepts list of file names instead of glob string
        """

        def add_leading_dimension(x):
            return x[None, ...]

        imread = imread or sk_imread
        name = 'imread-%s' % tokenize(filenames, map(os.path.getmtime, filenames))

        sample = imread(filenames[0])
        if preprocess:
            sample = preprocess(sample)

        keys = [(name, i) + (0,) * len(sample.shape) for i in range(len(filenames))]
        if preprocess:
            values = [(add_leading_dimension, (preprocess, (imread, fn)))
                      for fn in filenames]
        else:
            values = [(add_leading_dimension, (imread, fn))
                      for fn in filenames]
        dsk = dict(zip(keys, values))

        chunks = ((1,) * len(filenames),) + tuple((d,) for d in sample.shape)

        return da.Array(dsk, name, chunks, sample.dtype)

    def _plot(self, files):
        """
        plot random samples
        """
        # read data
        x_tr = da.from_array(h5py.File(files[0])['x'], chunks=1000)
        y_tr = da.from_array(h5py.File(files[1])['y'], chunks=1000)
        x_va = da.from_array(h5py.File(files[2])['x_va'], chunks=1000)
        y_va = da.from_array(h5py.File(files[3])['y_va'], chunks=1000)
        print("TRAINING SAMPLES:", x_tr.shape, "TRAINING LABELS:", y_tr.shape)
        print("VALIDATION SAMPLES:", x_va.shape, "VALIDATION LABELS:", y_va.shape)
        x_tr /= 255.
        x_va /= 255.
        self._plot_samples(x_va, y_va, 'Validation', shape=(2, 2))
        self._plot_samples(x_tr, y_tr, 'Training', shape=(2, 2))

    @staticmethod
    def _plot_samples(data, labels, title, shape=(2, 2)):
        """plots random samples from h5 files"""
        h = shape[0]
        w = shape[1]
        n = h * w
        rands = [random.randint(0, len(data)) for _ in range(n)]
        plt.set_cmap('gray')
        plt.suptitle(title)
        for i in range(1, n + 1):
            label = np.array(labels[rands[i - 1]])
            plt.subplot(h, w, i)
            plt.title("Sample " + str(rands[i - 1]) + " " + str(label))
            plt.axis('off')
            plt.imshow(data[rands[i - 1], :, :, 0], )
        plt.show()
