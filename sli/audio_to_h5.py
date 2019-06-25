import numpy as np
import pandas as pd
import os
import dask.array as da
import h5py
from skimage.io import imread as sk_imread
from dask.base import tokenize


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


class AudioToH5Converter:

    def __init__(self, path: str, audios: str, val_part: float = 0.25, transpose: bool = True, verbose: bool = False):
        """
        Initialize audio to h5 converter
        :param path: working path
        :param audios: path to file-lang correspondence csv-file
        :param val_part: validation set part
        :param transpose transpose picture or not
        :param verbose: verbose output
        """
        self.path = path
        self.audios = audios
        self.verbose = verbose
        self.val_part = val_part
        self.transpose = transpose

    def convert(self):
        self._get_h5_datasets()

    def _del_previous_data(self, *name):
        """removes old data files"""
        if len(name) == 0:
            name = ["temp.h5", "x_tr.h5", "y_tr.h5", "x_va.h5", "y_va.h5"]
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
            arr = _imread(filenames, preprocess=np.transpose)
        else:
            arr = _imread(filenames)
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
        y = y.reindex_axis(sorted(y.columns), axis=1).values
        y = da.from_array(y, chunks=1000)

        # data
        temp_out_data = os.path.join(self.path, "temp.h5")
        count = self._pics_to_h5(fl, temp_out_data, "temp")
        x = h5py.File(temp_out_data)["temp"]
        x = da.from_array(x, chunks=1000)

        # calculate test/validation sizes
        va_size = int(self.val_part * count)
        tr_size = count - va_size

        # sampling
        shfl = np.random.permutation(count)
        tr_idx = shfl[:tr_size]
        va_idx = shfl[tr_size:tr_size + va_size]

        print("VALIDATION SET SIZE", va_size)
        print("TRAINING SET SIZE", tr_size)

        x[tr_idx].to_hdf5(os.path.join(self.path, "x_tr.h5"), 'x_tr')
        y[tr_idx].to_hdf5(os.path.join(self.path, "y_tr.h5"), 'y_tr')
        x[va_idx].to_hdf5(os.path.join(self.path, "x_va.h5"), 'x_va')
        y[va_idx].to_hdf5(os.path.join(self.path, "y_va.h5"), 'y_va')
        if self.verbose:
            print("TRAINING DATA SHAPE:", x[tr_idx].shape)
            print("TRAINING LABELS SHAPE:", y[tr_idx].shape)
            print("VALIDATION DATA SHAPE:", x[va_idx].shape)
            print("VALIDATION LABELS SHAPE:", y[va_idx].shape)

        del x
        self._del_previous_data("temp.h5")
        print()
