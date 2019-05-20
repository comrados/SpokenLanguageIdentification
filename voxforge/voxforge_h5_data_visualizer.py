import random
import matplotlib.pyplot as plt
import h5py
import dask.array as da
import os
import numpy as np


def plot_samples(data, labels, shape=(2, 2)):
    h = shape[0]
    w = shape[1]
    n = h * w
    rands = [random.randint(0, len(data)) for _ in range(n)]
    plt.set_cmap('gnuplot2')
    for i in range(1, n + 1):
        label = np.array(labels[rands[i - 1]])
        plt.subplot(h, w, i)
        plt.title("Sample " + str(rands[i - 1]) + " " + str(label))
        plt.axis('off')
        plt.imshow(data[rands[i - 1], :, :])
    plt.show()


def main():
    path = r"D:/speechrecogn/voxforge/pics/"

    x_tr = da.from_array(h5py.File(os.path.join(path, "x_tr.h5"))['x_tr'], chunks=1000)
    y_tr = da.from_array(h5py.File(os.path.join(path, "y_tr.h5"))['y_tr'], chunks=1000)
    print(x_tr.shape, y_tr.shape)
    x_va = da.from_array(h5py.File(os.path.join(path, "x_va.h5"))['x_va'], chunks=1000)
    y_va = da.from_array(h5py.File(os.path.join(path, "y_va.h5"))['y_va'], chunks=1000)
    print(x_va.shape, y_va.shape)
    x_tr /= 255.
    x_va /= 255.
    plot_samples(x_va, y_va, shape=(2, 2))
    plot_samples(x_tr, y_tr, shape=(2, 2))


if __name__ == "__main__":
    main()
