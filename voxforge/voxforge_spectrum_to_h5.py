import numpy as np
import pandas as pd
import os
import dask.array.image as dai
import dask.array as da
import h5py


def del_previous_data(path):
    try:
        os.remove(os.path.join(path, "data.h5"))
        os.remove(os.path.join(path, "x_tr.h5"))
        os.remove(os.path.join(path, "y_tr.h5"))
        os.remove(os.path.join(path, "x_va.h5"))
        os.remove(os.path.join(path, "y_va.h5"))
    except Exception as err:
        print(err)

def jpgs_to_h5(source, target, name):
    dai.imread(source + '*.png').to_hdf5(target, name)
    return len(os.listdir(source))


def get_h5_dataset(path, val_part=0.25):    
    path_files_list = os.path.join(path, "files_list.csv")
    path_pics = os.path.join(path, "out/")
    path_out_data = os.path.join(path, "data.h5")
    data_name = 'data'
    count = jpgs_to_h5(path_pics, path_out_data, data_name)
    
    y = pd.read_csv(path_files_list)['lang']
    y = pd.get_dummies(y)
    y = y.reindex_axis(sorted(y.columns), axis=1)
    y = y.values
    y = da.from_array(y, chunks=1000)
    
    x = h5py.File(path_out_data)[data_name]
    x = da.from_array(x, chunks=1000)    
    
    va_size = int(val_part*count)
    tr_size = count - va_size
    
    shfl = np.random.permutation(count)
    tr_idx = shfl[:tr_size]
    va_idx = shfl[tr_size:tr_size+va_size]
    x[tr_idx].to_hdf5(os.path.join(path, "x_tr.h5"), 'x_tr')
    y[tr_idx].to_hdf5(os.path.join(path, "y_tr.h5"), 'y_tr')
    x[va_idx].to_hdf5(os.path.join(path, "x_va.h5"), 'x_va')
    y[va_idx].to_hdf5(os.path.join(path, "y_va.h5"), 'y_va')


import random
import matplotlib.pyplot as plt

def plot_samples(data, shape=(2, 2)):
    h = shape[0]
    w = shape[1]
    n = h*w
    rands = [random.randint(0, len(data)) for _ in range(n)]
    counter = 0
    for i in range(1, n+1):
            plt.subplot(h, w, i)
            plt.imshow(data[rands[counter], :, :])
            counter += 1            
    plt.show()


def main():
    path = r"D:/speechrecogn/voxforge/pics/"
    
    del_previous_data(path)
    
    get_h5_dataset(path, val_part=0.25)
    
    
    x_tr = da.from_array(h5py.File(os.path.join(path, "x_tr.h5"))['x_tr'], chunks=1000)
    y_tr = da.from_array(h5py.File(os.path.join(path, "y_tr.h5"))['y_tr'], chunks=1000)
    print(x_tr.shape, y_tr.shape)
    x_va = da.from_array(h5py.File(os.path.join(path, "x_va.h5"))['x_va'], chunks=1000)
    y_va = da.from_array(h5py.File(os.path.join(path, "y_va.h5"))['y_va'], chunks=1000)
    print(x_va.shape, y_va.shape)
    x_tr /= 255.
    x_va /= 255.
    plot_samples(x_va, shape=(2, 2))
    plot_samples(x_tr, shape=(2, 2))

if __name__ == "__main__":
    main()
    