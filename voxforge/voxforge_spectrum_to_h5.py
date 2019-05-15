import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import os
import librosa as lr
import shutil
import dask.array.image as dai
import dask.dataframe as dd
import dask as da
import h5py
import glob


in_dim = (192,192,1)
out_dim = 176
batch_size = 32
mp3_path = 'data/mp3/'
tr_path = 'data/train/'
va_path = 'data/valid/'
te_path = 'data/test/'
data_size = 66176
tr_size = 52800
va_size = 4576
te_size = 8800


def mp3_to_img(path, height=192, width=192):
    signal, sr = lr.load(path, res_type='kaiser_fast')
    hl = signal.shape[0]//(width*1.1) #this will cut away 5% from start and end
    spec = lr.feature.melspectrogram(signal, n_mels=height, hop_length=int(hl))
    img = lr.logamplitude(spec)**2
    start = (img.shape[1] - width) // 2
    return img[:, start:start+width]


def process_audio(in_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    files = glob.glob(in_folder + '*.mp3')
    start = len(in_folder)
    for file in files:
        img = mp3_to_img(file)
        sp.misc.imsave(out_folder + file[start:] + '.jpg', img)


def process_audio_with_classes(in_folder, out_folder, labels):
    os.makedirs(out_folder, exist_ok=True)
    for i in range(len(labels['Sample Filename'])):
        file = labels['Sample Filename'][i]
        lang = labels['Language'][i]
        os.makedirs(out_folder + lang, exist_ok=True)
        img = mp3_to_img(in_folder + file)
        sp.misc.imsave(out_folder + lang + '/' + file + '.jpg', img)


def jpgs_to_h5(source, target, name):
    langs = []
    paths = [os.path.join(source, "validation"), os.path.join(source, "train")]    
    for path in paths:
        for lang in os.listdir(path):
            lang_path = os.path.join(path, lang)
            d = dai.imread(lang_path + r"/*.png")
            print(dai.imread(lang_path + r"/*.png"))
            for file in os.listdir(lang_path):
                file_path = os.path.join(lang_path, file)
                
                langs.append(lang)
    return langs, d
    
path = r"D:/speechrecogn/voxforge/pics"
val_path = os.path.join(path, "validation")
train_path = os.path.join(path, "train")

langs, d = jpgs_to_h5(r"D:\speechrecogn\voxforge\pics", r"D:\speechrecogn\voxforge\pics\data.h5", 'data')
