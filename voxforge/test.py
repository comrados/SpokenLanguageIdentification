import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import os
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import hilbert


def calc_fft(rate, samples):
    n = len(samples)
    freq = fftfreq(n, d=1 / rate)
    y = fft(samples)
    return y, freq


def cut_freq(fft, freq, bot, top):
    for idx, f in enumerate(freq):
        if abs(f) >= top:
            fft[idx] = 0
        if abs(f) <= bot:
            fft[idx] = 0
    return fft


def filter_via_fft(signal, rate, bot=100, top=7000):
    signal_fft, freq = calc_fft(rate, signal)
    filtered_fft = cut_freq(signal_fft, freq, bot, top)
    signal_ifft = ifft(filtered_fft)
    return signal_ifft.real


def envelope(y, rate, threshold, len_part):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate * len_part), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean


def plot_fft(fft, freq):
    n = len(fft) // 2
    plt.plot(freq[0:n], np.abs(fft)[0:n])
    plt.title("FFT length: " + str(n))
    plt.show()


def plot_signal(signal, freq):
    plt.plot(np.linspace(0.0, len(signal) / freq, len(signal)), signal)
    plt.title("Signal length: " + str(len(signal)))
    plt.show()


def plot_signal_with_envelope(signal, envelope, threshold=300):
    plt.plot(signal, label='signal')
    plt.plot(envelope, label='envelope')
    plt.plot([0, len(signal) - 1], [threshold, threshold], label='threshold')
    plt.legend()
    plt.title("Signal length: " + str(len(signal)))
    plt.show()


orig_path = r"D:/speechrecogn/voxforge/audios"
clean_path = r"D:/speechrecogn/voxforge/audios_clean"
lang = r"ru"

file = r"kn0pka-20110505-hic-ru_0030.wav"
file = r"Leonid-20130928-kfg-ru_0014.wav"

orig = os.path.join(orig_path, lang, 'wav', file)
clean = os.path.join(clean_path, lang, file)

rate_o, samples_o = wav.read(orig)
rate_c, samples_c = wav.read(clean)

###############################################################################

# plot_signal(samples_o, 16000)

fft_o, freq_o = calc_fft(16000, samples_o)
# plot_fft(fft_o, freq_o)

###############################################################################

# plot_signal(samples_c, 16000)

fft_c, freq_c = calc_fft(16000, samples_c)
# plot_fft(fft_c, freq_c)

###############################################################################

top, bot = 7000, 100

out = r"D:/out_" + str(bot) + "_" + str(top) + ".wav"

y = cut_freq(fft_c, freq_c, bot, top)

plot_fft(y, freq_c)

ifft_c = ifft(y)

filtered = filter_via_fft(samples_c, rate_c)

mask, ansig = envelope(filtered, 16000, 300, 0.1)
out_sig = filtered[mask].astype(np.int16)

plot_signal_with_envelope(filtered, ansig)

plot_signal(filtered, 16000)
plot_signal(out_sig, 16000)
print(file, '{:.2f}'.format(len(out_sig) / len(ifft_c)))

# wav.write(out, 16000, out_sig)

###############################################################################

analytic_signal = hilbert(samples_o, N=int(len(samples_o)))
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * 16000)

mask, ansig = envelope(samples_o, 16000, 300, 0.1)

plot_signal_with_envelope(samples_o, amplitude_envelope)
plot_signal_with_envelope(samples_o, ansig)

plt.plot(samples_o[60000:60100])
plt.plot(amplitude_envelope[60000:60100])
plt.show()

###############################################################################

df = pd.read_csv(r"D:\speechrecogn\voxforge\audios_list.csv")

for index, row in df.iterrows():
    file = row['file']
    basename = os.path.basename(file)
    lang = row['lang']

import random

# random.seed(0)

for i in range(10):
    print(random.randint(0, 101))

###############################################################################

import librosa
import numpy as np
import matplotlib.pyplot as plt
import os


def plot(*series, alpha=0.5):
    for s in series:
        plt.plot(s, alpha=alpha)
    plt.grid()
    plt.show()


def plot_drc(threshold=-35, scale=2, direction='upward'):
    dbs = np.linspace(-0.1, -100, num=1000)
    new = np.linspace(-0.1, -100, num=1000)
    if direction is 'downward':
        mask = np.where(new > threshold, True, False)
    else:
        mask = np.where(new < threshold, True, False)
    new[mask] = new[mask] * scale - threshold * (scale - 1)
    plt.plot(dbs, new)
    plt.grid()
    plt.show()


plot_drc(threshold=-35, scale=1.5, direction='downward')
plot_drc(threshold=-35, scale=5, direction='upward')


def dynamic_range_compression(dbs, threshold, scale=2, direction='upward'):
    new = np.copy(dbs)
    if direction is 'downward':
        mask = np.where(new > threshold, True, False)
    else:
        mask = np.where(new < threshold, True, False)
    new[mask] = new[mask] * scale - threshold * (scale - 1)
    return new


clean_path = r"D:/speechrecogn/voxforge/audios_clean"
orig_path = r"D:/speechrecogn/voxforge/audios"
lang = 'ru'

file = r"kn0pka-20110505-hic-ru_0030.wav"
# file = r"Leonid-20130928-kfg-ru_0014.wav"

orig = os.path.join(orig_path, lang, 'wav', file)
clean = os.path.join(clean_path, lang, file)

samples, rate = librosa.core.load(clean, sr=None)

a_s = np.abs(samples)

mask = np.where(samples < 0, True, False)

dbs = librosa.core.amplitude_to_db(samples)

dbs_new = dynamic_range_compression(dbs, -35, 5, direction='upward')
# dbs_new = dynamic_range_compression(dbs_new, -35, 1.1, direction='downward')

plot(samples * 10, dbs, dbs_new)
plot(dbs - dbs_new)

amps = librosa.core.db_to_amplitude(dbs)
amps_new = librosa.core.db_to_amplitude(dbs_new)

amps[mask] = amps[mask] * (-1)
amps_new[mask] = amps_new[mask] * (-1)

plot(amps, amps_new)

plot(amps - amps_new)

librosa.output.write_wav(r"D:/test_rosa.wav", amps, rate)
librosa.output.write_wav(r"D:/test_rosa_new.wav", amps_new, rate)

np.mean(dbs)

###############################################################################

import numpy as np
import pandas as pd
import os
import dask.array.image as dai
import dask.array as da
import h5py
import dask.base as db

arr = dai.imread(r"D:\speechrecogn\voxforge\audios_spec\*.png", preprocess=np.transpose)
arr = dai.imread(r"D:\speechrecogn\voxforge\audios_spec\*.png")

from glob import glob
import os
from skimage.io import imread as sk_imread
from dask.array import Array
from dask.base import tokenize
import pandas as pd
import numpy as np


def imread(filenames, imread=None, preprocess=None):
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

    return Array(dsk, name, chunks, sample.dtype)


df = pd.read_csv(r"D:\speechrecogn\voxforge\audios_spec.csv")
for idx, row in df.iterrows():
    file_path = row['file']
    file = os.path.basename(file_path)
    filename, file_extension = os.path.splitext(file)
    lang = row['lang']

fl = df['file'].values.tolist()

arr2 = imread(fl, preprocess=np.transpose)

print(imread(fl, preprocess=np.transpose))
print(imread(fl))


def a(*a):
    print(a, len(a))


a(1, 1, 2)

###############################################################################

import pandas as pd
import numpy as np

df = pd.read_csv(r"D:\speechrecogn\voxforge\audios_clean_list.csv")

df = df.sample(frac=1, random_state=0).reset_index(drop=True)

dic = {u: 0 for u in df['lang'].unique()}


def get_min_unique_counts(df):
    """returns min wav files quantity"""
    return df.groupby('lang')['file'].nunique().min()


uniques = get_min_unique_counts(df)
