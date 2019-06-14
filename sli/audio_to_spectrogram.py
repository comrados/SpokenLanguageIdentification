import librosa as lr

import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
import random


def normalize(arr):
    max_value = np.max(arr)
    min_value = np.min(arr)
    for i in range(len(arr)):
        arr[i] = (arr[i] - min_value) / (max_value - min_value)
    return arr


def scale(arr, k):
    arr = arr * k
    return arr.astype(np.int32)


def get_patches(spec, n, l):
    h, w = spec.shape
    patches = []
    offsets = []
    i = 0
    while i < n:
        offset = random.randint(0, w - l - 1)
        if offset in offsets:
            continue
        else:
            i += 1
            offsets.append(offset)
            offsets.append(offset + l)
            patches.append(spec[:, offset:offset + l])
    return patches, offsets


def plot_patches(patches):
    for i in range(1, len(patches) + 1):
        plt.subplot(1, len(patches), i)
        plt.axis('off')
        plt.imshow(patches[i - 1], origin="lower")
    plt.show()


def plot_spec(spec, offsets):
    new = np.copy(spec)
    new[:, offsets] = 255
    plt.subplot(2, 1, 1)
    plt.imshow(new, origin="lower")
    plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.axis('off')
    plt.imshow(spec, origin="lower")
    plt.show()


audio_paths = [r"D:/speechrecogn/voxforge/audios_clean\ru_kn0pka-20110505-hic-ru_0034.wav",
               r"D:/speechrecogn/voxforge/audios_clean\de_Black_Galaxy-20080530-xzb-de11-101.wav"]

n_mels = 100
min_time = 2.5
time = 25
min_frames = int(min_time * 1000 / time)

nn_frame_width = 1
patch_width = int(nn_frame_width * 1000 / time)

for audio_path in audio_paths:
    plt.set_cmap('viridis')

    sr, y = wav.read(audio_path)
    hop = int(sr / 1000 * time)

    y = y.astype(np.float32)

    spec = lr.feature.melspectrogram(y, n_mels=n_mels, hop_length=hop)
    img = lr.core.amplitude_to_db(spec)

    img = np.abs(img)

    norm = normalize(img)
    scaled = scale(norm, 255)

    patches, offsets = get_patches(scaled, 3, patch_width)
    plot_patches(patches)

    plot_spec(scaled, offsets)

    plt.imsave(r"D:/test.png", scaled)