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
    arr = arr*k
    return arr.astype(np.int32)

def get_offsets(w, l, n, mode='random'):
    if mode is 'random':
        return get_random_offsets(w, l, n)
    elif mode is 'uniform':
        return get_uniform_offsets(w, l, n)
    elif mode is 'gauss':
        return get_gauss_offsets(w, l, n)
    else:
        return get_random_offsets(w, l, n)

def get_random_offsets(w, l, n):
    offsets = []
    i = 0
    while i < n:
        offset = random.randint(0, w-l-1)
        if offset in offsets:
            continue
        else:
            i += 1
            offsets.append(offset)
            offsets.append(offset+l)
    return offsets

def get_uniform_offsets(w, l, n):
    starts = np.linspace(0, w-l-1, num=n, dtype=int)
    offsets = []
    for start in starts:
        offsets.append(start)
        offsets.append(start+l)
    return offsets

def get_gauss_offsets(w, l, n):
    offsets = []
    i = 0
    while i < n:
        # distribution with mean (w-l-1)/2 and 3-sigma interval (w-l-1)/2
        offset = int(np.minimum(np.maximum(0, np.random.normal((w-l-1)/2, (w-l-1)/2/3)), w-l-1))
        if offset in offsets:
            continue
        else:
            i += 1
            offsets.append(offset)
            offsets.append(offset+l)
    return offsets

def get_patches(spec, n, l, mode='random'):
    h, w = spec.shape
    patches = []
    offsets = get_offsets(w, l, n, mode=mode)
    for i in range(0, len(offsets), 2):
        patches.append(spec[:, offsets[i]:offsets[i+1]])
    return patches, offsets

def plot_patches(patches):    
    for i in range(1, len(patches)+1):
        plt.subplot(1, len(patches), i)
        plt.axis('off')
        plt.imshow(patches[i-1], origin="lower")
    plt.show()
    
def plot_spec(spec, offsets):
    new = np.copy(spec)
    for i in range(0, len(offsets), 2):
        color = random.randint(200, 256)       
        new[:,offsets[i]] = color
        new[:,offsets[i+1]] = color
    plt.subplot(2, 1, 1)
    plt.imshow(new, origin="lower")
    plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.axis('off')
    plt.imshow(spec, origin="lower")
    plt.show()

audio_paths = [r"D:/speechrecogn/voxforge/audios_clean\de_Black_Galaxy-20080530-xzb-de11-101.wav",
               r"D:/speechrecogn/voxforge/audios_clean\ru_kn0pka-20110505-hic-ru_0034.wav"]

n_mels=100
time = 25
n = 10

nn_frame_width = 1
patch_width = int(nn_frame_width*1000/time)


for audio_path in audio_paths:
    plt.set_cmap('viridis')
    
    sr, y = wav.read(audio_path)
    y, sr = lr.load(audio_path, sr=None)
    hop = int(sr/1000*time)
    
    y = y.astype(np.float32)
    
    spec = lr.feature.melspectrogram(y, n_mels=n_mels, hop_length=hop)
    img = lr.core.amplitude_to_db(spec)
    
    img = np.abs(img)
    
    norm = normalize(img)
    scaled = scale(norm, 255)
    
    patches, offsets = get_patches(scaled, n, patch_width, mode='random')
    plot_patches(patches)   
    
    plot_spec(scaled, offsets)
    
    plt.imsave(r"D:/test.png", scaled, origin="lower")

val = np.minimum(np.maximum(0, np.random.normal((w-l-1)/2, (w-l-1)/2/2)), w-l-1)


np.random.normal(0, 1)