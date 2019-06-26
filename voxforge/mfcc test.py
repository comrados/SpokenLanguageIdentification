import librosa as lr

import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
import random
from PIL import Image
from PIL import ImageOps


def normalize(arr):
    max_value = np.max(arr)
    min_value = np.min(arr)
    arr = (arr - min_value) / (max_value - min_value)
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
    
def plot_patches2(patches):    
    for i in range(1, len(patches)+1):
        plt.subplot(1, len(patches), i)
        plt.axis('off')
        plt.imshow(patches[i-1])
    plt.show()
    
def plot_spec(spec, offsets):
    new = np.copy(spec)
    for i in range(0, len(offsets), 2):
        color = random.randint(200, 256)       
        new[:,offsets[i]] = color
        new[:,offsets[i+1]] = color
    plt.subplot(1, 2, 1)
    plt.imshow(new, origin="lower")
    plt.axis('off')
    plt.title('with offsets')
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(spec, origin="lower")
    plt.title('pure')
    plt.show()

audio_paths = [r"D:/speechrecogn/voxforge/audios_clean\de_Black_Galaxy-20080530-xzb-de11-101.wav"]

n_mels=200
time = 25
n = 10

nn_frame_width = 1
patch_width = int(nn_frame_width*1000/time)


for audio_path in audio_paths:
    plt.set_cmap('binary')
    
    sr, y = wav.read(audio_path)
    
    
    y, sr = lr.load(audio_path, sr=None)
    hop = int(sr/1000*time) # hop length (samples)
    spec = lr.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop)
    db = lr.power_to_db(spec, ref=np.max)
    # rescale magnitudes: 0 to 255
    scaled = scale(normalize(db), 255).astype(np.uint8)
    
    patches, offsets = get_patches(scaled, n, patch_width, mode='random')
    plot_patches(patches)   
    
    plot_spec(scaled, offsets)
    
    im = Image.fromarray(scaled, mode='L')
    
    im = ImageOps.flip(im)
    
    im.show()
    
    plt.imsave(r"D:/test.png", scaled, origin="lower")





im = Image.open(r"D:\speechrecogn\voxforge\audio_spec_full\de_de_ralfherzog-20080215-de120-de120-86.png")


import librosa
import librosa.display
import pandas as pd
import os

y, sr = lr.load(audio_path, sr=None)

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=200)
db = librosa.power_to_db(S, ref=np.max)  
db = scale(normalize(db), 255).astype(np.int16) 

print(np.min(db), np.max(db))

plt.set_cmap('binary')
plot_patches([db])

plt.figure(figsize=(10, 4))
librosa.display.specshow(db, y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()



df = pd.read_csv(r"D:\speechrecogn\voxforge\audios_clean_list.csv")
for index, row in df.iterrows():
    file_path = row['file']
    file = os.path.basename(file_path)
    filename, file_extension = os.path.splitext(file)
    lang = row['lang']
    
    y, sr = lr.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=200)
    db = librosa.power_to_db(S, ref=np.max)  
    
    db = scale(normalize(db), 255).astype(np.int16)     
    
    print(np.min(db), np.max(db))
