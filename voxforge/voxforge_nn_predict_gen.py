import tensorflow as tf
from numpy.lib import stride_tricks
import numpy as np
from PIL import Image
import scipy.io.wavfile as wav
import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


def stft(sig, frame_size, overlap_fac=0.75, window=np.hanning):
    """short time fourier transform of audio signal """
    win = window(frame_size)
    hop_size = int(frame_size - np.floor(overlap_fac * frame_size))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frame_size / 2.0))), sig)
    # cols for windowing
    cols = int(np.ceil((len(samples) - frame_size) / float(hop_size)) + 1)
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frame_size))

    frames = stride_tricks.as_strided(samples, shape=(cols, frame_size),
                                      strides=(samples.strides[0] * hop_size, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)


def logscale_spec(spec, sr=16000, factor=20., alpha=1.0, f0=0.9, fmax=1):
    """scale frequency axis logarithmically """
    # spec = spec[:, 0:513]
    timebins, freqbins = np.shape(spec)
    scale = np.linspace(0, 1, freqbins)  # ** factor

    # http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=650310&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel4%2F89%2F14168%2F00650310
    scale = np.array(
        [x * alpha if x <= f0 else (fmax - alpha * f0) / (fmax - f0) * (x - f0) + alpha * f0 for x in scale])
    scale *= (freqbins - 1) / max(scale)

    newspec = np.complex128(np.zeros([timebins, freqbins]))
    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
    freqs = [0.0 for _ in range(freqbins)]
    totw = [0.0 for _ in range(freqbins)]
    for i in range(0, freqbins):
        if i < 1 or i + 1 >= freqbins:
            newspec[:, i] += spec[:, i]
            freqs[i] += allfreqs[i]
            totw[i] += 1.0
            continue
        else:
            # scale[15] = 17.2
            w_up = scale[i] - np.floor(scale[i])
            w_down = 1 - w_up
            j = int(np.floor(scale[i]))

            newspec[:, j] += w_down * spec[:, i]
            freqs[j] += w_down * allfreqs[i]
            totw[j] += w_down

            newspec[:, j + 1] += w_up * spec[:, i]
            freqs[j + 1] += w_up * allfreqs[i]
            totw[j + 1] += w_up

    for i in range(len(freqs)):
        if totw[i] > 1e-6:
            freqs[i] /= totw[i]

    return newspec, freqs


def plotstft(audiopath, binsize=2 ** 10, name='tmp.png', alpha=1, img_size=None):
    """plot spectrogram"""
    samplerate, samples = wav.read(audiopath)
    # samples = samples[:, channel]
    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1, sr=samplerate, alpha=alpha)
    sshow = sshow[2:, :]
    ims = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel
    timebins, freqbins = np.shape(ims)

    offset = np.random.randint(ims.shape[0] - 151)

    ims = ims[offset:offset + 150, 0:513]  # 0-11khz, ~9s interval

    ims = np.transpose(ims)

    image = Image.fromarray(ims)
    if img_size is not None:
        image = image.resize(img_size)
    image = image.convert('L')
    image.save(name)


def get_wavs_from_dirs(dirs):
    "get all wavs from given directories"
    wavs = []
    for d in dirs:
        files = os.listdir(d)
        for file in files:
            if file.endswith('.wav'):
                wavs.append(os.path.join(d, file))
    return wavs


def generate_sprectra(audios, temp_path, count):
    "generate random spectra of wav to feed into NN"
    dic = {}
    if not os.path.exists(os.path.join(temp_path, "out")):
        os.makedirs(os.path.join(temp_path, "out"))
    for idx, audio in enumerate(audios):
        dic[idx] = audio
        for i in range(count):
            alpha = np.random.uniform(0.9, 1.1)
            plotstft(audio, alpha=alpha, name=os.path.join(temp_path, "out", str(idx) + '_' +str(i) +'.png'))
    return dic


def n_largest_setarr(a, n=1):
    "n-largest element of each row is set to 1, other elements - 0"
    # a : Input array
    # n : We want n-max element position to be set to 1
    out = np.zeros_like(a)
    out[np.arange(len(a)), np.argpartition(a,-n, axis=1)[:,-n]] = 1
    return out


def predict(model, path_predict, count, remove_temp=True):
    "predict label for picture"
    predict_datagen = ImageDataGenerator(rescale=1./255)
    predict_generator = predict_datagen.flow_from_directory(
            path_predict,
            target_size=(150, 513),
            color_mode='grayscale',
            shuffle=False,
            class_mode='categorical',
            batch_size=count)
    res = model.predict_generator(predict_generator, verbose=1)
    if remove_temp:
        shutil.rmtree(path_predict)
    fin = []    
    for i in range (0, len(res), count):
        t = res[i:i+count]
        fin.append(np.sum(n_largest_setarr(t), axis=0)/count)        
    fin = np.array(fin)
    return fin


def print_res(predicts, d):
    "print results"
    for key, value in d.items():
        print(predicts[key], ":", value)

def main():
    ru1 = r"D:\audios\ru\16000"
    de1 = r"D:\audios\de\16000"
    dirs = [ru1, de1]
    temp = r"D:\speechrecogn\temp"
    
    count = 10
    wavs = get_wavs_from_dirs(dirs)
    
    dic = generate_sprectra(wavs, temp, count)    
    model = tf.keras.models.load_model(r"D:\speechrecogn\model8.57-0.272-0.901.hdf5")    
    res = predict(model, temp, count) 
    
    print_res(res, dic)
    
if __name__ == "__main__":
    main()

