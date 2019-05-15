import numpy as np
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from PIL import Image
import os
import random
from IPython.display import clear_output


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
        if (totw[i] > 1e-6):
            freqs[i] /= totw[i]

    return newspec, freqs


def plotstft(audiopath, binsize=2 ** 10, name='tmp.png', alpha=1, quantity=1, img_size=None):
    """plot spectrogram"""
    samplerate, samples = wav.read(audiopath)
    # samples = samples[:, channel]
    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1, sr=samplerate, alpha=alpha)
    sshow = sshow[2:, :]
    ims = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel
    timebins, freqbins = np.shape(ims)

    offset = np.random.randint(ims.shape[0] - 151)

    # print(ims.shape)

    ims = ims[offset:offset + 150, 0:513]  # 0-11khz, ~9s interval

    ims = np.transpose(ims)

    # print(ims.shape)

    image = Image.fromarray(ims)
    if img_size is not None:
        image = image.resize(img_size)
    image = image.convert('L')
    image.save(name)


def convert_to_images(path, validation_part=0.2, img_size=None):
    """converts wavs to images"""
    audios_path = check_path([path, "audios"])
    pics_path = check_path([path, "pics"])
    pics_data_path = check_path([pics_path, "train"])
    pics_validation_path = check_path([pics_path, "validation"])
    max_count = get_min_files_count(audios_path)
    print(max_count, (1 - validation_part), max_count * (1 - validation_part))
    print('Max numbers of files wil be used:', max_count)
    for folder in os.listdir(audios_path):
        print("\nOutputting from folder:", folder.upper())
        lang_folder = check_path([audios_path, folder])
        # out_folder = check_path([pics_path, folder])
        out_data_folder = check_path([pics_data_path, folder])
        out_validation_folder = check_path([pics_validation_path, folder])
        if os.path.isdir(lang_folder):
            wav_folder = check_path([lang_folder, "wav"])
            count = 0
            wav_files = os.listdir(wav_folder)
            random.shuffle(wav_files)
            for i in range(max_count):
                file = wav_files[i]
                file_path = os.path.join(wav_folder, file)
                count += 1
                if count > max_count:
                    break
                if os.path.isfile(file_path):
                    file_name, file_extension = os.path.splitext(file)
                    clear_output(wait=True)
                    print(folder.upper() + ' (' + str(count) + '/' + str(max_count) + '):', file, end='\r')
                    quantity = 10
                    for idx in range(quantity):
                        if i < max_count * (1 - validation_part):
                            out_file = os.path.join(out_data_folder, file_name + "_" + str(idx) + ".png")
                        else:
                            out_file = os.path.join(out_validation_folder, file_name + "_" + str(idx) + ".png")
                        alpha = np.random.uniform(0.9, 1.1)
                        plotstft(file_path, name=out_file, alpha=alpha, quantity=quantity, img_size=img_size)


def check_path(elements):
    """checks and creates path"""
    res = elements[0]
    for i in range(1, len(elements)):
        res = os.path.join(res, elements[i])
    if not os.path.exists(res):
        os.makedirs(res)
    return res


def get_min_files_count(audios_path):
    """returns min wav files quantity"""
    min_count = 1000000
    for folder in os.listdir(audios_path):
        lang_folder = check_path([audios_path, folder])
        if os.path.isdir(lang_folder):
            wav_folder = check_path([lang_folder, "wav"])
            wav_files = os.listdir(wav_folder)
            print(len(wav_files), min_count)
            if len(wav_files) < min_count:
                min_count = len(wav_files)
    return min_count


def main():
    path = r"D:/speechrecogn/voxforge/"
    convert_to_images(path, validation_part=0.2, img_size=(150, 150))


if __name__ == "__main__":
    main()
