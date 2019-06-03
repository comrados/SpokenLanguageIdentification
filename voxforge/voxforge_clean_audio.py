import numpy as np
import scipy.io.wavfile as wav
import os
import pandas as pd
from scipy.fftpack import fft, ifft, fftfreq


def envelope(y, rate, threshold, len_part):
    "moving average window"
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate*len_part), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


def calc_fft(rate, samples):
    "calculate fft and frequencies"
    n = len(samples)
    freq = fftfreq(n, d=1/rate)
    y = fft(samples)
    return y, freq


def cut_freq(fft, freq, low, hi):
    "nullify fft frequencies that larger and smaller than thresholds"
    idxs = freq >= hi
    fft[idxs] = 0
    idxs = freq <= low
    fft[idxs] = 0
    return fft


def filter_via_fft(signal, rate, low, hi):
    "filter signal via fft frequencies cutting"
    signal_fft, freq = calc_fft(rate, signal)
    filtered_fft = cut_freq(signal_fft, freq, low, hi)
    signal_ifft = ifft(filtered_fft)
    return signal_ifft.real.astype(np.int16)


def clean_audio(path, out="audios_clean", one_folder=False, silence=200, len_part=0.25, low=100, hi=7000):
    """converts wavs to images"""
    files_list = []
    audios_path = check_path(path, "audios")
    out_path = check_path(path, out)
    for folder in os.listdir(audios_path):
        print("\nOutputting from folder:", folder.upper())
        if not one_folder:
            out_path = check_path(path, out, folder)
        lang_folder = check_path(audios_path, folder)
        if os.path.isdir(lang_folder):
            wav_folder = check_path(lang_folder, "wav")
            wav_files = os.listdir(wav_folder)
            for file in wav_files:                
                file_path = os.path.join(wav_folder, file)
                if one_folder:
                    file = folder + '_' + file
                if os.path.isfile(file_path):
                    rate, signal = wav.read(file_path)
                    if max(signal) <= silence:
                        continue
                    mask = envelope(signal, rate, silence, len_part)
                    if len(signal[mask]) <= rate/10:
                        continue
                    ratio = len(signal[mask])/len(signal)
                    flag = '1'
                    if ratio >= 0.75:
                        mask = envelope(signal, rate, silence+100, len_part)
                        ratio = len(signal[mask])/len(signal)
                        flag = '2'
                        if ratio >= 0.65:
                            signal = filter_via_fft(signal, rate, low, hi)
                            mask = envelope(signal, rate, silence+100, len_part)
                            ratio = len(signal[mask])/len(signal)
                            flag = '3'
                    print(flag, '{:.2f}'.format(ratio), file)
                    wav.write(os.path.join(out_path, file), rate, signal[mask])                      
                    temp = {'file': file, 'lang': folder.upper()}
                    files_list.append(temp)
    df = pd.DataFrame.from_dict(files_list)
    df.to_csv(os.path.join(path, "clean_files_list.csv"), index=False)


def check_path(*elements):
    """checks and creates path"""
    res = elements[0]
    for i in range(1, len(elements)):
        res = os.path.join(res, elements[i])
    if not os.path.exists(res):
        os.makedirs(res)
    return res


def main():
    path = r"D:/speechrecogn/voxforge/"
    clean_audio(path, out="audios_clean", one_folder=False, silence=200, len_part=0.25, low=100, hi=7000)


if __name__ == "__main__":
    main()
