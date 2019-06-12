import numpy as np
import scipy.io.wavfile as wav
import os
import pandas as pd
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import butter, lfilter

import matplotlib.pyplot as plt
import time


def envelope(y, rate, threshold, len_part):
    """moving average window"""
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate * len_part), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


def calc_fft(rate, samples):
    """calculate fft and frequencies"""
    n = len(samples)
    freq = fftfreq(n, d=1 / rate)
    y = fft(samples)
    return y, freq


def cut_freq(fft, freq, low, hi):
    """nullify fft frequencies that larger and smaller than thresholds"""
    idxs = freq >= hi
    fft[idxs] = 0
    idxs = freq <= low
    fft[idxs] = 0
    return fft


def fft_filter(signal, rate, low, hi):
    """filter signal via fft frequencies cutting"""
    signal_fft, freq = calc_fft(rate, signal)
    filtered_fft = cut_freq(signal_fft, freq, low, hi)
    signal_ifft = ifft(filtered_fft)
    return signal_ifft.real.astype(np.int16)


def butter_bandpass(lowcut, highcut, fs, order=5):
    """butterwoth bandpass"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """butterwoth bandpass filter"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y.astype(np.int16)


def plot(orig, signal, mask, s='title'):
    """plot signals"""
    plt.title(s)
    plt.plot(orig, alpha=0.75, label='orig')
    signal[np.invert(mask)] = 0
    plt.plot(signal, alpha=0.75, label='converted')
    plt.legend()
    plt.show()
    time.sleep(0.01)


def mul_sig_silence(signal, min_silence):
    """scales signals (and silence values)"""
    k = 25000 / max(abs(signal))
    if k < 1:
        return signal, min_silence
    signal = (signal * k).astype(np.int16)
    min_silence = int(min_silence * k)
    return signal, min_silence


def apply_filter(signal, rate, low, hi, f, min_silence, len_part):
    """applies filter"""
    if f == 'fft':
        signal = fft_filter(signal, rate, low, hi)
    if f == 'butter':
        signal = butter_bandpass_filter(signal, low, hi, rate, order=6)
    mask = envelope(signal, rate, min_silence, len_part)
    return signal, mask


def clear_audio(file_path, min_silence, silence_part, amp_mag, low, hi, f, len_part, min_time, plotting):
    """
    cleans audio file
    
    :param file_path path to audio file
    :param min_silence minimum silence level
    :param silence_part recalculated silence part from maximum value of amplitude (won't be less than min_silence)
    :param len_part length in seconds of moving window for signal enveloping
    :param f bandpass filter type ('butter' - butterworth, 'fft' - via fourier transform)
    :param low frequency of filter's low pass
    :param hi frequency of filter's high pass
    :param min_time minimal length of processed audio to save
    :param amp_mag amplitude magnification (multiplies audio signal amplitude to increase volume)
    :param plotting plot original and modified signals
    """
    rate, signal = wav.read(file_path)
    signal_orig = signal

    if max(abs(signal)) <= min_silence:
        return

    if max(abs(signal)) * silence_part > min_silence:
        min_silence = max(abs(signal)) * silence_part

    if amp_mag:
        signal, scaled_min_silence = mul_sig_silence(signal, min_silence)

    signal, mask = apply_filter(signal, rate, low, hi, f, scaled_min_silence, len_part)
    
    return signal_orig, signal, mask, rate, scaled_min_silence


def output(out_path, file, res, min_time, plotting):
    """saves clean audio, plots, writes console messages"""
    signal_orig, signal, mask, rate, scaled_min_silence = res                    
    ratio = len(signal[mask]) / len(signal)
                    
    if ratio <= 0.3 or len(signal[mask]) < int(rate * min_time):
        return
                    
    s = '{:.2f}'.format(ratio) + ' {:.2f}'.format(scaled_min_silence) + ' ' + file
    print(s)
    if plotting:
        plot(signal_orig, signal, mask, s=s)
    wav.write(os.path.join(out_path, file), rate, signal[mask])


def clean_audios(path, out="audios_clean", one_folder=False, min_silence=200, silence_part=0.01, len_part=0.1,
                 f='butter', low=100, hi=7000, min_time=3., amp_mag=True, plotting=False):
    """
    cleans audio files in folders
    
    :param path audio files path
    :param out output folder name
    :param one_folder output to one ore multiple folders
    :param min_silence minimum silence level
    :param silence_part recalculated silence part from maximum value of amplitude (won't be less than min_silence)
    :param len_part length in seconds of moving window for signal enveloping
    :param f bandpass filter type ('butter' - butterworth, 'fft' - via fourier transform)
    :param low frequency of filter's low pass
    :param hi frequency of filter's high pass
    :param min_time minimal length of processed audio to save
    :param amp_mag amplitude magnification (multiplies audio signal amplitude to increase volume)
    :param plotting plot original and modified signals
    """
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
                    res = clear_audio(file_path, min_silence, silence_part, amp_mag, low, hi, f, len_part, min_time, plotting)                    
                    if not res:
                        continue
                    
                    output(out_path, file, res, min_time, plotting)

                    temp = {'file': file, 'lang': folder.lower()}
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
    clean_audios(path, out="audios_clean", one_folder=False, min_silence=200, silence_part=0.01, len_part=0.25,
                 min_time=2.5, low=100, hi=7000, amp_mag=True, plotting=True)


if __name__ == "__main__":
    main()
