import numpy as np
import scipy.io.wavfile as wav
import os
import pandas as pd
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import butter, filtfilt
from . import utils

import matplotlib.pyplot as plt
import time


def _calc_fft(rate, samples):
    """calculate fft and frequencies"""
    n = len(samples)
    freq = fftfreq(n, d=1 / rate)
    y = fft(samples)
    return y, freq


def _cut_freq(fft, freq, low, hi):
    """nullify fft frequencies that larger and smaller than thresholds"""
    idxs = freq >= hi
    fft[idxs] = 0
    idxs = freq <= low
    fft[idxs] = 0
    return fft


def _fft_filter(signal, rate, low, hi):
    """filter signal via fft frequencies cutting"""
    signal_fft, freq = _calc_fft(rate, signal)
    filtered_fft = _cut_freq(signal_fft, freq, low, hi)
    signal_ifft = ifft(filtered_fft)
    return signal_ifft.real.astype(np.int16)


def _butter_bandpass(lowcut, highcut, fs, order=5):
    """butterwoth bandpass"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def _butter_bandpass_filter(signal, rate, low, hi, order=6):
    """butterwoth bandpass filter"""
    b, a = _butter_bandpass(low, hi, rate, order=order)
    y = filtfilt(b, a, signal)
    return y.astype(np.int16)


def _mul_sig_silence(signal, min_silence):
    """scales signals (and silence values)"""
    k = 25000 / max(abs(signal))
    if k < 1:
        return signal, min_silence
    signal = (signal * k).astype(np.int16)
    min_silence = int(min_silence * k)
    return signal, min_silence


class AudioCleaner:

    def __init__(self, path, audios_dirty, audios_clean="audios_clean", one_folder=False, min_silence=200,
                 silence_part=0.01, len_part=0.25, min_time=2.5, f='butter', low=100, hi=7000, amp_mag=True,
                 plotting=False):
        """
        :param path: working path
        :param audios_dirty: path to file-lang correspondence csv-file
        :param audios_clean: output folder name (will be created in path)
        :param one_folder: output to one or multiple (the resulting number of folders equals to number of languages)
        :param min_silence: minimum silence level
        :param silence_part: recalculated silence part from maximum value of amplitude (won't be less than min_silence)
        :param len_part: length in seconds of moving window for signal enveloping
        :param f: bandpass filter type ('butter' - butterworth, 'fft' - via fourier transform)
        :param low: frequency of filter's lowcut
        :param hi: frequency of filter's highcut
        :param min_time: minimal length of processed audio to save
        :param amp_mag: amplitude magnification (multiplies audio signal amplitude to increase volume)
        :param plotting: plot original and modified signals
        """
        self.path = path
        self.audios_dirty = audios_dirty
        self.audios_clean = audios_clean
        self.one_folder = one_folder
        self.min_silence = min_silence
        self.silence_part = silence_part
        self.len_part = len_part
        self.min_time = min_time
        self.f = f
        self.low = low
        self.hi = hi
        self.amp_mag = amp_mag
        self.plotting = plotting
        self.file_lang_list = []

    def _envelope(self, signal, rate, threshold):
        """moving average window"""
        mask = []
        signal = pd.Series(signal).apply(np.abs)
        signal_means = signal.rolling(window=int(rate * self.len_part), min_periods=1, center=True).mean()
        for mean in signal_means:
            if mean > threshold:
                mask.append(True)
            else:
                mask.append(False)
        return mask

    def _plot(self, orig, signal, mask, s='title'):
        """plot signals"""
        plt.title(s)
        plt.plot(self, orig, alpha=0.75, label='orig')
        signal[np.invert(mask)] = 0
        plt.plot(signal, alpha=0.75, label='converted')
        plt.legend()
        plt.show()
        time.sleep(0.1)

    def _apply_filter(self, signal, rate, min_silence):
        """applies filter"""
        if self.f == 'fft':
            signal = _fft_filter(signal, rate, self.low, self.hi)
        if self.f == 'butter':
            signal = _butter_bandpass_filter(signal, rate, self.low, self.hi)
        mask = self._envelope(signal, rate, min_silence)
        return signal, mask

    def _clean_audio(self, file_path):
        """
        cleans audio file
        """
        rate, signal = wav.read(file_path)
        signal_orig = signal

        if max(abs(signal)) <= self.min_silence:
            print('TOO QUIET:', os.path.basename(file_path))
            return

        scaled_min_silence = self.min_silence
        if max(abs(signal)) * self.silence_part > scaled_min_silence:
            scaled_min_silence = max(abs(signal)) * self.silence_part

        if self.amp_mag:
            signal, scaled_min_silence = _mul_sig_silence(signal, scaled_min_silence)

        signal, mask = self._apply_filter(signal, rate, scaled_min_silence)

        return signal_orig, signal, mask, rate, scaled_min_silence

    def _output(self, out_path, file, res, min_time, lang, plotting):
        """saves clean audio, plots, writes console messages"""
        signal_orig, signal, mask, rate, scaled_min_silence = res
        ratio = len(signal[mask]) / len(signal)

        if len(signal[mask]) < int(rate * min_time):
            print('TOO SHORT:', '{:.2f}'.format(len(signal[mask]) / rate), os.path.basename(file))
            return

        s = '{:.2f}'.format(ratio) + ' {:.2f}'.format(scaled_min_silence) + ' ' + file
        print(s)
        if plotting:
            self._plot(signal_orig, signal, mask, s=s)
        wav.write(os.path.join(out_path, file), rate, signal[mask])
        self.file_lang_list.append({'file': os.path.join(out_path, file), 'lang': lang})

    def clean(self):
        """
        cleans audio files in folders
        """
        df = pd.read_csv(self.audios_dirty)
        for index, row in df.iterrows():
            file_path = row['file']
            file = os.path.basename(file_path)
            lang = row['lang']
            out_path = utils.check_path(self.path, self.audios_clean)
            if self.one_folder:
                file = lang + '_' + file
            else:
                out_path = utils.check_path(self.path, self.audios_clean, lang)
            if os.path.isfile(file_path):
                res = self._clean_audio(file_path)
                if not res:
                    continue
                self._output(out_path, file, res, self.min_time, lang, self.plotting)

        return utils.files_langs_to_csv(self.file_lang_list, self.path, "audios_clean_list.csv")
