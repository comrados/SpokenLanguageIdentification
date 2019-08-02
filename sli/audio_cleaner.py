import numpy as np
import librosa
import os
import pandas as pd
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import butter, filtfilt
from . import utils
import matplotlib.pyplot as plt
import time


class AudioCleaner:

    def __init__(self, path: str, audios_dirty: str, audios_clean: str = "audios_clean", one_folder: bool = False,
                 min_silence: float = 0.01, len_part: float = 0.25, min_time: float = 2.5, plotting: bool = False,
                 f: dict = {'type': 'butter', 'low': 100, 'high': 7000}, verbose: bool = False):
        """
        Initialize audio cleaner

        :param path: working path
        :param audios_dirty: path to file-lang correspondence csv-file
        :param audios_clean: output folder name (will be created in path)
        :param one_folder: output to one or multiple (the resulting number of folders equals to number of languages)
        :param min_silence: minimum silence level (0.0 to 1.0)
        :param len_part: length in seconds of moving window for signal enveloping
        :param f: bandpass filter, dict attr: 'type' - ('butter' or 'fft'), 'low' - lowcut, 'high' highcut
        :param min_time: minimal length of processed audio to save
        :param plotting: plot original and modified signals
        :param verbose: verbose output
        """
        self.path = path
        self.audios_dirty = audios_dirty
        self.audios_clean = audios_clean
        self.one_folder = one_folder
        self.min_silence = min_silence
        self.len_part = len_part
        self.min_time = min_time
        self.f = f
        self.plotting = plotting
        self.file_lang_list = []
        self.verbose = verbose

    def _envelope(self, signal, rate, threshold):
        """moving average window"""
        signal = pd.Series(signal).apply(np.abs)
        signal_means = signal.rolling(window=int(rate * self.len_part), min_periods=1, center=True).mean()
        return np.where(signal_means > threshold, True, False)

    def _apply_filter(self, signal, rate, min_silence):
        """applies filter"""
        if self.f['type'] == 'fft':
            signal = self._fft_filter(signal, rate, self.f['low'], self.f['high'])
        elif self.f['type'] == 'butter':
            signal = self._butter_bandpass_filter(signal, rate, self.f['low'], self.f['high'])
        else:
            pass
        mask = self._envelope(signal, rate, min_silence)
        return signal, mask

    def _clean_audio(self, file_path):
        """
        cleans audio file
        """
        try:
            signal, rate = librosa.core.load(file_path, sr=None)
            signal_orig = np.copy(signal)

            if np.max(np.abs(signal)) <= self.min_silence:
                if self.verbose:
                    print('TOO QUIET:', os.path.basename(file_path), end="\r")
                return

            # filter
            signal, mask = self._apply_filter(signal, rate, self.min_silence)

            # amplitude magnification
            signal, k = utils.scale_signal_ampl(signal)
            scaled_min_silence = self.min_silence * k

            return signal_orig, signal, mask, rate, scaled_min_silence
        except:
            print("FAILED TO CLEAN", file_path)
            return

    def _output(self, out_path, file, res, lang):
        """saves clean audio, plots, writes console messages"""
        signal_orig, signal, mask, rate, scaled_min_silence = res
        ratio = len(signal[mask]) / len(signal)

        if len(signal[mask]) < int(rate * self.min_time):
            if self.verbose:
                print('TOO SHORT:', '{:.2f}'.format(len(signal[mask]) / rate), os.path.basename(file), end="\r")
            return

        s = '{:.2f}'.format(ratio) + ' {:.4f}'.format(scaled_min_silence) + ' ' + file
        if self.verbose:
            print(s, end="\r")
        if self.plotting:
            self._plot(signal_orig, signal, mask, rate, s=s)
        utils.save_audio(os.path.join(out_path, file), rate, signal[mask])
        self.file_lang_list.append({'file': os.path.join(out_path, file), 'lang': lang})

    def clean(self):
        """
        cleans audio files in folders
        """
        df = pd.read_csv(self.audios_dirty)
        print("CLEANING", df.shape[0], "FILES")
        for idx, row in df.iterrows():
            file_path = row['file']
            file = os.path.basename(file_path)
            lang = row['lang']
            out_path = utils.check_path(self.path, self.audios_clean)

            # saving path
            if self.one_folder:
                file = lang + '_' + file
            else:
                out_path = utils.check_path(self.path, self.audios_clean, lang)

            # clean file
            if os.path.isfile(file_path):
                res = self._clean_audio(file_path)
                if not res:
                    continue
                self._output(out_path, file, res, lang)

            # output
            if idx + 1 == df.shape[0] or (idx + 1) % 100 == 0:
                print("FILES CLEANED:", idx + 1)
        print()
        return utils.files_langs_to_csv(self.file_lang_list, self.path, "audios_clean_list.csv")

    @staticmethod
    def _calc_fft(rate, samples):
        """calculate fft and frequencies"""
        n = len(samples)
        freq = fftfreq(n, d=1 / rate)
        y = fft(samples)
        return y, freq

    @staticmethod
    def _cut_freq(fft, freq, low, hi):
        """nullify fft frequencies that larger and smaller than thresholds"""
        ids = freq >= hi
        fft[ids] = 0
        ids = freq <= low
        fft[ids] = 0
        return fft

    def _fft_filter(self, signal, rate, low, hi):
        """filter signal via fft frequencies cutting"""
        signal_fft, freq = self._calc_fft(rate, signal)
        filtered_fft = self._cut_freq(signal_fft, freq, low, hi)
        signal_ifft = ifft(filtered_fft)
        return signal_ifft.real

    @staticmethod
    def _butter_bandpass(lowcut, highcut, fs, order=5):
        """butterwoth bandpass"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def _butter_bandpass_filter(self, signal, rate, low, hi, order=6):
        """butterwoth bandpass filter"""
        b, a = self._butter_bandpass(low, hi, rate, order=order)
        y = filtfilt(b, a, signal)
        return y

    @staticmethod
    def _plot(orig, signal, mask, rate, s='title'):
        """plot signals"""
        plt.title(s)
        plt.plot(np.arange(len(signal))[mask] / rate, signal[mask], 'lime', alpha=0.5, label='converted')
        plt.plot(np.arange(len(orig)) / rate, orig, 'blue', alpha=0.5, label='orig')
        plt.grid()
        plt.legend()
        plt.show()
        time.sleep(0.1)
