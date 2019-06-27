import numpy as np
import librosa
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
    ids = freq >= hi
    fft[ids] = 0
    ids = freq <= low
    fft[ids] = 0
    return fft


def _fft_filter(signal, rate, low, hi):
    """filter signal via fft frequencies cutting"""
    signal_fft, freq = _calc_fft(rate, signal)
    filtered_fft = _cut_freq(signal_fft, freq, low, hi)
    signal_ifft = ifft(filtered_fft)
    return signal_ifft.real


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
    return y


def _mul_sig_silence(signal, min_silence):
    """scales signals (and silence values)"""
    k = 0.9 / np.max(np.abs(signal))
    signal = (signal * k)
    min_silence = min_silence * k
    return signal, min_silence


def _plot(orig, signal, mask, rate, s='title'):
    """plot signals"""
    plt.title(s)
    plt.plot(np.arange(len(signal))[mask] / rate, signal[mask], 'lime', alpha=0.5, label='converted')
    plt.plot(np.arange(len(orig)) / rate, orig, 'blue', alpha=0.5, label='orig')
    plt.grid()
    plt.legend()
    plt.show()
    time.sleep(0.1)


def _convert_to_16bit_pcm(signal):
    """convert floating point wav to 16-bit pcm"""
    ids = signal >= 0
    signal[ids] = signal[ids] * 32767
    ids = signal < 0
    signal[ids] = signal[ids] * 32768
    return signal.astype(np.int16)


def _save_audio(path, rate, signal):
    """save audio with optional conversion to 16-bit pcm"""
    if 0.0 <= np.max(np.abs(signal)) <= 1.0:
        wav.write(path, rate, _convert_to_16bit_pcm(signal))
    else:
        wav.write(path, rate, signal)


def _drc_hard_knee(dbs, threshold, scale=2, direction='up'):
    """hard knee dynamic range compression filter"""
    new = np.copy(dbs)
    if direction is 'down':
        mask = np.where(new > threshold, True, False)
    else:
        mask = np.where(new < threshold, True, False)
    new[mask] = new[mask] * scale - threshold * (scale - 1)
    return new


class AudioCleaner:

    def __init__(self, path: str, audios_dirty: str, audios_clean: str = "audios_clean", one_folder: bool = False,
                 min_silence: float = 0.01, len_part: float = 0.25, min_time: float = 2.5, f: dict = None,
                 amp_mag: bool = True, drc: bool = False, drc_param: list = None, plotting: bool = False,
                 verbose: bool = False):
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
        :param amp_mag: amplitude magnification (multiplies audio signal amplitude to increase volume)
        :param drc: toggle dynamic range compression (DRC)
        :param drc_param: DRC parameters array of tuples [(scale1, direction1)] (e.g.: [(5, 'up'), (1.1, 'down')])
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
        self.amp_mag = amp_mag
        self.drc = drc
        self.drc_param = drc_param
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
            signal = _fft_filter(signal, rate, self.f['low'], self.f['high'])
        elif self.f['type'] == 'butter':
            signal = _butter_bandpass_filter(signal, rate, self.f['low'], self.f['high'])
        else:
            pass
        mask = self._envelope(signal, rate, min_silence)
        return signal, mask

    def _dynamic_range_compression(self, signal):
        """dynamic range compression"""
        mask = np.where(signal < 0, True, False)

        dbs = librosa.core.amplitude_to_db(signal)
        threshold = np.mean(dbs)

        if not self.drc_param:
            return signal

        for param in self.drc_param:
            dbs = _drc_hard_knee(dbs, threshold, param[0], direction=param[1])

        amps_new = librosa.core.db_to_amplitude(dbs)
        amps_new[mask] = amps_new[mask] * (-1)
        return amps_new

    def _clean_audio(self, file_path):
        """
        cleans audio file
        """
        signal, rate = librosa.core.load(file_path, sr=None)
        signal_orig = np.copy(signal)

        # dynamic range compression
        if self.drc:
            signal = self._dynamic_range_compression(signal)

        if np.max(np.abs(signal)) <= self.min_silence:
            if self.verbose:
                print('TOO QUIET:', os.path.basename(file_path), end="\r")
            return

        # filter
        signal, mask = self._apply_filter(signal, rate, self.min_silence)

        # amplitude magnification
        scaled_min_silence = self.min_silence
        if self.amp_mag:
            signal, scaled_min_silence = _mul_sig_silence(signal, self.min_silence)

        return signal_orig, signal, mask, rate, scaled_min_silence

    def _output(self, out_path, file, res, min_time, lang, plotting):
        """saves clean audio, plots, writes console messages"""
        signal_orig, signal, mask, rate, scaled_min_silence = res
        ratio = len(signal[mask]) / len(signal)

        if len(signal[mask]) < int(rate * min_time):
            if self.verbose:
                print('TOO SHORT:', '{:.2f}'.format(len(signal[mask]) / rate), os.path.basename(file), end="\r")
            return

        s = '{:.2f}'.format(ratio) + ' {:.4f}'.format(scaled_min_silence) + ' ' + file
        if self.verbose:
            print(s, end="\r")
        if plotting:
            _plot(signal_orig, signal, mask, rate, s=s)
        _save_audio(os.path.join(out_path, file), rate, signal[mask])
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
                self._output(out_path, file, res, self.min_time, lang, self.plotting)

            # output
            if idx + 1 == df.shape[0] or (idx + 1) % 100 == 0:
                print("FILES CLEANED:", idx + 1)
        print()
        return utils.files_langs_to_csv(self.file_lang_list, self.path, "audios_clean_list.csv")
