import os
import pandas as pd
import numpy as np
import scipy.io.wavfile as wav


def check_path(*elements):
    """checks and creates path"""
    res = elements[0]
    for i in range(1, len(elements)):
        res = os.path.join(res, elements[i])
    if not os.path.exists(res):
        os.makedirs(res)
    return res


def files_langs_to_csv(files_list, path, csv_name):
    """file-langs lists to csv"""
    if len(files_list) > 0:
        df = pd.DataFrame.from_dict(files_list)
        df.to_csv(os.path.join(path, csv_name), index=False)
        return os.path.join(path, csv_name)
    else:
        return None


def arr_to_csv(arr, path, csv_name):
    """writes array to csv"""
    df = pd.DataFrame(np.array(arr))
    df.to_csv(os.path.join(path, csv_name), index=False)


def scale_signal_ampl(signal):
    """scales signals (and silence values)"""
    k = 0.9 / np.max(np.abs(signal))
    signal = (signal * k)
    return signal, k


def save_audio(path, rate, signal):
    """save audio with optional conversion to 16-bit pcm"""
    if 0.0 <= np.max(np.abs(signal)) <= 1.0:
        wav.write(path, rate, _convert_to_16bit_pcm(signal))
    else:
        wav.write(path, rate, signal)


def _convert_to_16bit_pcm(signal):
    """convert floating point wav to 16-bit pcm"""
    ids = signal >= 0
    signal[ids] = signal[ids] * 32767
    ids = signal < 0
    signal[ids] = signal[ids] * 32768
    return signal.astype(np.int16)
