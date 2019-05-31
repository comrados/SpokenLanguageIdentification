import numpy as np
import scipy.io.wavfile as wav
import os
import pandas as pd



def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/5), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


def remove_silence(path, out="audios_clean", one_folder=False):
    """converts wavs to images"""
    files_list = []
    audios_path = check_path(path, "audios")
    pics_path = check_path(path, "pics")    
    #max_count = get_min_files_count(audios_path)
    out_path = check_path(path, out)
    for folder in os.listdir(audios_path):
        print("\nOutputting from folder:", folder.upper())
        if not one_folder:
            out_path = check_path(path, out, folder)
        lang_folder = check_path(audios_path, folder)
        # out_folder = check_path([pics_path, folder])
        if os.path.isdir(lang_folder):
            wav_folder = check_path(lang_folder, "wav")
            wav_files = os.listdir(wav_folder)
            for file in wav_files:                
                file_path = os.path.join(wav_folder, file)
                if one_folder:
                    file = folder + '_' + file
                if os.path.isfile(file_path):
                    rate, signal = wav.read(file_path)
                    mask = envelope(signal, rate, 200)
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



path = r"D:/speechrecogn/voxforge/"
#convert_to_images(path, validation_part=0.2, img_size=(150, 150))
remove_silence(path, out="audios_clean", one_folder=True)
#convert_to_images2(path, out="out_150x150", img_size=(150, 150))
