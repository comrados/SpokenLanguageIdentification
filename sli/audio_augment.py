import pandas as pd
import os
import librosa as lr
import numpy as np
import random
from . import utils


class AudioAugmentor:

    def __init__(self, path: str, audios: str, audios_augmented: str = "audios_augmented", one_folder: bool = False,
                 min_time: float = 2.5, n_files: int = 3, speed: tuple = (0.9, 1.1), pitch_shift: tuple = (-0.3, 0.3),
                 drc: list = None, seed: int = None, verbose: bool = False):
        """
        Initialize audio cleaner

        :param path: working path
        :param audios: path to file-lang correspondence csv-file
        :param audios_augmented: output folder name (will be created in path)
        :param one_folder: output to one or multiple (the resulting number of folders equals to number of languages)
        :param min_time: minimal length of processed audio to save
        :param n_files: number of augmented files, returned from the original (including original one)
        :param speed: randomly manipulate the speed inside of the given range (e.g. (0.9, 1.1)), None to ignore
        :param pitch_shift: randomly manipulate pitch inside of the given range (e.g. (-1, 1)), None to ignore
        :param drc:  dynamic range compression (DRC) parameters [(scale1, direction1)]
                    (e.g.: [(5, 'up'), (1.1, 'down')]), None to ignore
        :param speed: random generation seed
        :param verbose: verbose output
        """
        self.path = path
        self.audios = audios
        self.audios_augmented = audios_augmented
        self.one_folder = one_folder
        self.min_time = min_time
        self.n_files = n_files
        self.file_lang_list = []
        self.speed = speed
        self.pitch_shift = pitch_shift
        self.drc = drc
        self.verbose = verbose
        random.seed(seed)

    def augment(self):
        """run the augmentation"""
        df = pd.read_csv(self.audios)
        print("AUGMENTING", df.shape[0], "FILES")
        if self.verbose:
            print("SHIFTING PITCH BETWEEN: {0:+.3f} AND {1:+.3f}".format(*self.pitch_shift))
            print("CHANGING SPEED BETWEEN: {0:.3f} AND {1:.3f}".format(*self.speed))
        for idx, row in df.iterrows():
            file_path = row['file']
            file = os.path.basename(file_path)
            lang = row['lang']
            out_path = utils.check_path(self.path, self.audios_augmented)

            # saving path
            if self.one_folder:
                if not file.startswith(lang + '_'):
                    file = lang + '_' + file
            else:
                out_path = utils.check_path(self.path, self.audios_augmented, lang)

            # augment
            audios = self._augment_audio(file_path)

            # output
            if audios:
                for i, audio in enumerate(audios):
                    name, ext = os.path.splitext(file)
                    new_name = name + '_' + str(i) + ext
                    self._output(out_path, new_name, lang, audio[0], audio[1])
                    if self.verbose:
                        print("PITCH: {:+.3f},".format(audio[3]), "SPEED: {:.3f},".format(audio[2]), "NAME:", new_name)

            if idx + 1 == df.shape[0] or (idx + 1) % 100 == 0:
                print("FILES AUGMENTED:", idx + 1)
        print()
        return utils.files_langs_to_csv(self.file_lang_list, self.path, "audios_augmented_list.csv")

    def _augment_audio(self, file_path):
        """augment one audio and return list of signals (including the original one)"""
        try:
            audios = []
            orig, rate = lr.load(file_path, sr=None)
            audios.append((rate, orig, 1, 0))

            if self.pitch_shift or self.speed:
                for i in range(1, self.n_files):
                    audio = orig[:]
                    pitch, speed = 0, 1
                    # change pitch
                    if self.pitch_shift:
                        audio, pitch = self._pitch_shift(audio, rate)
                    # change speed
                    if self.speed:
                        audio, speed = self._speed_change(audio)
                    # dynamic range compression
                    if self.drc:
                        audio = self._dynamic_range_compression(audio)

                    # checks if length of the converted file satisfies conditions
                    if len(audio) < int(rate * self.min_time):
                        i -= 1
                    else:
                        audios.append((rate, audio, speed, pitch))

            return audios
        except:
            print("FAILED TO AUGMENT", file_path)
            return

    def _output(self, out_path, file, lang, rate, audio):
        """saves clean audio, plots, writes console messages"""



        utils.save_audio(os.path.join(out_path, file), rate, audio)
        self.file_lang_list.append({'file': os.path.join(out_path, file), 'lang': lang})

    def _pitch_shift(self, audio, fs):
        """pitch shifting by random factor"""
        pitch = random.uniform(self.pitch_shift[0], self.pitch_shift[1])
        audio = lr.effects.pitch_shift(audio, fs, n_steps=pitch)
        return audio, pitch

    def _speed_change(self, audio):
        """speed manipulation by random factor"""
        speed = random.uniform(self.speed[0], self.speed[1])
        indices = np.round(np.arange(0, len(audio), speed))
        indices = indices[indices < len(audio)].astype(int)
        return audio[indices.astype(int)], speed

    def _dynamic_range_compression(self, signal):
        """dynamic range compression"""
        mask = np.where(signal < 0, True, False)

        dbs = lr.core.amplitude_to_db(signal)
        threshold = np.mean(dbs)

        if not self.drc:
            return signal

        for param in self.drc:
            dbs = self._drc_hard_knee(dbs, threshold, param[0], direction=param[1])

        amps_new = lr.core.db_to_amplitude(dbs)
        amps_new[mask] = amps_new[mask] * (-1)
        return amps_new

    @staticmethod
    def _drc_hard_knee(dbs, threshold, scale=2, direction='up'):
        """hard knee dynamic range compression filter"""
        new = np.copy(dbs)
        if direction is 'down':
            mask = np.where(new > threshold, True, False)
        else:
            mask = np.where(new < threshold, True, False)
        new[mask] = new[mask] * scale - threshold * (scale - 1)
        return new
