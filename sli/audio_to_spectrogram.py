import librosa as lr
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import os

from . import utils


def _get_random_offsets(w, l, n):
    """random offsets"""
    offsets = []
    i = 0
    while i < n:
        offset = random.randint(0, w - l - 1)
        if offset in offsets:
            continue
        else:
            i += 1
            offsets.append(offset)
            offsets.append(offset + l)
    return offsets


def _get_uniform_offsets(w, l, n):
    """uniformly distributed offsets"""
    starts = np.linspace(0, w - l - 1, num=n, dtype=int)
    offsets = []
    for start in starts:
        offsets.append(start)
        offsets.append(start + l)
    return offsets


def _get_gauss_offsets(w, l, n):
    """normally distributed offsets"""
    offsets = []
    i = 0
    while i < n:
        # distribution with mean (w-l-1)/2 and 3-sigma interval (w-l-1)/2
        offset = int(np.minimum(np.maximum(0, np.random.normal((w - l - 1) / 2, (w - l - 1) / 2 / 3)), w - l - 1))
        if offset in offsets:
            continue
        else:
            i += 1
            offsets.append(offset)
            offsets.append(offset + l)
    return offsets


def _scale_arr(arr, k):
    """scale array values by factor k, return as int"""
    arr = arr * k
    return arr.astype(np.int32)


def _normalize(arr):
    """normalize array values, normalized values range from 0 to 1"""
    max_value = np.max(arr)
    min_value = np.min(arr)
    return (arr - min_value) / (max_value - min_value)


def _save_spectrogram_img(path, arr):
    """save spectrogram as image"""
    plt.imsave(path, arr, origin="lower")


def _plot_patches(patches, sampling):
    """plot patches"""
    for i in range(1, len(patches) + 1):
        plt.subplot(1, len(patches), i)
        plt.axis('off')
        plt.title(str(i+1))
        plt.imshow(patches[i - 1], origin="lower")
    plt.suptitle(str(len(patches)) + ' spectrogram patches, ' + sampling + ' sampling')
    plt.show()


def _plot_spec(spec, offsets):
    """plot spectrogram"""
    new = np.copy(spec)
    new[:, offsets] = 255
    plt.subplot(1, 2, 1)
    plt.imshow(new, origin="lower")
    plt.axis('off')
    plt.title('With offsets')
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(spec, origin="lower")
    plt.title('Pure')
    plt.suptitle('Full spectrogram')
    plt.show()


class AudioSpectrumExtractor:

    def __init__(self, path: str, audios: str, spec: str = "audios_spec", frame_length: float = 25.0, n_mels: int = 200,
                 n_patches: int = 10, patch_length: float = 1.0, patch_sampling_mode: str = 'random',
                 save_full_spec: str = None, plotting: bool = False, verbose: bool = False):
        """
        Initialize audio spectrogram extractor

        :param path: working path
        :param audios: path to file-lang correspondence csv-file
        :param spec: spectrograms folder name
        :param frame_length: spectrogram frame length in milliseconds
        :param n_mels: quantity of spectrogram mels (spectrogram height)
        :param n_patches: number of patches extracted from each spectrogram
        :param patch_length: length of patch in seconds
        :param patch_sampling_mode: patch sampling: 'random' - random, 'gauss' - normal, 'uniform' - uniformly separated
        :param save_full_spec: path to save full spectrograms, doesn't save if None
        :param plotting: plot spectrogram and patches
        :param verbose: verbose output
        """
        self.path = path
        self.audios = audios
        self.spec = spec
        self.frame_length = frame_length
        self.n_mels = n_mels
        self.n_patches = n_patches
        self.patch_length = patch_length
        self.patch_width = int(patch_length * 1000 / self.frame_length)
        self.patch_sampling_mode = patch_sampling_mode
        self.save_full_spec = save_full_spec
        if self.save_full_spec:
            utils.check_path(self.path, self.save_full_spec)
        utils.check_path(self.path, self.spec)
        self.plotting = plotting
        self.verbose = verbose
        self.spec_lang_list = []
        self.spec_full_lang_list = []

    def _get_offsets(self, w):
        """wrapper for offsets acquisition"""
        if self.patch_sampling_mode is 'random':
            return _get_random_offsets(w, self.patch_width, self.n_patches)
        elif self.patch_sampling_mode is 'uniform':
            return _get_uniform_offsets(w, self.patch_width, self.n_patches)
        elif self.patch_sampling_mode is 'gauss':
            return _get_gauss_offsets(w, self.patch_width, self.n_patches)
        else:
            return _get_random_offsets(w, self.patch_width, self.n_patches)

    def _get_spectrogram(self, file_path):
        """
        get scaled (from 0 to 255) spectrogram
        https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html
        """
        y, sr = lr.load(file_path, sr=None)
        hop = int(sr / 1000 * self.frame_length)  # hop length (samples)
        spec = lr.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, hop_length=hop)
        db = lr.power_to_db(spec, ref=np.max)
        # rescale magnitudes: 0 to 255
        scaled = _scale_arr(_normalize(db), 255).astype(np.int16)
        return scaled

    def _get_patches(self, spec):
        """get spectrogram patches"""
        h, w = spec.shape
        patches = []
        offsets = self._get_offsets(w)
        for i in range(0, len(offsets), 2):
            patches.append(spec[:, offsets[i]:offsets[i + 1]])
        return patches, offsets

    def _save_spec(self, patch, filename, lang, num=None):
        """get name of spectrogram image and save it"""
        if filename.startswith(lang + '_'):
            new_name = filename
        else:
            new_name = lang + '_' + filename
        spec_folder = self.save_full_spec
        if num:
            new_name += '_' + str(num)
            spec_folder = self.spec
        new_name += '.png'
        path = os.path.join(self.path, spec_folder, new_name)
        _save_spectrogram_img(path, patch)
        if num:
            self.spec_lang_list.append({'file': path, 'lang': lang})
        else:
            self.spec_full_lang_list.append({'file': path, 'lang': lang})

    def extract(self):
        """calculate spectrogram and extract patches"""
        # read files
        df = pd.read_csv(self.audios)
        print("CONVERTING", df.shape[0], "FILES INTO SPECTROGRAMS")
        for idx, row in df.iterrows():
            file_path = row['file']
            file = os.path.basename(file_path)
            filename, file_extension = os.path.splitext(file)
            lang = row['lang']
            if self.verbose:
                print(idx+1, file_path, end="\r")

            # get scaled (from 0 to 255) spectrogram
            scaled = self._get_spectrogram(file_path)

            # get patches from spectrogram
            patches, offsets = self._get_patches(scaled)

            # save spectrogram patches
            plt.set_cmap('binary')  # grayscale colormap
            for i, patch in enumerate(patches):
                self._save_spec(patch, filename, lang, num=i+1)

            # save full spectrogram
            if self.save_full_spec:
                self._save_spec(scaled, filename, lang)

            # output
            if idx + 1 == df.shape[0] or (idx + 1) % 100 == 0:
                print("FILES CONVERTED:", idx + 1)

            # plotting
            if self.plotting:
                _plot_patches(patches, self.patch_sampling_mode)
                _plot_spec(scaled, offsets)

        print()
        spec_csv = utils.files_langs_to_csv(self.spec_lang_list, self.path, "audios_spec.csv")
        spec_full_csv = utils.files_langs_to_csv(self.spec_full_lang_list, self.path, "audios_spec_full.csv")
        return spec_csv, spec_full_csv

        # TODO saving to h5
