import librosa as lr
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import os
from PIL import Image, ImageOps
import h5py
from sklearn.utils.class_weight import compute_sample_weight
from . import utils


class AudioSpectrumExtractor:

    def __init__(self, path: str, audios: str, spec: str = "audios_spec", balanced: bool = True, n_patches: int = 10,
                 seed: int = None, frame_length: float = 25.0, patch_height: int = 100, patch_length: float = 1.0,
                 save_as: str = 'both', patch_sampling: str = 'random', save_full_spec: str = None, invert_colors=False,
                 h5_val_part: float = 0.15, h5_weights: bool = True, plotting: bool = False, verbose: bool = False):
        """
        Initialize audio spectrogram extractor

        :param path: working path
        :param audios: path to file-lang correspondence csv-file
        :param spec: spectrograms folder name
        :param balanced: balance number of spectrograms per language
        :param n_patches: number of patches extracted from each spectrogram
        :param seed: seed for files' order shuffling, default - None, if None - random
        :param frame_length: spectrogram frame length in milliseconds
        :param patch_height: spectrogram height (quantity of spectrogram mels)
        :param patch_length: length of patch in seconds
        :param save_as: save as images - 'img', h5-file - 'h5', both - 'both'
        :param patch_sampling: patch sampling: 'random' - random, 'gauss' - normal, 'uniform' - uniformly separated
        :param save_full_spec: path to save full spectrograms, doesn't save if None
        :param invert_colors: spectrogram background color ('white or black')
        :param h5_val_part: validation part (only for h5-file)
        :param h5_weights: toggle sample weights writing
        :param plotting: plot spectrogram and patches
        :param verbose: verbose output
        """
        self.path = path
        self.audios = audios
        self.spec = spec
        self.balanced = balanced
        self.n_patches = n_patches
        self.frame_length = frame_length
        self.patch_height = patch_height
        self.patch_length = patch_length
        self.patch_width = int(patch_length * 1000 / self.frame_length)
        self.save_as = save_as
        self.patch_sampling = patch_sampling
        self.seed = seed
        self.save_full_spec = save_full_spec
        if self.save_full_spec:
            utils.check_path(self.path, self.save_full_spec)
        utils.check_path(self.path, self.spec)
        self.invert_colors = invert_colors
        self.h5_val_part = h5_val_part
        self.plotting = plotting
        self.verbose = verbose
        self.spec_lang_list = []
        self.spec_full_lang_list = []
        self.h5_weights = h5_weights
        self.h5_file = None
        self.h5_buffer_x = []
        self.h5_buffer_y = []
        self.h5_buffer_w = []

    def _get_offsets(self, w):
        """wrapper for offsets acquisition"""
        if self.patch_sampling is 'random':
            return self._get_random_offsets(w, self.patch_width, self.n_patches)
        elif self.patch_sampling is 'uniform':
            return self._get_uniform_offsets(w, self.patch_width, self.n_patches)
        elif self.patch_sampling is 'gauss':
            return self._get_gauss_offsets(w, self.patch_width, self.n_patches)
        else:
            return self._get_random_offsets(w, self.patch_width, self.n_patches)

    def _get_spectrogram(self, file_path):
        """
        get scaled (from 0 to 255) spectrogram
        https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html
        """
        y, sr = lr.load(file_path, sr=None)
        hop = int(sr / 1000 * self.frame_length)  # hop length (samples)
        spec = lr.feature.melspectrogram(y=y, sr=sr, n_mels=self.patch_height, hop_length=hop)
        db = lr.power_to_db(spec, ref=np.max)
        # rescale magnitudes: 0 to 255
        norm = self._normalize(db)
        if self.invert_colors:
            norm = 1 - norm
        return norm

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
        self._save_spectrogram_img(path, patch)
        if num:
            self.spec_lang_list.append({'file': path, 'lang': lang})
        else:
            self.spec_full_lang_list.append({'file': path, 'lang': lang})

    def _read_and_shuffle_df(self):
        """read and shuffle dataframe"""
        df = pd.read_csv(self.audios)  # read
        if self.h5_weights:
            df['weight'] = compute_sample_weight('balanced', df['lang'])
        return df.sample(frac=1, random_state=self.seed).reset_index(drop=True)  # shuffle

    def _init_h5(self, tags, max_count, h, w, lang_count):
        """init h5 file"""
        self.h5_file.require_dataset(tags[0], (max_count, h, w, 1), maxshape=(max_count, h, w, 1), dtype=np.float16)
        self.h5_file.require_dataset(tags[1], (max_count, lang_count), maxshape=(max_count, lang_count),
                                     dtype=np.float16)
        if self.h5_weights:
            self.h5_file.require_dataset(tags[2], (max_count,), maxshape=(max_count,), dtype=np.float16)

    def _try_write_to_buffer(self, entry, lang_dummy, weight, buff_size=1000):
        """writes to the temporary buffer"""
        if len(self.h5_buffer_x) == len(self.h5_buffer_y) == len(self.h5_buffer_w) and len(self.h5_buffer_y) < buff_size:
            self.h5_buffer_x.append(np.flip(entry, axis=0) / 255.)
            self.h5_buffer_y.append(lang_dummy)
            if self.h5_weights:
                self.h5_buffer_w.append(weight)
            return True
        if len(self.h5_buffer_x) != len(self.h5_buffer_y):
            raise Exception("Something went wrong")
        else:
            return False

    def _flush_buffer(self, tags, count):
        """flush buffer"""
        self.h5_buffer_x = np.array(self.h5_buffer_x, dtype=np.float16)
        self.h5_buffer_x = self.h5_buffer_x.reshape(self.h5_buffer_x.shape + (1,))
        self.h5_buffer_y = np.array(self.h5_buffer_y)

        self.h5_file[tags[0]][count:count + len(self.h5_buffer_x)] = self.h5_buffer_x
        self.h5_file[tags[1]][count:count + len(self.h5_buffer_y)] = self.h5_buffer_y

        if self.h5_weights:
            self.h5_buffer_w = np.array(self.h5_buffer_w)
            self.h5_file[tags[2]][count:count + len(self.h5_buffer_w)] = self.h5_buffer_w
            self.h5_file[tags[2]].flush()
            self.h5_buffer_w = []

        count = count + len(self.h5_buffer_x)

        self.h5_file[tags[0]].flush()
        self.h5_file[tags[1]].flush()
        self.h5_buffer_x = []
        self.h5_buffer_y = []

        return count

    def _write_to_h5(self, tags, entry, lang_dummy, weight, count, force_flush=False):
        """buffer-wise writing to hd5"""
        if self._try_write_to_buffer(entry, lang_dummy, weight) and not force_flush:
            pass
        else:
            count = self._flush_buffer(tags, count)
            self._try_write_to_buffer(entry, lang_dummy, weight)
        return count

    def _extract_part(self, df, part, tags, mirror_df=False):
        """extract part spectrograms"""
        if mirror_df:
            df = self._mirror_df(df)
        thresholds = self._get_thresholds(df, part)  # number of files drawn for each language
        uniques = sorted(df['lang'].unique())
        counts = {u: 0 for u in uniques}  # dict for saving counts of processed files
        count_specs = 0  # spectrograms count
        print("CONVERTING UP TO", df.shape[0], "FILES INTO SPECTROGRAMS")
        print("LANGUAGES:", len(counts))
        print("MIN FILES' COUNT PER LANGUAGE:", thresholds)
        if self.save_as in ('both', 'h5'):
            if self.balanced:
                max_count = sum(thresholds.values()) * self.n_patches
            else:
                max_count = sum(thresholds.values()) * self.n_patches
            self._init_h5(tags, max_count, self.patch_height, self.patch_width, len(counts))
        for idx, row in df.iterrows():
            file_path = row['file']
            file = os.path.basename(file_path)
            filename, file_extension = os.path.splitext(file)
            lang = row['lang']
            weight = row['weight']
            if self.verbose:
                print(idx + 1, file_path, end="\r")

            # checking counts and threshold (for balanced dataset)
            if counts[lang] >= thresholds[lang]:
                continue

            # get scaled (from 0 to 255) spectrogram
            norm = self._get_spectrogram(file_path)

            # get patches from spectrogram
            patches, offsets = self._get_patches(norm)

            if self.save_as in ('both', 'img'):
                # save spectrogram patches
                plt.set_cmap('binary')  # grayscale colormap
                for i, patch in enumerate(patches):
                    self._save_spec(self._scale_to_img(patch), filename, lang, num=i + 1)

                # save full spectrogram
                if self.save_full_spec:
                    self._save_spec(self._scale_to_img(norm), filename, lang)
            if self.save_as in ('both', 'h5'):
                for patch in patches:
                    count_specs = self._write_to_h5(tags, patch, self._lang_dummy(lang, uniques), weight, count_specs)
                pass
            else:
                raise ("Wrong type of spectrograms saving: " + self.save_as)

            # output
            if idx + 1 == df.shape[0] or (idx + 1) % 100 == 0:
                print("FILES PROCESSED:", idx + 1)

            # plotting
            if self.plotting:
                self._plot_patches(patches, self.patch_sampling)
                self._plot_spec(norm, offsets)

            # increasing counter
            counts[lang] += 1

        print("FILES CONVERTED TOTAL:", sum(counts.values()))
        print("FILES CONVERTED FOR EACH LANGUAGE:", counts)

        if self.save_as in ('both', 'h5'):
            count_specs = self._flush_buffer(tags, count_specs)
            print("SPECTROGRAMS FLUSHED TO H5-FILE:", count_specs)

    def extract(self):
        """calculate spectrogram and extract patches"""
        # read files
        df = self._read_and_shuffle_df()
        if self.save_as in ('both', 'h5'):
            try:
                os.remove(os.path.join(self.path, 'data.hdf5'))
            except Exception as err:
                print("Impossible to remove file:", os.path.join(self.path, 'data.hdf5'), err)
            self.h5_file = h5py.File(os.path.join(self.path, 'data.hdf5'), 'a')

        tr = ["x", "y"]  # training tags
        va = ["x_va", "y_va"]  # validation tags
        if self.h5_weights:
            tr.append("w")
            va.append("w_va")

        self._extract_part(df, 1.0 - self.h5_val_part, tr)  # training part
        self._extract_part(df, self.h5_val_part, va, mirror_df=True)  # validation part

        spec_csv = utils.files_langs_to_csv(self.spec_lang_list, self.path, "audios_spec.csv")
        spec_full_csv = utils.files_langs_to_csv(self.spec_full_lang_list, self.path, "audios_spec_full.csv")
        if self.save_as in ('both', 'h5'):
            if self.verbose:
                if self.h5_weights:
                    print("TRAINING SHAPE", self.h5_file["x"].shape, self.h5_file["y"].shape, self.h5_file["w"].shape)
                    print("VALIDATION SHAPE", self.h5_file["x_va"].shape, self.h5_file["y_va"].shape,
                          self.h5_file["w_va"].shape)
                else:
                    print("TRAINING SHAPE", self.h5_file["x"].shape, self.h5_file["y"].shape)
                    print("VALIDATION SHAPE", self.h5_file["x_va"].shape, self.h5_file["y_va"].shape)
            self.h5_file.close()
        print()
        return spec_csv, spec_full_csv

    def _get_thresholds(self, df, part):
        """counts how many files of each lang to convert"""
        if self.balanced:
            counts = df.groupby('lang')['file'].nunique()
            min_count = counts.min()
            t = pd.Series(index=counts.index)  # empty series
            t = t.fillna(min_count) * part  # fill empty values and multiply by part
            return t.astype(int).to_dict()  # convert
        else:
            counts = df.groupby('lang')['file'].nunique()
            counts = counts * part
            return counts.astype(int).to_dict()

    @staticmethod
    def _mirror_df(df):
        return df.reindex(index=df.index[::-1]).reset_index(drop=True)

    @staticmethod
    def _lang_dummy(lang, uniques):
        arr = np.zeros(len(uniques), dtype=np.int8)
        arr[uniques.index(lang)] = 1
        return arr

    @staticmethod
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

    @staticmethod
    def _get_uniform_offsets(w, l, n):
        """uniformly distributed offsets"""
        starts = np.linspace(0, w - l - 1, num=n, dtype=int)
        offsets = []
        for start in starts:
            offsets.append(start)
            offsets.append(start + l)
        return offsets

    @staticmethod
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

    @staticmethod
    def _scale_to_img(arr):
        """scale array values by factor k, return as uint8"""
        arr = arr * 255
        return arr.astype(np.uint8)

    @staticmethod
    def _normalize(arr):
        """normalize array values, normalized values range from 0 to 1"""
        max_value = np.max(arr)
        min_value = np.min(arr)
        return ((arr - min_value) / (max_value - min_value)).astype(np.float16)

    @staticmethod
    def _plot_patches(patches, sampling):
        """plot patches"""
        for i in range(1, len(patches) + 1):
            plt.subplot(1, len(patches), i)
            plt.axis('off')
            plt.title(str(i + 1))
            plt.imshow(patches[i - 1], origin="lower")
        plt.suptitle(str(len(patches)) + ' spectrogram patches, ' + sampling + ' sampling')
        plt.show()

    @staticmethod
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

    @staticmethod
    def _save_spectrogram_img(path, arr):
        """save spectrogram as image"""
        img = ImageOps.flip(Image.fromarray(arr, mode='L'))
        img.save(path)
