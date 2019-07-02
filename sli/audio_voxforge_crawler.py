import re
from urllib import request
import shutil
import random
import os
import tarfile
from pydub import AudioSegment
from . import utils


def _get_file_names_from_url(url):
    files = []
    response = request.urlopen(url)
    html = response.read().decode('utf-8')

    pattern = '(?i)<A HREF="(.*?)">(.*?)</A>'

    for filename in re.findall(pattern, html):
        if filename[0] == filename[1]:
            files.append(filename[0])
    return files


def _to_wav(file, name, ext):
    """convert file to wav, if not wav"""
    new_name = name + ".wav"
    audio = AudioSegment.from_file(file, format=ext[1:])
    audio.export(new_name, format="wav")
    os.remove(file)
    return new_name


class AudioCrawlerVoxforge:

    def __init__(self, links_dict: dict, path: str, audios: str = "audios", archives="archives", limit: int = 100,
                 seed: int = None, extraction_mode: str = 'one', verbose: bool = False):
        """
        Initialize audio downloader

        :param links_dict: dictionary with links of format: {'ln1': '"http://link1.com', 'ln2': '"http://link2.com'}
        :param path: working path
        :param audios: folder for extracted wav-files (will be created in given path)
        :param archives: folder for downloaded archives with audios
        :param limit: downloading limit
        :param seed: seed for random order of downloading (original order if None)
        :param extraction_mode: 'one' - (default) extract to one directory, create name-lang correspondence table,
                                'many' - extract each files of each language to separate directory
        :param verbose: verbose output
        """

        self.links_dict = links_dict
        self.path = path
        self.audios = audios
        self.archives = archives
        self.limit = limit
        self.seed = seed
        if self.seed:
            random.seed(self.seed)
        self.out_path = utils.check_path(path, audios)
        self.extraction_mode = extraction_mode
        self.file_lang_list = []
        self.verbose = verbose

    def crawl(self):
        self._voxforge_download()
        return self._extract_files()

    def _download_files(self, files, limit, out, value):
        """download files from web-page"""
        j = 0
        for file in os.listdir(out):
            if os.path.isfile(os.path.join(out, file)):
                j += 1
        print('FILES ALREADY EXIST', str(j))
        for file in files:
            if not os.path.exists(os.path.join(out, file)):
                if j == limit or j % 10 == 0:
                    print("FILES DOWNLOADED:", j)
                try:
                    j += 1
                    if j > limit:
                        break
                    if self.verbose:
                        print('(' + str(j) + '/' + str(len(files)) + '):', file)
                    out_file = open(os.path.join(out, file), 'wb')
                    response = request.urlopen(value + file)
                    shutil.copyfileobj(response, out_file)
                    out_file.close()
                except:
                    j -= 1
                    print("FAILED TO GET:", file)

    def _voxforge_download(self):
        """parse given link to find all files, matching the pattern, download files"""

        for lang, url in self.links_dict.items():
            out = utils.check_path(self.out_path, lang, self.archives)

            files = _get_file_names_from_url(url)

            print('FOUND TOTAL', len(files), 'FILES IN:', url)
            print('DOWNLOADING UP TO', self.limit, 'FILES TO:', out)

            random.shuffle(files)  # shuffle download order

            self._download_files(files, self.limit, out, url)
            print()

    def _extract_file_many(self, path, file, lang, out_path):
        """extract wav files from archive"""
        name, ext = os.path.splitext(file)
        try:
            tar = tarfile.open(os.path.join(path, file), 'r')
            for item in tar:
                if item.isfile():
                    full_name = os.path.join(out_path, name + '-' + os.path.basename(item.name))
                    file_name, file_extension = os.path.splitext(full_name)
                    if len(file_extension) != 0 and file_extension != '.txt':
                        if not os.path.exists(full_name):
                            out = open(full_name, 'wb+')
                            out.write(tar.extractfile(item).read())
                            out.close()
                            if file_extension != '.wav':
                                full_name = _to_wav(full_name, file_name, file_extension)
                        self.file_lang_list.append({"file": full_name, "lang": lang})
            tar.close()
        except:
            print("FAILED TO EXTRACT FROM:", os.path.join(path, file))

    def _extract_file_one(self, path, file, lang):
        self._extract_file_many(path, file, lang, self.out_path)

    def _extract_files(self):
        """extract files from archives in given directory"""
        folders = []
        for folder in os.listdir(self.out_path):
            lang = folder
            folder = os.path.join(self.out_path, folder)
            if os.path.isdir(folder):
                folders.append((folder, lang))
        for folder, lang in folders:
            archives_folder = os.path.join(folder, self.archives)
            files = os.listdir(archives_folder)
            out_folder = self.out_path
            if self.extraction_mode is 'many':
                out_folder = utils.check_path(folder, "wav")
            self._iterate_files(files, archives_folder, out_folder, lang)
        return utils.files_langs_to_csv(self.file_lang_list, self.path, "audios_list.csv")

    def _iterate_files(self, files, folder, out_folder, lang):
        """iterates through files in subdirectories"""

        print('EXTRACTING', len(files), 'FROM:', folder)
        for i, file in enumerate(files):
            if os.path.isfile(os.path.join(folder, file)):
                if self.extraction_mode is 'many':
                    if self.verbose:
                        print('FILE', str(i + 1) + ':', file, 'to', out_folder, end="\r")
                    self._extract_file_many(folder, file, lang, out_folder)
                else:
                    if self.verbose:
                        print('FILE', str(i + 1) + ':', file, 'to', self.out_path, end="\r")
                    self._extract_file_one(folder, file, lang)
            if i + 1 == len(files) or (i + 1) % 10 == 0:
                print("FILES EXTRACTED:", i + 1)
        print()
