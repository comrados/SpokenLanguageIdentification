import re
from urllib import request
import shutil
import random
import os
import tarfile
from pydub import AudioSegment


class VoxforgeAudioCrawler:

    def __init__(self, links_dict, out_path, limit=100):

        """
        initialize audio downloader

        :param links_dict: dictionary with links of format {'lang': '"http://www.link.com'}
        :param out_path: output folder for downloaded files
        :param limit: downloading limit
        """

        self.links_dict = links_dict
        self.out_path = out_path
        self.limit = limit

    def crawl(self):
        self.voxforge_download()
        self.extract_files()

    @staticmethod
    def download_files(files, limit, out, value):
        """download files from web-page"""

        j = 0
        for file in os.listdir(out):
            if os.path.isfile(os.path.join(out, file)):
                j += 1
        print('FILES ALREADY EXIST', (str(j)))
        if j > limit:
            return
        for file in files:
            if not os.path.exists(os.path.join(out, file)):
                j += 1
                if j > limit:
                    break
                print('(' + str(j) + '/' + str(len(files)) + '):', file)
                out_file = open(os.path.join(out, file), 'wb')
                response = request.urlopen(value + file)
                shutil.copyfileobj(response, out_file)
                out_file.close()

    def voxforge_download(self):
        """parse given link to find all files, matching the pattern, download files"""

        for key, value in self.links_dict.items():

            out = os.path.join(self.out_path, key, "archives")  # output folder
            if not os.path.exists(out):
                os.makedirs(out)

            response = request.urlopen(value)
            html = response.read().decode('utf-8')

            files = []
            pattern = '(?i)<A HREF="(.*?)">(.*?)</A>'

            for filename in re.findall(pattern, html):
                if filename[0] == filename[1]:
                    files.append(filename[0])

            print('FOUND TOTAL', len(files), 'FILES IN:', value)
            print('DOWNLOADING UP TO', self.limit, 'FILES TO:', out)

            random.shuffle(files)  # shuffled downloads list

            self.download_files(files, self.limit, out, value)
            print()

    def extract_file(self, path, file, out_path):
        """extract wav files from archive"""

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        name, ext = os.path.splitext(file)
        tar = tarfile.open(path + '/' + file, 'r')
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
                            self.to_wav(full_name, file_name, file_extension)
        tar.close()

    @staticmethod
    def to_wav(file, name, ext):
        """convert file to wav, if not wav"""

        new_name = name + ".wav"
        audio = AudioSegment.from_file(file, format=ext[1:])
        audio.export(new_name, format="wav")
        os.remove(file)

    def extract_files(self):
        """extract files from archives in given directory"""

        folders = []
        for folder in os.listdir(self.out_path):
            folder = os.path.join(self.out_path, folder)
            if os.path.isdir(folder):
                folders.append(folder)
        for folder in folders:
            archives_folder = os.path.join(folder, "archives")
            files = os.listdir(archives_folder)
            self.iterate_files(files, archives_folder, os.path.join(folder, "wav"))

    def iterate_files(self, files, folder, out_folder):
        """iterates through files in subdirectories"""

        print('EXTRACTING', len(files), 'FROM:', folder)
        for i, file in enumerate(files):
            if os.path.isfile(os.path.join(folder, file)):
                print('FILE', str(i+1)+':', file)
                self.extract_file(folder, file, out_folder)
        print()

