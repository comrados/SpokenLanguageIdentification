import re
import urllib
import shutil
import random
import os
from IPython.display import clear_output


def voxforge_download(links, out_path, limit=100, randomize_order=True):
    """parse given link to find all files, matching the pattern, download files"""
    dirs = []

    for key, value in links.items():

        out = os.path.join(out_path, key, "archives")  # output folder
        if not os.path.exists(out):
            os.makedirs(out)

        dirs.append(out)

        response = urllib.request.urlopen(value)
        html = response.read().decode('utf-8')

        global downloads
        files = []
        pattern = '(?i)<A HREF="(.*?)">(.*?)</A>'

        for filename in re.findall(pattern, html):
            if filename[0] == filename[1]:
                files.append(filename[0])

        print('FOUND TOTAL', len(files), 'FILES IN:', value)
        print('DOWNLOADING UP TO', limit, 'FILES TO:', out)

        random.shuffle(files)  # shuffled downloads list

        download_files(files, limit, out, value)
        print()

    return dirs


def download_files(files, limit, out, value):
    """download iles from web-page"""
    j = 0
    for file in os.listdir(out):
        if os.path.isfile(os.path.join(out, file)):
            j += 1
    if j > limit:
        print('FILES ALREADY EXIST', (str(j)))
        return
    for file in files:
        if not os.path.exists(os.path.join(out, file)):
            j += 1
            if j > limit:
                break
            print('(' + str(j) + '/' + str(len(files)) + '):', file)
            out_file = open(os.path.join(out, file), 'wb')
            response = urllib.request.urlopen(value + file)
            shutil.copyfileobj(response, out_file)
            out_file.close()


def main():
    # links to parse in search of files
    links = {'de': "http://www.repository.voxforge1.org/downloads/de/Trunk/Audio/Main/16kHz_16bit/",
             'en': "http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/",
             'ru': "http://www.repository.voxforge1.org/downloads/Russian/Trunk/Audio/Main/16kHz_16bit/"}

    out_path = r"D:/speechrecogn/voxforge/audios/"  # output root directory

    folders = voxforge_download(links, out_path, limit=5)


if __name__ == "__main__":
    main()
