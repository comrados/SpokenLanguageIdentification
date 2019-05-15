import os
import tarfile
from pydub import AudioSegment


def extract_file(path, file, out_path):
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
                        full_name = to_wav(full_name, file_name, file_extension)
                    if os.path.getsize(full_name) < 1000 * 100:
                        os.remove(full_name)
    tar.close()


def to_wav(file, name, ext):
    """convert file to wav, if not wav"""
    new_name = name + ".wav"
    audio = AudioSegment.from_file(file, format=ext[1:])
    audio.export(new_name, format="wav")
    os.remove(file)
    return new_name


def extract_files(directory):
    """extract files from archives in given directory"""
    dic = {}
    folders = []
    for folder in os.listdir(directory):
        folder = os.path.join(directory, folder)
        if os.path.isdir(folder):
            folders.append(folder)
    for folder in folders:
        archives_folder = os.path.join(folder, "archives")
        files = os.listdir(archives_folder)
        iterate_files(files, archives_folder, os.path.join(folder, "wav"))
    return dic


def iterate_files(files, folder, out_folder):
    """iterates through files in subdirectories"""
    print('EXTRACTING FROM:', folder)
    for file in files:
        if os.path.isfile(os.path.join(folder, file)):
            print('FILE:', file, end='\r')
            extract_file(folder, file, out_folder)
    print()


def main():
    path = "D:/speechrecogn/voxforge/audios/"  # path with files

    extracted = extract_files(path)


if __name__ == "__main__":
    main()
