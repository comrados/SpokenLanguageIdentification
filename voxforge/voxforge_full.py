import re
import urllib
import shutil
import random
import os
import tarfile
from pydub import AudioSegment

def voxforge_download(links, out_path, limit=100, randomize_order=True):
    dirs = []
    
    for key, value in links.items():
        
        out = out_path+key+"/" # output folder
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
        
        print('Found total', len(files),'files in', value)
        print('Saving total', limit,'files to', out)
        
        random.shuffle(files) # shuffled downloads list
        
        download_files(files, limit, out, value)        
        print()
        
    return dirs


def download_files(files, limit, out, value):
    i = 0
    j = 0
    for file in os.listdir(out):
        if os.path.isfile(out+file):
            j += 1
    for file in files:
        if not os.path.exists(out+file):
            i += 1
            j += 1
            if i > limit:
                break 
            print('('+str(j)+ '/' +str(len(files))+ '):', file)    
            out_file = open(out+file, 'wb')
            response = urllib.request.urlopen(value+file)
            shutil.copyfileobj(response, out_file)
            out_file.close()

    
def extract_file(path, file, out_path=None, remove_archives=False):
    if out_path is None:
        out_path = path
    if not os.path.exists(out_path):
            os.makedirs(out_path)
    name, ext = os.path.splitext(file)
    tar = tarfile.open(path+'/'+file, 'r')
    for item in tar:
        if item.isfile():
            full_name = os.path.join(out_path,name+'-'+os.path.basename(item.name))
            file_name, file_extension = os.path.splitext(full_name)
            if len(file_extension) != 0 and file_extension != '.txt':
                if not os.path.exists(full_name):
                    out = open(full_name, 'wb+')
                    out.write(tar.extractfile(item).read())
                    out.close()
                    if file_extension != '.wav':
                        full_name = to_wav(full_name, file_name, file_extension)
                    if os.path.getsize(full_name)<1000*100:
                        os.remove(full_name)
                    
                    
    tar.close()
    if remove_archives:
        os.remove(path+file)
        
        
def to_wav(file, name, ext):
    new_name = name+".wav"
    audio = AudioSegment.from_file(file, format=ext[1:])
    audio.export(new_name,format="wav")
    os.remove(file)
    return new_name


def extract_files(directory, out_folders=None, remove_archives=False):
    dic = {}
    folders = []
    for folder in os.listdir(directory):
        folder = os.path.join(directory, folder)
        if os.path.isdir(folder):
            folders.append(folder)
    for folder in folders:
        files = os.listdir(folder)
        for file in files:
            if os.path.isfile(os.path.join(folder, file)):
                out_dir = None
                if out_folders is not None:                
                    out_dir = folder+'/'+out_folders
                extract_file(folder, file, out_path=out_dir, remove_archives=remove_archives)
    return dic
    

def main():
    links = {}

    links['de'] = "http://www.repository.voxforge1.org/downloads/de/Trunk/Audio/Main/16kHz_16bit/"
    links['en'] = "http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/"
    links['ru'] = "http://www.repository.voxforge1.org/downloads/Russian/Trunk/Audio/Main/16kHz_16bit/"
    
    out_path = "D:/speechrecogn/voxforge/"
    
    folders = voxforge_download(links, out_path, limit=1)
    
    extracted = extract_files(out_path, out_folders='out', remove_archives=False)
    

if __name__ == "__main__":
    main()
