from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from pydub import AudioSegment
import uuid
from PIL import Image

audio = AudioSegment.from_wav(r"D:/speechrecogn/voxforge/audios\ru\wav\uvgeek-20150608-etf-ru_0089.wav")

data = np.array(audio.get_array_of_samples())/audio.max_possible_amplitude

(rate,sig) = wav.read(file_path)
mfcc_feat = mfcc(sig,rate)

print(mfcc_feat)
plt.plot(mfcc_feat)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

(rate,sig) = wav.read(file_path)
mfcc_feat = mfcc(sig,rate, nfft=1024)

ig, ax = plt.subplots()
mfcc_data= np.swapaxes(mfcc_feat, 0 ,1)
cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
ax.set_title('MFCC')
#Showing mfcc_data
plt.show()
#Showing mfcc_feat
plt.plot(mfcc_feat)
plt.show()


import datetime

str(datetime.datetime.now())



d = datetime.datetime(2018, 1, 1)
print(d)

while d < datetime.datetime(2018, 1, 2):
    d += datetime.timedelta(minutes=15)
    print(d)

t1 = (1, 2, 3 ,4)
t2 = (100,) + t1 + (100,)


from scipy import signal
import matplotlib.pyplot as plt
from pydub import AudioSegment
import scipy.io.wavfile as wav

file = r"D:/speechrecogn/voxforge/audios\ru\wav\uvgeek-20150608-etf-ru_0089.wav"

audio = AudioSegment.from_wav(file)
samplerate, samples = wav.read(file)

samples = samples / 32768.0
print(max(samples))

f, t, Sxx = signal.spectrogram(samples, 16000)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

di = 11
dsi = uuid.UUID("00000000-0000-0000-0000-000000000000")
print(str(dsi))
tu = (214353, 3465432)
limit = "LIMIT 10000"

params = {"di": str(di), "dsi": str(dsi), "from": str(tu[0]), "to": str(tu[1]), "limit": limit}

print(params)

query = "SELECT * FROM data WHERE device_id={di} ".format(**params)
query += "and data_source_id={dsi} ".format(**params)
query += "and time_upload >= '{from}' and time_upload < '{to}' ".format(**params)
query += "{limit} ALLOW FILTERING".format(**params)

print(query)

query = "SELECT * FROM data WHERE device_id={di} and data_source_id={dsi} and time_upload >= '{from}' and time_upload < '{to}' {limit} ALLOW FILTERING".format(**params)

print(query)

x1 = [123, 123,5456]
x2=['asd', 'fgb', 'klojkhg']

y = [(123,564),(43,76),(987,545)] 
y = list(y)

x2.extend(x1)

x3 = list(x1)

'%.4f' % 25.5

img = Image.open(r"D:\speechrecogn\voxforge\pics\train\en\1028-20100710-hne-ar-03_1.png")