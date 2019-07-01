import sli

links = {'de': "http://www.repository.voxforge1.org/downloads/de/Trunk/Audio/Main/16kHz_16bit/",
         'en': "http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/",
         'ru': "http://www.repository.voxforge1.org/downloads/Russian/Trunk/Audio/Main/16kHz_16bit/"}
path = r"D:/speechrecogn/voxforge/"
audios = "audios"
archives = "archives"

crawler = sli.AudioCrawlerVoxforge(links, path, audios, archives, limit=7, extraction_mode='many')
dirty = crawler.crawl()

########################################################################################################################

audios_clean = "audios_clean"
one_folder = True
min_silence = 0.01
len_part = 0.25
min_time = 2.5
f = {'type': 'butter', 'low': 100, 'high': 7000}
amp_mag = True
drc = False
drc_param = [(5, 'up'), (1.1, 'down')]
plotting = False

cleaner = sli.AudioCleaner(path, dirty, audios_clean, one_folder, min_silence, len_part, min_time, f,
                           amp_mag, drc, drc_param, plotting)
clean = cleaner.clean()

########################################################################################################################

spectre = sli.AudioSpectrumExtractor(path, clean, "audios_spec", save_full_spec="audios_spec_full", seed=0)
patches, specs = spectre.extract()

########################################################################################################################

converter = sli.AudioToH5Converter(path, patches, verbose=True, plotting=False)
files = converter.convert()

########################################################################################################################

list_files = {'x': r'D:\speechrecogn\voxforge\x.h5',
              'y': r'D:\speechrecogn\voxforge\y.h5'}

path = r"D:/speechrecogn/voxforge/"

model = r'D:\speechrecogn\voxforge\models\model-best.hdf5'

model1 = r'models/model1.hdf5'
model2 = r'models/model2.hdf5'
model3 = r'models/model3.hdf5'
model4 = r'models/model4.hdf5'

nn = sli.AudioLangRecognitionNN(path, model=model)

pr, pr_l, ev = nn.predict(list_files)

print(ev)
