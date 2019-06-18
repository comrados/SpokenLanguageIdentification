import sli

links = {'de': "http://www.repository.voxforge1.org/downloads/de/Trunk/Audio/Main/16kHz_16bit/",
         'en': "http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/",
         'ru': "http://www.repository.voxforge1.org/downloads/Russian/Trunk/Audio/Main/16kHz_16bit/"}
path = r"D:/speechrecogn/voxforge/"
audios = "audios"
archives = "archives"

crawler = sli.AudioCrawlerVoxforge(links, path, audios, archives, limit=7, extraction_mode='many')
dirty = crawler.crawl()


audios_clean = "audios_clean"
one_folder = True
min_silence = 0.01
len_part = 0.25
min_time = 2.5
f = 'butter'
low = 100
hi = 7000
amp_mag = True
drc = False
drc_param = [(5, 'up'), (1.1, 'down')]
plotting = False


cleaner = sli.AudioCleaner(path, dirty, audios_clean, one_folder, min_silence, len_part, min_time, f, low, hi,
                           amp_mag, drc, drc_param, plotting)
clean = cleaner.clean()


spectre = sli.AudioSpectrumExtractor(path, clean, "audio_spec", save_full_spec="audio_spec_full")

spectre.extract()
