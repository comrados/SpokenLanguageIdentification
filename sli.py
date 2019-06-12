import sli

links = {'de': "http://www.repository.voxforge1.org/downloads/de/Trunk/Audio/Main/16kHz_16bit/",
             'en': "http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/",
             'ru': "http://www.repository.voxforge1.org/downloads/Russian/Trunk/Audio/Main/16kHz_16bit/"}

out_path = r"D:/speechrecogn/voxforge/audios/"

crawler = sli.VoxforgeAudioCrawler(links, out_path, limit=7)

crawler.crawl()
