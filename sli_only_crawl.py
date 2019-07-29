import sli

links = {'de': "http://www.repository.voxforge1.org/downloads/de/Trunk/Audio/Main/16kHz_16bit/",  # german
         'en': "http://www.repository.voxforge1.org/downloads/en/Trunk/Audio/Main/16kHz_16bit/",  # english
         'ru': "http://www.repository.voxforge1.org/downloads/ru/Trunk/Audio/Main/16kHz_16bit/",  # russian
         'bg': "http://www.repository.voxforge1.org/downloads/bg/Trunk/Audio/Main/16kHz_16bit/",  # bulgarian
         'ca': "http://www.repository.voxforge1.org/downloads/ca/Trunk/Audio/Main/16kHz_16bit/",  # catalan
         'el': "http://www.repository.voxforge1.org/downloads/el/Trunk/Audio/Main/16kHz_16bit/",  # greek
         'es': "http://www.repository.voxforge1.org/downloads/es/Trunk/Audio/Main/16kHz_16bit/",  # spanish
         'fa': "http://www.repository.voxforge1.org/downloads/fa/Trunk/Audio/Main/16kHz_16bit/",  # persian (arabic)
         'fr': "http://www.repository.voxforge1.org/downloads/fr/Trunk/Audio/Main/16kHz_16bit/",  # french
         'he': "http://www.repository.voxforge1.org/downloads/he/Trunk/Audio/Main/16kHz_16bit/",  # hebrew
         'it': "http://www.repository.voxforge1.org/downloads/it/Trunk/Audio/Main/16kHz_16bit/",  # italian
         'nl': "http://www.repository.voxforge1.org/downloads/nl/Trunk/Audio/Main/16kHz_16bit/",  # dutch
         'pt': "http://www.repository.voxforge1.org/downloads/pt/Trunk/Audio/Main/16kHz_16bit/",  # portuguese
         'sq': "http://www.repository.voxforge1.org/downloads/sq/Trunk/Audio/Main/16kHz_16bit/",  # albanian
         'tr': "http://www.repository.voxforge1.org/downloads/tr/Trunk/Audio/Main/16kHz_16bit/",  # turkish
         'uk': "http://www.repository.voxforge1.org/downloads/uk/Trunk/Audio/Main/16kHz_16bit/"}  # ukrainian


path = r"D:/speechrecogn/voxforge/"
audios = "audios"
archives = "archives"

crawler = sli.AudioCrawlerVoxforge(links, path, audios, archives, limit=50, extraction_mode='many')

top_links = crawler.get_top_n(n=10)

print(top_links)

dirty = crawler.crawl()

print(dirty)
