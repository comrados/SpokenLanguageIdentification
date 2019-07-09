import sli


path = r"D:/speechrecogn/voxforge/"
dirty = r"D:/speechrecogn/voxforge/audios_list.csv"
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
