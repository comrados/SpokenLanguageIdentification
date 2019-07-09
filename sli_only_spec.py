import sli

clean = r"D:/speechrecogn/voxforge/audios_clean_list.csv"
path = r"D:/speechrecogn/voxforge"

spectre = sli.AudioSpectrumExtractor(path, clean, "audios_spec", save_full_spec="audios_spec_full", seed=0,
                                     balanced=False, verbose=True, h5_val_part=0.1, patch_length=2)

spectre.get_stats()

patches, specs = spectre.extract()
