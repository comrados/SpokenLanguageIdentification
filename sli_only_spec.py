import sli

clean = r"D:/speechrecogn/voxforge/audios_clean_list.csv"
path = r"D:/speechrecogn/voxforge"

spectre = sli.AudioSpectrumExtractor(path, clean, "audios_spec", save_full_spec="audios_spec_full", seed=0,
                                     balanced=True, verbose=True, h5_val_part=0, patch_length=2, n_patches=10,
                                     h5_name="data_balanced_2sec_50_1.hdf5", save_as="h5", balanced_threshold=100,
                                     h5_weights=False)

spectre.get_stats()

patches, specs = spectre.extract()
