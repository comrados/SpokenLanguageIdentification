import sli

clean = r"D:/speechrecogn/voxforge/audios_augmented_list.csv"
path = r"D:/speechrecogn/voxforge"

spectre = sli.AudioSpectrumExtractor(path, clean, "audios_spec", save_full_spec=None, seed=0,
                                     verbose=True, h5_val_part=0, patch_length=2, n_patches=10,
                                     h5_name="data_balanced_2sec_100_aug_1.hdf5", save_as="h5",
                                     h5_weights=True, patch_sampling='gauss', balanced=False)

spectre.get_stats()

patches, specs = spectre.extract()
