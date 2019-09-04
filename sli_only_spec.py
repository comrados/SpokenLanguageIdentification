import sli

clean = r"D:/speechrecogn/voxforge/audios_augmented_list.csv"
path = r"D:/speechrecogn/voxforge"

spectre = sli.AudioSpectrumExtractor(path, clean, "audios_spec", save_full_spec=None, seed=0,
                                     verbose=True, patch_length=2, n_patches=10,
                                     h5_name="data_2sec_100_aug_2.hdf5", save_as="h5",
                                     h5_weights=True, patch_sampling='consequent', balanced=False,
                                     consequent_step=2)

spectre.get_stats()

patches, specs = spectre.extract()
