import sli


path = r"D:/speechrecogn/voxforge/"
clean = r"D:/speechrecogn/voxforge/audios_clean_list.csv"
audios_aug = "audios_augmented"
one_folder = True


augmentor = sli.AudioAugmentor(path, clean, audios_aug, one_folder, verbose=True)

augmented = augmentor.augment()
