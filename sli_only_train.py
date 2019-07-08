import sli

path = r"D:/speechrecogn/voxforge/"
file = r"D:/speechrecogn/voxforge/data.hdf5"


nn = sli.AudioLangRecognitionNN(path, threshold=0.9, epochs=10)

history = nn.train(file)

print(history)