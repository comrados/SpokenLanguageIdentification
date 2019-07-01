import sli

list_files = {'x': r'D:\speechrecogn\voxforge\x.h5',
              'y': r'D:\speechrecogn\voxforge\y.h5'}

path = r"D:/speechrecogn/voxforge/"

model = r'D:\speechrecogn\voxforge\models\model-best.hdf5'

model1 = r'models/model1.hdf5'
model2 = r'models/model2.hdf5'
model3 = r'models/model3.hdf5'
model4 = r'models/model4.hdf5'

models = [model1, model2, model3, model4]

res = []

nn = sli.AudioLangRecognitionNN(path, model=model)

for m in models:
    pr, pr_l, ev = nn.predict(list_files, model=m, save='both')
    res.append(ev)

print(res)
