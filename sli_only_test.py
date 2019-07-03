import sli
import os

list_files = {'x': r'D:\speechrecogn\voxforge\x.h5',
              'y': r'D:\speechrecogn\voxforge\y.h5'}

path = r"D:/speechrecogn/voxforge/"

dir = r'models/'
dir2_4 = r'D:\speechrecogn\voxforge\models\2-4'
dir5 = r'D:\speechrecogn\voxforge\models\5'

mp = dir5

models = os.listdir(mp)

res = []

nn = sli.AudioLangRecognitionNN(path)

for m in models:
    pr, pr_l, ev = nn.predict(list_files, model=os.path.join(mp, m), save='both')
    res.append([m, ev])

for r in res:
    print(*r)
