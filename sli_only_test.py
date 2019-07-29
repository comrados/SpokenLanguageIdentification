import sli
import os

# list_files = {'x': r'D:\speechrecogn\voxforge\x.h5', 'y': r'D:\speechrecogn\voxforge\y.h5'}

# list_files = r'D:\speechrecogn\voxforge\data.hdf5'
# list_files = r'D:\speechrecogn\voxforge\data_balanced.hdf5'

# list_files = r'D:\speechrecogn\voxforge\data_balanced_2sec.hdf5'
# list_files = r'D:\speechrecogn\voxforge\data_balanced_2sec_100_1.hdf5'
# list_files = r'D:\speechrecogn\voxforge\data_balanced_2sec_100_2.hdf5'
# list_files = r'D:\speechrecogn\voxforge\data_balanced_2sec_100_3.hdf5'
list_files = r'D:\speechrecogn\voxforge\data_balanced_2sec_50_1.hdf5'

path = r"D:/speechrecogn/voxforge/"

dir_ = r'models/'
dir2_4 = r'D:\speechrecogn\voxforge\models\2-4'
dir5 = r'D:\speechrecogn\voxforge\models\5'
dir6_7 = r'D:\speechrecogn\voxforge\models\6-7'  # 30%
dir_inception = r"D:\speechrecogn\voxforge\models\incept"

mp = dir_inception

models = os.listdir(mp)

res = []

nn = sli.AudioLangRecognitionNN(path)

for i, m in enumerate(models):
    print(i+1, 'of', len(models))
    pr, pr_l, ev = nn.predict(list_files, model=os.path.join(mp, m), save='both')
    res.append([m, *ev])

print("\n"+list_files+"\n")
for r in res:
    print('{0}, loss: {1:.3f}, acc: {2:.3f}'.format(*r))
