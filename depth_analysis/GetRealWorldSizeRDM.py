import numpy as np
import glob
import re
import pandas as pd

tsv_data = pd.read_csv('size_meanRatings.tsv', sep='\t', header=0)
names_fromtsv = np.array(tsv_data)[:, 1]
sizes_fromtsv = np.array(tsv_data)[:, 11]

sizes = np.zeros([200])
for i in range(200):
    str1 = glob.glob('../images/test_images/' + str(i + 1).zfill(5) + '*/*.jpg')
    str1 = str(str1[0])
    str2 = '.*'+str(i+1).zfill(5)+'_(.*)' +'\_.*'
    name = str(re.findall(str2, str1)[0])
    length = int((len(name)-1)*0.5)
    name = name[:length]
    print(name)
    index = np.where(names_fromtsv == name)[0]
    sizes[i] = sizes_fromtsv[index]

np.save('RDMs/RealWorldSizes.npy', sizes)

sizes = np.load('RDMs/RealWorldSizes.npy')

sorted_indices = np.argsort(sizes)
print(np.where(sorted_indices == 74)[0][0], np.where(sorted_indices == 105)[0][0], np.where(sorted_indices == 135)[0][0])
print(sizes[74], sizes[105], sizes[135])
print(sizes[74]-sizes[105], sizes[74]-sizes[135], sizes[105]-sizes[135])

rdm = np.zeros([200, 200])

for i in range(200):
    for j in range(200):
        if i > j:
            rdm[i, j] = abs(sizes[i] - sizes[j])
            rdm[j, i] = rdm[i, j]

np.save('RDMs/RealWorldSizeRDM.npy', rdm)

from neurora.rsa_plot import plot_rdm

plot_rdm(rdm, percentile=True)