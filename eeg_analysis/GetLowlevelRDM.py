import numpy as np
from scipy.stats import pearsonr
from PIL import Image, ImageOps

rdm = np.zeros([200, 200])

for i in range(200):
    print(i)
    for j in range(200):
        if i > j:
            img1 = np.array(Image.open('../images/test_images/renamed/' + str(i) + '.jpg'))
            img2 = np.array(Image.open('../images/test_images/renamed/' + str(j) + '.jpg'))
            rdm[i, j] = 1 - pearsonr(np.reshape(img1, [750000]), np.reshape(img2, [750000]))[0]
            rdm[j, i] = rdm[i, j]

np.save('RDMs/LowLevelRDM.npy', rdm)
print(rdm[74, 105], rdm[74, 135], rdm[105, 135])

rdm = np.zeros([200, 200])

for i in range(200):
    print(i)
    for j in range(200):
        if i > j:
            img1 = np.array(Image.open('../images/test_images/renamed/' + str(i) + 'co.jpg'))
            img2 = np.array(Image.open('../images/test_images/renamed/' + str(j) + 'co.jpg'))
            rdm[i, j] = 1 - pearsonr(np.reshape(img1, [750000]), np.reshape(img2, [750000]))[0]
            rdm[j, i] = rdm[i, j]

np.save('RDMs/LowLevelRDM_objonly.npy', rdm)

rdm = np.zeros([200, 200])

for i in range(200):
    print(i)
    for j in range(200):
        if i > j:
            img1 = np.array(Image.open('../images/test_images/renamed/' + str(i) + 'noo.jpg'))
            img2 = np.array(Image.open('../images/test_images/renamed/' + str(j) + 'noo.jpg'))
            rdm[i, j] = 1 - pearsonr(np.reshape(img1, [750000]), np.reshape(img2, [750000]))[0]
            rdm[j, i] = rdm[i, j]

np.save('RDMs/LowLevelRDM_bgonly.npy', rdm)