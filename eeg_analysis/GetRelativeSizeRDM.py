import numpy as np

from PIL import Image

sizes = np.zeros([200])
for i in range(200):
    img = Image.open('../images/test_images/obj/' + str(i) + '.png')
    x, y = np.array(img).shape[:2]
    sizes[i] = np.sqrt(x*x + y*y)
    print(sizes[i])

np.save('RDMs/RelativeSizes.npy', sizes)

sizes = np.load('RDMs/RelativeSizes.npy')

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

np.save('RDMs/RelativeSizeRDM.npy', rdm)

from neurora.rsa_plot import plot_rdm

plot_rdm(rdm, percentile=True)