import numpy as np

from PIL import Image

rel_sizes = np.load('RDMs/RelativeSizes.npy')
rw_sizes = np.load('RDMs/RealWorldSizes.npy')
depths = rw_sizes/rel_sizes

np.save('RDMs/RealWorldDepths.npy', depths)

depths = np.load('RDMs/RealWorldDepths.npy')

sorted_indices = np.argsort(depths)
print(sorted_indices)
print(np.where(sorted_indices == 74)[0][0], np.where(sorted_indices == 105)[0][0], np.where(sorted_indices == 135)[0][0])
print(depths[74], depths[105], depths[135])
print(depths[74]-depths[105], depths[74]-depths[135], depths[105]-depths[135])

rdm = np.zeros([200, 200])

for i in range(200):
    for j in range(200):
        if i > j:
            rdm[i, j] = abs(depths[i] - depths[j])
            rdm[j, i] = rdm[i, j]

np.save('RDMs/RealWorldDepthRDM.npy', rdm)

from neurora.rsa_plot import plot_rdm

plot_rdm(rdm, percentile=True)