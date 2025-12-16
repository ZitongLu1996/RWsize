import numpy as np
import pingouin as pg
import pandas as pd

indices = [4, 5, 27, 28, 29, 31, 47, 49, 59, 70,
           80, 84, 97, 100, 112, 113, 119, 121, 123, 124,
           125, 130, 131, 139, 146, 154, 157, 158, 161, 162,
           194, 195, 198]

llrdm = np.load('RDMs/LowLevelRDM.npy')
rlsizerdm = np.load('RDMs/RelativeSizeRDM.npy')
rwsizerdm = np.load('RDMs/RealWorldSizeRDM.npy')
rwdepthrdm = np.load('RDMs/RealWorldDepthRDM.npy')

llrdm = np.delete(llrdm, indices, axis=0)
llrdm = np.delete(llrdm, indices, axis=1)
rlsizerdm = np.delete(rlsizerdm, indices, axis=0)
rlsizerdm = np.delete(rlsizerdm, indices, axis=1)
rwsizerdm = np.delete(rwsizerdm, indices, axis=0)
rwsizerdm = np.delete(rwsizerdm, indices, axis=1)
rwdepthrdm = np.delete(rwdepthrdm, indices, axis=0)
rwdepthrdm = np.delete(rwdepthrdm, indices, axis=1)

llv = np.zeros([13861])
rlsizev = np.zeros([13861])
rwsizev = np.zeros([13861])
rwdepthv = np.zeros([13861])
index = 0
for i in range(167):
    for j in range(167):
        if i > j:
            llv[index] = llrdm[i, j]
            rlsizev[index] = rlsizerdm[i, j]
            rwsizev[index] = rwsizerdm[i, j]
            rwdepthv[index] = rwdepthrdm[i, j]
            index = index + 1

partial_corrs = np.zeros([4, 10, 100])

for sub in range(10):
    subEEGRDMs = np.load('RDMs/eegrdms_sub' + str(sub + 1).zfill(2) + '.npy')
    subEEGRDMs = np.delete(subEEGRDMs, indices, axis=1)
    subEEGRDMs = np.delete(subEEGRDMs, indices, axis=2)
    print(subEEGRDMs.shape)
    subEEGv = np.zeros([100, 13861])
    index = 0
    for i in range(167):
        for j in range(167):
            if i > j:
                subEEGv[:, index] = subEEGRDMs[:, i, j]
                index = index + 1
    for t in range(100):
        eegv = subEEGv[t]
        data = {'eeg': eegv,
                'lowlevel': llv,
                'rlsize': rlsizev,
                'rwsize': rwsizev,
                'rwdepth': rwdepthv,}
        df = pd.DataFrame(data, columns=['eeg', 'lowlevel', 'rlsize', 'rwsize', 'rwdepth'])
        stats = pg.partial_corr(data=df, x='lowlevel', y='eeg', covar=['rlsize', 'rwsize', 'rwdepth'], alternative='greater', method='spearman')
        partial_corrs[0, sub, t] = stats['r'][0]
        stats = pg.partial_corr(data=df, x='rlsize', y='eeg', covar=['lowlevel', 'rwsize', 'rwdepth'], alternative='greater', method='spearman')
        partial_corrs[1, sub, t] = stats['r'][0]
        stats = pg.partial_corr(data=df, x='rwsize', y='eeg', covar=['lowlevel', 'rlsize', 'rwdepth'], alternative='greater', method='spearman')
        partial_corrs[2, sub, t] = stats['r'][0]
        stats = pg.partial_corr(data=df, x='rwdepth', y='eeg', covar=['lowlevel', 'rlsize', 'rwsize'], alternative='greater', method='spearman')
        partial_corrs[3, sub, t] = stats['r'][0]

np.save('EEG_partial_corrs_new_SingleObj.npy', partial_corrs)