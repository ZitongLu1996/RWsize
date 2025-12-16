import numpy as np
import pingouin as pg
import pandas as pd

earlyrdm = np.load('ann_rdms/resnet_early.npy')
laterdm = np.load('ann_rdms/resnet_late.npy')
earlyv = np.zeros([19900])
latev = np.zeros([19900])
index = 0
for i in range(200):
    for j in range(200):
        if i > j:
            earlyv[index] = earlyrdm[i, j]
            latev[index] = laterdm[i, j]
            index += 1

partial_corrs = np.zeros([2, 10, 100])

for sub in range(10):
    subEEGRDMs = np.load('../depth_analysis/RDMs/eegrdms_sub' + str(sub + 1).zfill(2) + '.npy')
    print(subEEGRDMs.shape)
    subEEGv = np.zeros([100, 19900])
    index = 0
    for i in range(200):
        for j in range(200):
            if i > j:
                subEEGv[:, index] = subEEGRDMs[:, i, j]
                index = index + 1
    for t in range(100):
        eegv = subEEGv[t]
        data = {'eeg': eegv,
                'early': earlyv,
                'late': latev,}
        df = pd.DataFrame(data, columns=['eeg', 'early', 'late'])
        stats = pg.partial_corr(data=df, x='early', y='eeg', covar=['late'], alternative='greater', method='spearman')
        partial_corrs[0, sub, t] = stats['r'][0]
        stats = pg.partial_corr(data=df, x='late', y='eeg', covar=['early'], alternative='greater', method='spearman')
        partial_corrs[1, sub, t] = stats['r'][0]

np.save('eeg_resnet_partial_corrs.npy', partial_corrs)

earlyrdm = np.load('ann_rdms/clip_resnet_early.npy')
laterdm = np.load('ann_rdms/clip_resnet_late.npy')
earlyv = np.zeros([19900])
latev = np.zeros([19900])
index = 0
for i in range(200):
    for j in range(200):
        if i > j:
            earlyv[index] = earlyrdm[i, j]
            latev[index] = laterdm[i, j]
            index += 1

partial_corrs = np.zeros([2, 10, 100])

for sub in range(10):
    subEEGRDMs = np.load('../depth_analysis/RDMs/eegrdms_sub' + str(sub + 1).zfill(2) + '.npy')
    print(subEEGRDMs.shape)
    subEEGv = np.zeros([100, 19900])
    index = 0
    for i in range(200):
        for j in range(200):
            if i > j:
                subEEGv[:, index] = subEEGRDMs[:, i, j]
                index = index + 1
    for t in range(100):
        eegv = subEEGv[t]
        data = {'eeg': eegv,
                'early': earlyv,
                'late': latev,}
        df = pd.DataFrame(data, columns=['eeg', 'early', 'late'])
        stats = pg.partial_corr(data=df, x='early', y='eeg', covar=['late'], alternative='greater', method='spearman')
        partial_corrs[0, sub, t] = stats['r'][0]
        stats = pg.partial_corr(data=df, x='late', y='eeg', covar=['early'], alternative='greater', method='spearman')
        partial_corrs[1, sub, t] = stats['r'][0]

np.save('eeg_clip_partial_corrs.npy', partial_corrs)