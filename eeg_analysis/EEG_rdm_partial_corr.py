import numpy as np
import pingouin as pg
import pandas as pd

"""#llrdm = np.load('RDMs/LowLevelRDM.npy')
rlsizerdm = np.load('RDMs/RelativeSizeRDM.npy')
rwsizerdm = np.load('RDMs/RealWorldSizeRDM.npy')
rwdepthrdm = np.load('RDMs/RealWorldDepthRDM.npy')
#llv = np.zeros([19900])
rlsizev = np.zeros([19900])
rwsizev = np.zeros([19900])
rwdepthv = np.zeros([19900])
index = 0
for i in range(200):
    for j in range(200):
        if i > j:
            #llv[index] = llrdm[i, j]
            rlsizev[index] = rlsizerdm[i, j]
            rwsizev[index] = rwsizerdm[i, j]
            rwdepthv[index] = rwdepthrdm[i, j]
            index = index + 1

partial_corrs = np.zeros([3, 10, 100])

for sub in range(10):
    subEEGRDMs = np.load('RDMs/eegrdms_sub' + str(sub + 1).zfill(2) + '.npy')
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
                #'lowlevel': llv,
                'rlsize': rlsizev,
                'rwsize': rwsizev,
                'rwdepth': rwdepthv,}
        df = pd.DataFrame(data, columns=['eeg', 'rlsize', 'rwsize', 'rwdepth'])
        stats = pg.partial_corr(data=df, x='rlsize', y='eeg', x_covar=['rwsize', 'rwdepth'], alternative='greater', method='spearman')
        partial_corrs[0, sub, t] = stats['r'][0]
        stats = pg.partial_corr(data=df, x='rwsize', y='eeg', x_covar=['rlsize', 'rwdepth'], alternative='greater', method='spearman')
        partial_corrs[1, sub, t] = stats['r'][0]
        stats = pg.partial_corr(data=df, x='rwdepth', y='eeg', x_covar=['rlsize', 'rwsize'], alternative='greater', method='spearman')
        partial_corrs[2, sub, t] = stats['r'][0]

np.save('EEG_partial_corrs.npy', partial_corrs)"""

rlsizerdm = np.load('RDMs/RelativeSizeRDM.npy')
rwsizerdm = np.load('RDMs/RealWorldSizeRDM.npy')
rwdepthrdm = np.load('RDMs/RealWorldDepthRDM.npy')
semanticrdm = np.load('../ann_comparisons/ann_rdms/w2v.npy')
#llv = np.zeros([19900])
rlsizev = np.zeros([19900])
rwsizev = np.zeros([19900])
rwdepthv = np.zeros([19900])
semanticv = np.zeros([19900])
index = 0
for i in range(200):
    for j in range(200):
        if i > j:
            #llv[index] = llrdm[i, j]
            rlsizev[index] = rlsizerdm[i, j]
            rwsizev[index] = rwsizerdm[i, j]
            rwdepthv[index] = rwdepthrdm[i, j]
            semanticv[index] = semanticrdm[i, j]
            index = index + 1

partial_corrs = np.zeros([4, 10, 100])

for sub in range(10):
    subEEGRDMs = np.load('RDMs/eegrdms_sub' + str(sub + 1).zfill(2) + '.npy')
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
                #'lowlevel': llv,
                'rlsize': rlsizev,
                'rwsize': rwsizev,
                'rwdepth': rwdepthv,
                'semantic': semanticv,}
        df = pd.DataFrame(data, columns=['eeg', 'rlsize', 'rwsize', 'rwdepth', 'semantic'])
        stats = pg.partial_corr(data=df, x='rlsize', y='eeg', x_covar=['rwsize', 'rwdepth', 'semantic'], alternative='greater', method='spearman')
        partial_corrs[0, sub, t] = stats['r'][0]
        stats = pg.partial_corr(data=df, x='rwsize', y='eeg', x_covar=['rlsize', 'rwdepth', 'semantic'], alternative='greater', method='spearman')
        partial_corrs[1, sub, t] = stats['r'][0]
        stats = pg.partial_corr(data=df, x='rwdepth', y='eeg', x_covar=['rlsize', 'rwsize', 'semantic'], alternative='greater', method='spearman')
        partial_corrs[2, sub, t] = stats['r'][0]
        stats = pg.partial_corr(data=df, x='semantic', y='eeg', x_covar=['rlsize', 'rwsize', 'rwdepth'], alternative='greater', method='spearman')
        partial_corrs[3, sub, t] = stats['r'][0]

np.save('EEG_partial_corrs_new_withsemantic.npy', partial_corrs)

"""partial_corrs = np.zeros([4, 10, 1000, 100])

for sub in range(10):
    subEEGRDMs = np.load('RDMs/eegrdms_sub' + str(sub + 1).zfill(2) + '.npy')
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
        for k in range(1000):
            np.random.shuffle(eegv)
            data = {'eeg': eegv,
                    'lowlevel': llv,
                    'rlsize': rlsizev,
                    'rwsize': rwsizev,
                    'rwdepth': rwdepthv,}
            df = pd.DataFrame(data, columns=['eeg', 'lowlevel', 'rlsize', 'rwsize', 'rwdepth'])
            stats = pg.partial_corr(data=df, x='lowlevel', y='eeg', covar=['rlsize', 'rwsize', 'rwdepth'], alternative='greater', method='spearman')
            partial_corrs[0, sub, k, t] = stats['r'][0]
            stats = pg.partial_corr(data=df, x='rlsize', y='eeg', covar=['lowlevel', 'rwsize', 'rwdepth'], alternative='greater', method='spearman')
            partial_corrs[1, sub, k, t] = stats['r'][0]
            stats = pg.partial_corr(data=df, x='rwsize', y='eeg', covar=['lowlevel', 'rlsize', 'rwdepth'], alternative='greater', method='spearman')
            partial_corrs[2, sub, k, t] = stats['r'][0]
            stats = pg.partial_corr(data=df, x='rwdepth', y='eeg', covar=['lowlevel', 'rlsize', 'rwsize'], alternative='greater', method='spearman')
            partial_corrs[3, sub, k, t] = stats['r'][0]

np.save('EEG_partial_corrs_null.npy', partial_corrs)"""