import numpy as np
import pingouin as pg
import pandas as pd
from scipy.stats import spearmanr

llrdm = np.load('RDMs/LowLevelRDM.npy')
rlsizerdm = np.load('RDMs/RelativeSizeRDM.npy')
rwsizerdm = np.load('RDMs/RealWorldSizeRDM.npy')
rwdepthrdm = np.load('RDMs/RealWorldDepthRDM.npy')
llv = np.zeros([19900])
rlsizev = np.zeros([19900])
rwsizev = np.zeros([19900])
rwdepthv = np.zeros([19900])
index = 0
for i in range(200):
    for j in range(200):
        if i > j:
            llv[index] = llrdm[i, j]
            rlsizev[index] = rlsizerdm[i, j]
            rwsizev[index] = rwsizerdm[i, j]
            rwdepthv[index] = rwdepthrdm[i, j]
            index = index + 1

corrs = np.zeros([4, 10, 100])

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
        corrs[0, sub, t] = spearmanr(eegv, llv)[0]
        corrs[1, sub, t] = spearmanr(eegv, rlsizev)[0]
        corrs[2, sub, t] = spearmanr(eegv, rwsizev)[0]
        corrs[3, sub, t] = spearmanr(eegv, rwdepthv)[0]

np.save('EEG_corrs.npy', corrs)

corrs = np.zeros([4, 10, 1000, 100])

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
            corrs[0, sub, k, t] = spearmanr(eegv, llv)[0]
            corrs[1, sub, k, t] = spearmanr(eegv, rlsizev)[0]
            corrs[2, sub, k, t] = spearmanr(eegv, rwsizev)[0]
            corrs[3, sub, k, t] = spearmanr(eegv, rwdepthv)[0]

np.save('EEG_corrs_null.npy', corrs)