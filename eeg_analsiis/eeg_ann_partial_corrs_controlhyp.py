import numpy as np
import pingouin as pg
import pandas as pd

anns = ['resnet_early', 'resnet_late', 'w2v', 'clip_resnet_early', 'clip_resnet_late', 'cornet_early', 'cornet_late']
corrs = np.zeros([7, 10, 100])
for k in range(7):
    rdm_dnn = np.load('ann_rdms/' + anns[k] + '.npy')

    rlsizerdm = np.load('../depth_analysis/RDMs/RelativeSizeRDM.npy')
    rwsizerdm = np.load('../depth_analysis/RDMs/RealWorldSizeRDM.npy')
    rwdepthrdm = np.load('../depth_analysis/RDMs/RealWorldDepthRDM.npy')
    rlsizev = np.zeros([19900])
    rwsizev = np.zeros([19900])
    rwdepthv = np.zeros([19900])
    dnnv = np.zeros([19900])
    index = 0
    for i in range(200):
        for j in range(200):
            if i > j:
                # llv[index] = llrdm[i, j]
                rlsizev[index] = rlsizerdm[i, j]
                rwsizev[index] = rwsizerdm[i, j]
                rwdepthv[index] = rwdepthrdm[i, j]
                dnnv[index] = rdm_dnn[i, j]
                index = index + 1

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
                    # 'lowlevel': llv,
                    'rlsize': rlsizev,
                    'rwsize': rwsizev,
                    'rwdepth': rwdepthv,
                    'dnn': dnnv, }
            df = pd.DataFrame(data, columns=['eeg', 'rlsize', 'rwsize', 'rwdepth', 'dnn'])
            stats = pg.partial_corr(data=df, x='dnn', y='eeg', x_covar=['rlsize', 'rwsize', 'rwdepth'],
                                    alternative='greater', method='spearman')
            corrs[k, sub, t] = stats['r'][0]

np.save('rsa_partial_corrs_controlhyp.npy', corrs)