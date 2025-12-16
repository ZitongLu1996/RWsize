import numpy as np
import glob
import re
import pandas as pd
from gensim import models
import torch
import torch.nn as nn

word2vec_path = 'GoogleNews-vectors-negative300.bin.gz'
weights = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
#embedding = nn.Embedding.from_pretrained(weights)

tsv_data = pd.read_csv('../depth_analysis/size_meanRatings.tsv', sep='\t', header=0)
names_fromtsv = np.array(tsv_data)[:, 1]
sizes_fromtsv = np.array(tsv_data)[:, 11]


vs = np.zeros([200, 300])

sizes = np.zeros([200])
m = 0
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
    if name in weights.index_to_key:
        name = name
    else:
        name = 'unk'
        m += 1
    print(weights[name])
    vs[i] = weights[name]
    print(weights[name].shape)

np.save('w2v_features/features.npy', vs)

features = np.load('w2v_features/features.npy')
print(features)

from thingsvision.core.rsa import compute_rdm, correlate_rdms

rdm_dnn = compute_rdm(features, method='correlation')

print(rdm_dnn.shape)
print(rdm_dnn)

np.save('ann_rdms/w2v.npy', rdm_dnn)

from neurora.rsa_plot import plot_rdm

plot_rdm(rdm_dnn, percentile=True)

from neurora.corr_cal_by_rdm import rdms_corr

eegrdms = np.zeros([10, 100, 200, 200])
for sub in range(10):
    eegrdms[sub] = np.load('../depth_analysis/RDMs/eegrdms_sub' + str(sub + 1).zfill(2) + '.npy')

corrs = rdms_corr(rdm_dnn, eegrdms)

from neurora.rsa_plot import plot_tbytsim_withstats

plot_tbytsim_withstats(corrs, start_time=-0.2, end_time=0.8, time_interval=0.01, xlim=[-0.2, 0.8], cbpt=False,
                        ylim=[-0.05, 0.2], smooth=False, p=0.05, avgshow=True)