import torch
from thingsvision import get_extractor
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader

model_name = 'resnet101'
source = 'torchvision'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

extractor = get_extractor(
    model_name = model_name,
    source = source,
    device = device,
    pretrained = True
)

root='../images/test_images/forloading_objectonly'
batch_size = 32

dataset = ImageDataset(
  root=root,
  out_path='resnet_features_objonly',
  backend=extractor.get_backend(),
  transforms=extractor.get_transformations()
)

batches = DataLoader(
  dataset=dataset,
  batch_size=batch_size,
  backend=extractor.get_backend()
)

info = extractor.show_model()
print(info)

module_name = 'avgpool'

features = extractor.extract_features(
  batches=batches,
  module_name=module_name,
  flatten_acts=True  # flatten 2D feature maps from convolutional layer
)

save_features(features, out_path='resnet_late_features_objonly', file_format='npy')

import numpy as np

features = np.load('resnet_late_features_objonly/features.npy')
print(features)

from thingsvision.core.rsa import compute_rdm, correlate_rdms

rdm_dnn = compute_rdm(features, method='correlation')

print(rdm_dnn.shape)
print(rdm_dnn)

np.save('ann_rdms/resnet_late_objonly.npy', rdm_dnn)

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


module_name = 'maxpool'

features = extractor.extract_features(
  batches=batches,
  module_name=module_name,
  flatten_acts=True  # flatten 2D feature maps from convolutional layer
)

save_features(features, out_path='resnet_early_features_objonly', file_format='npy')

import numpy as np

features = np.load('resnet_early_features_objonly/features.npy')
print(features)

from thingsvision.core.rsa import compute_rdm, correlate_rdms

rdm_dnn = compute_rdm(features, method='correlation')

print(rdm_dnn.shape)
print(rdm_dnn)

np.save('ann_rdms/resnet_early_objonly.npy', rdm_dnn)

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