import torch
import numpy as np
from thingsvision import get_extractor
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader
from thingsvision.core.rsa import compute_rdm, correlate_rdms

model_name = 'OpenCLIP'
source = 'custom'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_parameters = {
    'variant' : 'RN101',
    'dataset' : 'yfcc15m'
}

extractor = get_extractor(
  model_name=model_name,
  source=source,
  device=device,
  pretrained=True,
  model_parameters=model_parameters
)

info = extractor.show_model()
print(info)

root='../../THINGS_EEG2/images/test_images/forloading'
batch_size = 32

dataset = ImageDataset(
  root=root,
  out_path='clip_resnet_features',
  backend=extractor.get_backend(),
  transforms=extractor.get_transformations()
)

batches = DataLoader(
  dataset=dataset,
  batch_size=batch_size,
  backend=extractor.get_backend()
)

module_names = ['visual.avgpool', 'visual.layer1', 'visual.layer2', 'visual.layer3',
                'visual.layer4', 'visual.attnpool']

layeri = 0
for module_name in module_names:
    features = extractor.extract_features(
        batches=batches,
        module_name=module_name,
        flatten_acts=True  # flatten 2D feature maps from convolutional layer
    )

    layeri += 1

    save_features(features, out_path='clip_resnet_features/layer' + str(layeri),
                  file_format='npy')

    features = np.load('clip_resnet_features/layer'+str(layeri)+'/features.npy')
    print(features)

    rdm_dnn = compute_rdm(features, method='correlation')

    np.save('ann_rdms/clip_resnet_layer'+str(layeri)+'.npy', rdm_dnn)