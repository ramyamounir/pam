import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import sys; sys.path.append('./')
from src.utils.sdr import SDR
from src.models.pam import PamModel
from src.models.mcahn import ModernAsymmetricHopfieldNetwork
from src.models.single_tpc import SingleLayertPC
from src.utils.exps import accuracy_SDR, accuracy_POLAR

resize_shape = (64,64)
N_c = resize_shape[0]*resize_shape[1]

def imgs2sdr(imgs):
    sdrs = []
    for img in imgs[::2]:
        img = img.reshape(-1)
        sdrs.append(SDR(N=img.shape[0], ix=torch.where(img>128)[0]))
    return sdrs


def plot_binary_imgs(sdrs):
    fig, axes = plt.subplots(1, len(sdrs), figsize=(10,2))
    for i, sdr in enumerate(sdrs):
        img = sdr.bin.float().reshape(resize_shape)
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    plt.subplots_adjust(wspace=0)
    plt.show()



def get_pam_model():

    transition_configs = {
            'synaptic_density': 1.0,
            'eta_inc': 0.1,
            'eta_dec': 0.0,
            'eta_decay': 0.0,
            'init_mean': -0.8,
            'init_std': 0.1,
            'threshold': 2.5,
            }

    emission_configs = {
            'synaptic_density': 1.0,
            'eta_inc': 0.1,
            'eta_dec': 0.1,
            'eta_decay': 0.0,
            'init_mean': 0.0,
            'init_std': 0.1,
            'threshold': 0.25,
            }

    return PamModel(N_c=N_c, N_k=8, W=int(N_c*0.05), transition_configs=transition_configs, emission_configs=emission_configs)

def get_hn_model(d):

    configs = {
            'data_type': 'binary',
            'sep': d,
            }

    return ModernAsymmetricHopfieldNetwork(input_size=N_c, **configs)


def get_pc_model():

    configs = {
            'lr': 1e-4 ,
            'data_type': 'binary',
            'learn_iters': 800,
            }

    return SingleLayertPC(input_size=N_c, **configs)


dataset = torchvision.datasets.MovingMNIST(root = 'data/m_mnist', split=None, download=True, transform=torchvision.transforms.Resize(resize_shape))
seq1 = imgs2sdr(dataset[0])
seq1_polar = torch.stack([x.bin.float() for x in seq1])*2.0-1.0

pc_model = get_pc_model()
pc_model.train_seq(seq1_polar)
recall = pc_model.recall_seq(seq1_polar, query='offline')
iou1 = accuracy_POLAR(seq1_polar[1:], recall[1:])
print(iou1)

# hn_model = get_hn_model(d=1)
# recall = hn_model.recall_seq(seq1_polar, query='offline')
# iou1 = accuracy_POLAR(seq1_polar[1:], recall[1:])
# print(iou1)


# pm.learn_sequence(seq1)

# rec1 = pm.generate_sequence_offline(seq1[0], len(seq1)-1)
# plot_binary_imgs(rec1)

# iou1 = accuracy_SDR(seq1[1:], rec1)

# print(iou1)




