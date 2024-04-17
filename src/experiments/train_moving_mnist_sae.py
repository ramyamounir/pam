import torch, torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import Subset
import os, json
import numpy as np
from tqdm import tqdm

import sys; sys.path.append('./')
from src.utils.exps import checkdir, set_seed, ImagesDatset
from src.datasets.sparse_autoencoder import SAE_Trainer


def main(save_base_dir):

    dataset = torchvision.datasets.MovingMNIST(root = 'data/m_mnist', split=None, download=True, transform=lambda x: ((x.float()/255.0)*2.0)-1.0)
    dataset = torch.utils.data.Subset(dataset, torch.randperm(len(dataset))[:10])

    imgs = torch.cat([im for im in dataset], dim=0)
    imgs_dataset = ImagesDatset(imgs)


    data_loader = torch.utils.data.DataLoader(imgs_dataset, batch_size=10, shuffle=True, num_workers=1)
    sae_trainer = SAE_Trainer(N_c=100, W=5, img_height=imgs.shape[-2], img_width=imgs.shape[-1], n_channels=imgs.shape[-3], data_loader=data_loader, save_base_dir=save_base_dir)
    sae_trainer.train(n_epochs=1000, log_images_every=10)

    # save data
    data_loader = torch.utils.data.DataLoader(imgs_dataset, batch_size=10, shuffle=False, num_workers=1)
    sae = sae_trainer.SAE
    sae.load_state_dict(torch.load(f'{save_base_dir}/weights.pth'))

    imgs, sdrs, recons = [], [], []
    for im, _ in data_loader:

        im = im.cuda()
        imgs.append(im)

        sdr = sae.encode(im)
        sdrs.append(sdr)

        recon = sae.decode(sdr)
        recons.append(recon)


    imgs = torch.cat(imgs, dim=0).reshape(10, 20, 1, 64, 64)
    sdrs = torch.cat(sdrs, dim=0).reshape(10, 20, 100)
    recons = torch.cat(recons, dim=0).reshape(10, 20, 1, 64, 64)

    data = dict(imgs = imgs, sdrs = sdrs, recons = recons)
    torch.save(data, f'{save_base_dir}/data.pth')

if __name__ == "__main__":

    save_base_dir = f'results/{os.path.splitext(os.path.basename(__file__))[0]}/run_001'
    assert checkdir(save_base_dir, careful=True), f'path {save_base_dir} exists'

    set_seed(0)
    main(save_base_dir=save_base_dir)



