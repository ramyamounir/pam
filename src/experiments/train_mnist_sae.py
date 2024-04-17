import torch, torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import Subset
import os, json
import numpy as np
from tqdm import tqdm

import sys; sys.path.append('./')
from src.utils.exps import checkdir, set_seed
from src.datasets.sparse_autoencoder import SAE_Trainer


def main(save_base_dir):

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32,32)), lambda x:(x*2.0)-1.0])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataset = torch.utils.data.Subset(dataset, torch.randperm(len(dataset))[:100])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)

    sae_trainer = SAE_Trainer(N_c=100, W=5, img_height=32, img_width=32, n_channels=1, data_loader=data_loader, save_base_dir=save_base_dir)
    sae_trainer.train(n_epochs=1000, log_images_every=2)

    # save data
    sae = sae_trainer.SAE
    sae.load_state_dict(torch.load(f'{save_base_dir}/weights.pth'))

    imgs, sdrs, recons = [], [], []
    for im, _ in data_loader:
        imgs.append(im)

        im = im.cuda()
        sdr = sae.encode(im)
        recon = sae.decode(sdr)

        sdrs.append(sdr)
        recons.append(recon)


    imgs = torch.cat(imgs, dim=0)
    sdrs = torch.cat(sdrs, dim=0)
    recons = torch.cat(recons, dim=0)

    data = dict(imgs = imgs, sdrs = sdrs, recons = recons)
    torch.save(data, f'{save_base_dir}/data.pth')


if __name__ == "__main__":

    save_base_dir = f'results/{os.path.splitext(os.path.basename(__file__))[0]}/run_001'
    assert checkdir(save_base_dir, careful=True), f'path {save_base_dir} exists'

    set_seed(0)
    main(save_base_dir=save_base_dir)



