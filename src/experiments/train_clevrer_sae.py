import torch, torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import Subset
import os, json, subprocess
import numpy as np
from tqdm import tqdm
from glob import glob

import sys; sys.path.append('./')
from src.utils.exps import checkdir, set_seed, ImagesDatset, video_extract
from src.datasets.sparse_autoencoder import SAE_Trainer

def extract_vids(vids, ixs, save_base_dir):
    print('extracting videos. . .')
    for ix in tqdm(ixs):
        video_input = vids[ix]
        video_output = f'{save_base_dir}/imgs/vid_{str(ix).zfill(3)}/'
        video_extract(video_input, video_output, 0.1, 192, 256, 'png')

def read_imgs(img_paths):
    imgs = torch.stack([torchvision.io.read_image(path) for path in img_paths])
    imgs = ((imgs/255.0)*2.0)-1.0
    return imgs

def main(save_base_dir):

    vids_path = glob('./data/clevrer/*/*.mp4')
    ixs = torch.randperm(10).numpy().tolist()
    extract_vids(vids_path, ixs, save_base_dir)

    full_imgs = read_imgs(glob(f'{save_base_dir}/imgs/vid_*/*.png'))
    imgs_dataset = ImagesDatset(full_imgs)
    data_loader = torch.utils.data.DataLoader(imgs_dataset, batch_size=10, shuffle=True, num_workers=1)

    sae_trainer = SAE_Trainer(N_c=200, W=100, img_height=full_imgs.shape[-2], img_width=full_imgs.shape[-1], n_channels=full_imgs.shape[-3], data_loader=data_loader, save_base_dir=save_base_dir)
    sae_trainer.train(n_epochs=100, log_images_every=10)

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


    imgs = torch.cat(imgs, dim=0).reshape(10, 51, 3, 192, 256)
    sdrs = torch.cat(sdrs, dim=0).reshape(10, 51, 200)
    recons = torch.cat(recons, dim=0).reshape(10, 51, 3, 192, 256)

    data = dict(imgs = imgs, sdrs = sdrs, recons = recons)
    torch.save(data, f'{save_base_dir}/data.pth')

if __name__ == "__main__":

    save_base_dir = f'results/{os.path.splitext(os.path.basename(__file__))[0]}/run_002'
    assert checkdir(save_base_dir, careful=False), f'path {save_base_dir} exists'

    set_seed(0)
    main(save_base_dir=save_base_dir)



