import torch, torchvision
import torch.nn.functional as F
import os, json
import numpy as np
from tqdm import tqdm
from torchvision.utils import make_grid

import sys; sys.path.append('./')
from src.utils.sdr import SDR
from src.utils.exps import checkdir, set_seed
from src.datasets.binary import generate_video_SDR_from_data, add_noise_SDR_patterns
from src.datasets.sparse_autoencoder import SparseAutoEncoder



def compute(N, W, data_base_dir, noise, seed):

    # get data with seed
    set_seed(seed)
    data = torch.load(os.path.join(data_base_dir, 'data.pth'))
    imgs, sdrs, recons = generate_video_SDR_from_data(data, 2, seq_id=None)

    sae = SparseAutoEncoder(N_c=N, W=W, img_height=imgs[0].shape[-2], img_width=imgs[0].shape[-1], n_channels=imgs[0].shape[-3])
    sae.load_state_dict(torch.load(os.path.join(data_base_dir, 'weights.pth')))

    noisy_imgs = []
    for e in noise:
        noisy_sdrs = add_noise_SDR_patterns(sdrs, int(e*W))
        X_polar = torch.stack([x.bin.float() for x in noisy_sdrs])*2.0-1.0
        noisy_imgs.append(((sae.decode(X_polar[-1].unsqueeze(0))+1.0)/2.0))

    noisy_imgs = torch.cat(noisy_imgs, dim=0)
    noisy_rendered = make_grid((noisy_imgs*255.0).cpu().to(torch.uint8), nrow=1)
    return noisy_rendered




def main(save_base_dir, data_base_dir_5, data_base_dir_25, data_base_dir_50, N, seed):
    os.makedirs(save_base_dir, exist_ok=True)

    for W, data_base_dir in zip([5, 25, 50], [data_base_dir_5, data_base_dir_25, data_base_dir_50]):
        imgs = compute(N, W, data_base_dir, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], seed)
        torchvision.io.write_png(imgs, os.path.join(save_base_dir, f'{str(W).zfill(2)}_{str(seed).zfill(3)}.png'))




if __name__ == "__main__":


    save_base_dir = f'results/{os.path.splitext(os.path.basename(__file__))[0]}/run_001'
    assert checkdir(save_base_dir, careful=False), f'path {save_base_dir} exists'

    for i in range(10):
        main(save_base_dir=os.path.join(save_base_dir, 'clevrer'), 
             data_base_dir_5=f'results/train_clevrer_sae/run_001', 
             data_base_dir_25=f'results/train_clevrer_sae/run_003', 
             data_base_dir_50=f'results/train_clevrer_sae/run_002', 
             N=200,
             seed=i)

