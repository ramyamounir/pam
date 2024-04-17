import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import Subset

from tqdm import tqdm
from tbw import TBWrapper, TBType
import os, math
from collections import deque
import matplotlib.pyplot as plt

import sys; sys.path.append('./')


class CNNEncoder(nn.Module):
    def __init__(self, N_c, img_height, img_width, n_channels=3):
        super().__init__()
        self.N_c = N_c
        self.img_height = img_height
        self.img_width = img_width
        self.n_channels = n_channels

        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(128 * int(img_height/8) * int(img_width/8), N_c)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * int(self.img_height/8) * int(self.img_width/8))
        x = F.tanh(self.fc(x))
        return x

class CNNDecoder(nn.Module):
    def __init__(self, N_c, img_height, img_width, n_channels=3):
        super().__init__()
        self.N_c = N_c
        self.img_height = img_height
        self.img_width = img_width
        self.n_channels = n_channels

        self.fc = nn.Linear(N_c, 128 * int(img_height/8) * int(img_width/8))  # Fully connected layer to upscale the feature vector
        self.conv_transpose1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # Upsampling layer 1
        self.conv_transpose2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  # Upsampling layer 2
        self.conv_transpose3 = nn.ConvTranspose2d(32, n_channels, kernel_size=3, stride=2, padding=1, output_padding=1)   # Upsampling layer 3

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, int(self.img_height/8), int(self.img_width/8))
        x = F.relu(self.conv_transpose1(x))
        x = F.relu(self.conv_transpose2(x))
        x = self.conv_transpose3(x)
        x = F.tanh(x) # maybe removeeeeeeeeee
        return x

class SparseAutoEncoder(nn.Module):
    def __init__(self, N_c, W, img_height, img_width, n_channels):
        super().__init__()
        self.encoder = CNNEncoder(N_c, img_height, img_width, n_channels)
        self.decoder = CNNDecoder(N_c, img_height, img_width, n_channels)
        self.W = W
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-4)

    def optimize(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def forward(self, imgs, strength):
        emb = self.encoder(imgs)
        emb_bin = self.binarize(emb, strength=strength)
        recon = self.decoder(emb_bin)

        recon_loss = F.mse_loss(imgs, recon)
        emb_loss = F.mse_loss(emb, emb_bin)
        self.optimize(0.5*recon_loss + 0.5*emb_loss)
        return emb_bin, recon, recon_loss, emb_loss

    def binarize(self, emb, strength):

        topk_indices = torch.topk(emb, k=self.W, dim=1)[1]
        mask = torch.full_like(emb, -1)
        mask.scatter_(1, topk_indices, 1.0)
        new_emb = emb - emb + mask
        emb = strength*new_emb + (1-strength)*emb
        return emb 

    def encode(self, img):
        with torch.no_grad():
            emb = self.encoder(img)
            emb_bin = self.binarize(emb, strength=1.0)

        return emb_bin 

    def decode(self, sdr):
        with torch.no_grad():
            img = self.decoder(sdr)
        return img

class SAE_Trainer():
    def __init__(self, N_c, W, img_height, img_width, n_channels, data_loader, save_base_dir):
        self.save_base_dir = save_base_dir
        self.data_loader = data_loader
        self.create_writers()

        self.SAE = SparseAutoEncoder(N_c=N_c, W=W, img_height=img_height, img_width=img_width, n_channels=n_channels).cuda()

    def create_writers(self):
        self.writer = TBWrapper(os.path.join(self.save_base_dir, 'logs'))

        self.writer(TBType.SCALAR, 'train/recon_loss')
        self.writer(TBType.SCALAR, 'train/emb_loss')
        self.writer(TBType.SCALAR, 'train/bin_sum')
        self.writer(TBType.SCALAR, 'train/bin_str')
        self.writer(TBType.IMAGE, 'train/bin_imgs')

        self.writer(TBType.SCALAR, 'test/infer_loss')
        self.writer(TBType.IMAGE, 'test/infer_imgs')


    def infer(self, log_images=False):

        counter = 0
        loss = 0.0
        for imgs, _ in self.data_loader:

            imgs = imgs.cuda()
            sdrs = self.SAE.encode(imgs)
            recon = self.SAE.decode(sdrs)

            loss += F.mse_loss(imgs, recon)

            counter += 1
            if log_images: 
                im, rec = (imgs[0]+1.0)/2.0, (recon[0]+1.0)/2.0
                self.writer['test/infer_imgs'](make_grid([im, rec], nrow=2))

        loss /= counter
        self.writer['test/infer_loss'](loss)

        return loss


    def train_one_epoch(self, strength, log_images_every=1000):

        counter = 0
        for imgs, _ in tqdm(self.data_loader):
            imgs = imgs.cuda()
            emb, recon, recon_loss, emb_loss = self.SAE(imgs, strength=strength)

            # logging
            self.writer['train/bin_str'](strength)
            self.writer['train/bin_sum'](((emb+1.0)/2.0).sum()/emb.shape[0])
            self.writer['train/recon_loss'](recon_loss)
            self.writer['train/emb_loss'](emb_loss)

            if counter==log_images_every:
                im, rec = (imgs[0]+1.0)/2.0, (recon[0]+1.0)/2.0
                self.writer['train/bin_imgs'](make_grid([im, rec], nrow=2))
                counter = 0
            else: counter += 1



    def train(self, n_epochs, log_images_every=1000):

        # train and save
        prev_inf_loss = math.inf
        for e in range(n_epochs):

            strength = torch.clamp(torch.tensor((e+1)/n_epochs), 0.0, 1.0)
            self.train_one_epoch(strength, log_images_every)

            inf_loss = self.infer(log_images=False)
            if inf_loss < prev_inf_loss:
                torch.save(self.SAE.state_dict(), os.path.join(self.save_base_dir, 'weights.pth'))
                prev_inf_loss = inf_loss

        # load and test
        self.SAE.load_state_dict(torch.load(os.path.join(self.save_base_dir, 'weights.pth')))
        inf_loss = self.infer(log_images=True)
        print(f"Inference Loss: {inf_loss}")




