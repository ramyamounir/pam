import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import Subset

from tqdm import tqdm
from tbw import TBWrapper, TBType
import os
from collections import deque

import sys; sys.path.append('./')
from src.experiments.utils import checkdir


class CNNEncoder(nn.Module):
    def __init__(self, dim, n_channels=3):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(128 * 4 * 4, dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 4 * 4)
        x = F.tanh(self.fc(x))
        return x

class CNNDecoder(nn.Module):
    def __init__(self, dim, n_channels=3):
        super().__init__()
        self.fc = nn.Linear(dim, 128 * 4 * 4)  # Fully connected layer to upscale the feature vector
        self.conv_transpose1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0)  # Upsampling layer 1 ### 1 for CIFAR
        self.conv_transpose2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  # Upsampling layer 2
        self.conv_transpose3 = nn.ConvTranspose2d(32, n_channels, kernel_size=3, stride=2, padding=1, output_padding=1)   # Upsampling layer 3

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 4, 4)
        x = F.relu(self.conv_transpose1(x))
        x = F.relu(self.conv_transpose2(x))
        x = self.conv_transpose3(x)
        return x

class BinAutoEncoder(nn.Module):
    def __init__(self, dim, sparsity, n_channels=1):
        super().__init__()
        self.encoder = CNNEncoder(dim, n_channels)
        self.decoder = CNNDecoder(dim, n_channels)
        self.sparsity = sparsity
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-4)

    def optimize(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def forward(self, imgs, bin_strength):
        emb = self.encoder(imgs)
        emb_bin = self.binarize(emb, BS=bin_strength)
        recon = self.decoder(emb_bin)

        loss = F.mse_loss(imgs, recon)
        self.optimize(loss)

        return emb_bin, loss, recon

    def binarize(self, emb, BS):
        inv_mask = 1.0/emb

        wh_ix = torch.topk(emb, k=emb.shape[-1]-self.sparsity, dim=-1, largest=False).indices
        for wx, wh in enumerate(wh_ix):
            inv_mask[wx][wh] *= 0.0

        new_emb = emb*inv_mask
        emb = BS*new_emb + (1-BS)*emb
        return emb 

    def encode(self, img):
        with torch.no_grad():
            emb = self.encoder(img)
            emb_bin = self.binarize(emb, BS=1.0)

        return emb_bin 

    def decode(self, sdr):
        with torch.no_grad():
            img = self.decoder(sdr)
        return img



class MNISTae():
    def __init__(self, save_base_dir):
        self.save_base_dir = save_base_dir

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        # trainset = Subset(trainset, range(1000))
        self.trainloader_dense = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=1)

        # self.active_duty_cycle = deque(maxlen=1000)


        self.binAutoEncoder = BinAutoEncoder(dim=512, sparsity=10, n_channels=1).cuda()
        self.create_writers()

    def create_writers(self):
        self.writer = TBWrapper(os.path.join(self.save_base_dir, 'logs'))
        self.writer(TBType.SCALAR, 'bin_loss')
        self.writer(TBType.SCALAR, 'bin_sum')
        self.writer(TBType.SCALAR, 'bin_str')
        # self.writer(TBType.HISTOGRAM, 'bin_hist')
        self.writer(TBType.IMAGE, 'bin_imgs')
        self.writer(TBType.IMAGE, 'bin_infer_imgs')


        self.writer(TBType.SCALAR, 'infer_loss')


    def train_bin(self, n_epochs):

        counter = 0
        for e in range(n_epochs):
            for imgs, _ in tqdm(self.trainloader_dense):

                imgs = imgs.cuda()

                # warmup_epochs = 20
                # bs = torch.clamp(1-torch.exp(torch.tensor(-10.0*counter/((warmup_epochs*len(self.trainloader_dense))-1))), 0.0, 1.0)
                # if e == warmup_epochs-1: bs = 1.0

                bs = torch.clamp(torch.tensor((e+1)/n_epochs), 0.0, 1.0)
                bs = 1.0

                enc, loss, recon = self.binAutoEncoder(imgs, bin_strength=bs)

                # logging
                # self.active_duty_cycle.append(new_enc)
                # self.writer['bin_hist'](torch.stack(list(self.active_duty_cycle)).mean(0))
                self.writer['bin_loss'](loss)
                self.writer['bin_str'](bs)
                self.writer['bin_sum'](enc.sum())
                if counter%1000==0: self.writer['bin_imgs'](make_grid([imgs[0], recon[0]], nrow=2))
                counter += 1

            self.simple_infer()

    def infer_bin(self):
        
        counter = 0
        for imgs, _ in tqdm(self.trainloader_dense):

            imgs = imgs.cuda()
            sdrs = self.binAutoEncoder.encode(imgs)
            recon = self.binAutoEncoder.decode(sdrs)

            # logging
            self.writer['bin_loss'](F.mse_loss(recon, imgs))
            self.writer['bin_sum'](sdrs.sum())
            if counter%10==0: 
                self.writer['bin_infer_imgs'](make_grid([imgs[0], recon[0]], nrow=2))
            counter += 1

    def simple_infer(self):

        counter = 0
        loss = 0.0
        for imgs, _ in tqdm(self.trainloader_dense):

            imgs = imgs.cuda()
            sdrs = self.binAutoEncoder.encode(imgs)
            recon = self.binAutoEncoder.decode(sdrs)

            loss += F.mse_loss(imgs, recon)
            counter += 1

        self.writer['infer_loss'](loss/counter)


    def train(self):
        self.train_bin(n_epochs=10)
        self.infer_bin()
        self.save()

    def save(self):
        torch.save(self.binAutoEncoder.state_dict(), os.path.join(self.save_base_dir, 'weights.pth'))
        


if __name__ == "__main__":

    save_base_dir = f'results/{os.path.splitext(os.path.basename(__file__))[0]}/run_001'
    assert checkdir(save_base_dir, careful=False), f'path {save_base_dir} exists'

    ae = MNISTae(save_base_dir)
    ae.train()


