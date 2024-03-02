import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from tbw import TBWrapper, TBType

import sys;sys.path.append('..')
from basics import SDR

# Define data transformations to apply
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image tensors
])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
images = torch.stack([dataset[i][0] for i in range(10)])
embs = [SDR(512, S=10) for _ in range(10)]

def get_sdr(embs, ix, noise=0):
    return torch.stack([embs[i].add_noise(n=noise).bin for i in ix])


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(512, 128 * 4 * 4)  # Fully connected layer to upscale the feature vector
        self.conv_transpose1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # Upsampling layer 1
        self.conv_transpose2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # Upsampling layer 2
        self.conv_transpose3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)   # Upsampling layer 3

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 4, 4)  # Reshape to (batch_size, channels, height, width)
        x = torch.relu(self.conv_transpose1(x))
        x = torch.relu(self.conv_transpose2(x))
        x = torch.tanh(self.conv_transpose3(x))  # Apply tanh activation for output in range [-1, 1]
        return x

model = Decoder()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
writer = TBWrapper('logs/decoder')

writer(TBType.SCALAR, 'loss')
writer(TBType.SCALAR, 'loss_noisy')

writer(TBType.IMAGE, 'images')
writer(TBType.IMAGE, 'images_noisy')


def test():
    labels = images

    inputs = get_sdr(embs, torch.arange(10), noise=0).float()
    preds = model(inputs)
    loss = F.mse_loss(preds, labels)
    writer['loss'](loss)
    writer['images'](torchvision.transforms.functional.resize(torchvision.utils.make_grid(torch.cat([preds,labels]), nrow=10), size=128))


    inputs = get_sdr(embs, torch.arange(10), noise=100).float()
    preds = model(inputs)
    loss = F.mse_loss(preds, labels)
    writer['loss_noisy'](loss)
    writer['images_noisy'](torchvision.transforms.functional.resize(torchvision.utils.make_grid(torch.cat([model(inputs),labels]), nrow=10), size=128))



# train
for i in range(5000):
    batch_ix = torch.randperm(10)[:5]

    inputs = get_sdr(embs, batch_ix).float()
    labels = images[batch_ix]

    preds = model(inputs)
    loss = F.mse_loss(preds, labels)

    optim.zero_grad()
    loss.backward()
    optim.step()

    if i%100 == 0:
        test()











