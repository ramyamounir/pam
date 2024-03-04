import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from tbw import TBWrapper, TBType
from create_vocab import Vocab
from tqdm import tqdm


class CNNDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 128 * 4 * 4)  # Fully connected layer to upscale the feature vector
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

class Trainer():
    def __init__(self, embs):
        self.model = CNNDecoder(dim=embs.shape[-1])
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL Image to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image tensors
        ])
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

        self.embs = embs
        self.images = torch.stack([dataset[i][0] for i in range(len(embs))])
        self.create_writers()

    def create_writers(self):
        self.writer = TBWrapper('logs/decoder')
        self.writer(TBType.SCALAR, 'loss')
        self.writer(TBType.IMAGE, 'images')
        self.writer(TBType.IMAGE, 'images_noisy')

    def get_batch_ix(self, batch_size):
        shuffled_ix = torch.randperm(len(self.embs))
        for i in range(0, len(self.embs), batch_size):
            yield shuffled_ix[i:min(i+batch_size, len(self.embs))]

    def train(self, num_epochs, batch_size):
        for i in tqdm(range(num_epochs)):
            l, c = 0.0, 0
            for batch_ix in self.get_batch_ix(batch_size):

                inputs = self.embs[batch_ix]
                labels = self.images[batch_ix]

                preds = self.model(inputs)
                loss = F.mse_loss(preds, labels)
                l += loss; c += 1

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            self.writer['loss'](l/c)
            if i%10==0:
                self.visualize()


    def visualize(self):

        labels = self.images[:10]
        res = [labels]
        for std in torch.linspace(0.0, 1.0, 5):

            inputs = self.embs[:10] + torch.normal(mean=0, std=std, size=self.embs[:10].size())
            preds = self.model(inputs)
            res.append(preds)

        res = torch.cat(res)
        self.writer['images_noisy'](torchvision.transforms.functional.resize(torchvision.utils.make_grid(res, nrow=10), size=256*3))


class DecoderCIFAR():
    def __init__(self, vocab_path, num_points):
        self.vocab_path = vocab_path
        self.num_points = num_points

        self.embs = Vocab.load_from(vocab_path)[:num_points]
        self.trainer = Trainer(self.embs)

    def train(self, **kwargs):
        self.trainer.train(**kwargs)

    def decode(self, inp):
        decoded = self.trainer.model(inp)

    def save(self, path):
        to_save = dict(vocab_path=self.vocab_path, num_points=self.num_points)
        to_save['state_dict'] = self.trainer.model.state_dict()
        torch.save(to_save, path)

    @staticmethod
    def load_from(path):
        loaded = torch.load(path)
        d = DecoderCIFAR(vocab_path=loaded['vocab_path'], num_points=loaded['num_points'])
        d.trainer.model.load_state_dict(loaded['state_dict'])
        return d

if __name__ == "__main__":

    cifar_decoder = DecoderCIFAR('saved/sparse.pth', 100)
    cifar_decoder.train(num_epochs=100, batch_size=10)
    cifar_decoder.save('saved/cifar.pth')

    c = DecoderCIFAR.load_from('saved/cifar.pth')
    c.train(num_epochs=1, batch_size=10)


