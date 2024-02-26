import torch
from dataset import TextDataset
from layers import Layer4

from tqdm import tqdm
from english_words import get_english_words_set

class Trainer():
    def __init__(self, D, K, S, T):
        self.D=D
        self.K=K
        self.S=S
        self.T=T

        self.td = TextDataset(dim=self.D, S=self.S)
        self.l4 = Layer4(num_base_neurons=self.D, num_neurons_per_minicolumn=self.K, sparsity=0.02, connections_density=0.5, connections_decay=1.0)
        self.parameters = ['td', 'l4']

    def inference(self, data):
        for d in data:
            for c in d:
                input_sdr = self.td.encode(c)
                results = self.l4(input_sdr=input_sdr, train=False, gen=True)

                print(f' "{c}" ->', self.td.decode(results['predicted'], overlap=self.T), end=' ')
                if results['generated'] != None: print(f'generated -> {self.td.decode(results["generated"], overlap=self.T)}', end=' ')
                if results['boundary']: print(' BOUNDARY')
                else: print('\n')

            print('\n')
            self.l4.reset()

    def train(self, data, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            for d in data:
                for c in d:
                    input_sdr = self.td.encode(c)
                    results = self.l4(input_sdr=input_sdr, train=True)

                self.l4.reset()

    def save(self, path=None):
        to_save = {}
        for p in self.parameters:
            attr = getattr(self, p)
            to_save[p] = attr if isinstance(attr, torch.Tensor) else attr.save(path=None)

        if path != None: torch.save(to_save, path)
        else: return to_save


    def load(self, path):
        parameters = torch.load(path) if isinstance(path, str) else path

        for name, weight in parameters.items():
            attr = getattr(self, name)
            if isinstance(attr, torch.Tensor):
                attr = weight
            else:
                attr.load(weight)




# data = ['cato', 'cari', 'cano', 'cab', 'com']
# data = ['caorto', 'coarti']
# data = ['cato', 'coti']
# data = ['can', 'cat', 'can', 'cab', 'cam', 'can']
# data = ['raro']
# data = ['kalolibaldor']
# data = list(get_english_words_set(['web2'], lower=True))

data = open('words.txt').read().splitlines()
num_words = 10
num_epochs = 100
data = data[:num_words]


trainer = Trainer(D=128, K=8, S=10, T=8)

trainer.load('here.pth')
trainer.inference(data)

# trainer.train(data, num_epochs)
# trainer.inference(data)
# trainer.save('here.pth')



