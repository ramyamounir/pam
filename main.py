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
        self.l4 = Layer4(num_base_neurons=self.D, num_neurons_per_minicolumn=self.K, sparsity=S, connections_density=0.5, connections_decay=1.0)
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
                setattr(self, name, weight)
            else:
                attr.load(weight)

    def generate(self, letters, num_gen=1):

        result_set = set()

        for g in range(num_gen):

            self.l4.reset()
            output = ''

            # predict
            for l_i, l in enumerate(letters):
                input_sdr = self.td.encode(l)
                results = self.l4(input_sdr=input_sdr, train=False, gen=(l_i == len(letters)-1))
                if results['boundary']: break
                output += l

            if len(output) != len(letters): 
                result_set.add(output)
                break

            # generate
            while True:
                generated = results['generated']
                if len(generated) == 0: break
                decoded_gen = self.td.decode(generated, overlap=self.T)

                if len(decoded_gen)>1: 
                    output += decoded_gen[0]
                    print('WARNING: Multiple generated')
                elif len(decoded_gen) == 1:
                    output += decoded_gen[0]
                elif len(decoded_gen) == 0:
                    break

                results = self.l4(input_sdr=generated, train=False, gen=True)

            result_set.add(output)

        return result_set



        # # generate
        # while True:
        #     results = self.l4(input_sdr=input_sdr, train=False, gen=True)
        #     gen = self.td.decode(results['generated'], overlap=8)
        #     if len(gen) == 0: valid=False;break
        #     print(gen, results['generated'])
        #     input_sdr = results['generated']




# data = ['cato', 'cari', 'cano', 'cab', 'com']
# data = ['caorto', 'coarti']
# data = ['cato', 'coti']
# data = ['can', 'cat', 'can', 'cab', 'cam', 'can']
# data = ['raro']
# data = ['kalolibaldor']
# data = list(get_english_words_set(['web2'], lower=True))

data = open('words.txt').read().splitlines()
num_words = 100
num_epochs = 1000
data = data[:num_words]


trainer = Trainer(D=128, K=8, S=10, T=8)

# trainer.train(data, num_epochs)
# trainer.inference(data)
# trainer.save('test.pth')


trainer.load('test.pth')
# trainer.inference(data)
res = trainer.generate('t', num_gen=10)
print(res)










