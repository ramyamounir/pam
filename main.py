import torch
from dataset import TextDataset
from layers import Layer4

from tqdm import tqdm
from english_words import get_english_words_set


# data = ['cato', 'cari', 'cano', 'cab', 'com']
# data = ['caorto', 'coarti']
# data = ['cato', 'coti']
# data = ['can', 'cat', 'can', 'cab', 'cam', 'can']
# data = ['raro']
# data = ['kalolibaldor']
# data = list(get_english_words_set(['web2'], lower=True))
data = open('words.txt').read().splitlines()

dim=128
S=10
T=8
K=8
num_words = 50

td = TextDataset(dim=dim, S=S)
l4 = Layer4(num_base_neurons=dim, num_neurons_per_minicolumn=K, sparsity=0.02, connections_density=0.5, connections_decay=1.0)


def inference():
    for d in data[:num_words]:
        for c in d:
            input_sdr = td.encode(c)
            results = l4(input_sdr=input_sdr, train=False, gen=True)

            print(f' "{c}" ->', td.decode(results['predicted'], overlap=T), end=' ')

            if results['generated'] != None: print(f'generated -> {td.decode(results["generated"], overlap=T)}', end=' ')

            if results['boundary']: print(' BOUNDARY')
            else: print('\n')

        print('\n')
        l4.reset()


for epoch in tqdm(range(1000)):
    for d in data[:num_words]:
        for c in d:
            input_sdr = td.encode(c)
            results = l4(input_sdr=input_sdr, train=True)

        l4.reset()

inference()


