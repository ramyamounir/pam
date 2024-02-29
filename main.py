import torch
from dataset import TextDataset
from layers import Layer4
from collections import deque

from tbw import TBWrapper, TBType
from tqdm import tqdm

class Trainer():
    def __init__(self, D, K, S, T, L, log_dir='logs/model_1'):
        self.D=D
        self.K=K
        self.S=S
        self.T=T
        self.L=L
        self.create_writers(log_dir)

        self.td = TextDataset(dim=self.D, S=self.S)
        self.l4 = Layer4(num_base_neurons=self.D, num_neurons_per_minicolumn=self.K, sparsity=S, connections_density=0.5, connections_decay=1e-4, learning_rate=L)
        self.parameters = ['td', 'l4']

    def create_writers(self, log_dir):
        self.chars = ""
        self.words_deque = deque(maxlen=100)

        self.writer = TBWrapper(log_dir)
        self.writer(TBType.TEXT, 'segmentation')
        # self.writer(TBType.TEXT, 'generation')
        # self.writer(TBType.TEXT, 'noise removal')
        # self.writer(TBType.TEXT, 'correction')

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

    def log(self, letter, boundary):
        if boundary:
            self.words_deque.append(self.chars)
            self.chars = ""

        self.chars += letter

        if len(self.words_deque) == 100:
            print(self.words_deque)
            quit()


    def train(self, data, num_epochs):

        # log_segmentation = ''
        # counter = 0

        for epoch in tqdm(range(num_epochs)):
            for d in data:
                for c in d:
                    input_sdr = self.td.encode(c)
                    results = self.l4(input_sdr=input_sdr, train=True)

                    # self.log(c, results['boundary'])

                    # if 0 <= counter%1000 <= 100:
                    #     if results['boundary']:
                    #         log_segmentation += "|"
                    #     log_segmentation += c

                    #     if counter %1000 == 100:
                    #         self.writer['segmentation'](log_segmentation)
                    #         log_segmentation = ''

                    # counter += 1

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


    def remove_noise(self, letters, n=1):

        self.l4.reset()
        input_sdr = self.td.encode(letters[0])
        results = self.l4(input_sdr=input_sdr, train=False, gen=False)
        output = letters[0]

        for l in letters[1:]:
            input_sdr = self.td.encode(l)
            gen = self.l4.generate_from(results['predicted'], input_sdr.add_noise(n))
            decoded_gen = self.td.decode(gen, overlap=self.T)

            if len(decoded_gen)>1: 
                output += decoded_gen[0]
                print('WARNING: Multiple generated')
            elif len(decoded_gen) == 1:
                output += decoded_gen[0]
            elif len(decoded_gen) == 0:
                break


            results = self.l4(input_sdr=gen, train=False, gen=False)

        return output






# data = ['cato', 'cari', 'cano', 'cab', 'com']
# data = ['caortoti', 'coartito']
# data = ['Hi, my name is Ramy']
# data = ['cato', 'coti']
# data = ['can', 'cat', 'can', 'cab', 'cam', 'can']
# data = ['raro']
# data = ['kalolibaldor']

data = open('data/words.txt').read().splitlines()
# data = open('data/hunger.txt').read().splitlines()
# data = [' '.join(open('data/hunger.txt').read().splitlines())[:10_000]]

num_words = 100
num_epochs = 200
data = data[:num_words]


trainer = Trainer(D=128, K=8, S=10, T=8, L=1)

# trainer.train(data, num_epochs)
# trainer.save('saved_models/words_100.pth')
# trainer.inference(data)
# quit()


trainer.load('saved_models/words_100.pth')
# trainer.inference(data)


# word = 'Hi, my name is'
# n=90
# counter = 0

# for _ in range(10):
#     res = trainer.remove_noise(word, n=n)
#     print(res)
#     if res == word:
#         counter += 1

# print(counter/10)

starting = 't'
data_filtered = set([word for word in data if word.startswith(starting)])
res = trainer.generate(starting, num_gen=1000)

print(len(data_filtered.intersection(res))/len(data_filtered))
print(res)
print(data_filtered)










