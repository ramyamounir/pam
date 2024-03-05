import torch
from glob import glob
from tqdm import tqdm
import os

import sys; sys.path.append('./')
from basics import SDR
from layers import Layer4
from experiments.base.generate_data import Generator
from utils import checkdir

class Runner():
    def __init__(self, seq_file, N, S, K, epochs, test_type, save_every=10):
        self.seq_file = seq_file
        self.N = N
        self.S = S
        self.K = K
        self.epochs = epochs
        self.test_type = test_type
        self.save_every = save_every

        self.gen = Generator.load_from(seq_file)
        self.results = {}

    def create_model(self):
        return Layer4(self.N, self.K, self.S, 0.5, 1e-4, 1.0)

    def store(self, dataset_ix, epoch):
        tag = f'{str(dataset_ix).zfill(2)},{str(epoch).zfill(len(str(self.epochs)))}'
        self.results[tag] = self.test()

    def test(self):

        self.model.reset()
        results = []

        if self.test_type == 'AP':
            for ix, (sdr, b) in enumerate(self.gen):
                if ix > 0: sdr = res['predicted']
                res = self.model(sdr, train=False, gen=True)
                results.append(res['predicted'].save())
                if torch.numel(res['predicted'].ix) == 0:break
                print(res['predicted']) 
        elif self.test_type == 'AG':
            for ix, (sdr, b) in enumerate(self.gen):
                if ix > 0: sdr = res['generated']
                res = self.model(sdr, train=False, gen=True)
                results.append(res['generated'].save())
                print(res['generated']) 
        elif self.test_type == 'NA':
            for ix, (sdr, b) in enumerate(self.gen):
                res = self.model(sdr, train=False, gen=True)
                results.append(res['predicted'].save())
                print(res['predicted'])
        else:
            raise NotImplementedError("Kind not implemented")

        return results

    def run(self):

        for dataset_ix in range(self.gen.seqs.shape[0]):
            self.gen.set_dataset(dataset_ix)
            self.model = self.create_model()

            for e in tqdm(range(self.epochs)):

                for sdr, b in self.gen:
                    res = self.model(sdr, train=True, gen=True)
                    if res['boundary']:
                        break

                if e%self.save_every==0: 
                    self.store(dataset_ix, e)

                self.model.reset()

        return self.results


def run_experiments(K_range=[8,12,16], test_types=['AP', 'AG', 'NA'], epochs=100):

    save_base_dir = 'results/sequence_memory/solo'
    for seq_file in sorted(glob('results/sequence_memory/gt/seq_*.pth')):
        print(f'working on {seq_file} . . .')
        result_path = os.path.join(save_base_dir, os.path.splitext(os.path.basename(seq_file))[0])

        for K in K_range:
            result_path = os.path.join(result_path, f'K_{K}')
            checkdir(result_path)

            for test_type in test_types:
                filename = f'{test_type}.pth'
                results = Runner(seq_file, 128, 12, K, epochs, test_type, 10).run()
                torch.save(results, os.path.join(result_path, filename))






