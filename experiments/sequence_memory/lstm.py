import torch
from glob import glob
from tqdm import tqdm
import os

import sys; sys.path.append('./')
from basics import SDR
from experiments.base.generate_data import Generator
from experiments.base.base_models import LSTMSequenceMemory, SequenceMemory
from utils import checkdir



class Runner():
    def __init__(self, seq_file, N, S, L, epochs, test_type, save_every=10):
        self.seq_file = seq_file
        self.N = N
        self.S = S
        self.L = L
        self.epochs = epochs
        self.test_type = test_type
        self.save_every = save_every

        self.gen = Generator.load_from(seq_file)
        self.results = {}

    def create_model(self):
        memory = LSTMSequenceMemory(dim=self.N, num_layers=self.L)
        return SequenceMemory(memory=memory, pred_threshold=0.01)

    def store(self, dataset_ix, epoch):
        tag = f'{str(dataset_ix).zfill(2)},{str(epoch).zfill(len(str(self.epochs)))}'
        self.results[tag] = self.test()

    def test(self):

        self.model.reset()
        results = []

        if self.test_type == 'AP':
            for ix, (sdr, b) in enumerate(self.gen):
                if ix > 0: sdr = pred_sdr
                pred, _, _ = self.model(sdr.bin.float().unsqueeze(0))
                pred_sdr = SDR.from_bin((pred.squeeze(0)>0.5).bool())
                results.append(pred_sdr.save())
        elif self.test_type == 'NA':
            for ix, (sdr, b) in enumerate(self.gen):
                pred, _, _ = self.model(sdr.bin.float().unsqueeze(0))
                pred_sdr = SDR.from_bin((pred.squeeze(0)>0.5).bool())
                results.append(pred_sdr.save())
        else:
            raise NotImplementedError("Kind not implemented")

        return results

    def run(self):

        for dataset_ix in range(self.gen.seqs.shape[0]):
            self.gen.set_dataset(dataset_ix)
            self.model = self.create_model()

            for e in tqdm(range(self.epochs)):

                for ix, (sdr, b) in enumerate(self.gen):
                    prediction, boundary, loss = self.model(sdr.bin.float().unsqueeze(0))

                if e%self.save_every==0: 
                    self.store(dataset_ix, e)

                self.model.reset()

        return self.results


def run_experiments(L_range=[2], test_types=['AP', 'NA'], epochs=100):

    save_base_dir = 'results/sequence_memory/lstm'
    checkdir(save_base_dir)

    for seq_file in sorted(glob('results/sequence_memory/gt/seq_*.pth')):
        print(f'working on {seq_file} . . .')
        result_path = os.path.join(save_base_dir, os.path.splitext(os.path.basename(seq_file))[0])

        for L in L_range:
            result_path = os.path.join(result_path, f'L_{L}')
            checkdir(result_path)

            for test_type in test_types:
                filename = f'{test_type}.pth'
                results = Runner(seq_file, 128, 12, L, epochs, test_type, 10).run()
                torch.save(results, os.path.join(result_path, filename))


