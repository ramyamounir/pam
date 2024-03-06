import torch
from glob import glob
from tqdm import tqdm
import os

import sys; sys.path.append('./')
from basics import SDR
from layers import Layer4, Layer4Continual
from experiments.base.generate_data import Generator
from utils import checkdir

from dask import delayed as dd
from dask import compute as dc

class Runner():
    def __init__(self, 
                 seq_file, 
                 K, 
                 test_type,
                 conn_den=0.5,
                 conn_dec=0.0,
                 lr=1.0
                 ):
        self.seq_file = seq_file
        self.K = K
        self.test_type = test_type
        self.conn_den = conn_den
        self.conn_dec = conn_dec
        self.lr = lr

        self.gen = Generator.load_from(seq_file)
        self.N = self.gen.vocab.dim
        self.S = self.gen.vocab.sparsity

        self.results = {}

    def create_model(self):
        return Layer4Continual(self.N, self.K, self.S, self.conn_den, self.conn_dec, self.lr)

    def store(self, dataset_ix):
        self.results[f'{str(dataset_ix).zfill(2)}'] = self.test()

    def test(self):

        self.model.reset()
        results = []

        if self.test_type == 'AP':
            for ix, (sdr, b) in enumerate(self.gen):
                if ix > 0: sdr = res['predicted']
                res = self.model(sdr, train=False, gen=True)
                results.append(res['predicted'].save())
        elif self.test_type == 'AG':
            for ix, (sdr, b) in enumerate(self.gen):
                if ix > 0: sdr = res['generated']
                res = self.model(sdr, train=False, gen=True)
                results.append(res['generated'].save())
        elif self.test_type == 'NA':
            for ix, (sdr, b) in enumerate(self.gen):
                res = self.model(sdr, train=False, gen=True)
                results.append(res['predicted'].save())
        else:
            raise NotImplementedError("Kind not implemented")

        return results

    def run(self):

        for dataset_ix in range(self.gen.seqs.shape[0]):
            self.gen.set_dataset(dataset_ix)
            self.model = self.create_model()

            for sdr, b in self.gen:
                res = self.model(sdr, train=True, gen=True)

            self.model.reset()
            self.store(dataset_ix)

        return self.results


def run_experiment(args, exp_id):

    exp_id = str(exp_id).zfill(3)
    print(f'started experiment {exp_id} . . .')
    save_dir = os.path.join(save_base_dir, exp_id)
    checkdir(save_dir)

    runner = Runner(args['seq_file'], 
                        args['K'], 
                        args['test_type'],
                        args['conn_den'],
                        args['conn_dec'],
                        args['lr'],
                        )

    results = runner.run()

    args.update(
            dict(vocab_path=runner.gen.vocab_path,
                 vocab_len=runner.gen.vocab_len, 
                 len_seq=runner.gen.len_seq, 
                 num_seq=runner.gen.num_seq, 
                 num_datasets=runner.gen.num_datasets)
            )

    to_save = dict(args=args, vals=results)
    torch.save(to_save, os.path.join(save_dir, 'result.pth'))
    print(f'finished experiment {exp_id} . . .')



save_base_dir = 'results/sequence_memory/solo'
checkdir(save_base_dir)
args = dict(conn_den=0.5, conn_dec=0.0, lr=1.0)

K_range = [8,16,24,32]
test_types = ['AP', 'NA']
seq_files = sorted(glob('results/sequence_memory/gt/seq_*.pth'))

# K_range = [8,24]
# test_types = ['AP', 'NA']
# seq_files = sorted(glob('results/sequence_memory/gt/seq_*.pth'))[:3]

fns = []
for K in K_range:
    for test_type in test_types:
        for seq_file in seq_files:
            args.update(dict(K=K, test_type=test_type, seq_file=seq_file))
            fns.append(dd(run_experiment)(args, len(fns)))

dc(*fns)

