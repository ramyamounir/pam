import torch
from glob import glob
from tqdm import tqdm
import os

import sys; sys.path.append('./')
from basics import SDR
from layers import Layer4, Layer4Continual, Layer4ContinualGen
from experiments.base.generate_data import Generator
from utils import checkdir

from dask import delayed as dd
from dask import compute as dc

class Runner():
    def __init__(self, 
                 seq_file, 
                 K, 
                 num_gen,
                 cap,
                 stream,
                 conn_den=0.5,
                 conn_dec=0.0,
                 lr=1.0
                 ):

        self.seq_file = seq_file
        self.K = K
        self.num_gen = num_gen
        self.cap = cap
        self.stream = stream
        self.conn_den = conn_den
        self.conn_dec = conn_dec
        self.lr = lr

        self.gen = Generator.load_from(seq_file)
        self.N = self.gen.vocab.dim
        self.S = self.gen.vocab.sparsity

        self.epochs = max(int(cap*(1.0-stream)),1)
        self.persistance = max(int(cap*stream),1)

        self.results = {}

    def create_model(self):
        return Layer4ContinualGen(self.N, self.K, self.S, self.conn_den, self.conn_dec, self.lr, self.persistance)

    def store(self, dataset_ix):
        self.results[f'{str(dataset_ix).zfill(2)}'] = self.test()

    def test(self):

        self.model.reset()
        seqs = []

        for _ in tqdm(range(self.num_gen)):
            seq = []
            counter = 0
            for ix, (sdr, b) in enumerate(self.gen):
                if counter == 0: seq.append(sdr.save())
                else: sdr = res['generated']

                res = self.model(sdr, train=False)

                if b: 
                    seqs.append(seq)
                    self.model.reset()
                    counter = 0
                    seq = []
                else: 
                    seq.append(res['generated'].save())
                    counter += 1

        return seqs


    def run(self):

        for dataset_ix in range(self.gen.seqs.shape[0]):
            self.gen.set_dataset(dataset_ix)
            self.model = self.create_model()

            # for e in tqdm(range(self.epochs)):
            for e in range(self.epochs):
                # for sdr_ix, (sdr, b) in tqdm(enumerate(self.gen), total=self.gen.seqs.shape[-1]*self.gen.seqs.shape[-2]):
                for sdr_ix, (sdr, b) in enumerate(self.gen):
                    res = self.model(sdr, train=True)
                    if b: self.model.reset()

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
                        args['num_gen'],
                        args['cap'],
                        args['stream'],
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


save_base_dir = 'results/sequence_generation/solo/run_001'
assert checkdir(save_base_dir, careful=False), f'path {save_base_dir} exists'

# K_range = [8,16,24,32]
# test_types = ['AP', 'NA']
# seq_files = sorted(glob('results/sequence_generation/gt/voc_001/seq_*.pth'))
# stream_range = torch.linspace(0.0, 1.0, 5)[-1:]
# args = dict(conn_den=0.5, conn_dec=0.0, lr=1.0, cap=100)


K_range = [8]
seq_files = sorted(glob('results/sequence_generation/gt/voc_001/seq_*.pth'))[0:1]
stream_range = torch.linspace(0.0, 1.0, 5)[-1:]
args = dict(conn_den=0.5, conn_dec=0.0, lr=1.0, cap=100, num_gen=10)


fns = []
for K in K_range:
    for seq_file in seq_files:
        for stream in stream_range:
            args.update(dict(K=K, seq_file=seq_file, stream=stream))
            fns.append(dd(run_experiment)(args, len(fns)))


dc(*fns)

