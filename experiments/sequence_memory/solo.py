import torch
from glob import glob

import sys; sys.path.append('./')
from basics import SDR
from layers import Layer4
from experiments.base.generate_data import Generator


seq_files = sorted(glob('results/sequence_memory/seq_*.pth'))

for seq_file in seq_files:
    print(f'seq file = {seq_file}')
    gen = Generator.load_from(seq_files[0])

    for dataset_ix in range(gen.seqs.shape[0]):
        print(f'dataset = {dataset_ix}')
        gen.set_dataset(dataset_ix)

        for K in [8, 12, 16]:
            print(f'K = {K}')
            model = Layer4(num_base_neurons=128, num_neurons_per_minicolumn=K, sparsity=12, connections_density=0.5, connections_decay=1e-4, learning_rate=1.0)

            for e in range(100):
                print(f'epoch = {e}')
                for sdr, b in gen:
                    res = model(sdr, train=True, gen=True)
                    if b: break

                model.reset()

