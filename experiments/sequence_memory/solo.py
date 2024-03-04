import torch
from glob import glob
from tqdm import tqdm

import sys; sys.path.append('./')
from basics import SDR
from layers import Layer4
from experiments.base.generate_data import Generator



results = {}

def store(seq_file, dataset_ix, neurons_per_minicolumn, epoch, res):
    tag = f'{str(dataset_ix).zfill(2)},{str(neurons_per_minicolumn).zfill(2)},{epoch}'
    if seq_file not in results: results[seq_file] = {}
    results[seq_file][tag] = res

def test(model, gen, kind='AP'):
    model.reset()

    results = []

    if kind == 'AP':
        for ix, (sdr, b) in enumerate(gen):
            if ix > 0: sdr = res['predicted']
            res = model(sdr, train=False, gen=True)
            results.append(res['predicted'].save())
    elif kind == 'AG':
        for ix, (sdr, b) in enumerate(gen):
            if ix > 0: sdr = res['generated']
            res = model(sdr, train=False, gen=True)
            results.append(res['generated'].save())
    elif kind == 'NA':
        for ix, (sdr, b) in enumerate(gen):
            res = model(sdr, train=False, gen=True)
            results.append(res['predicted'].save())
    else:
        raise NotImplementedError("Kind not implemented")

    return results


for seq_file in sorted(glob('results/sequence_memory/seq_*.pth')):
    print(f'seq file = {seq_file}')

    gen = Generator.load_from(seq_file)

    for dataset_ix in range(gen.seqs.shape[0]):
        print(f'dataset = {dataset_ix}')
        gen.set_dataset(dataset_ix)

        for K in [8, 12, 16]:
            print(f'K = {K}')
            model = Layer4(num_base_neurons=128, num_neurons_per_minicolumn=K, sparsity=12, connections_density=0.5, connections_decay=1e-4, learning_rate=1.0)

            for e in tqdm(range(300)):

                for sdr, b in gen:
                    res = model(sdr, train=True, gen=True)

                if e%10==0: 
                    gen_results = dict(AP=test(model, gen, kind='AP'), AG=test(model, gen, kind='AG'), NA=test(model, gen, kind='NA'))
                    store(seq_file, dataset_ix, K, e, gen_results)

                model.reset()


# need to save autoregressive predictive, autoregressive generated, non-autoregressive sequences
torch.save(results, 'results/sequence_memory/solo_results.pth')



