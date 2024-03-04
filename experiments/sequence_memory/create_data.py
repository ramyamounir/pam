import sys; sys.path.append('./')
from experiments.base.create_vocab import Vocab
from experiments.base.generate_data import Generator


# create vocab
v = Vocab(kind='sparse', dim=128, size=1000, sparsity=0.1)
v.create()
v.save('results/sequence_memory/sparse_100_128_10.pth')


# create sequence
for l in range(10,110,10):
    g = Generator(vocab_path='results/sequence_memory/sparse_100_128_10.pth', vocab_len=10, len_seq=l, num_seq=1, num_datasets=10)
    g.create()
    g.save(f'results/sequence_memory/seq_{str(l).zfill(3)}.pth')




