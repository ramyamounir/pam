import sys; sys.path.append('./')
from experiments.base.create_vocab import Vocab
from experiments.base.generate_data import Generator
from utils import checkdir

# create directory
checkdir('results/sequence_memory/gt')

# create vocab
v = Vocab(kind='sparse', dim=256, size=1000, sparsity=10)
v.create()
v.save('results/sequence_memory/gt/sparse_1000_256_10.pth')


# create sequence
for l in range(10,110,10):
    g = Generator(vocab_path='results/sequence_memory/gt/sparse_1000_256_10.pth', vocab_len=20, len_seq=l, num_seq=1, num_datasets=1)
    g.create()
    g.save(f'results/sequence_memory/gt/seq_{str(l).zfill(3)}.pth')



