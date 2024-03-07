import sys; sys.path.append('./')
from experiments.base.create_vocab import Vocab
from experiments.base.generate_data import Generator
from utils import checkdir
import os

# create directory
vocab_dir = 'results/sequence_generation/gt/voc_001'
assert checkdir(vocab_dir, careful=True), f'path {vocab_dir} exists'

# create vocab
v = Vocab(kind='sparse', dim=256, size=1000, sparsity=10)
v.create()
v.save(os.path.join(vocab_dir, 'sparse_1000_256_10.pth'))

# create sequence
n_range = [10,50,100]
for n in n_range:
    g = Generator(vocab_path=os.path.join(vocab_dir, 'sparse_1000_256_10.pth'), vocab_len=20, len_seq=5, num_seq=n, num_datasets=3)
    g.create()
    g.save(os.path.join(vocab_dir, f'seq_{str(n).zfill(len(str(n_range[-1])))}.pth'))



