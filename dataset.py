import torch
import string
from basics import SDR


class TextDataset():
    def __init__(self, dim=512, S=10):
        # self.letters = string.printable
        self.letters = string.ascii_lowercase
        self.letter2id = {l:i for i,l in enumerate(self.letters)}
        self.id2letter = {i:l for i,l in enumerate(self.letters)}


        self.sdrs = [SDR(dim, S) for _ in self.letters]


    def encode(self, letter):
        return self.sdrs[self.letter2id[letter.lower()]]

    def decode(self, sdr, overlap=8):
        overlaps = torch.tensor([sdr.overlap(s) for s in self.sdrs])
        sort_val, sort_ix = torch.sort(overlaps, descending=True)
        return [self.id2letter[i.item()] for i in sort_ix[sort_val>=overlap]]

    def save(self, path=None):
        to_save = []
        for sdr in self.sdrs:
            to_save.append(sdr.save(path=None))
        return to_save

    def load(self, path):
        parameters = torch.load(path) if isinstance(path, str) else path
        for sdr, par in zip(self.sdrs, parameters):
            sdr.load(par)




if __name__ == "__main__":
    td = TextDataset()


