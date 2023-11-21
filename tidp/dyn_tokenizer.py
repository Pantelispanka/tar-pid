import pandas as pd
import torch


class DynamicTokenizer:
    smiles_all = []
    _t2i = {}
    _i2t = {}
    chars = set()

    charset = [' ', '#', ')', '(', '+', '-', '/', '1', '3', '2', '5', '4', '7', '6', '8', '9', '0'
        , '=', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'S', '[', ']', '\\', 'c', 'K', "%", 'p', 'L'
        , 'l', 'o', 'n', 's', 'r', 'P', 'i', 'a', '.', 't', 'e', 'T', 'Z', 'M', 'g', '']

    def __init__(self, path, col="SMILES"):
        self.path = path
        self.col = col
        self.mol_length = 120
        self.read_csv()

    def read_csv(self):
        self.df = pd.read_csv(self.path)

    def get_smiles(self):
        smiles = self.df[[self.col]]
        for s in smiles[self.col]:
            self.smiles_all.append(s)

    def create_vocab(self):
        for smile in self.smiles_all:
            for c in smile:
                self.chars.add(c)
            self.chars.add("")

    def get_vocab_size(self):
        return len(self.charset)
        # return len(self.chars)

    def create_t2i(self):
        for index, c in enumerate(self.charset):
            self._t2i[c] = index
        # for index, c in enumerate(self.chars):
        #     self._t2i[c] = index

    def create_i2t(self):
        for index, c in enumerate(self.charset):
            self._i2t[index] = c
        # for index, c in enumerate(self.chars):
        #     self._i2t[index] = c

    def encode(self, sm_list):

        res = []
        for smile in sm_list:
            indexes = []
            for c in smile:
                indexes.append(self._t2i[c])
            size = len(indexes)
            if size > self.mol_length:
                indexes = indexes[:120]
            pad = self.mol_length - size
            pads = [self._t2i[" "] for x in range(pad)]
            indexes.extend(pads)
            res.append(indexes)
        return torch.Tensor(res)

    def decode(self, tensor):
        smiles_res = []
        for i in range(tensor.shape[0]):
            cur_sm = ''
            for t in tensor[i].detach().cpu().numpy():
                cur_sm += self._i2t[int(t)]
            cur_sm = cur_sm.replace(" ", "")
            smiles_res.append(cur_sm)

        return smiles_res
