import torch
import re


class Tokenizer:

    _atoms = ['He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ar',
              'Ca', 'Ti', 'Cr', 'Fe', 'Ni', 'Cu', 'Ga', 'Ge', 'As', 'Se',
              'Br', 'Kr', 'Rb', 'Sr', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
              'Pd', 'Ag', 'Cd', 'Sb', 'Te', 'Xe', 'Ba', 'La', 'Ce', 'Pr',
              'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Er', 'Tm', 'Yb',
              'Lu', 'Hf', 'Ta', 'Re', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',
              'Bi', 'At', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'Pu', 'Am', 'Cm',
              'Bk', 'Cf', 'Es', 'Fm', 'Md', 'Lr', 'Rf', 'Db', 'Sg', 'Mt',
              'Ds', 'Rg', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

    def __init__(self, mol_length: int = 40):
        self.mol_length = mol_length
        self._atoms_re = self.get_tokenizer_re(self._atoms)

    def get_tokenizer_re(self, atoms):
        return re.compile('(' + '|'.join(atoms) + r'|\%\d\d|.)')

    __i2t = {
        0: 'unused', 1: '>', 2: '<', 3: '2', 4: 'F', 5: 'Cl', 6: 'N',
        7: '[', 8: '6', 9: 'O', 10: 'c', 11: ']', 12: '#',
        13: '=', 14: '3', 15: ')', 16: '4', 17: '-', 18: 'n',
        19: 'o', 20: '5', 21: 'H', 22: '(', 23: 'C',
        24: '1', 25: 'S', 26: 's', 27: 'Br'
        , 28: '@', 29: '+', 30: '/'
        , 31: '7', 32: '\\', 33: '8', 34: '9', 35: 'P', 36: 'I', 37: '.'
        , 38: 'B', 39: 'e', 40: 'Se', 41: '%10', 42: 'Si', 43: 'p'
        , 44: 'b' , 45: '%11', 46: '%12'
        , 47: 'Be'
    }

    __t2i = {
        '>': 1, '<': 2, '2': 3, 'F': 4, 'Cl': 5, 'N': 6, '[': 7, '6': 8,
        'O': 9, 'c': 10, ']': 11, '#': 12, '=': 13, '3': 14, ')': 15,
        '4': 16, '-': 17, 'n': 18, 'o': 19, '5': 20, 'H': 21, '(': 22,
        'C': 23, '1': 24, 'S': 25, 's': 26, 'Br': 27
        , '@': 28, '+': 29, '/': 30
        , '7': 31, '\\': 32, '8': 33, '9': 34, 'P': 35, 'I': 36, '.': 37
        , 'B': 38, 'e': 39, 'Se': 40, '%10': 41, 'Si': 42, 'p': 43
        , 'b': 44, '%11': 45, '%12': 46
        , 'Be': 47
    }


    def smiles_tokenizer(self, line, atoms=None):
        """
        Tokenizes SMILES string atom-wise using regular expressions. While this
        method is fast, it may lead to some mistakes: Sn may be considered as Tin
        or as Sulfur with Nitrogen in aromatic cycle. Because of this, you should
        specify a set of two-letter atoms explicitly.

        Parameters:
             atoms: set of two-letter atoms for tokenization
        """
        if atoms is not None:
            reg = self.get_tokenizer_re(atoms)
        else:
            reg = self._atoms_re
        return reg.split(line)[1::2]


    def encode(self, sm_list):
        """
        Encoder list of smiles to tensor of tokens
        """
        res = []
        lens = []
        for s in sm_list:
            try:
                tokens = ([1] + [self.__t2i[tok]
                                 for tok in self.smiles_tokenizer(s)])[:self.mol_length - 1]
            except KeyError as e:
                print(str(e))
            lens.append(len(tokens))
            tokens += (self.mol_length - len(tokens)) * [2]
            res.append(tokens)
        t = torch.Tensor(res)
        return t, lens


    def decode(self, tokens_tensor):
        """
        Decodes from tensor of tokens to list of smiles
        """

        smiles_res = []

        for i in range(tokens_tensor.shape[0]):
            cur_sm = ''
            for t in tokens_tensor[i].detach().cpu().numpy():
                # try:
                if t == 2:
                    break
                elif t > 2:
                    cur_sm += self.__i2t[t]
                # except KeyError as e:
                #     cur_sm = cur_sm
            smiles_res.append(cur_sm)

        return smiles_res


    def get_vocab_size(self):
        return len(self.__i2t)
