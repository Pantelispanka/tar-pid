import torch

from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import re

class MolecularDataset(Dataset):
    def __init__(self, source="", column="Docking_molecule",
                 with_missings=False):

        self.source = source
        self.source_smiles = []
        self.source_props = []
        self.source_docked_confs = []

        self.with_missings = with_missings

        self.column = column
        self.len = 0
        self.df = pd.read_csv(self.source)
        self.read_data()

    def read_data(self):
        self.source_smiles = self.df[self.column][:140000].values
        cols = list(self.df)
        for col in cols:
            match = re.search(r"\d+", col)
            if match:
                self.source_docked_confs.append(col)
        self.len = len(self.df)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        df_t = self.df.iloc[idx]
        smile = df_t[self.column]
        vals = df_t[self.source_docked_confs].astype(float)
        return smile, torch.FloatTensor(vals.values)
