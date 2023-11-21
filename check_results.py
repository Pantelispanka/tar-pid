import pandas as pd
import re

from jaqpotpy.models import MolecularTorchGeometric, MolecularSKLearn
from jaqpotpy.models.torch_models import GCN_V1

from jaqpotpy.datasets import TorchGraphDataset, SmilesDataset
from jaqpotpy.models import Evaluator, AttentiveFP
from sklearn.metrics import max_error, mean_absolute_error, r2_score
from sklearn.metrics import matthews_corrcoef, confusion_matrix


from jaqpotpy.descriptors.molecular import MolGraphConvFeaturizer, PagtnMolGraphFeaturizer, TopologicalFingerprint

from torch_geometric.nn import GCNConv, GATConv, GINConv, GraphConv
from torch_geometric.nn import global_mean_pool

from torch.utils.data import DataLoader

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt


from sklearn.metrics import hamming_loss, zero_one_loss, accuracy_score, precision_score


# df = pd.read_csv("/Users/pantelispanka/Euclia/tidp/a-syn/a-syn-1000/data/all_2.csv", index_col=False)

df = pd.read_csv("/home/pantelispanka/tipd/res_final_6000.csv", index_col=False)

df_screened = pd.read_csv('/home/pantelispanka/tipd/screened_results.csv', index_col=False)

source_docked_confs = []
cols = list(df)
for col in cols:
    match = re.search(r"\d+", col)
    if match:
        source_docked_confs.append(col)

print(source_docked_confs)

for index, row in df.iterrows():
    vals = row[source_docked_confs]
    values = vals.values
    mean = values.mean()
    # if mean < -7.0:
    #     print(values.mean())
    #     print(values.std())
    #     print(vals.values)


for dc in source_docked_confs:
    # print(dc)
    dc_class = f"{dc}_class"
    # print(dc_class)
    quantile = df[dc].quantile(0.05)
    # print(quantile)

    df[dc_class] = [1 if val < quantile else 0 for val in df[dc]]
    df_screened[dc_class] = [1 if val < quantile else 0 for val in df_screened[dc]]

    # print(df[dc_class])
    # print(df[dc].quantile(0.1))

df.to_csv("/home/pantelispanka/tipd/res_final_6000_class.csv", index=False)
df_screened.to_csv("/home/pantelispanka/tipd/res_screened_class.csv", index=False)

source_docked_confs_class = []

cols = list(df)
for col in cols:
    match = re.search(r"class", col)
    if match:
        source_docked_confs_class.append(col)


naive = {}

for index, row in df.iterrows():
    index = row[source_docked_confs_class].sum()
    try:
        naive[index] += 1
    except KeyError as e:
        naive[index] = 1


screened = {}

for index, row in df_screened.iterrows():
    index = row[source_docked_confs_class].sum()
    try:
        screened[index] += 1
    except KeyError as e:
        screened[index] = 1


print(naive)
print(screened)

naive = dict(sorted(naive.items()))

screened = dict(sorted(screened.items()))

import matplotlib.pyplot as plt

plt.bar(list(naive.keys()), naive.values(), color='g')

plt.savefig('./naive.png')

plt.clf()

plt.bar(list(screened.keys()), screened.values(), color='g')


plt.savefig('./screened.png')