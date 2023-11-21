import pandas as pd
import re

from jaqpotpy.models import MolecularTorchGeometric, MolecularSKLearn
from jaqpotpy.models.torch_models import GCN_V1

from jaqpotpy.datasets import TorchGraphDataset, SmilesDataset
from jaqpotpy.models import Evaluator, AttentiveFP
from sklearn.metrics import max_error, mean_absolute_error, r2_score, confusion_matrix


from jaqpotpy.descriptors.molecular import MolGraphConvFeaturizer, PagtnMolGraphFeaturizer, TopologicalFingerprint

from torch_geometric.nn import GCNConv, GATConv, GINConv, GraphConv
from torch_geometric.nn import global_mean_pool


import torch
import torch.nn.functional as F


from sklearn.metrics import hamming_loss, zero_one_loss, accuracy_score, precision_score
from sklearn.metrics import matthews_corrcoef

# df = pd.read_csv("/Users/pantelispanka/Euclia/tidp/a-syn/a-syn-1000/data/all_2.csv", index_col=False)

df = pd.read_csv("/home/pantelispanka/tipd/all_2.csv", index_col=False)

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
    # print(df[dc_class])
    # print(df[dc].quantile(0.1))

print(list(df))

df.to_csv("/home/pantelispanka/tipd/all_2_class.csv", index=False)

# selected = {}
# selected['SMILES'] = []
# for dc in source_docked_confs:
#     first = True
#     print(dc)
#     dc_class = f"{dc}_class"
#     smiles_cl = f"{dc}_smile"
#     selected[smiles_cl] = []
#     selected[dc_class] = []
#     print(dc_class)
#     quantile = df[dc].quantile(0.1)
#     quantile_up = df[dc].quantile(0.4)
#     print(quantile)
#     for ind, val in enumerate(df[dc]):
#         if val < quantile:
#             selected[smiles_cl].append(df['SMILES'][ind])
#             # selected['SMILES'].append(df['SMILES'][ind])
#             selected[dc_class].append(1.0)
#         elif val > quantile_up:
#             selected[smiles_cl].append(df['SMILES'][ind])
#             # selected['SMILES'].append(df['SMILES'][ind])
#             selected[dc_class].append(0.0)
    # print(df[dc].quantile(0.1))


# all_smiles = []
# cols = 0
# for k in selected:
#     match = re.search(r"smile", k)
#     if match:
#         cols += 1
# print(cols)
# for smile in selected['129_a_smile']:
#     found_in = 0
#     for k in selected:
#         match = re.search(r"smile", k)
#         if match:
#             if smile in selected[k]:
#                 found_in += 1
#     if found_in == cols:
#         all_smiles.append(smile)
#
# print(len(all_smiles))

# smile_cols = []
#
# for k in selected:
#     match = re.search(r"smile", k)
#     if match:
#         smile_cols.append(k)
#
# class_cols = []
#
# for k in selected:
#     match = re.search(r"class", k)
#     if match:
#         class_cols.append(k)
#
#
# print(smile_cols)
# print(class_cols)
#
#
# selected_clear = {}
#
# for c in class_cols:
#     selected_clear[c] = []
#
#
# selected_clear['SMILES'] = []
# for s in all_smiles:
#     selected_clear['SMILES'].append(s)
#     for i, s_col in enumerate(smile_cols):
#         index = selected[s_col].index(s)
#         selected_clear[class_cols[i]].append(selected[class_cols[i]][index])
#         # print(s_col)


# df = pd.DataFrame.from_dict(selected_clear)

# df.to_csv("/Users/pantelispanka/Euclia/tidp/a-syn/a-syn-1000/data/all_class.csv", index=False)

print(len(df))

source_docked_confs_class = []

cols = list(df)
for col in cols:
    match = re.search(r"class", col)
    if match:
        source_docked_confs_class.append(col)

print(len(source_docked_confs_class))

from sklearn.model_selection import train_test_split

from rdkit import Chem

# df = df[:400]

train, test = train_test_split(df, test_size=0.1)
import math
#
for index, row in train.iterrows():
    if row[source_docked_confs_class].mean() > 0.1:
        for i in range(5):
            train = pd.concat([train, pd.DataFrame([row])], ignore_index=True)

mols = []
for smile in train['SMILES']:
    # mol = Chem.MolFromSmiles(smile)
    # m2 = Chem.AddHs(mol)
    # smile = Chem.MolToSmiles(m2)
    mols.append(smile)


outs_num = 20
ys = train[source_docked_confs_class[:outs_num]]

mols_t = []
for smile in test['SMILES']:
    # mol = Chem.MolFromSmiles(smile)
    # m2 = Chem.AddHs(mol)
    # smile = Chem.MolToSmiles(m2)
    mols_t.append(smile)

ys_t = test[source_docked_confs_class[:outs_num]]


# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.ensemble import RandomForestClassifier
#
# clf = MultiOutputClassifier(RandomForestClassifier())
#
# featurizer = TopologicalFingerprint()
# dataset = SmilesDataset(smiles=mols, y=ys, task='multitask', featurizer=featurizer)
# dataset_t = SmilesDataset(smiles=mols_t, y=ys_t, task='multitask', featurizer=featurizer)
# val = Evaluator()
# val.dataset = dataset_t
# val.register_scoring_function('Accuracy', accuracy_score)
# val.register_scoring_function('Hamming', hamming_loss)
# val.register_scoring_function('Zero loss', zero_one_loss)
#
# molecular_model = MolecularSKLearn(dataset=dataset, doa=None, model=clf, eval=val).fit()



class GCN_V1_MULTI(torch.nn.Module):
    def __init__(self, in_channels
                 , num_layers, hidden_channels, out_channels
                 , activation: torch.nn.Module = torch.nn.ReLU(), dropout=0.0, norm: torch.nn.Module = None, act_first=True):
        super(GCN_V1_MULTI, self).__init__()

        self.act = activation
        self.dropout = dropout
        self.act_first = act_first
        self.norm = norm

        # self.jaqpotpy_version = config.version

        self.layers = torch.nn.ModuleList([GCNConv(in_channels, hidden_channels).jittable()])
        for i in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_channels, hidden_channels).jittable())
        self.out = torch.nn.Linear(hidden_channels, out_channels)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
            if self.norm is not None and self.act_first is True:
                x = self.act(x)
                x = self.norm(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            elif self.norm is not None and self.act_first is False:
                x = self.norm(x)
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.out(x)
        x = self.sigmoid(x)
        return x


class GATConv_V1_MULTI(torch.nn.Module):

    def __init__(self, in_channels
                 , num_layers, hidden_channels, out_channels
                 , activation: torch.nn.Module = torch.nn.ReLU(), dropout=0.0, norm: torch.nn.Module = None, act_first=True):
        super(GATConv_V1_MULTI, self).__init__()

        self.act = activation
        self.dropout = dropout
        self.act_first = act_first
        self.norm = norm

        self.layers = torch.nn.ModuleList([GATConv(in_channels, hidden_channels).jittable()])
        for i in range(num_layers - 1):
            self.layers.append(GATConv(hidden_channels, hidden_channels).jittable())

       
        self.out = torch.nn.Linear(hidden_channels, 1024)
        self.out1 = torch.nn.Linear(1024, out_channels)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
            if self.norm is not None and self.act_first is True:
                x = self.act(x)
                x = self.norm(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            elif self.norm is not None and self.act_first is False:
                x = self.norm(x)
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.out(x)
        x = self.out1(x)
        x = self.sigmoid(x)
        return x


class GatedGraphConv_V1_MULTI(torch.nn.Module):

    def __init__(self, in_channels
                 , num_layers, hidden_channels, out_channels
                 , activation: torch.nn.Module = torch.nn.ReLU(), dropout=0.0, norm: torch.nn.Module = None, act_first=True):
        super(GatedGraphConv_V1_MULTI, self).__init__()

        self.act = activation
        self.dropout = dropout
        self.act_first = act_first
        self.norm = norm

        self.layers = torch.nn.ModuleList([GraphConv(in_channels, hidden_channels).jittable()])
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(hidden_channels, hidden_channels).jittable())
        self.out = torch.nn.Linear(hidden_channels, 442)
        self.out1 = torch.nn.Linear(442, out_channels)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
            if self.norm is not None and self.act_first is True:
                x = self.act(x)
                x = self.norm(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            elif self.norm is not None and self.act_first is False:
                x = self.norm(x)
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.out(x)
        x = self.out1(x)
        x = self.sigmoid(x)
        return x




featurizer = MolGraphConvFeaturizer(use_chirality=True, use_partial_charge=True)

dataset = TorchGraphDataset(smiles=mols, y=ys, task='multiclass'
                            , featurizer=featurizer)
dataset.create()


dataset_t = TorchGraphDataset(smiles=mols_t, y=ys_t, task='multiclass'
                            , featurizer=featurizer)
dataset_t.create()


val = Evaluator()
val.dataset = dataset_t
# model = GCN_V1_MULTI(33, 4, 40, 20, act_first=True)
model = GATConv_V1_MULTI(33, 6, 60, outs_num, act_first=True)
# model = GatedGraphConv_V1_MULTI(33, 2, 40, 20, act_first=True)

# model = AttentiveFP(in_channels=94, hidden_channels=40, out_channels=20, edge_dim=42, num_layers=2, num_timesteps=2).jittable()


optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# weights = torch.FloatTensor([0.04, 99.2])
# criterion = torch.nn.BCELoss(weight=weights)
# criterion = torch.nn.BCELoss(reduction='mean')
criterion = torch.nn.BCELoss()

# criterion = torch.nn.BCEWithLogitsLoss()

epochs = 3200
from torch_geometric.loader import DataLoader

train_batch = 820
test_batch = 500
train_loader = DataLoader(dataset=dataset,  batch_size=train_batch, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=dataset_t,  batch_size=test_batch, shuffle=True, num_workers=0)

# device = "cpu"
device = "cuda:0"
import numpy as np

model.to(device)

for i in range(epochs):
    epoch_loss = 0
    test_loss = 0

    out_train = np.empty((0,outs_num), int)
    truth_train = np.empty((0,outs_num), int)
    
    for data in train_loader:
        model.train()
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        try:
            edge_attributes = data.edge_attr.to(device)
        except AttributeError:
            edge_attributes = None
        batch = data.batch.to(device)
        y = data.y.to(device)
        if edge_attributes is not None:

            out = model(x, edge_index, edge_attributes, batch)
        else:
            out = model(x, edge_index, batch)
        loss = criterion(out, y)
        epoch_loss += loss

        output = torch.round(out)


        out_n = output.cpu().detach().numpy()
        truth_n = y.cpu().detach().numpy()

        out_train = np.vstack((out_train, out_n))
        truth_train = np.vstack((truth_train, truth_n))


        # for ind, _y in enumerate(y.detach().numpy()):
        #     for y_i in _y:
        #         if y_i == 1.0:
        #             # if accuracy_score(_y, output.detach().numpy()[ind]) > 0:
        #             if precision_score(_y, output.detach().numpy()[ind], zero_division = np.nan ) > 0:
        #                 print("CONTAINS ACTIVE ACCURACY TRAIN")
        #                 print(accuracy_score(_y, output.detach().numpy()[ind]))
        #                 print("PRECISION")
        #                 print(precision_score(_y, output.detach().numpy()[ind], zero_division = np.nan))
        #                 print("Y")
        #                 print(_y)
        #                 print("PRED")
        #                 print(output.detach().numpy()[ind])

                # hl = hamming_loss(y.detach().numpy(), output.detach().numpy())
                # z_o = zero_one_loss(y.detach().numpy(), output.detach().numpy())
                # print(f"Hamming Loss train {str(hl)}")
                # print(f"Zero One Loss train {str(z_o)}")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    mcc_train = []
    acc_train = []
    for ind in range(outs_num):
        out_ar = out_train[:,ind]
        truth_ar = truth_train[:,ind]
        # print("TRAIN")
        # print(confusion_matrix(truth_ar, out_ar))
        mcc_train.append(matthews_corrcoef(truth_ar, out_ar))
        acc_train.append(accuracy_score(truth_ar, out_ar))
    hl_train = hamming_loss(y.cpu().detach().numpy(), output.cpu().detach().numpy())
    z_o_train = zero_one_loss(y.cpu().detach().numpy(), output.cpu().detach().numpy())

    for data in test_loader:
        model.eval()
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        try:
            edge_attributes = data.edge_attr.to(device)
        except AttributeError:
            edge_attributes = None
        batch = data.batch.to(device)
        y = data.y.to(device)
        if edge_attributes is not None:
            out = model(x, edge_index, edge_attributes, batch)
        else:
            out = model(x, edge_index, batch)
        loss = criterion(out, y)
        test_loss += loss

        output = torch.round(out)

        out_n = output.cpu().detach().numpy()
        truth_n = y.cpu().detach().numpy()
        mcc = []
        acc = []
        for ind in range(outs_num):
            out_ar = out_n[:,ind]
            truth_ar = truth_n[:,ind]

            print("TEST")
            print(confusion_matrix(truth_ar, out_ar))

            acc.append(accuracy_score(truth_ar, out_ar))
            mcc.append(matthews_corrcoef(truth_ar, out_ar))



    #     for ind, _y in enumerate(y.cpu().detach().numpy()):
    #         if 1.0 in _y:
    #             if accuracy_score(_y, output.cpu().detach().numpy()[ind]) > 0:
    #                 in_positive_acc.append(accuracy_score(_y, output.cpu().detach().numpy()[ind]))
    #                 in_positive_pre.append(precision_score(_y, output.cpu().detach().numpy()[ind], zero_division = np.nan))


    #             # if precision_score(_y, output.detach().numpy()[ind], zero_division = np.nan ) > 0:
    #                 # print("CONTAINS ACTIVE ACCURACY TEST")
    #                 # print(accuracy_score(_y, output.cpu().detach().numpy()[ind]))
    #                 # print("PRECISION")
    #                 # print(precision_score(_y, output.cpu().detach().numpy()[ind], zero_division = np.nan))
    #                 # print("Y")
    #                 # print(_y)
    #                 # print("PRED")
    #                 # print(output.cpu().detach().numpy()[ind])
    #         if 1.0 not in _y:
    #             if accuracy_score(_y, output.cpu().detach().numpy()[ind]) > 0:
    #                 in_negative_acc.append(accuracy_score(_y, output.cpu().detach().numpy()[ind]))
    #                 in_negative_pre.append(precision_score(_y, output.cpu().detach().numpy()[ind], zero_division = np.nan))


    #             # if precision_score(_y, output.detach().numpy()[ind], zero_division = np.nan ) > 0:
    #                 # print("NON ACTIVE ACCURACY TEST")
    #                 # print(accuracy_score(_y, output.cpu().detach().numpy()[ind]))
    #                 # print("PRECISION")
    #                 # print(precision_score(_y, output.cpu().detach().numpy()[ind], zero_division = np.nan))
    #                 # print("Y")
    #                 # print(_y)
    #                 # print("PRED")
    #                 # print(output.cpu().detach().numpy()[ind])

    #             # hl = hamming_loss(y.detach().numpy(), output.detach().numpy())
    #             # z_o = zero_one_loss(y.detach().numpy(), output.detach().numpy())
    #             # print(f"Hamming Loss test {str(hl)}")
    #             # print(f"Zero One Loss test {str(z_o)}")
        hl = hamming_loss(y.cpu().detach().numpy(), output.cpu().detach().numpy())
        z_o = zero_one_loss(y.cpu().detach().numpy(), output.cpu().detach().numpy())
    # print(f"Zero one loss train {z_o_train}")
    # print(f"Hamming loss train {hl_train}")
    # print(f"Zero one loss {z_o}")
    # print(f"Hamming loss {hl}")



    # print(f"Negative accuracy {str(in_negative_acc)}, Negative precision {str(in_negative_pre)}")
    # print(f"Positive accuracy {str(in_positive_acc)}, Positive precision {str(in_positive_pre)}")
    print("TRAIN")
    print(f"Zero one loss train {z_o_train}")
    print(f"Hamming loss train {hl_train}")
    print(f"Epoch loss { str(epoch_loss / len(train_loader)) }")
    print(source_docked_confs_class)
    print(acc_train)
    print(mcc_train)


    print("TEST")
    print(f"Zero one loss {z_o}")
    print(f"Hamming loss {hl}")
    print(f"Test loss {str(test_loss / len(test_loader))}")
    print(source_docked_confs_class)
    print(acc)
    print(mcc)

# m = MolecularTorchGeometric(dataset=dataset
#                             , model_nn=model, eval=val
#                             , train_batch=320, test_batch=500
#                             , epochs=8000, optimizer=optimizer, criterion=criterion).fit()
#
# m.eval()
#
# model = m.create_molecular_model()
# model(mols[0])
# model(mols)
