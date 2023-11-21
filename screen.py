import torch
from jaqpotpy.descriptors.molecular import TopologicalFingerprint

import pandas as pd

from jaqpotpy.cfg import config

config.verbose = False

o = 0

smiles = []
with open("/home/pantelispanka/tipd/chembl_500k_train.csv", "r") as f:
    lines = f.readlines()
    for line in lines:
        splited = line.split(",")
        if o > 0 :
            smile = splited[2]
            smiles.append(smile)
        o += 1

o = 0

with open("/home/pantelispanka/tipd/dataset_v1.csv", "r") as f:
    lines = f.readlines()
    for line in lines:
        splited = line.split(",")
        if o > 0 :
            smile = splited[0]
            smiles.append(smile)
        o += 1


# feat = 512

class FFFingerprint(torch.nn.Module):
    def __init__(self):
        super(FFFingerprint, self).__init__()
        self.fc1 = torch.nn.Linear(1412, 1256)
        self.fc2 = torch.nn.Linear(1256, 1256)
        self.fc3 = torch.nn.Linear(1256, 1256)
        self.fc4 = torch.nn.Linear(1256, outs_num)
        self.dropout = torch.nn.Dropout(0.2)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        self.dropout(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)
        

        out = self.fc3(out)
        out = self.relu(out)

        out = self.fc4(out)
        self.dropout(out)
        out = self.sigmoid(out)
        return out
    

feat = TopologicalFingerprint(size=1412)

outs_num = 41

model = FFFingerprint()
model = torch.load("/home/pantelispanka/tipd/multi_model_05.ptt")
model.eval()


smiles_passed = []

mean_dock = []

sc = 0

for s in smiles:
    feats = feat.featurize(s)
    
    if len(feats[0]) > 0:
        sc += 1
        t = torch.from_numpy(feats[0]).double()
        
        t = t.to("cuda:0")
        out = model(t.float())
        output = torch.round(out)

        mean = torch.mean(output).item()
        if mean > 0.7 and len(s) < 120:
            mean_dock.append(mean)
            smiles_passed.append(s.strip("\n"))

smiles = {"SMILES": smiles_passed, "Mean_Dock": mean_dock}


df = pd.DataFrame.from_dict(smiles)

df.to_csv("./screened_05.csv", index=False)

print(f"SCREENED TOTAL {str(sc)}")
print(f"TOTAL FOUND {str(len(smiles_passed))}")