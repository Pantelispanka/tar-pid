import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('./chembl_500k_train.csv')

df_sampled = df.sample(1000)
df_sampled['sample'] = 'Randomly selected'


print(list(df_sampled))

dfs = pd.read_csv('./screened_05.csv')


data = {}
data['mol_wt'] = []
data['subset'] = []

for smile in dfs['SMILES']:
    m = Chem.MolFromSmiles(smile)
    data['mol_wt'].append(Descriptors.MolWt(m))
    data['subset'].append('Screened molecules')

for sm in df_sampled['SMILES']:
    m_ = Chem.MolFromSmiles(sm)
    data['mol_wt'].append(Descriptors.MolWt(m_))
    data['subset'].append('Random selected molecules')


df_all = df.from_dict(data)


sns.kdeplot(data=data, x='mol_wt', hue='subset')
plt.xlabel('Molecular weight')
plt.show()
