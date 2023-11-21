import pandas as pd
import re


import matplotlib.pyplot as plt
import seaborn as sns



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

naive_means = []
screened_means = []
for index, row in df.iterrows():
    vals = row[source_docked_confs]
    values = vals.values
    mean = values.mean()
    naive_means.append(mean)

for index, row in df_screened.iterrows():
    vals = row[source_docked_confs]
    values = vals.values
    mean = values.mean()
    screened_means.append(mean)

fig, ax = plt.subplots(figsize=(12, 6))
sns.kdeplot(data=naive_means,
            color='crimson', label='Naive docking mean energies', fill=True, ax=ax)
sns.kdeplot(data=screened_means,
            color='limegreen', label='Screened ligands mean energies', fill=True, ax=ax)
ax.legend()

plt.tight_layout()
plt.savefig('./kd_plots.png')
