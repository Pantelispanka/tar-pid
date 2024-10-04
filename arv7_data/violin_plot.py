import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch

df_screened = pd.read_csv('./screened_results.csv')
df_naive = pd.read_csv('./all_arv.csv')

quantiles = df_naive.quantile(0.05)
print(type(quantiles))



list_of_sites = ['200_KAR', '56_KAS', '290_KBA', '80_KAD', '322_KAK', '134_KAH'
    , '342_KAI', '168_KAK', '134_KAC', '304_KAQ'
    , '104_KAF', '22_KAJ', '216_KAN', '156_KAE', '204_KAM'
    , '58_KAF', '210_KAF', '154_KAN', '72_KAI', '178_KAA', '148_KAG'
    , '212_KAL', '268_KAL', '314_KAF', '296_KAW'
    , '182_KAD', '310_KAQ', '192_KAO'
    , '216_KAA', '222_KAV', '260_KAF', '284_KAJ', '92_KAM'
    , '8_KAI', '22_KAI', '34_KAS'
    , '340_KAH', '140_KAD', '140_KAC', '220_KAH', '286_KAM']

id = '286_KAM'
sites = list_of_sites[40:41]
quant = quantiles['286_KAM']
print(sites)

# sites = list_of_sites[30:41]

df_screened_cols = df_screened[sites].copy()
df_screened_cols = pd.melt(df_screened_cols, var_name='pocket id', value_name='binding energy (kcal/mol)')
df_screened_cols['smiles'] = 'screened'

df_naive_cols = df_naive[sites].copy()
df_naive_cols = pd.melt(df_naive_cols, var_name='pocket id', value_name='binding energy (kcal/mol)')
df_naive_cols['smiles'] = 'naive'

df_all = pd.concat([df_screened_cols, df_naive_cols])

print(df_all.smiles.unique())

print(df_all)

# sns.violinplot(data=df_all, split=True, inner="quart")
# sns.violinplot(data=df_all[['200_KAR', '56_KAS', '290_KBA', '80_KAD']], split=True, inner="quart")


ax = sns.violinplot(data=df_all, x="pocket id", y="binding energy (kcal/mol)", hue="smiles", split=True, inner="quart")


line75 = ax.axhline(y=quant, color="gold", label="original quantile")
patch = PathPatch(ax.collections[0].get_paths()[0], transform=ax.transData)
line75.set_clip_path(patch) # clip the line by the form of the violin

ax.legend()

plt.show()
