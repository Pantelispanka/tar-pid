import pyKVFinder
import toml
import seaborn as sns
import matplotlib.pyplot as plt

cavities_names = ['204_KAM', '134_KAC', '22_KAJ', '80_KAD', '154_KAN', '296_KAW', '92_KAM'
    , '140_KAC', '260_KAF', '192_KAO', '220_KAH', '310_KAQ', '104_KAF', '268_KAL', '304_KAQ', '342_KAI'
    , '290_KBA', '322_KAK', '210_KAF', '34_KAS', '178_KAA', '56_KAS', '58_KAF', '72_KAI', '8_KAI', '200_KAR'
    , '134_KAH', '212_KAL', '286_KAM', '182_KAD', '216_KAN', '284_KAJ', '148_KAG', '340_KAH'
    , '314_KAF', '222_KAV', '140_KAD', '168_KAK', '22_KAI', '156_KAE', '216_KAA']

cavs = {}

cavs['VOLUME'] = []
cavs['AREA'] = []
cavs['MAX_DEPTH'] = []
cavs['AVG_DEPTH'] = []
cavs['AVG_HYDROPATHY'] = []

for cav in cavities_names:
    pdb, cavity = cav.split("_")
    path = f'/Users/pantelispanka/miniconda3/envs/docking-examples//cavity_finder/test/arv7_data/binding_site_analysis/results/results_{pdb}_a.toml'
    res = toml.load(path)
    print(cav)
    print(res['RESULTS']['RESIDUES'][cavity])

    for k in res['RESULTS']:
        try:
            cavs[k].append(res['RESULTS'][k][cavity])
        except KeyError as e:
            pass



# sns.set()
#
# fig, axes = plt.subplots(1, 2, figsize=(128, 4))
# # fig.suptitle('Cavity property distributions')
#
# print(cavs['VOLUME'])
# sns.kdeplot(data=cavs, x='VOLUME')
# sns.kdeplot(data=cavs, x='AREA')
# sns.kdeplot(data=cavs, x='MAX_DEPTH')
# sns.kdeplot(data=cavs, x='AVG_DEPTH')
sns.kdeplot(data=cavs, x='AVG_HYDROPATHY')
plt.xlabel('Average hydropathy')
plt.show()
#
#
# plt.show()


for cav in cavs:
    print(cav)
