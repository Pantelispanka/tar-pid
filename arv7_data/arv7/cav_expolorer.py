import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import toml


path = '/Users/pantelispanka/miniconda3/envs/docking-examples/cavity_finder/test/arv7_data/arv7/results/toml/*.toml'
files = glob.glob(path)
print(len(files))

in_cavity = {}
pdbs = []

for res in files:
    pdb = res.split("/")[-1]
    results = toml.load(res)
    for cavity in results['RESULTS']['AVG_HYDROPATHY']:
        try:
            if results['RESULTS']['AVG_HYDROPATHY'][cavity] < -0.5:
                if results['RESULTS']['VOLUME'][cavity] > 180:
                    pdbs.append(results['FILES']['INPUT'])
                    for residue in results['RESULTS']['RESIDUES'][cavity]:
                        resid = residue[0] + "_" + residue[2]
                        try:
                            in_cavity[resid]['in_cavities'] += 1
                            in_cavity[resid]['cavity']['cavity_name'].append(cavity)
                            in_cavity[resid]['cavity']['cavity_struct'].append(pdb)
                        except KeyError as e:
                            in_cavity[resid] = {}
                            in_cavity[resid]['in_cavities'] = 1
                            in_cavity[resid]['cavity'] = {}
                            in_cavity[resid]['cavity']['cavity_name'] = [cavity]
                            in_cavity[resid]['cavity']['cavity_struct'] = [pdb]
        except Exception as e:
            pass

# f = open("./results_all.json")

# returns JSON object as
# a dictionary
# in_cavity = json.load(f)
print("---PDBS---")
print(pdbs)
print("---RES---")
residues = []
in_cavs = []
cav_structs = []
for key in in_cavity:
    if in_cavity[key]['in_cavities'] > 3:
        # print(in_cavity[key])
        # print(in_cavity[key]['cavity']['cavity_struct'])
        cav_structs.append(in_cavity[key]['cavity']['cavity_struct'])
        residues.append(key.split("_")[0])
        in_cavs.append(in_cavity[key]['in_cavities'])

print(cav_structs)
print(residues)
print(in_cavs)



fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(residues, in_cavs, color='maroon',
        width=0.8)

plt.xlabel("residues")
plt.ylabel("in cavities")
plt.title("Resids in cavities")
plt.show()