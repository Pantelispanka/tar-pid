
import sys,argparse,os
import pandas as pd
import numpy as np
import subprocess
import glob
import matplotlib.pyplot as plt
import matplotlib
import mdtraj as md
import argparse, sys
import statistics

#pdbs = ["../data/51_a.pdb", "../data/79_a.pdb", "../data/128_a.pdb", "../data/69_a.pdb", "../data/116_a.pdb", "../data/57_a.pdb", "../data/28_a.pdb", "../data/122_a.pdb", "../data/153_a.pdb", "../data/130_a.pdb", "../data/144_a.pdb", "../data/66_a.pdb", "../data/42_a.pdb", "../data/64_a.pdb", "../data/15_a.pdb", "../data/129_a.pdb", "../data/170_a.pdb", "../data/11_a.pdb", "../data/117_a.pdb", "../data/83_a.pdb"]
#print(pdbs)
#trajs=[]
#for i in range(0,len(pdbs)):
#   try:
#      traj = md.load_pdb(pdbs[i])
#      trajs.append(traj)
#   except KeyboardInterrupt:
#      raise
#   except Exception as e:
#      raise

#mtraj = md.join(trajs)
#mtraj.save('ensemble.xtc')

pdb='../../segment_5_input_af_0_sys.pdb'
xtc='../../segment_5_input_af_rebuilt.xtc'
#pdb=pdbs[0]
print(pdb)
traj = md.load(xtc, top=pdb, stride=1)
topology = md.load(pdb).topology


dssp=md.compute_dssp(traj,simplified='false')
print('dssp data shape', dssp.shape)

nresid=topology.n_residues
residuesALL=b = [i+1 for i in range(nresid)]
array2 = np.vstack((residuesALL,dssp))
dssp=np.c_[["" for x in range(len(array2))],array2]

df2=pd.DataFrame(data=dssp[1:,1:],
                index=dssp[1:,0],
                columns=dssp[0,1:])

print(pd.DataFrame(data=dssp[1:,1:],
                  index=dssp[1:,0],
                  columns=dssp[0,1:]))

df2.to_csv('dssp.dat', header=True, index=None, sep=' ', mode='w')


print(len(dssp)-1)

dsspC_resid=[]
dsspH_resid=[]
dsspE_resid=[]

for resid in range(1,nresid+1):
    dummyC=0
    dummyH=0
    dummyE=0
    for t in range(0,len(dssp)-1):
        t_i=t+1
       #print(weights_10[t],sasa[t_i][resid])
        if (dssp[t_i][resid] =='C'):
            dummyC+=float(1/len(traj))
        elif(dssp[t_i][resid] =='H'):
            dummyH+=float(1/len(traj))
        elif(dssp[t_i][resid] =='E'):
            dummyE+=float(1/len(traj))
        
    dsspC_resid.append(dummyC)
    dsspH_resid.append(dummyH)
    dsspE_resid.append(dummyE)
    print(resid,dummyC,dummyH,dummyE)



print(len(dssp)-1)


f=open('helix.dat','w')
for i in range(0,len(residuesALL)):
        print(residuesALL[i])
        f.write(str(residuesALL[i])+" "+str(dsspH_resid[i])+"\n")
f.close()


f=open('coil.dat','w')
for i in range(0,len(residuesALL)):
        print(residuesALL[i])
        f.write(str(residuesALL[i])+" "+str(dsspC_resid[i])+"\n")
f.close()

f=open('bsheet.dat','w')
for i in range(0,len(residuesALL)):
        print(residuesALL[i])
        f.write(str(residuesALL[i])+" "+str(dsspE_resid[i])+"\n")
f.close()

###
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(residuesALL,dsspC_resid,color="black",alpha=0.5)
plt.ylim(0,1)
plt.xlim(1,644)

plt.yticks(fontsize=15)
plt.xticks(fontsize=15)

plt.xlabel('Residue',fontsize=15)
plt.ylabel('% coil',fontsize=15)
#plt.legend(fontsize=20,loc=1,prop={'size': 5})
ax.tick_params(axis='both', labelsize=15)
ax.legend=None
plt.savefig('coil.pdf',bbox_inches='tight')
plt.clf()
plt.close()



#######
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(residuesALL,dsspH_resid,color="black",alpha=0.5)
plt.ylim(0,1.)
plt.xlim(1,644)


plt.xlabel('Residue',fontsize=15)
plt.ylabel('% helix',fontsize=15)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)

#plt.legend(fontsize=20,loc=1,prop={'size': 5})
ax.tick_params(axis='both', labelsize=15)
ax.legend=None
plt.savefig('helix.pdf',bbox_inches='tight')
plt.clf()
plt.close()


#######
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(residuesALL,dsspE_resid,color="black",alpha=0.5)
plt.ylim(0,1)
plt.xlim(1,644)

plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlabel('Residue',fontsize=15)
plt.ylabel('% β-sheet',fontsize=15)
#plt.legend(fontsize=20,loc=1,prop={'size': 5})
ax.tick_params(axis='both', labelsize=15)
ax.legend=None
plt.savefig('beta.pdf',bbox_inches='tight')
plt.clf()
plt.close()
####
