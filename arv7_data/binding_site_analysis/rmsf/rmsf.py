import numpy as np
from matplotlib import  pyplot as plt


x=np.loadtxt("rmsf.xvg")[:, 0]
y=np.loadtxt("rmsf.xvg")[:, 1]
print(x)
print(y)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
#ax = fig.add_axes([0,0,1,1])
#ax.scatter(x,y, color="r",alpha=0.3)
#plt.ylim(0,1.2)
ax.bar(x,y,color="black",alpha=0.5)
#plt.plot(x,y)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlabel('Residue',fontsize=15)
plt.ylabel('RMSF (nm)',fontsize=15)
ax.tick_params(axis='both', labelsize=15)
ax.legend=None
plt.savefig('rmsf.pdf',bbox_inches='tight')
#plt.savefig('rmsf.png')

