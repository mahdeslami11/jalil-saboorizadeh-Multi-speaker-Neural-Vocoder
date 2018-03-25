import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pl

cc=np.load('fvnormMUSA.npy')
#for i in range(len(cc)):

 #   print(cc[i])
#cc=cc.reshape(-1,40)
print('shape', cc.shape )
print(cc.shape)
print(cc)
#cc=cc[10]
bins = np.linspace(0, 0.2, 300)
fig = pl.hist(cc,bins)

pl.title('Mean')
pl.xlabel("value")
pl.ylabel("Frequency")
pl.savefig("fv2normMUSA.png")


