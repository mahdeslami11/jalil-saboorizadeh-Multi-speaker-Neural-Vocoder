import torch
import numpy as np
from   torch.utils.data import Dataset, DataLoader
import utils

class myDataset(Dataset):

    def __init__(self):
        ratio_min=0
        ratio_max=0.8
        
        npynamecond= 'conditioning'+'_'+str(ratio_min)+'_'+str(ratio_max)+'.npy'
        npynamedata= 'cdatamatrix'+'_'+str(ratio_min)+'_'+str(ratio_max)+'.npy'
        self.cond=np.load(npynamecond)
        self.data=np.load(npynamedata)
        self.num_samples=self.data.shape[0]*self.data.shape[1]

    def __getitem__(self, index):
        if index < 0 or index >= 100000000:
            raise IndexError
        nbatch, sample_in_batch = divmod(index, 128)
        print('sample in batch', sample_in_batch)
        begseq  = nbatch * 1040 + 80
        print('begseq', begseq)
        fromseq = begseq - 80
        print( 'fromseq' , fromseq )
        toseq   = begseq + 1040
        print( 'toseq' , toseq )
        reset =False
        
        data = self.data[sample_in_batch][nbatch][fromseq:toseq-1]
        print('data get item ', data)
        target = self.data[sample_in_batch][nbatch][begseq:toseq]
        print('target', target)
        return (data, target)

    def __len__(self):
        return self.num_samples

dset = myDataset()
print('Size dataset:', len(dset))

dloader = DataLoader(dset, 128, shuffle=False, drop_last = True)

print ('END LOADING DATA', '*'*40)
test=dset[1]
print(test)
iteration, num_epochs = (0,2)
for epoch in range(num_epochs):
    print('ep')
    for (iteration, fulldata) in enumerate(dloader, iteration + 1):
        print('op')
        (data, target) = fulldata


        print('Epoch=', epoch, ' Iteration=', iteration)

        data2 = data
        print('Data', data2)


        target2 = target
        print('Target', target2)
