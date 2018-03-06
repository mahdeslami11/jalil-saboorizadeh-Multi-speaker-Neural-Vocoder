#!/usr/bin/python3


import numpy as np
import torch
from   torch.utils.data import Dataset, DataLoader

import datetime
from timeit import default_timer as timer


show_dataset   = False
show_dataloader= False
show_3rd_seq   = True


##################################################

class myDataset(Dataset):

    def __init__(self, params, total_signal_samples = 221):
        super().__init__()

        print('\nCreate dataset ','-'*60)
        print(datetime.datetime.now())
        start = timer()
        
        
        self.seq_len      =  params.get('seq_len',  6)
        self.overlap      =  params.get('overlap',  2)
        self.split_target =  params.get('split_target', True)
        # seq_len/cond_len must be exact (e.g. 6 signal_samples, and 1 cond for each 2 signal_samples) 
        self.cond_len     =  self.seq_len//self.overlap

        # reply to create 'vectors'
        cond_dim   =  params.get('cond_dim', 2)
        
        self.batch_size = params.get('batch_size', 4)

        print('\ntotal signal_samples:', total_signal_samples,
              '\nseq_len:', self.seq_len,
              '\noverlap:', self.overlap,
              '\nComplete seq:', total_signal_samples//self.seq_len,
              '\nbatch_size:', self.batch_size,
              '\n')


        # Read the data
        self.data =  np.arange(total_signal_samples)
        

        # split total signal_samples in batch_size sequences
        # we need to overlap 'signal_samples', but to simplify prepare the data,
        # we do not overlap between the diff. batch sequences
        # Eg. nbatch_size=2, sample 2 values, overlap 1
        # signal samples: 0 1 2 3 4 5 6 7 8 9
        # batch seq 0): 0 1 2 3 4 => (0),1,2 - (2),3,4,
        # batch seq 1): 5 6 7 8 9 => (5),6,7 - (7),8,9
        

        num_seq = (total_signal_samples-self.batch_size * self.overlap)//(self.batch_size*self.seq_len)
        self.num_samples = self.batch_size * num_seq

        print('Num. samples:', self.num_samples)
        
        # Cut the last ones
        total_signal_samples = self.num_samples * self.seq_len + self.batch_size * self.overlap
        print('Cut from: ', self.data.size, ' signal_samples to ', total_signal_samples)
        print ('Total Signal_Samples: ', total_signal_samples)
        print ('   => batch_size=', self.batch_size,
               ' x (nsamples/batch =', self.num_samples//self.batch_size,
               ' x seqlen =', self.seq_len,
               ' + overlap =', self.overlap, ')')

        self.data = self.data[:total_signal_samples]

        # Read the conditioning
        self.cond = np.arange(-100, -100-self.num_samples*self.cond_len,-1)
        
        # repeat cond so that it is a vector of cond_len, same value.
        self.cond = np.broadcast_to(self.cond, (cond_dim,len(self.cond))).swapaxes(1, 0)

        print('n cond = ', len(self.cond))
        print('from ', self.cond[0], 'to', self.cond[-1])


        print('data shape (raw vector):', self.data.shape)        
        print('cond shape (raw 2d matrix):', self.cond.shape)        

        self.data = self.data.reshape(self.batch_size, -1)
        print('data shape:', self.data.shape)        

        # self.cond = self.cond.reshape(self.batch_size, -1, self.cond_len, cond_dim)
        self.cond = self.cond.reshape(self.batch_size, -1, cond_dim)
        print('cond shape', self.cond.shape)
        print('\nEllapsed time: ', timer() - start, 's.')

        print('Dataset created','-'*60,'\n')
        
    def __getitem__(self, index):
        if index < 0 or index >= self.num_samples:
            raise IndexError

        nbatch, sample_in_batch = divmod(index, self.batch_size)
        begseq  = nbatch * self.seq_len + self.overlap
        fromseq = begseq - self.overlap
        toseq   = begseq + self.seq_len

        fromcond  = nbatch * self.cond_len + 1
        tocond    = fromcond + self.cond_len
        
        if self.split_target:
            return (self.cond[sample_in_batch,fromcond:tocond], self.data[sample_in_batch,fromseq:toseq-1],
                    self.data[sample_in_batch, begseq:toseq])
        else:
            return (self.cond[sample_in_batch,fromcond:tocond], self.data[sample_in_batch,fromseq:toseq])

    def __len__(self):
        return self.num_samples


##################################################
    
# seq_len/overlap must be exact (e.g. seq_len=8 signal_samples, and overlap=2)
# There will be one cond for each overlap signal_samples, cond_len = 4(=> 1 cond for each 3 signal_samples) 
params = {
    'batch_size'  :    5,
    'seq_len'     :    9,
    'overlap'     :    3,
    'cond_dim'    :    2,
    'split_target':    True,
}



dset = myDataset(params)
print('Size dataset:', len(dset))

def __getitem__(self, index):
        if index < 0 or index >= self.num_samples:
            raise IndexError
        nbatch, sample_in_batch = divmod(index, self.batch_size)
        print('sample in batch', sample_in_batch)
        begseq  = nbatch * self.seq_len + self.overlap_len
        print('begseq', begseq)
        fromseq = begseq - self.overlap_len
        print( 'fromseq' , fromseq ) 
        toseq   = begseq + self.seq_len
        print( 'toseq' , toseq ) 
        reset =False
        fromcond  = nbatch * self.cond_len + 1
        tocond    = fromcond + self.cond_len
        data = self.quantize(torch.from_numpy(self.data[sample_in_batch][nbatch][fromseq:toseq-1]), self.q_levels)
        print('data get item ', data)
        cond = torch.from_numpy(self.cond[sample_in_batch][nbatch][fromcond:tocond])
        print('cnd', cond)
        target = self.quantize(torch.from_numpy(self.data[sample_in_batch][nbatch][begseq:toseq]), self.q_levels)
        print('target', target)
        return (data, target)
        #return (self.cond[sample_in_batch,nbatch], self.data[sample_in_batch,nbatch])

dloader = DataLoader(dset, batch_size=params['batch_size'], shuffle=False, drop_last = True)

print('data test', dset[1])

print ('END LOADING DATA', '*'*40)

if show_3rd_seq:
    print ('Show 3rd seq in batch during dataloader iteration\n', '-'*60)
    iteration, num_epochs = (0,2)
    for epoch in range(num_epochs):
        for (iteration, fulldata) in enumerate(dloader, iteration + 1):
            if dset.split_target:
                (cond, data, target) = fulldata
            else:
                (cond, data) = fulldata
                
            print('Epoch=', epoch, ' Iteration=', iteration)

            data2 = data[2]
            print('Data', data2)

            if dset.split_target:
                target2 = target[2]
                print('Target', target2)

            
            cond2 = cond[2]
            print('Cond')
            for n,c in enumerate(cond2):
                print(n, c, sep=": ", end=';')
            print('\n')



if show_dataset:
    print ('Show dataset iteration\n', '-'*60, len(dset))
    for i,seq in enumerate(dset):
        print('Sample', i, ':', seq)
    print('='*60)


if show_dataloader:
    print ('Show dataloader iteration\n', '-'*60)
    iteration, num_epochs = (0,2)
    for epoch in range(num_epochs):
        for (iteration, fulldata) in enumerate(dloader, iteration + 1):
            if dset.split_target:
                (cond, data, target) = fulldata
            else:
                (cond, data) = fulldata

            print('Epoch=', epoch, ' Iteration=', iteration)
            print('Cond.=', cond)
            print('Data =', data)
            if dset.split_target:
                print('Target =', target)
