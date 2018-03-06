import utils

import torch
from torch.utils.data import (
    Dataset, DataLoader as DataLoaderBase
)

from librosa.core import load
from natsort import natsorted
from tempfile import TemporaryFile

import os
from os import listdir
from os.path import join
import glob
import numpy as np

class FolderDataset(Dataset):
# 'seq_len': 1024

    def __init__(self, path, overlap_len, q_levels, ratio_min=0, ratio_max=1):
        super().__init__()
        self.overlap_len = overlap_len
        self.q_levels = q_levels
        self.seq_len = 1040
        self.batch_size = 128
        self.data = []
        self.cond = []
        #self.cond = np.empty(shape=[0,40])
        max_cond=None
        nosync=True
        print(path)
        print('\nCreate dataset ','-'*60)
        file_names = natsorted(glob.glob(os.path.join(path, '*.wav')))
        file_ceps = natsorted(glob.glob(os.path.join(path, '*.cc')))
        file_f0 = natsorted(glob.glob(os.path.join(path, '*.lf0')))
        file_fv = natsorted(glob.glob(os.path.join(path, '*.fv')))

        self.file_names = file_names[
            int(ratio_min * len(file_names)) : int(ratio_max * len(file_names))
        ]
        self.file_ceps = file_ceps[
            int(ratio_min * len(file_ceps)) : int(ratio_max * len(file_ceps))
        ]
        self.file_f0 = file_f0[
            int(ratio_min * len(file_f0)) : int(ratio_max * len(file_f0))
        ]
        self.file_fv  = file_fv [
            int(ratio_min * len(file_fv )) : int(ratio_max * len(file_fv ))
        ]
         
    
        num_files = len(self.file_names)
        print(num_files)
        for i in range(num_files):
            print(file_names[i])
            (d, _) = load(file_names[i], sr=None, mono=True)
            num_samples = d.shape[0]
            print ('num samples', num_samples)
            c = np.loadtxt(file_ceps[i])
            print('c', c.shape)
            (num_ceps, _) = c.shape
            print('num ceps', num_ceps)
            if nosync:
                oversize = num_samples%80
                print('oversize', oversize)
                if oversize >= 60:
                     zeros = 80 - oversize
                     d = np.append(d, np.zeros(zeros))
                     print('c shape over', c.shape)
                     print('samples', d.shape)
                     print('oversize >60')
                if oversize <= 60 and oversize != 0:
                    d = d[:-oversize]
                    c = c[:-1][:]
                    print('samples', d.shape)
                    print('c shape no over', c.shape)
                    print('oversize <60')
            else:
                truncate = num_ceps*80
                d = d[:truncate]
                
               
            
            self.data = np.append(self.data, d)
            self.cond = np.concatenate((self.cond, c), axis=0)
            print('cond', self.cond.shape)
        total_samples=self.data.shape[0]
        total_ceps=self.cond.shape[0]

        print('\ntotal samples:', total_samples,
              '\nseq_len:', self.seq_len,
              '\nComplete seq:', total_samples//self.seq_len,
              '\nbatch_size:', self.batch_size,
              '\n')
        self.num_samples = self.batch_size * (total_samples//(self.batch_size*self.seq_len*80))
        num_conditioning= self.seq_len/80
        total_samples = self.num_samples * self.seq_len * 80
        print('total samples', total_samples)
        total_conditioning = total_samples//80
        print('total conditioning', total_conditioning)
        print('conditioning', self.cond.shape)
        self.data = self.data[:total_samples]
        self.cond = self.cond[:total_conditioning]
        self.data = self.data[:self.num_samples * self.seq_len].reshape(self.batch_size, -1, self.seq_len)
        print('cond', self.cond)
        print('cons shape', self.cond.shape)
        print('data shape:', self.data.shape)
        self.cond = self.cond.reshape(self.batch_size, -1, self.seq_len//2, 40)
        if max_cond == None:
            max_cond=np.amax(self.cond, axis=0)
            min_cond=np.amin(self.cond, axis=0)   # Minima along the first axis
        else:
            max_cond=max_cond
            min_cond=min_cond   # Minima along the first axis
        self.cond = (self.cond-min_cond)/(max_cond-min_cond) 
        print('cond', max_cond)
        print('min', min_cond)
        print('cond reshape', self.cond.shape)
        np.save('conditioning.npy', self.cond)
        np.save('datamatrix.npy', self.data)
        print('Dataset created','-'*60,'\n')
        


    def __getitem__(self, index):
        nbatch, sample_in_batch = divmod(index, self.batch_size)
        #return (self.cond[sample_in_batch,nbatch], self.data[sample_in_batch,nbatch])
        return (self.data[sample_in_batch,nbatch])


    def __len__(self):
        return len(self.file_names)



