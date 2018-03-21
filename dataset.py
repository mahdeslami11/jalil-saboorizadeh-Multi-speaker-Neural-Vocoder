import utils

import torch
from torch.utils.data import Dataset

from librosa.core import load
from natsort import natsorted

import os
import glob
import numpy as np
from interpolate import interpolation


class FolderDataset(Dataset):   # 'seq_len': 1024

    def __init__(self, datasets_path, path, cond_path, overlap_len, q_levels, ulaw, seq_len, batch_size,
                 max_cond=None, min_cond=None):
        super().__init__()
        self.overlap_len = overlap_len
        self.q_levels = q_levels
        self.ulaw = ulaw
        if ulaw:
            self.quantize = utils.uquantize
        else:
            self.quantize = utils.linear_quantize
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.data = []
        self.cond = []
        # arreclar aquest cond borrar la primera
        self.cond = np.empty(shape=[0, 43])
        self.cond_len = 80 
        self.max_cond = max_cond
        self.min_cond = min_cond
        
        create_dataset = True
        nosync = True
        print('Extracting wav from: ', path)
        print('Extracting conditioning from: ', cond_path)
        print('\nCreate dataset ', '-'*60)
        if create_dataset:

            file_names = open(datasets_path + 'wav_train.list', 'r').read().splitlines()

            num_files = len(file_names)

            for i in range(num_files):
                # Load WAV
                print(self.file_names[i] + '.wav')
                (d, _) = load(path + file_names[i] + '.wav', sr=None, mono=True)
                num_samples = d.shape[0]

                # Load CC conditioning
                print('Load cepstrum {0}:{1}'.format(i, file_names[i]) + '.cc')
                c = np.loadtxt(cond_path + file_names[i] + '.cc')
                c = c.reshape(-1, 40)
                (num_ceps, _) = c.shape

                # Load LF0 conditioning
                f0file = np.loadtxt(cond_path + file_names[i] + '.lf0')
                # interp es la senyal interpolada, uv es el flag (UV) de mom el deixem
                f0, uv = interpolation(f0file, -10000000000)
                num_f0 = f0.shape[0]
                f0 = f0.reshape((num_f0, 1))

                # Load GV conditioning
                fvfile = np.loadtxt(cond_path + file_names[i] + '.gv')
                fv, uv = interpolation(fvfile, 1e3)
                num_fv = fv.shape[0]
                uv = uv.reshape((num_fv, 1))
                fv = fv.reshape((num_fv, 1))

                if nosync:
                    oversize = num_samples % 80
                    print('oversize', oversize)
                    if oversize >= 60:
                        zeros = 80 - oversize
                        d = np.append(d, np.zeros(zeros))
                        # print('c shape over', c.shape)
                        # print('samples', d.shape)
                        # print('oversize >60')
                    if oversize <= 60 and oversize != 0:
                        d = d[:-oversize]
                        c = c[:-1][:]
                        f0 = f0[:-1]
                        fv = fv[:-1]
                        uv = uv[:-1]
                    # if oversize != 0:
                    #   d = d[:-oversize]
                    #   c = c[:-1][:]
                    #   f0 = f0[:-1]
                    #   fv = fv[:-1]
                    #   uv = uv[:-1]
                    print('Theoretically', d.shape[0]/80)
                    print('real cond', c.shape)
                    if (d.shape[0]/80) != c.shape[0]:
                        print('ERROR in file: {0}: shape={1}'.format(cond_path + file_names[i] +
                                                                     '.cc', d.shape[0]), '_'*50)
                        # c = c[:-1][:]
                        # f0 = f0[:-1]
                        # fv = fv[:-1]
                        # uv = uv[:-1]
                        # print('Theoretically edited', d.shape[0]/80)
                        # print('real cond edited', c.shape)

                    # if (d.shape[0]/80) !=( c.shape[0]):
                        # print('error')
                        # break
                        # print('samples', d.shape)
                        # print('c shape no over', c.shape)
                        # print('f0', f0.shape)
                        # print('oversize <60')
                else:
                    truncate = num_ceps*80
                    d = d[:truncate]
                if not ulaw:
                    d = self.quantize(torch.from_numpy(d), self.q_levels).numpy()
                condi = np.concatenate((c, f0), axis=1)
                condi = np.concatenate((condi, fv), axis=1)
                condi = np.concatenate((condi, uv), axis=1)
                # print('shape', condi.shape)
                # self.data = np.append(self.data, self.quantize(torch.from_numpy(d), self.q_levels))
                self.data = np.append(self.data, d)
                # print('data shape', self.data.shape)
                self.cond = np.concatenate((self.cond, condi), axis=0)
                # print('cond', self.cond.shape)
            total_samples = self.data.shape[0]
            total_cond = self.cond.shape[0]
            dim_cond = self.cond.shape[1]
            print('total dades', total_samples)
            complete_seq = total_samples//(self.seq_len+self.overlap_len)
            # print('\ntotal samples:', total_samples,
            # '\nseq_len:', self.seq_len, '\nlon_seq:', self.seq_len + self.overlap_len,
            # '\nComplete', complete_seq, '\nbatch_size:', self.batch_size, '\n')
            
            lon_seq=self.seq_len+self.overlap_len
            self.num_samples = self.batch_size*(total_samples//(self.batch_size*lon_seq*self.cond_len))
            
            print('num samples', self.num_samples)
            #print(self.cond_len)
            num_conditioning = (self.seq_len + self.overlap_len)/self.cond_len
            self.total_samples = self.num_samples * (self.seq_len+self.overlap_len) * self.cond_len
            print('total samples', total_samples)
            total_conditioning = self.total_samples//self.cond_len
            print('total conditioning', total_conditioning)
            print('cond len', self.cond_len)
            #print('conditioning', self.cond.shape)
            self.data = self.data[:self.total_samples]
            print('dades tallades', self.data.shape)
            self.cond = self.cond[:total_conditioning]
            print('cond shape', self.cond.shape)
            self.data = self.data[:self.total_samples].reshape(self.batch_size, -1)
            print('dades shape', self.data.shape)
            if self.max_cond is None:
                self.max_cond=np.amax(self.cond, axis=0)
                self.min_cond=np.amin(self.cond, axis=0) 
            self.cond = (self.cond-self.min_cond)/(self.max_cond-self.min_cond)
            evalpar=False	# Georgina's v. was True to output shapes
            if evalpar:
                print('shape', self.cond.shape)
                cc=self.cond[:,0:39]
                fvv=self.cond[:,41]
                f00=self.cond[:,40]

                print('shape cc', cc.shape)
                medcc=cc.mean()
                medf0=f00.mean()
                medfv=fvv.mean()
                print('Mean cc', medcc)
                print('Mean f0',medf0)
                print('Mean fv', medfv)
                varcc=np.var(cc)
                varfv=np.var(fvv)
                varf0=np.var(f00)
                print('Var cc', varcc)
                print('Var f0',varf0)
                print('Var fv', varfv)
                #cc=cc.reshape(-1)
                fvv=fvv.reshape(-1)
                f00=f00.reshape(-1)
                np.save('ccnormshape', cc)
                #np.save('fvnorm', fvv)
                #np.save('f0norm', f00)
                quit()



           # print('cons shape', self.cond.shape)
           # print('data shape:', self.data.shape)
            #self.cond = self.cond.reshape(self.batch_size, -1, (self.seq_len + self.overlap_len)//self.cond_len, dim_cond)
            self.cond = self.cond[:total_conditioning].reshape(self.batch_size, -1, dim_cond)

           # print('cond', self.max_cond)
           # print('min', self.min_cond)
            print('cond reshape', self.cond.shape)
            print('total index', (self.total_samples)//(self.seq_len)) 
            npynamecond= 'conditioning'+'_'+str(ratio_min)+'_'+str(ratio_max)+'.npy'
            npynamedata='cdatamatrix'+'_'+str(ratio_min)+'_'+str(ratio_max)+'.npy'
            npymax = cond_path + '/max'
            npymin= cond_path + '/min' 
            np.save(npynamecond, self.cond)
            np.save(npynamedata, self.data)
            np.save(npymax, self.max_cond)
            np.save(npymin, self.min_cond)
        else:
            file_names = natsorted(glob.glob(os.path.join(path, '*.wav')))
            self.file_names = file_names[int(ratio_min * len(file_names)) : int(ratio_max * len(file_names))]
            npynamecond= cond_path + 'conditioning'+'_'+str(ratio_min)+'_'+str(ratio_max)+'.npy'
            npynamedata= cond_path + 'cdatamatrix'+'_'+str(ratio_min)+'_'+str(ratio_max)+'.npy'
            self.cond=np.load(npynamecond)
            self.data=np.load(npynamedata)
            self.max_cond=np.load('max.npy')
            self.min_cond=np.load('min.npy')
        print('Dataset created','-'*60,'\n')



    def __getitem__(self, index):
        verbose=False
        nbatch, sample_in_batch = divmod(index, self.batch_size)
        # print('sample in batch', sample_in_batch)
        # print('nbatch', nbatch)
        begseq  = nbatch * self.seq_len + self.overlap_len
        fromseq = begseq - self.overlap_len
        toseq   = begseq + self.seq_len
        #quantdata = self.quantize(torch.from_numpy(self.data[sample_in_batch][fromseq:toseq]), self.q_levels)
        #data = quantdata[:-1]
        #target = quantdata[self.overlap_len:]
        # print('begseq', begseq)
        # print( 'fromseq' , fromseq ) 
        if not self.ulaw:
            data = torch.from_numpy(self.data[sample_in_batch][fromseq:toseq-1]).long()
            target = torch.from_numpy(self.data[sample_in_batch][begseq:toseq]).long()
        cond_batch=self.cond.shape[1]
        # print('cond in batch', cond_batch)
        cond_in_seq= (self.seq_len)//self.cond_len
        #print('cond in seq', cond_in_seq)
        # print( 'toseq' , toseq ) 
        if nbatch == 0:
            reset = True
            fromcond  =  nbatch * cond_in_seq + 1
        else:
            reset =False
            fromcond  =  nbatch * cond_in_seq + 2
            
        if verbose: 
            print('batch', nbatch)
            print('sample in batch', sample_in_batch)
            print('from cond', fromcond)
            print('to cond', tocond)
        tocond    = fromcond + cond_in_seq 
        # data = self.quantize(torch.from_numpy(self.data[sample_in_batch][nbatch][fromseq:toseq-1]), self.q_levels)
        if self.ulaw:
            data = self.quantize(torch.from_numpy(self.data[sample_in_batch][fromseq:toseq-1]), self.q_levels)
            target = self.quantize(torch.from_numpy(self.data[sample_in_batch][begseq:toseq]), self.q_levels)

         #cond = torch.from_numpy(self.cond[sample_in_batch][nbatch][fromcond:tocond])
        cond = torch.from_numpy(self.cond[sample_in_batch][fromcond:tocond])
        if verbose:
            print('data get item ', data.size())
            print('cnd shape getitem', cond.size())
        # target = self.quantize(torch.from_numpy(self.data[sample_in_batch][nbatch][begseq:toseq]), self.q_levels)
        # print('target', target)
        return data, reset, target, cond
        #return (self.cond[sample_in_batch,nbatch], self.data[sample_in_batch,nbatch])

    def __len__(self):
        return self.total_samples//self.seq_len

    def cond_range(self):
        return self.max_cond, self.min_cond
