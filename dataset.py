import utils

import torch
from torch.utils.data import Dataset

from librosa.core import load

import os
import numpy as np
from interpolate import interpolation


class FolderDataset(Dataset):

    def __init__(self, datasets_path, path, cond_path, overlap_len, q_levels, ulaw, seq_len, batch_size,
                 max_cond=None, min_cond=None):
        super().__init__()

        # Define class variables from initialization parameters
        self.overlap_len = overlap_len
        self.q_levels = q_levels
        self.ulaw = ulaw
        if self.ulaw:
            self.quantize = utils.uquantize
        else:
            self.quantize = utils.linear_quantize
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.max_cond = max_cond
        self.min_cond = min_cond

        # Define sets of data and speaker IDs
        self.data = []
        self.global_spk = []

        # Define numpy array for conditioners of size 44 = 40 MFCC + 1 LF0 + 1FV + 1 U/V
        # Conditioners are computed by Ahocoder every 80 audio samples (windows of 5ms at 16kHz sample rate)
        self.cond = np.empty(shape=[0, 43])
        self.cond_len = 80

        # Define npy training dataset files
        npy_name_data = 'npy_datasets/train_data.npy'
        npy_name_cond = 'npy_datasets/train_conditioners.npy'
        npy_name_spk = 'npy_datasets/train_speakers.npy'

        # Check if dataset has to be created
        files = [npy_name_data, npy_name_cond, npy_name_spk]
        create_dataset = len(files) != len([f for f in files if os.path.isfile(f)])

        nosync = True
        print('Extracting wav from: ', path)
        print('Extracting conditioning from: ', cond_path)
        print('\nCreate dataset ', '-'*60)

        if create_dataset:
            # Get file names from train list
            file_names = open(datasets_path + 'wav_train.list', 'r').read().splitlines()

            # Search for unique speakers in list and sort them
            spk = list()
            for file in file_names:
                current_spk = file[0:2]
                if current_spk not in spk:
                    spk.append(current_spk)
            spk.sort()

            # Load each of the files from the list. Note that extension has to be added
            for file in file_names:
                # Load WAV
                print(file + '.wav')
                (d, _) = load(path + file + '.wav', sr=None, mono=True)
                num_samples = d.shape[0]

                # Load CC conditioner
                c = np.loadtxt(cond_path + file + '.cc')
                if c.shape[1] != 40:
                    print('Error in cc conditioner dimension')
                    quit()
                c = c.reshape(-1, c.shape[1])
                (num_ceps, _) = c.shape

                # Load LF0 conditioner
                f0file = np.loadtxt(cond_path + file + '.lf0')
                f0, _ = interpolation(f0file, -10000000000)
                f0 = f0.reshape(f0.shape[0], 1)

                # Load FV conditioner
                fvfile = np.loadtxt(cond_path + file + '.gv')
                fv, uv = interpolation(fvfile, 1e3)
                num_fv = fv.shape[0]
                uv = uv.reshape(num_fv, 1)
                fv = fv.reshape(num_fv, 1)

                # Load speaker conditioner
                speaker = spk.index(file[0:2])

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
                    if (d.shape[0]/80) != c.shape[0]:
                        print('ERROR in file: {0}: shape={1}'.format(cond_path + file +
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

                # Concatenate all speech conditioners
                cond = np.concatenate((c, f0), axis=1)
                cond = np.concatenate((cond, fv), axis=1)
                cond = np.concatenate((cond, uv), axis=1)

                # Append/Concatenate current audio file, speech conditioners and speaker ID
                self.data = np.append(self.data, d)
                self.cond = np.concatenate((self.cond, cond), axis=0)
                self.global_spk = np.append(self.global_spk, speaker)

            total_samples = self.data.shape[0]
            dim_cond = self.cond.shape[1]
            print('Total samples: ', total_samples)
            
            lon_seq = self.seq_len+self.overlap_len
            self.num_samples = self.batch_size*(total_samples//(self.batch_size*lon_seq*self.cond_len))
            
            print('Number of samples (1 audio file): ', self.num_samples)
            num_conditioning = (self.seq_len + self.overlap_len)/self.cond_len
            self.total_samples = self.num_samples * (self.seq_len+self.overlap_len) * self.cond_len
            print('total samples', total_samples)
            total_conditioning = self.total_samples//self.cond_len
            print('total conditioning', total_conditioning)
            print('cond len', self.cond_len)
            # print('conditioning', self.cond.shape)
            self.data = self.data[:self.total_samples]
            print('dades tallades', self.data.shape)
            self.cond = self.cond[:total_conditioning]
            print('cond shape', self.cond.shape)
            self.data = self.data[:self.total_samples].reshape(self.batch_size, -1)
            print('dades shape', self.data.shape)

            # Normalize conditioners
            if self.max_cond is None:
                self.max_cond = np.amax(self.cond, axis=0)
                self.min_cond = np.amin(self.cond, axis=0)
            self.cond = (self.cond-self.min_cond)/(self.max_cond-self.min_cond)

            evalpar = False     # Georgina's v. was True to output shapes
            if evalpar:
                print('shape', self.cond.shape)
                cc = self.cond[:, 0:39]
                fvv = self.cond[:, 41]
                f00 = self.cond[:, 40]

                print('shape cc', cc.shape)
                medcc = cc.mean()
                medf0 = f00.mean()
                medfv = fvv.mean()
                print('Mean cc', medcc)
                print('Mean f0', medf0)
                print('Mean fv', medfv)
                varcc = np.var(cc)
                varfv = np.var(fvv)
                varf0 = np.var(f00)
                print('Var cc', varcc)
                print('Var f0', varf0)
                print('Var fv', varfv)
                # cc=cc.reshape(-1)
                fvv = fvv.reshape(-1)
                f00 = f00.reshape(-1)
                np.save('ccnormshape', cc)
                # np.save('fvnorm', fvv)
                # np.save('f0norm', f00)
                quit()

            self.cond = self.cond[:total_conditioning].reshape(self.batch_size, -1, dim_cond)

            print('cond reshape', self.cond.shape)
            print('total index', self.total_samples//self.seq_len)

            # Save training dataset
            np.save(npy_name_data, self.data)
            np.save(npy_name_cond, self.cond)
            np.save(npy_name_spk, self.global_spk)

        else:
            # Load previously created training dataset
            self.data = np.load(npy_name_data)
            self.cond = np.load(npy_name_cond)
            self.global_spk = np.load(npy_name_spk)

        print('Dataset created', '-'*60, '\n')

    def __getitem__(self, index):
        verbose = False

        # Compute which sample within n_batch has to be returned given an index
        n_batch, sample_in_batch = divmod(index, self.batch_size)

        # Compute start and end for both input data and target sequences
        start_data = n_batch * self.seq_len
        start_target = start_data + self.overlap_len
        end_target = start_target + self.seq_len

        if not self.ulaw:
            data = torch.from_numpy(self.data[sample_in_batch][start_data:end_target-1]).long()
            target = torch.from_numpy(self.data[sample_in_batch][start_target:end_target]).long()
        else:
            data = self.quantize(torch.from_numpy(self.data[sample_in_batch][start_data:end_target-1]), self.q_levels)
            target = self.quantize(torch.from_numpy(self.data[sample_in_batch][start_target:end_target]), self.q_levels)

        # Count number of acoustic parameters computations in a sequence (1 computation every 80 audio samples)
        cond_in_seq = self.seq_len//self.cond_len

        if n_batch == 0:        # Reset all hidden states to avoid predicting with non-related samples
            reset = True
            from_cond = n_batch * cond_in_seq + 1
        else:
            reset = False
            from_cond = n_batch * cond_in_seq + 2

        to_cond = from_cond + cond_in_seq

        if verbose:
            print('batch', n_batch)
            print('sample in batch', sample_in_batch)
            print('from cond', from_cond)
            print('to cond', to_cond)

        cond = torch.from_numpy(self.cond[sample_in_batch][from_cond:to_cond])

        if verbose:
            print('data get item ', data.size())
            print('cnd shape getitem', cond.size())

        # Get the speaker ID
        spk = self.global_spk[sample_in_batch]

        return data, reset, target, cond, spk

    def __len__(self):
        return self.total_samples//self.seq_len

    def cond_range(self):
        # Compute conditioners range if not done
        if self.max_cond is None:
            self.max_cond = np.amax(self.cond, axis=0)
            self.min_cond = np.amin(self.cond, axis=0)
        return self.max_cond, self.min_cond
