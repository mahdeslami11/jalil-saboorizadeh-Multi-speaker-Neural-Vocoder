import utils

import torch
from torch.utils.data import Dataset

from librosa.core import load

import os
import numpy as np
from interpolate import interpolation


class FolderDataset(Dataset):

    def __init__(self, datasets_path, path, cond_path, overlap_len, q_levels, ulaw, seq_len, batch_size, cond_dim,
                 cond_len, partition):
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

        # Define sets of data, conditioners and speaker IDs
        self.data = []
        self.global_spk = []
        self.cond_dim = cond_dim
        self.cond_len = cond_len
        self.cond = np.empty(shape=[0, self.cond_dim])

        # Define npy training dataset file names
        npy_name_data = 'npy_datasets/' + partition + '_data.npy'
        npy_name_cond = 'npy_datasets/' + partition + '_conditioners.npy'
        npy_name_spk = 'npy_datasets/' + partition + '_speakers.npy'

        # Define npy file names with maximum and minimum values of de-normalized conditioners
        npy_name_min_max_cond = 'npy_datasets/min_max_joint_cond.npy'

        # Define npy file name with array of unique speakers in dataset
        npy_name_spk_id = 'npy_datasets/spk_id.npy'

        # Check if dataset has to be created
        files = [npy_name_data, npy_name_cond, npy_name_spk]
        create_dataset = len(files) != len([f for f in files if os.path.isfile(f)])

        nosync = True

        if create_dataset:
            print('Create ' + partition + ' dataset', '-' * 60, '\n')
            print('Extracting wav from: ', path)
            print('Extracting conditioning from: ', cond_path)
            print('List of files is: wav_' + partition + '.list')

            # Get file names from partition's list list
            file_names = open(datasets_path + 'wav_' + partition + '.list', 'r').read().splitlines()

            if not os.path.isfile(npy_name_spk_id):
                # Search for unique speakers in list and sort them
                spk_list = list()
                for file in file_names:
                    current_spk = file[0:2]
                    if current_spk not in spk_list:
                        spk_list.append(current_spk)
                spk_list.sort()
                spk = np.asarray(spk_list)
                np.save(npy_name_spk_id, spk)
            else:
                spk = np.load(npy_name_spk_id)

            # Load each of the files from the list. Note that extension has to be added
            for file in file_names:
                # Load WAV
                print(file + '.wav')
                (d, _) = load(path + file + '.wav', sr=None, mono=True)
                num_samples = d.shape[0]

                # Load CC conditioner
                c = np.loadtxt(cond_path + file + '.cc')
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

                # Load speaker conditioner (index where the ID is located)
                speaker = np.where(spk == file[0:2])[0][0]
                speaker = np.repeat(speaker, num_fv)

                if nosync:
                    oversize = num_samples % 80
                    print('oversize', oversize)
                    if oversize >= 60:
                        zeros = 80 - oversize
                        d = np.append(d, np.zeros(zeros))
                    if oversize <= 60 and oversize != 0:
                        d = d[:-oversize]
                        c = c[:-1][:]
                        f0 = f0[:-1]
                        fv = fv[:-1]
                        uv = uv[:-1]
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
            self.total_samples = self.num_samples * (self.seq_len+self.overlap_len) * self.cond_len
            total_conditioning = self.total_samples//self.cond_len
            self.data = self.data[:self.total_samples]
            self.cond = self.cond[:total_conditioning]
            self.data = self.data[:self.total_samples].reshape(self.batch_size, -1)

            self.length = self.total_samples // self.seq_len

            self.cond = self.cond[:total_conditioning].reshape(self.batch_size, -1, dim_cond)

            self.global_spk = self.global_spk[:total_conditioning].reshape(self.batch_size, -1)

            # Save maximum and minimum of de-normalized conditioners for conditions of train partition
            if partition == 'train' and not os.path.isfile(npy_name_min_max_cond):
                # Compute maximum and minimum of de-normalized conditioners for conditions of train partition
                print('Computing maximum and minimum values for each speaker of training dataset.')
                self.max_cond = np.amax(self.cond)
                self.min_cond = np.amin(self.cond)
                np.save(npy_name_min_max_cond, np.array([self.min_cond, self.max_cond]))

            # Load maximum and minimum of de-normalized conditioners
            else:
                self.min_cond = np.load(npy_name_min_max_cond)[0]
                self.max_cond = np.load(npy_name_min_max_cond)[1]

            # Normalize conditioners with absolute maximum and minimum for each speaker of training partition
            print('Normalizing conditioners.')
            self.cond = (self.cond - self.min_cond) / (self.max_cond - self.min_cond)

            # Save partition's dataset
            np.save(npy_name_data, self.data)
            np.save(npy_name_cond, self.cond)
            np.save(npy_name_spk, self.global_spk)

            print('Dataset created for ' + partition + ' partition', '-' * 60, '\n')

        else:
            # Load previously created training dataset
            self.data = np.load(npy_name_data)
            self.cond = np.load(npy_name_cond)
            self.global_spk = np.load(npy_name_spk)

            # Load maximum and minimum of de-normalized conditioners
            self.min_cond = np.load(npy_name_min_max_cond)[0]
            self.max_cond = np.load(npy_name_min_max_cond)[1]

            # Compute length for current partition
            self.length = np.prod(self.data.shape) // self.seq_len

            print('Data shape:', self.data.shape)
            print('Conditioners shape:', self.cond.shape)
            print('Global speaker shape:', self.global_spk.shape)

            print('Dataset loaded for ' + partition + ' partition', '-' * 60, '\n')

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
            from_cond = n_batch * cond_in_seq + 1

        to_cond = from_cond + cond_in_seq

        if verbose:
            print('batch', n_batch)
            print('sample in batch', sample_in_batch)
            print('from cond', from_cond)
            print('to cond', to_cond)

        cond = torch.from_numpy(self.cond[sample_in_batch][from_cond:to_cond])

        # Get the speaker ID for each conditioner in the sequence
        global_spk = self.global_spk[sample_in_batch][from_cond:to_cond]

        # Assume most repeated speaker as it doesn't matter on transitions from one audio to another
        global_spk = np.argmax(np.bincount(global_spk.astype(int)))

        spk = torch.from_numpy(np.array([global_spk]))

        if verbose:
            print('data size: ', data.size(), 'with sequence length: ', self.seq_len, 'and overlap: ', self.overlap_len)
            print('conditioner size: ', cond.size())
            print('speaker size: ', spk.size())

        return data, reset, target, cond, spk

    def __len__(self):
        return self.length
