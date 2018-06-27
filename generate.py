from model import SampleRNN, Predictor, Generator

import torch

import re
import sys
import numpy as np
import argparse
from librosa.output import write_wav

import os
from interpolate import interpolation

import random

from tensorboardX import SummaryWriter

default_params = {
    # model parameters
    'n_rnn': 1,
    'dim': 1024,
    'learn_h0': True,
    'ulaw': True,
    'q_levels': 256,
    'weight_norm': False,
    'seq_len': 1040,
    'batch_size': 128,
    'look_ahead': False,
    'qrnn': False,
    'val_frac': 0.1,
    'test_frac': 0.1,
    'cond_dim': 43,         # Conditioners of size 43 = 40 MFCC + 1 LF0 + 1FV + 1 U/V
    'norm_ind': False,      # If true, normalization is done independent by speaker. If false, normalization is joint
    'static_spk': False,    # If true, training is only done with one speaker

    # training parameters
    'sample_rate': 16000,
    'n_samples': 1,
    'sample_length': 80000,
    'seed': 77977,
    'cond': 0,

    # generator parameters
    'datasets_path': '/veu/tfgveu7/project/tcstar/',
    'cond_set': 'cond/'
}


def init_random_seed(seed, cuda):
    print('Seed:', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def as_type(var, target_type):
    case = str(target_type).split('\'')[1].split('\'')[0]
    if case == 'bool':
        return var[0] == 'T'
    elif case == 'int':
        return int(var)
    elif case == 'float':
        return float(var)
    elif case == 'list':
        return list(map(int, var.split(',')))
    else:
        return var


def load_model(checkpoint_path):
    model_pattern = '.*ep{}-it{}'

    checkpoint_name = os.path.basename(checkpoint_path)
    match = re.match(
        model_pattern.format(r'(\d+)', r'(\d+)'),
        checkpoint_name
    )
    if match:
        epoch = int(match.group(1))
        iteration = int(match.group(2))
    else:
        epoch, iteration = (0, 0)

    return torch.load(checkpoint_path), epoch, iteration


class RunGenerator:
    def __init__(self, model, sample_rate, cuda, epoch, cond, spk_list, speaker,
                 checkpoints_path, original_name, writer):
        self.generate = Generator(model, cuda)
        self.sample_rate = sample_rate
        self.cuda = cuda
        self.epoch = epoch
        self.cond = cond
        self.speaker = speaker
        self.writer = writer
        self.original_name = original_name

        path_split = checkpoints_path.split('/')
        self.filename = '/'.join(path_split[:2]) + '/samples/' + path_split[-1] + '_file-' + \
                        self.original_name + '_spk-' + spk_list[self.speaker] + '.wav'
        print('Generating file', self.filename)

    def __call__(self, n_samples, sample_length, cond, speaker):
        print('Generate', n_samples, 'of length', sample_length)
        samples = self.generate(n_samples, sample_length, cond, speaker, self.writer, self.original_name).cpu().numpy()
        for i in range(n_samples):
            print(self.filename)

            write_wav(
                self.filename,
                samples[i, :], sr=self.sample_rate
            )


def main(frame_sizes, **params):

    use_cuda = torch.cuda.is_available()

    params = dict(
        default_params,
        frame_sizes=frame_sizes, 
        **params
    )

    # Redefine parameters listed in the experiment directory and separated with '~'
    for i in params['model'].split('/')[1].split('~'):
        param = i.split(':')
        if param[0] in params:
            params[param[0]] = as_type(param[1], type(params[param[0]]))
    # Define npy file names with maximum and minimum values of de-normalized conditioners
    npy_name_min_max_cond = 'npy_datasets/min_max' + params['norm_ind'] * '_ind' + (not params['norm_ind']) * '_joint' \
                            + params['static_spk'] * '_static' + '.npy'

    # Define npy file name with array of unique speakers in dataset
    npy_name_spk_id = 'npy_datasets/spk_id.npy'

    # Get file names from partition's list
    file_names = open(str(params['datasets_path']) +
                      'generate_cond_vocoder.list', 'r').read().splitlines()

    spk_names = open(str(params['datasets_path']) +
                     'generate_spk_vocoder.list', 'r').read().splitlines()

    datasets_path = os.path.join(params['datasets_path'], params['cond_set'])

    spk = np.load(npy_name_spk_id)

    if len(spk_names) != len(file_names):
        print('Length of speaker file do not match length of conditioner file.')
        quit()

    print('Generating', len(file_names), 'audio files')

    writer = SummaryWriter(log_dir='generator_samplernn')
    
    for i in range(len(file_names)):
        print('Generating Audio', i)
        print('Generating...', file_names[i])

        # Load CC conditioner
        c = np.loadtxt(datasets_path + file_names[i] + '.cc')

        # Load LF0 conditioner
        f0file = np.loadtxt(datasets_path + file_names[i] + '.lf0')
        f0, _ = interpolation(f0file, -10000000000)
        f0 = f0.reshape(f0.shape[0], 1)

        # Load FV conditioner
        fvfile = np.loadtxt(datasets_path + file_names[i] + '.gv')
        fv, uv = interpolation(fvfile, 1e3)
        num_fv = fv.shape[0]
        uv = uv.reshape(num_fv, 1)
        fv = fv.reshape(num_fv, 1)

        # Load speaker conditioner
        speaker = np.where(spk == spk_names[i])[0][0]

        cond = np.concatenate((c, f0), axis=1)
        cond = np.concatenate((cond, fv), axis=1)
        cond = np.concatenate((cond, uv), axis=1)

        # Load maximum and minimum of de-normalized conditioners
        min_cond = np.load(npy_name_min_max_cond)[0]
        max_cond = np.load(npy_name_min_max_cond)[1]

        # Normalize conditioners with absolute maximum and minimum for each speaker of training partition
        if params['norm_ind']:
            print('Normalizing conditioners for each speaker of training dataset')
            cond = (cond - min_cond[speaker]) / (max_cond[speaker] - min_cond[speaker])
        else:
            print('Normalizing conditioners jointly')
            cond = (cond - min_cond) / (max_cond - min_cond)

        print('Shape cond', cond.shape)
        if params['look_ahead']:
            delayed = np.copy(cond)
            delayed[:-1, :] = delayed[1:, :]
            cond = np.concatenate((cond, delayed), axis=1)
            print('Shape cond after look ahead', cond.shape)

        print(cond.shape)
        seed = params.get('seed')
        init_random_seed(seed, use_cuda)

        spk_dim = len([i for i in os.listdir(os.path.join(params['datasets_path'], params['cond_set']))
                       if os.path.islink(os.path.join(params['datasets_path'], params['cond_set']) + '/' + i)])

        print('Start Generate SampleRNN')
        model = SampleRNN(
            frame_sizes=params['frame_sizes'],
            n_rnn=params['n_rnn'],
            dim=params['dim'],
            learn_h0=params['learn_h0'],
            q_levels=params['q_levels'],
            ulaw=params['ulaw'],
            weight_norm=params['weight_norm'],
            cond_dim=params['cond_dim']*(1+params['look_ahead']),
            spk_dim=spk_dim,
            qrnn=params['qrnn']
        )
        print(model)
        
        if use_cuda:
            model = model.cuda()
            predictor = Predictor(model).cuda()
        else:
            predictor = Predictor(model)

        f_name = params['model']
        model_data = load_model(f_name)
    
        if model_data is None:
            sys.exit('ERROR: Model not found in' + str(f_name))
        (state_dict, epoch_index, iteration) = model_data
        print('OK: Read model', f_name, '(epoch:', epoch_index, ')')
        print(state_dict)
        predictor.load_state_dict(state_dict)

        original_name = file_names[i].split('/')[1]
        if original_name == "..":
            original_name = file_names[i].split('/')[3]

        generator = RunGenerator(
            model=model,
            sample_rate=params['sample_rate'],
            cuda=use_cuda,
            epoch=epoch_index,
            cond=cond,
            spk_list=spk,
            speaker=speaker,
            checkpoints_path=f_name,
            original_name=original_name,
            writer=writer
         )

        generator(params['n_samples'], params['sample_length'], cond, speaker)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS
    )

    def parse_bool(arg):
        arg = arg.lower()
        if 'true'.startswith(arg):
            return True
        elif 'false'.startswith(arg):
            return False
        else:
            raise ValueError()

    parser.add_argument(
        '--frame_sizes', nargs='+', type=int, required=True,
        help='frame sizes in terms of the number of lower tier frames, \
              starting from the lowest RNN tier'
    )
    parser.add_argument(
        '--model', required=True,
        help='model (including path)'
    )
    parser.add_argument(
        '--n_rnn', type=int, help='number of RNN layers in each tier'
    )
    parser.add_argument(
        '--dim', type=int, help='number of neurons in every RNN and MLP layer'
    )
    parser.add_argument(
        '--learn_h0', type=parse_bool,
        help='whether to learn the initial states of RNNs'
    )
    parser.add_argument(
        '--ulaw', type=parse_bool,
        help='use  u-law quantization'
    )
    parser.add_argument(
        '--q_levels', type=int,
        help='number of bins in quantization of audio samples'
    )
    parser.add_argument(
        '--seq_len', type=int,
        help='how many samples to include in each truncated BPTT pass'
    )
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument(
        '--datasets_path', help='path to the directory to find the conditioning'
    )
    parser.add_argument(
        '--cond_set',
        help='cond_set name - name of a directory in the conditioning sets path \
                 (settable by --datasets_path)'
    )
    parser.add_argument(
        '--sample_rate', type=int,
        help='sample rate of the training data and generated sound'
    )
    parser.add_argument(
        '--n_samples', type=int,
        help='number of samples to generate in each epoch'
    )
    parser.add_argument(
        '--sample_length', type=int,
        help='length of each generated sample (in samples)'
    )
    parser.add_argument(
        '--norm_ind', type=parse_bool,
        help='Apply conditioner normalization independently by speaker or jointly if false'
    )
    parser.add_argument(
        '--look_ahead', type=float,
        help='Take conditioners from current and next frame'
    )
    parser.add_argument(
        '--static_spk', type=parse_bool,
        help='Only train with one speaker'
    )
    parser.add_argument(
        '--seed', type=int,
        help='seed initialization of random generator'
    )
    parser.add_argument(
        '--weight_norm', type=parse_bool,
        help='Apply weight normalization'
    )

    parser.set_defaults(**default_params)

    main(**vars(parser.parse_args()))
