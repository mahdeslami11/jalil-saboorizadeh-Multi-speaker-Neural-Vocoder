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
    'cond_dim': 43,

    # training parameters
    'sample_rate': 16000,
    'n_samples': 1,
    'sample_length': 80000,
    'seed': 77977,
    'cond': 0,

    # generator parameters
    'output_path': 'generated/',
    'cond_path': '/veu/tfgveu7/project/tcstar/',
    'cond_set': 'cond/'
}


def init_random_seed(seed, cuda):
    print('Seed:', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def as_type(var, target_type):
    case = str(target_type).split('\'')[1].split('\'')[0]
    if case == 'boolean':
        return var == 'True'
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
    def __init__(self, model, samples_path, sample_rate, cuda, epoch, cond, speaker, checkpoints_path, g):
        self.generate = Generator(model, cuda)
        self.samples_path = samples_path
        self.sample_rate = sample_rate
        self.cuda = cuda
        self.epoch = epoch
        self.cond = cond
        self.speaker = speaker

        g = str(g)

        m = re.search('/exp:(.+?)/checkpoints', checkpoints_path)
        if m:
            found = m.group(1)
            self.pattern = 'model_' + found + 'gen-ep{}-g' + g + '.wav'
            print('Generating file', self.pattern)

    def __call__(self, n_samples, sample_length, cond, speaker):
        print('Generate', n_samples, 'of length', sample_length)
        samples = self.generate(n_samples, sample_length, cond, speaker).cpu().numpy()
        maxv = np.iinfo(np.int16).max
        for i in range(n_samples):
            filename = os.path.join(self.samples_path, self.pattern.format(self.epoch, i))
            print(filename)

            write_wav(
                filename,
                (samples[i, :] * maxv).astype(np.int16), sr=self.sample_rate
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
    npy_name_min_max_cond = 'npy_datasets/min_max_joint.npy'

    # Define npy file name with array of unique speakers in dataset
    npy_name_spk_id = 'npy_datasets/spk_id.npy'

    # Get file names from partition's list list
    partition = 'train'

    file_names = open(str(params['cond_path']) +
                      'wav_' + partition + '.list', 'r').read().splitlines()

    cond_path = os.path.join(params['cond_path'], params['cond_set'])

    spk = np.load(npy_name_spk_id)

    i = np.array([-1, -2, -3, -4, -5])

    print('Generating', len(i), 'audio files')
    cont = 0
    
    for i in np.nditer(i):
        cont = cont+1
        print('Generating Audio', i)
        print('Generating...', file_names[i])

        # Load CC conditioner
        c = np.loadtxt(cond_path + file_names[i] + '.cc')

        # Load LF0 conditioner
        f0file = np.loadtxt(cond_path + file_names[i] + '.lf0')
        f0, _ = interpolation(f0file, -10000000000)
        f0 = f0.reshape(f0.shape[0], 1)

        # Load FV conditioner
        fvfile = np.loadtxt(cond_path + file_names[i] + '.gv')
        fv, uv = interpolation(fvfile, 1e3)
        num_fv = fv.shape[0]
        uv = uv.reshape(num_fv, 1)
        fv = fv.reshape(num_fv, 1)

        # Load speaker conditioner
        speaker = np.where(spk == file_names[i][0:2])[0][0]

        cond = np.concatenate((c, f0), axis=1)
        cond = np.concatenate((cond, fv), axis=1)
        cond = np.concatenate((cond, uv), axis=1)

        # Load maximum and minimum of de-normalized conditioners
        # Load maximum and minimum of de-normalized conditioners
        min_cond = np.load(npy_name_min_max_cond)[0]
        max_cond = np.load(npy_name_min_max_cond)[1]

        # Normalize conditioners with absolute maximum and minimum for each speaker of training partition
        print('Normalizing conditioners.')
        cond = (cond - min_cond) / (max_cond - min_cond)

        if params['look_ahead']:
            delayed = np.copy(cond)
            delayed[:, :-1, :] = delayed[:, 1:, :]
            cond = np.concatenate((cond, delayed), axis=2)

        print('shape cond', cond.shape)
        # min_cond=np.load(params['cond_path']+'/min.npy')
        # max_cond=np.load(params['cond_path']+'/max.npy')
        # print('max cond', max_cond.shape)
        # cond =  (cond-min_cond)/(max_cond-min_cond)
        seed = params.get('seed')
        init_random_seed(seed, use_cuda)

        output_path = params['output_path']
        ensure_dir_exists(output_path)

        spk_dim = len([i for i in os.listdir(os.path.join(params['cond_path'], params['cond_set']))
                       if os.path.islink(os.path.join(params['cond_path'], params['cond_set']) + '/' + i)])

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

        generator = RunGenerator(
            model,
            output_path,
            params['sample_rate'],
            use_cuda,
            epoch=epoch_index,
            cond=cond,
            speaker=speaker,
            checkpoints_path=f_name,
            g=cont
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
        '--output_path', help='path to the directory to save the generated samples to'
    )
    parser.add_argument(
        '--cond_path', help='path to the directory to find the conditioning'
    )
    parser.add_argument(
        '--cond_set',
        help='cond_set name - name of a directory in the conditioning sets path \
                 (settable by --cond_path)'
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
        '--look_ahead', type=float,
        help='Take conditioners from current and next frame'
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
