from model import SampleRNN, Predictor, Generator

import torch

import os
import sys
import re
import numpy as np
import argparse
from librosa.output import write_wav
from natsort import natsorted
import os
from os import listdir
from os.path import join
import glob
from interpolate import interpolation, linear_interpolation

import random       # just to set seed, which I don't know if ever used 8-|


# From the moment, keep parameters to find the model
# change!


default_params = {
    # model parameters
    'n_rnn': 2,
    'dim': 1024,
    'learn_h0': True,
    'ulaw': True,
    'q_levels': 256,
    'seq_len': 1040,
    'batch_size': 128,
    'val_frac': 0.1,
    'test_frac': 0.1,
    'cond_dim': 43,
    'weight_norm': True,

    # training parameters
    'sample_rate': 16000,
    'n_samples': 1,
    'sample_length': 40000,
    'seed': 77977,
    'cond': 0,

    # generator parameters
    'output_path': '.',
    'cond_path': './datasets/73'
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


def load_model(checkpoint_path):
    model_pattern = '.*ep{}-it{}'

    checkpoint_name = os.path.basename(checkpoint_path)
    match = re.match(
        model_pattern.format(r'(\d+)', r'(\d+)'),
        checkpoint_name
    )
    if match:
        epoch     = int(match.group(1))
        iteration = int(match.group(2))
    else:
        epoch, iteration = (0,0)
        
    return (torch.load(checkpoint_path), epoch, iteration)


class RunGenerator:
    def __init__(self, model, samples_path, sample_rate, cuda, epoch, cond):
        self.generate = Generator(model, cuda)
        self.samples_path = samples_path
        self.sample_rate = sample_rate
        self.cuda = cuda
        self.epoch = epoch
        self.pattern = 'gensyn-ep{}-g{}.wav'
        self.cond = cond
        

    def __call__(self, n_samples, sample_length, cond):
        print('Generate', n_samples, 'of length', sample_length)
        samples = self.generate(n_samples, sample_length, cond).cpu().numpy()
        maxv =  np.iinfo(np.int16).max
        for i in range(n_samples):
            filename = os.path.join(self.samples_path, self.pattern.format(self.epoch, i))
            print(filename)

            write_wav(
                filename,
                (samples[i, :]  * maxv).astype(np.int16), sr=self.sample_rate
            )


def main(frame_sizes, **params):


    use_cuda = torch.cuda.is_available()
 
    params = dict(
        default_params,
        frame_sizes=frame_sizes, 
        **params
    )
    ratio_min=0.9
    ratio_max=1
    file_ceps = natsorted(glob.glob(os.path.join(params['cond_path'], '*.cc')))
    file_f0 = natsorted(glob.glob(os.path.join(params['cond_path'], '*.lf0')))
    file_fv = natsorted(glob.glob(os.path.join(params['cond_path'], '*.gv')))
    file_ceps = file_ceps[
        int(ratio_min  * len(file_ceps)) : int(ratio_max  * len(file_ceps))
        ]
    file_f0 = file_f0[
        int(ratio_min * len(file_f0)) : int(ratio_max * len(file_f0))
        ]
    file_fv  = file_fv [
        int(ratio_min * len(file_fv )) : int(ratio_max * len(file_fv ))
        ]
    i=2

    print('Generating...', file_ceps[i])
    c = np.loadtxt(file_ceps[i])
    f0file = np.loadtxt(file_f0[i])
    #interp es la senyal interpolada, uv es el flag (UV) de mom el deixem
    f0, uv = interpolation(f0file, -10000000000)
    print('f0 uv', uv)
    print('shape', uv.shape)
    num_f0 = f0.shape[0]
    f0 = f0.reshape((num_f0,1))
    fvfile = np.loadtxt(file_fv[i])
    fv, uv = interpolation(fvfile, 1e3)
    num_fv = fv.shape[0]
    fv = fv.reshape((num_fv,1))
    uv = uv.reshape((num_fv,1))
    print('fv uv', uv)
    cond = np.concatenate((c, f0), axis=1)
    cond = np.concatenate((cond, fv), axis=1)
    cond = np.concatenate((cond, uv), axis=1)

    print('shape cond', cond.shape)
    min_cond=np.load(params['cond_path'] + '/min.npy')
    max_cond=np.load(params['cond_path'] + '/max.npy')
    print('max cond', max_cond.shape)
    cond =  (cond-min_cond)/(max_cond-min_cond)
    seed = params.get('seed')
    init_random_seed(seed, use_cuda)

    output_path = params['output_path']
    ensure_dir_exists(output_path)
    
    print('Start Generate SampleRNN')
    model = SampleRNN(
        frame_sizes=params['frame_sizes'],
        n_rnn=params['n_rnn'],
        dim=params['dim'],
        learn_h0=params['learn_h0'],
        q_levels=params['q_levels'],
        ulaw=params['ulaw'],
        weight_norm = params['weight_norm']


    )
    print(model)
    
    if use_cuda:
        model = model.cuda()
        predictor = Predictor(model).cuda()
    else:
        predictor = Predictor(model)
        

    fname = params['model']
    model_data = load_model(fname)
    
    if model_data is None:
        sys.exit('ERROR: Model not found in', fname)        
    (state_dict, epoch_index, iteration) = model_data
    print('OK: Read model', fname, '(epoch:',epoch_index, ')')
    print(state_dict)
    predictor.load_state_dict(state_dict)
    

    generator = RunGenerator(model, 
                             os.path.join(output_path, 'samples'),
                             params['sample_rate'],
                             use_cuda,
                             epoch=epoch_index, 
                             cond=cond)   

    generator(params['n_samples'],
              params['sample_length'], cond)

    

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
        '--seed', type=int,
        help='seed initialization of random generator'
    )
    parser.add_argument(
        '--weight_norm', type=parse_bool,
        help='Apply weight normalitzation'
    )

    

    parser.set_defaults(**default_params)

    main(**vars(parser.parse_args()))
