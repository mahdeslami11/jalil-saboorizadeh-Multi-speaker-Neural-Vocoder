from model import SampleRNN, Predictor
from optim import gradient_clipping
from nn import sequence_nll_loss_bits
from trainer import Trainer
from trainer.plugins import (
    TrainingLossMonitor, ValidationPlugin, AbsoluteTimeMonitor, SaverPlugin,
    GeneratorPlugin, StatsPlugin
)
from dataset import FolderDataset
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import MultiStepLR

import torch
from torch.utils.trainer.plugins import Logger

from natsort import natsorted

from functools import reduce
import os
import shutil
import sys
from glob import glob
import re
import argparse

import random
import numpy as np


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
    'val_frac': 0.1,
    'test_frac': 0.1,
    'qrnn': False,

    # training parameters
    'keep_old_checkpoints': False,
    'datasets_path': 'datasets',
    'cond_path': 'datasets',
    'results_path': 'results',
    'condset': '73',
    'epoch_limit': 1000,
    'resume': True,
    'sample_rate': 16000,
    'n_samples': 1,
    'sample_length': 80000,
    'loss_smoothing': 0.99,
    'seed': 77977,
    'model': None,
    'scheduler': False
}
tag_params = [
    'exp', 'frame_sizes', 'n_rnn', 'dim', 'learn_h0', 'ulaw', 'q_levels', 'seq_len',
    'batch_size', 'dataset', 'condset', 'val_frac', 'test_frac', 'seed', 'weight_norm', 'qrnn', 'scheduler'
    ]


def make_tag(params):
    def to_string(value):
        if isinstance(value, bool):
            return 'T' if value else 'F'
        elif isinstance(value, list):
            return ','.join(map(to_string, value))
        else:
            return str(value)

    return '-'.join(
        key + ':' + to_string(params[key])
        for key in tag_params
        if key not in default_params or params[key] != default_params[key]
    )


def setup_results_dir(params):
    def ensure_dir_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)

    tag = make_tag(params)
    results_path = os.path.abspath(params['results_path'])
    print('results path', results_path)
    ensure_dir_exists(results_path)
    results_path = os.path.join(results_path, tag)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    elif not params['resume']:
        shutil.rmtree(results_path)
        os.makedirs(results_path)

    for subdir in ['checkpoints', 'samples']:
        ensure_dir_exists(os.path.join(results_path, subdir))

    return results_path


def load_last_checkpoint(checkpoints_path):
    checkpoints_pattern = os.path.join(
        checkpoints_path, SaverPlugin.last_pattern.format('*', '*')
    )
    checkpoint_paths = natsorted(glob(checkpoints_pattern))
    if len(checkpoint_paths) > 0:
        checkpoint_path = checkpoint_paths[-1]
        checkpoint_name = os.path.basename(checkpoint_path)
        match = re.match(
            SaverPlugin.last_pattern.format(r'(\d+)', r'(\d+)'),
            checkpoint_name
        )
        epoch = int(match.group(1))
        iteration = int(match.group(2))
        return (torch.load(checkpoint_path), epoch, iteration)
    else:
        return None


def tee_stdout(log_path):
    log_file = open(log_path, 'a', 1)
    stdout = sys.stdout

    class Tee:

        def write(self, string):
            log_file.write(string)
            stdout.write(string)

        def flush(self):
            log_file.flush()
            stdout.flush()

    sys.stdout = Tee()


def init_random_seed(seed, cuda):
    print('seed', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


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
        epoch, iteration = (0,0)
                                                                
    return (torch.load(checkpoint_path), epoch, iteration)


def make_data_loader(overlap_len, params):
    path = os.path.join(params['datasets_path'], params['dataset'])
    cond_path = os.path.join(params['cond_path'], params['condset'])
    print('cond path', cond_path)

    def data_loader(split_from, split_to, max_cond, min_cond):
        #if 'max_cond' in kwargs:
           # max_cond=kwargs['max_cond']
           # min_cond=kwargs['min_cond']
        #else:
           # max_cond=None
           # min_cond=None

        dataset = FolderDataset(params['datasets_path'], path, cond_path, overlap_len, params['q_levels'], params['ulaw'],
                                params['seq_len'], params['batch_size'], max_cond, min_cond)

        (max_cond, min_cond)= dataset.cond_range()
        return (DataLoader(dataset, batch_size=params['batch_size'], shuffle=False, drop_last=True, num_workers=2),
                max_cond, min_cond)
    return data_loader


def main(exp, frame_sizes, dataset, **params):
    scheduler = True
    use_cuda = torch.cuda.is_available()
    print('Start samplernn')
    params = dict(
        default_params,
        exp=exp, frame_sizes=frame_sizes, dataset=dataset,
        **params
    )
    seed = params.get('seed')
    init_random_seed(seed, use_cuda)

    results_path = setup_results_dir(params)
    tee_stdout(os.path.join(results_path, 'log'))

    print('Create model')
    model = SampleRNN(
        frame_sizes=params['frame_sizes'],
        n_rnn=params['n_rnn'],
        dim=params['dim'],
        learn_h0=params['learn_h0'],
        q_levels=params['q_levels'],
        ulaw=params['ulaw'],
        weight_norm=params['weight_norm'],
        qrnn=params['qrnn']
    )
    if use_cuda:
        model=model.cuda()
        predictor = Predictor(model).cuda()
    else:
        predictor = Predictor(model)

    print('Done!')
    fname = params['model']
    if fname is not None:
        print('pre train whith', fname)
        model_data = load_model(fname)
        if model_data is None:
            sys.exit('ERROR: Model not found in', fname)        
        (state_dict, epoch_index, iteration) = model_data
        print('OK: Read model', fname, '(epoch:',epoch_index, ')')
        print(state_dict)
        predictor.load_state_dict(state_dict)
    print('predictor', predictor)
    for name,param in predictor.named_parameters():
        print(name, param.size())

    # optimizer = gradient_clipping(torch.optim.Rprop(predictor.parameters(), lr=0.001, etas=(0.5, 1.2), step_sizes=(1e-06, 50)))
    
    optimizer = torch.optim.Adam(predictor.parameters())
    if params['scheduler']:
        scheduler = MultiStepLR(optimizer, milestones=[15,35], gamma=0.1)
    optimizer = gradient_clipping(optimizer)
    print('Saving results in path', results_path)
    print('Read data')
    data_loader = make_data_loader(model.lookback, params)
    print('Done!')
    test_split = 1 - params['test_frac']
    val_split = test_split - params['val_frac']
    datafull = data_loader(0, val_split, None, None)
    datas = datafull[0]
    max_cond = datafull[1]
    min_cond = datafull[2]

    show_dataset=False
    if show_dataset:
        for i, full in enumerate(datas):
            print('dataloader---------------------------------------')
            print('batch', i)
            #print('max', max_cond)
            #print('min', min_cond)
            (data, reset, target, cond) = full           
            print('Data', data.size())
            print('Target', target.size())
            #print('Cond', cond.size())
            #print('reset', reset)
      #scheduler,
    if not params['scheduler']:    
        scheduler=None
    if use_cuda:
        cuda=True
    else:
        cuda = False
    trainer = Trainer(
        predictor, sequence_nll_loss_bits, optimizer,  datas, cuda, scheduler

    )


    checkpoints_path = os.path.join(results_path, 'checkpoints')
    checkpoint_data = load_last_checkpoint(checkpoints_path)
    if checkpoint_data is not None:
        (state_dict, epoch, iteration) = checkpoint_data
        trainer.epochs = epoch
        trainer.iterations = iteration
        predictor.load_state_dict(state_dict)

    trainer.register_plugin(TrainingLossMonitor(
        smoothing=params['loss_smoothing']
    ))
    trainer.register_plugin(ValidationPlugin(
        data_loader(val_split, test_split, max_cond, min_cond)[0],
        data_loader(test_split, 1, max_cond, min_cond)[0]
    ))
    trainer.register_plugin(AbsoluteTimeMonitor())
    trainer.register_plugin(SaverPlugin(
        checkpoints_path, params['keep_old_checkpoints']
    ))
    #
    #Generate=False
    #if Generate:
    #datas=data_loader(test_split, 1, max_cond, min_cond)[0]

       # for i, full in enumerate(datas):
       #     print('Generating---------------------------------------')
       #     print('batch', i)
       #     (data, reset, target, cond) = full    
       #     trainer.register_plugin(GeneratorPlugin(
       #         os.path.join(results_path, 'samples'), params['n_samples'],
       #         params['sample_length'], params['sample_rate'], cond
       #     ))
    trainer.register_plugin(
        Logger([
            'training_loss',
            'validation_loss',
            'test_loss',
            'time'
        ])
    )
    
    trainer.register_plugin(StatsPlugin(
        results_path,
        iteration_fields=[
            'training_loss',
            ('training_loss', 'running_avg'),
            'time'
        ],
        epoch_fields=[
            'validation_loss',
            'test_loss',
            'time'
        ],
        plots={
            'loss': {
                'x': 'iteration',
                'ys': [
                    'training_loss',
                    ('training_loss', 'running_avg'),
                    'validation_loss',
                    'test_loss',
                ],
                'log_y': True
            }
        }
    ))
    
    trainer.run(params['epoch_limit'])


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

    parser.add_argument('--exp', required=True, help='experiment name')
    parser.add_argument(
        '--frame_sizes', nargs='+', type=int, required=True,
        help='frame sizes in terms of the number of lower tier frames, \
              starting from the lowest RNN tier'
    )
    parser.add_argument(
        '--dataset', required=True,
        help='dataset name - name of a directory in the datasets path \
              (settable by --datasets_path)'
    )
    parser.add_argument(
        '--condset',
        help='condset name - name of a directory in the conditioningsets path \
              (settable by --cond_path)'
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
        '--val_frac', type=float,
        help='fraction of data to go into the validation set'
    )
    parser.add_argument(
        '--test_frac', type=float,
        help='fraction of data to go into the test set'
    )
    parser.add_argument(
        '--keep_old_checkpoints', type=parse_bool,
        help='whether to keep checkpoints from past epochs'
    )
    parser.add_argument(
        '--datasets_path', help='path to the directory containing datasets'
    )
    parser.add_argument(
        '--cond_path', help='path to the directory containing conditioningsets'
    )
    parser.add_argument(
        '--results_path', help='path to the directory to save the results to'
    )
    parser.add_argument('--epoch_limit', type=int, help='how many epochs to run')
    parser.add_argument(
        '--resume', type=parse_bool, default=True,
        help='whether to resume training from the last checkpoint'
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
        '--loss_smoothing', type=float,
        help='smoothing parameter of the exponential moving average over \
              training loss, used in the log and in the loss plot'
    )
    
    parser.add_argument(
        '--seed', type=int,
        help='seed init of random generator'
    
    )
    parser.add_argument(
        '--weight_norm', type = parse_bool,
        help ='Apply weight normalitzation to linear layers'
    )
    parser.add_argument(
        '--qrnn', type = parse_bool,
        help ='Use QRNN insted of RNN'
    )
    parser.add_argument(
        '--model',
        help='model (including path) to re train'
    )
    parser.add_argument(
        '--scheduler', type = parse_bool,
        help ='Use a variable learning rate'
    )
    parser.set_defaults(**default_params)

    main(**vars(parser.parse_args()))
