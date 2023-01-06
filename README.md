# Multi-speaker Neural Vocoder

Welcome to the repository of my Bachelor's Thesis. My Bachelor's thesis basically consisted in building a speech synthesizer based on Recurrent Neural Networks capable of speaking with different Spanish voices. The baseline code was the Pytorch implementation of the SampleRNN paper. Under the supervision of Dr. Antonio Bonafonte and the help of the PhD candidate Santiago Pascual, we adapted this code and introduced some improvements listed in a paper that was accepted for the IBERSPEECH conference of 2018. Take a look at both the [thesis and the paper](https://github.com/Barbany/Multi-speaker-Neural-Vocoder/tree/master/doc). Please cite this work if it's useful for your research:

```
@bachelorsthesis{barbany2018thesis,
  title={Multi-Speaker Neural Vocoder},
  author={Barbany, Oriol},
  year={2018},
  school={Universitat Polit{\`e}cnica de Catalunya}
}
```

```
@article{barbany2018paper,
author = {{Barbany}, Oriol and Bonafonte, Antonio and Pascual, Santiago},
journal = {IberSpeech},
title = {{Multi-Speaker Neural Vocoder}},
year = {2018}
}
```

The thesis also extended to voice conversion, which forced the incorporation of new architectures that are described on the report. You can find these last two modifications in branches of this repository. Nevertheless, note that the run file already allows to perform this experiments. See section [train your model](#train-your-model) for more details.

Hear the audio generated with our best model [here](https://github.com/Barbany/Multi-speaker-Neural-Vocoder/tree/master/samples). This is the model of the master tree in this repository with the modifications of speaker-dependent normalization and look ahead. The rest of the filename means that these samples were obtained with the "best" model, with how many epochs and iterations, and ends with the name of the original WAV file and the speaker identifier (note that there are audios of the 6 speakers of the dataset (see [dataset](#dataset) section for more information)). The algorithm stores both the best model, i.e. the one that has a lower value of the loss function, and the last one. When generating, one can choose which model to use.

* [Baseline model](#baseline-model)
* [Dependencies](#dependencies)
* [Dataset](#dataset)
* [Train your model](#train-your-model)

## Baseline model
The starting point of this project is an open-source PyTorch implementation of [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837) that can be found in https://github.com/deepsound-project/samplernn-pytorch.

![A visual representation of the SampleRNN architecture](http://deepsound.io/images/samplernn.png)

It's based on the reference implementation in Theano provided in the same paper: https://github.com/soroushmehr/sampleRNN_ICLR2017. Unlike the Theano version, PyTorch code allows training models with arbitrary number of tiers, whereas the original implementation allows maximum 3 tiers. For more details and motivation behind the implementation of this model to PyTorch, see the blog post: http://deepsound.io/samplernn_pytorch.html.

## Dependencies

This code requires Python 3.6 and PyTorch 0.4.0. Installation instructions for PyTorch are available on their website: http://pytorch.org/. Check your OS and CUDA version before downloading it. You can install the rest of the dependencies by running `pip install -r requirements.txt`. You will also have to download Ahocoder to get the acoustic conditioners of each of your audio files. It is also recommended to install soxi for manipulating and getting information of the files from the command line.

## Dataset

The dataset is from TC-STAR (Bonafonte et al.) and formed by 6 different professional Spanish speakers. The model is conditioned using the speech parameters extracted with Ahocoder (Bonafonte et al.) and they are located in the `tcstar/` directory. Note that in this directory, there are lists with all the files for each partition and CSVs with the duration of each audio file. There are also folders `cond/` and `wav/`, which have folders of each of the speakers with acoustic conditioners and WAV files respectively. Nevertheless, you can see that these speaker folders are symbolic links because the used database was not public. The purpose of this symlinks is therefore not providing the database (I would do it if I was allowed) but showing the structure of the files.

## Train your model

To train your model you need to run `run` followed by the name of the experiment {samplernn, samplernn-gan, bottle-neck}, e.g. `run samplernn-gan`. All model hyper-parameters as well as the speech conditioners' and wav directories are settable in the command line. Most hyper-parameters have sensible default values, so you don't need to provide all of them. Please see the shell script called for each of the experiments to see typical values of the input arguments of `train.py` or directly run `python train.py --help` for information about each of them. Note that you need to create an environment with all the dependencies, which activation binaries are originally located at `env/bin/activate`. Change this from the scripts if your location is different. Also change the location of your WAV files and conditioners if your folder structure differs from mine.

A run script to monitor the project with the tool TensorBoard is also provided. Check the location of your tensoboard binaries, your python interpreter and the desired log directory. Also note that TensorBoard runs on a given port indicated in the same script. Ensure that this port is not blocked and make an SSH tunnel if you are working on a remote machine.

The results - training log, loss plots, model checkpoints and generated samples will be saved in `results/`.

1-target :

The goal of this project was to build a speech synthesizer based on recurrent neural networks capable of speaking with different Spanish voices.

2-Description of innovation :

Speaker-dependent normalization was not enough for voice conversion purposes, so more complex architectures were pro- . Nevertheless, human listeners preferred the speech modeled with speaker-dependent normalization and, given the similarity of the features normalized for each speaker, a better quantization could be applied for coding applications or for de- ployment of neural networks with limited resources.Whilst the speaker-dependent normalization itself doesn’t seem to improve the results obtained with the classical speaker-independent feature scaling, when combined with the look ahead approach, it achieves a 4 score with the male balanced dataset. To sum up, with the combination of the two proposals,a state-of-the-art MOS score have been achieved for a multi- speaker speech synthesis system. Both of that approaches were novelties introduced in this thesis and results show that they could be beneficial to other TTS systems as well as for a bunch of other applications involving features from different sources and modeling of no-real-time sequences.

3-change source code : better source code in similar project :

    from __future__ import absolute_import
    from __future__ import division
    from __future__ import print_function

    import argparse
    import os
    import re
    import sys
    import scipy.io.wavfile
    from sklearn.preprocessing import StandardScaler
    import torch
    import torch.nn as nn
    import numpy as np
    import torch.optim as optim
    from torchvision import transforms

    from torch.utils.data import DataLoader
    from fftnet import FFTNet
    from dataset import CustomDataset
    from utils.utils import apply_moving_average, ExponentialMovingAverage, mu_law_decode, write_wav
    from utils import infolog
    from hparams import hparams, hparams_debug_string
    from tensorboardX import SummaryWriter
    log = infolog.log


    def save_checkpoint(device, hparams, model, optimizer, step, checkpoint_dir, ema=None):
    model = model.module if isinstance(model, nn.DataParallel) else model

    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "steps": step}
    checkpoint_path = os.path.join(
        checkpoint_dir, "model.ckpt-{}.pt".format(step))
    torch.save(checkpoint_state, checkpoint_path)
    log("Saved checkpoint: {}".format(checkpoint_path))

    if ema is not None:
        averaged_model = clone_as_averaged_model(device, hparams, model, ema)
        averaged_checkpoint_state = {
            "model": averaged_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "steps": step}
        checkpoint_path = os.path.join(
            checkpoint_dir, "model.ckpt-{}.ema.pt".format(step))
        torch.save(averaged_checkpoint_state, checkpoint_path)
        log("Saved averaged checkpoint: {}".format(checkpoint_path))

    def clone_as_averaged_model(device, hparams, model, ema):
    assert ema is not None
    averaged_model = create_model(hparams).to(device)
    averaged_model.load_state_dict(model.state_dict())
    for name, param in averaged_model.named_parameters():
        if name in ema.shadow:
            param.data = ema.shadow[name].clone()
    return averaged_model

    def create_model(hparams):
    if hparams.feature_type == 'mcc':
        lc_channel = hparams.mcep_dim + 3
    else:
        lc_channel = hparams.num_mels

    return FFTNet(n_stacks=hparams.n_stacks,
                  fft_channels=hparams.fft_channels,
                  quantization_channels=hparams.quantization_channels,
                  local_condition_channels=lc_channel)

    def train_fn(args):
    device = torch.device("cuda" if hparams.use_cuda else "cpu")
    upsample_factor = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)

    model = create_model(hparams)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

    if args.resume is not None:
        log("Resume checkpoint from: {}:".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint['steps']
    else:
        global_step = 0

    log("receptive field: {0} ({1:.2f}ms)".format(
        model.receptive_field, model.receptive_field / hparams.sample_rate * 1000))

    
    if hparams.feature_type == "mcc":
        scaler = StandardScaler()
        scaler.mean_ = np.load(os.path.join(args.data_dir, 'mean.npy'))
        scaler.scale_ = np.load(os.path.join(args.data_dir, 'scale.npy'))
        feat_transform = transforms.Compose([lambda x: scaler.transform(x)])
    else:
        feat_transform = None

    dataset = CustomDataset(meta_file=os.path.join(args.data_dir, 'train.txt'), 
                            receptive_field=model.receptive_field,
                            sample_size=hparams.sample_size,
                            upsample_factor=upsample_factor,
                            quantization_channels=hparams.quantization_channels,
                            use_local_condition=hparams.use_local_condition,
                            noise_injecting=hparams.noise_injecting,
                            feat_transform=feat_transform)

    dataloader = DataLoader(dataset, batch_size=hparams.batch_size,
                             shuffle=True, num_workers=args.num_workers,
                             pin_memory=True)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()

    ema = ExponentialMovingAverage(args.ema_decay)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    writer = SummaryWriter(args.checkpoint_dir)

    while global_step < hparams.training_steps:
        for i, data in enumerate(dataloader, 0):
            audio, target, local_condition = data
            target = target.squeeze(-1)
            local_condition = local_condition.transpose(1, 2)
            audio, target, h = audio.to(device), target.to(device), local_condition.to(device)

            optimizer.zero_grad()
            output = model(audio[:,:-1,:], h[:,:,1:])
            loss = criterion(output, target)
            log('step [%3d]: loss: %.3f' % (global_step, loss.item()))
            writer.add_scalar('loss', loss.item(), global_step)

            loss.backward()
            optimizer.step()

            # update moving average
            if ema is not None:
                apply_moving_average(model, ema)

            global_step += 1

            if global_step % hparams.checkpoint_interval == 0:
                save_checkpoint(device, hparams, model, optimizer, global_step, args.checkpoint_dir, ema)
                out = output[1,:,:]
                samples=out.argmax(0)
                waveform = mu_law_decode(np.asarray(samples[model.receptive_field:]),hparams.quantization_channels)
                write_wav(waveform, hparams.sample_rate, os.path.join(args.checkpoint_dir, "train_eval_{}.wav".format(global_step)))


    if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', default='',
        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--data_dir', default='training_data',
        help='Metadata file which contains the keys of audio and melspec')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
        help='Moving average decay rate.')
    parser.add_argument('--num_workers',type=int, default=4, 
        help='Number of dataloader workers.')
    parser.add_argument('--resume', type=str, default=None, 
        help='Checkpoint path to resume')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', 
        help='Directory to save checkpoints.')
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    infolog.init(os.path.join(args.checkpoint_dir, 'train.log'), 'FFTNET')
    hparams.parse(args.hparams)
    train_fn(args)

erja be poroje :

https://github.com/syang1993/FFTNet

https://gfx.cs.princeton.edu/pubs/Jin_2018_FAR/

4-The result of changing and improving the evaluation of the output audio file :

this project introduce FFTNet, a deep learning approach synthesizing audio waveforms. this approach builds on the recent WaveNet project, which showed that it was possible to synthesize a natural sounding audio waveform directly from a deep convolutional neural network. FFTNet offers two improvements over WaveNet. First it is substantially faster, allowing for real-time synthesis of audio waveforms. Second, when used as a vocoder, the resulting speech sounds more natural, as measured via a "mean opinion score" test. 

5-Reference to the project :

https://github.com/Barbany/Multi-speaker-Neural-Vocoder

6-introducing myself :

My name is Jalil Sabourizadeh and I am studying for a master's degree in medical engineering and this project is for the digital signal processing course.

7-The updated article file was uploaded on GitHub

8-My explanation files about the project in Google Drive :

https://drive.google.com/drive/folders/1LD2y1YpvpZu33SAhIlQrXnkPzrgX2BUj?usp=sharing

The proposal file for the project : 
