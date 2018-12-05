# Multi-speaker Neural Vocoder

Welcome to the repository of my Bachelor's Thesis. My Bachelor's thesis basically consisted in building a speech synthesizer based on Recurrent Neural Networks capable of speaking with different Spanish voices. The baseline code was the Pytorch implementation of the SampleRNN paper. Under the supervision of Dr. Antonio Bonafonte and the help of the PhD candidate Santiago Pascual, we adapted this code and introduced some improvements listed in a paper that was accepted for the IBERSPEECH conference of 2018. Take a look at both the [thesis and the paper](https://github.com/Barbany/Multi-speaker-Neural-Vocoder/tree/master/doc). Please cite this work if it's useful for this research:

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
author = {{Barbany}, Oriol, Bonafonte, Antonio and Pascual, Santiago},
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
