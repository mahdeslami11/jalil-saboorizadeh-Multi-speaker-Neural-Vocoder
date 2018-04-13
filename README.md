# Multi-speaker Neural Vocoder

The starting point of this project is an open-source PyTorch implementation of [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837).

![A visual representation of the SampleRNN architecture](http://deepsound.io/images/samplernn.png)

It's based on the reference implementation in Theano provided in the same paper: https://github.com/soroushmehr/sampleRNN_ICLR2017. Unlike the Theano version, PyTorch code allows training models with arbitrary number of tiers, whereas the original implementation allows maximum 3 tiers. However, it doesn't have weight normalization and doesn't allow using LSTM units (only GRU). For more details and motivation behind rewriting this model to PyTorch, see our blog post: http://deepsound.io/samplernn_pytorch.html.

## Dependencies

This code requires Python 3.5+ and PyTorch 0.1.12+. Installation instructions for PyTorch are available on their website: http://pytorch.org/. You can install the rest of the dependencies by running `pip install -r requirements.txt`.

## Datasets

The datasets are from TC-STAR (Bonafonte et al.) and formed by 6 different professional Spanish speakers. The model is conditioned using the speech parameters extracted with Ahocoder (Bonafonte et al.) and they are located with symbolic links in the `tcstar/` directory.

## Training

To train the model you need to run `run.sh`. All model hyper-parameters as well as the speech conditioners' and wav directories are settable in the command line. Most hyper-parameters have sensible default values, so you don't need to provide all of them.

The results - training log, loss plots, model checkpoints and generated samples will be saved in `results/`.
