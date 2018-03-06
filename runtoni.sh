#!/bin/bash

DB=73
name=${DB}ulaw

source ~/env/bin/activate
date
srun -pveu -J samplernn -c 2 --gres=gpu:1 --mem 102400 python3 -u train.py --ulaw True --exp test --frame_sizes 16 4 --n_rnn 2 --dataset 73
