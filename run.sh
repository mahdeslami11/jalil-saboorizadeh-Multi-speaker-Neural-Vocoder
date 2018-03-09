#!/bin/bash


PRG_PY=/home/usuaris/veu/antonio/samplernn_cond/train.py


DB=silent/F1/original

name=silentF1

#srun -pveu \rm -fR results/exp:${name}*$DB

CMD="python3 -u $PRG_PY --ulaw true --exp $name --frame_sizes 20 4 --seq_len 1040 --n_rnn 2 --dataset $DB --condset $DB  --epoch_limit 500"
#CMD="python3 -u train.py --ulaw  true --exp $name --frame_sizes 16 4 --seq_len 1024 --n_rnn 2 --dataset $DB --epoch_limit 500"


date


if [[ $HOSTNAME == d5lnx26* ]]; then
    { time $CMD  --results_path results26 ; } 2>&1
else
    srun -pveu -J $name -c 2 --gres=gpu:1 --mem 24000 $CMD
fi



