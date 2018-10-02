cd /veu/tfgveu7/project/

git fetch
git reset --hard origin/gan

cd /veu4/tfgveu7/

source /veu/tfgveu7/env/bin/activate

PRG_PY=/veu/tfgveu7/project/train.py

DB_PATH=/veu/tfgveu7/project/tcstar/

DB_WAV=wav/

DB_COND=cond/

name=gan

CMD="python3 -u $PRG_PY --ulaw true --exp $name --frame_sizes 20 4 --seq_len 1040 --n_rnn 2 --datasets_path $DB_PATH --cond_path $DB_PATH --batch_size 64 \
--dataset $DB_WAV --cond_set $DB_COND --epoch_limit 500 --learning_rate 1e-4 --weight_norm True --scheduler True --lambda_weight 0 0.01 50000 --ind_cond_dim 50"

date

if [[ $HOSTNAME == d5lnx26* ]]; then
      { time $CMD  --results_path results26 ; } 2>&1
else
      srun -pveu -J $name -c 2 --gres=gpu:1 --mem 60G $CMD
fi
