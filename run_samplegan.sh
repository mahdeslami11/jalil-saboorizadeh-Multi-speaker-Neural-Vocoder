git fetch
git reset --hard origin/gan

source env/bin/activate

PRG_PY=train.py

DB_PATH=tcstar/

DB_WAV=wav/

DB_COND=cond/

name=gan

date

python3 -u $PRG_PY --ulaw true --exp $name --frame_sizes 20 4 --seq_len 1040 --n_rnn 2 --datasets_path $DB_PATH --cond_path $DB_PATH --batch_size 64 \
--dataset $DB_WAV --cond_set $DB_COND --epoch_limit 500 --learning_rate 1e-4 --weight_norm True --scheduler True --lambda_weight 0 0.01 50000 --ind_cond_dim 50

