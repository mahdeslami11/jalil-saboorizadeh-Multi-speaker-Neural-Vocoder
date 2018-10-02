#!/bin/bash
listdir=$(ls results)
blue='\033[0;34m'
green='\033[0;32m'
nc='\033[0m'
count=1
echo -e "Choose one of the following experiments:"
for i in $listdir; do
	echo -e "$count:\t${blue}$i${nc}"
	((count=count+1))
done
read -n 2 -p "Index of the selection:" input
set -- $listdir
chosen_dir=${!input}
echo -e "\nYour selection is:\t${blue}$chosen_dir${nc}"
echo -e "\nChoose the index of either best or last checkpoint:"
checkdir=$(ls ~/project/veu4/results/$chosen_dir/checkpoints)
count=1
for i in $checkdir; do
	echo -e "$count:\t${green}$i${nc}"
	((count=count+1))
done
read -n 1 -p "Index of the selection:" input
set -- $checkdir
chosen_check=${!input}
echo -e "\nYour selection is:\t${green}$chosen_check${nc}\n"

exp_raw="$(cut -d ':' -f2 <<<"$chosen_dir")"
exp="$(cut -d '~' -f1 <<<"$exp_raw")"

if [ "${exp,,}" = "samplernn" ]; then
	branch="master"
elif [ "${exp,,}" = "gan" ]; then
	branch="gan"
elif [ "${exp,,}" = "neck" ]; then
	branch="bottle-neck"
else
	branch="Unknown"
fi

echo -e "Your branch based on the chosen experiment is ${green}$branch${nc}\n"

git fetch
git reset --hard origin/$branch

source env/bin/activate

PRG_PY=/veu/tfgveu7/project/generate.py

DB_PATH=/veu/tfgveu7/project/tcstar/

DB_COND=cond/

model="results/$chosen_dir/checkpoints/$chosen_check"

CMD="python3 -u $PRG_PY --model $model --ulaw true --frame_sizes 20 4 --seq_len 1040 --n_rnn 2 --datasets_path $DB_PATH --cond_set $DB_COND --weight_norm True"

date

if [[ $HOSTNAME == d5lnx26* ]]; then
    { time $CMD  --results_path results26 ; } 2>&1
else
    srun -pveu -J gen -c 2 --gres=gpu:1 --mem 10G $CMD
fi
