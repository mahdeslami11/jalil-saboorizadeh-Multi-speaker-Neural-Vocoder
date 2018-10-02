source env/bin/activate

program="/usr/anaconda3/bin/tensorboard"

name=tboard

CMD="usr/anaconda3/bin/python $program --logdir=gan --port=11222"

date

if [[ $HOSTNAME == d5lnx26* ]]; then
      { time $CMD  --results_path results26 ; } 2>&1
else
      srun -pveu -J $name $CMD
fi
