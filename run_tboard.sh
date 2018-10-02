source /veu/tfgveu7/env/bin/activate

program="/veu/spascual/anaconda3/bin/tensorboard"

name=tboard

CMD="/veu/spascual/anaconda3/bin/python $program --logdir=/veu4/tfgveu7/gan --port=11222"

date

if [[ $HOSTNAME == d5lnx26* ]]; then
      { time $CMD  --results_path results26 ; } 2>&1
else
      srun -pveu -J $name $CMD
fi
