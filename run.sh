#!/bin/bash
experiment=$1

if [ "${experiment,,}" = "samplernn" ]; then
	run_samplernn.sh
elif [ "${experiment,,}" = "samplernn-gan" ]; then
	run_samplegan.sh
elif [ "${experiment,,}" = "bottle-neck" ]; then
	run_sampleneck.sh
else
	echo -e "Please especify which experiment do you want to run. Possible experiments are:\n- SampleRNN\n- SampleRNN-GAN\n- Bottle-neck\n- Eigen-voice\n"
fi
