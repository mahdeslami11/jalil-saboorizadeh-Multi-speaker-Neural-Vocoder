#!/bin/bash
mkdir -p log_dir # directory for log files
#sbatch -pveu --array 1-1453%50 ahocoder_instance.sh
MSG="hola toni" sbatch -pveu --array 1-1921%50 ahocoder_instance.sh
