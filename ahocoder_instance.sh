#!/bin/bash



#SBATCH   -p veu
#SBATCH --mem=40M
#SBATCH --gres=gpu:0
#SBATCH --job-name ahocoder
#SBATCH --time 10:00
# -- SBATCH --nodes 1
#SBATCH --output log_dir/ahocoder-%a.log
#SBATCH --exclude=veuC01,veuC05

if [ -z "${SLURM_ARRAY_TASK_ID}" ]
then
    echo 1>&2 "Error: not running as a job array."
    echo "Check with file 4"
    ID=4
else
    ID=${SLURM_ARRAY_TASK_ID}
fi

echo "Array index: ${SLURM_ARRAY_TASK_ID}"



WAV_DIRECTORY=/veu4/antonio/db/silent/
AHO_DIRECTORY=/veu4/antonio/db/silent/
LIST_FILE=/veu4/antonio/db/silent/wav.list

AHOCODER=ahocoder
AHODECODER=ahodecoder
X2X=x2x



if [[ ! -f $LIST_FILE ]]; then
    echo "Not found $LIST_FILE" && exit 1
fi

FILENAME=$(head -$ID $LIST_FILE | tail -1 | perl -pe 's/\.wav$//i')

INPUT=$WAV_DIRECTORY/$FILENAME.wav
OUTPUT=$AHO_DIRECTORY/$FILENAME
mkdir -p $(dirname $OUTPUT)


echo $AHOCODER $INPUT $OUTPUT.lf0 $OUTPUT.mcp $OUTPUT.gv

$AHOCODER $INPUT $OUTPUT.raw.lf0 $OUTPUT.raw.cc $OUTPUT.raw.gv || exit 1
$X2X +fa   $OUTPUT.raw.lf0 > $OUTPUT.lf0 || exit 1
$X2X +fa   $OUTPUT.raw.gv > $OUTPUT.gv  || exit 1
$X2X +fa40 $OUTPUT.raw.cc > $OUTPUT.cc  || exit 1
\rm -f $OUTPUT.raw.lf0 $OUTPUT.raw.gv $OUTPUT.raw.cc
exit 0


echo "MSG $MSG $SLURM_ARRAY_TASK_ID"
echo '----------'
exit




