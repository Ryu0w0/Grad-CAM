#! /bin/bash
#SBATCH "--job-name=auto_en"
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output files/output/server_logs/job%J.out
#SBATCH --error files/output/server_logs/job%J.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:1

module load $1

# PARAMETERS
SAVE_KEY=$3
LOG_LEVEL="INFO"
USE_GPU=1
IS_REPRODUCIBLE=0
IS_LOCAL=0
DO_CV=0
DO_TEST=1
# MODEL
MODEL_CONFIG_KEY=$2
# TRAINING
USE_AUG=1
NUM_FOLDS=5
NUM_EPOCH=400
BATCH_SIZE=64
NUM_WORKERS=1
SAVE_IMG_PER_EPOCH=5
IS_SAVE_MODEL=1

python3 ./cnn_main.py $SAVE_KEY $LOG_LEVEL $USE_GPU $IS_REPRODUCIBLE $IS_LOCAL $MODEL_CONFIG_KEY $USE_AUG $NUM_FOLDS $NUM_EPOCH $BATCH_SIZE $NUM_WORKERS $SAVE_IMG_PER_EPOCH $DO_CV $DO_TEST $IS_SAVE_MODEL