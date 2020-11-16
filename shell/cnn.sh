#! /bin/bash
#SBATCH "--job-name=grad_cam"
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output files/output/server_logs/job%J.out
#SBATCH --error files/output/server_logs/job%J.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:1

module load $1

# PARAMETERS
export SAVE_KEY=$3
export LOG_LEVEL="INFO"
export USE_GPU=1
export IS_REPRODUCIBLE=0
export IS_LOCAL=0
export DO_CV=0
export DO_TEST=1
# MODEL
export MODEL_CONFIG_KEY=$2
# TRAINING
export USE_AUG=1
export NUM_FOLDS=5
export NUM_EPOCH=400
export BATCH_SIZE=64
export NUM_WORKERS=1
export SAVE_IMG_PER_EPOCH=5
export IS_SAVE_MODEL=1

python3 ./cnn_main.py
