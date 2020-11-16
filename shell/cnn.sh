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
save_key="-SAVE_KEY $3"
log_level="-LOG_LEVEL INFO"
use_gpu="-USE_GPU 1"
is_reproducible="-IS_REPRODUCIBLE 0"
is_local="-IS_LOCAL 0"
do_cv="-DO_CV 0"
do_test="-DO_TEST 1"
# MODEL
model_config_key="-MODEL_CONFIG_KEY $2"
# TRAINING
use_aug="-USE_AUG 1"
num_folds="-NUM_FOLDS 5"
num_epoch="-NUM_EPOCH 400"
batch_size="-BATCH_SIZE 64"
num_workers="-NUM_WORKERS 1"
save_img_per_epoch="-SAVE_IMG_PER_EPOCH 5"
is_save_model="-IS_SAVE_MODEL 1"

python3 ./cnn_main.py $save_key $log_level $use_gpu $is_reproducible $is_local $model_config_key $use_aug $num_folds $num_epoch $batch_size $num_workers $save_img_per_epoch $do_cv $do_test $is_save_model
