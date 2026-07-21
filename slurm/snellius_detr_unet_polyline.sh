#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=12:00:00
#SBATCH -o /home/%u/exps/hedge/detr_unet_polyline/1_%j.out

# Copy to $HOME/exps/hedge/detr_unet_polyline/<n>.sh, set the -o line and the
# exp name in the training script cfg to the same <n>, then: sbatch <n>.sh
# Outputs land in $HOME/exps/hedge/detr_unet_polyline/<n>/best_<n>.pt, next to
# <n>.sh and <n>_<jobid>.out, the same layout as semseg_unet.
#
# A100: 4 GPUs / 72 CPUs per node -> 18 CPUs per GPU.
# H100: 4 GPUs / 64 CPUs per node -> 16 CPUs per GPU (faster, longer queue).
# Paths come from hedge_seg/paths.py, so no path edits are needed between the
# local machine and the cluster. That file is printed below as well, since it
# now decides where data and checkpoints are read from and written to.

cd "$HOME/dev/hedge-seg"
echo $(date)
echo $(git log -1 --pretty=%h)

script_name=$HOME/dev/hedge-seg/scripts/train_detr_unet_polyline.py
echo "bash file ===>"
scontrol write batch_script $SLURM_JOB_ID - 2>/dev/null || echo "(could not dump batch script)"
echo "scripts ===>"
echo "paths.py:"
cat $HOME/dev/hedge-seg/hedge_seg/paths.py
echo "build_pdok_wms_dataset.py:"
cat $HOME/dev/hedge-seg/scripts/data/build_pdok_wms_dataset.py
echo "convert_pdok_polylines_to_detr_polyline.py:"
cat $HOME/dev/hedge-seg/scripts/data/convert_pdok_polylines_to_detr_polyline.py
echo $script_name:
cat $script_name

echo "cpu per node: $SLURM_CPUS_ON_NODE"
nvidia-smi

echo "source $HOME/.bashrc"
source $HOME/.bashrc
conda activate hedge
echo "activate my virtual env: $CONDA_DEFAULT_ENV"

echo "start training"
# python -u: without it stdout is block buffered (8 KB) because it goes to a
# file, so the .out log lags tens of epochs behind the run while tensorboard
# and checkpoints are current, which looks like a hung job.
PYTHONPATH=$HOME/dev/hedge-seg python -u $script_name
echo "end training"

echo $(date)
