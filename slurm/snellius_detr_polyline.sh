#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:00:05
#SBATCH -o /home/%u/exps/hedge/train_detr_dino_polyline_1_%j.out

cd "$HOME/dev/hedge-seg"
echo $(date)
echo $(git log -1 --pretty=%h)

script_name=$HOME/dev/hedge-seg/scripts/train_detr_dino_polyline.py
echo "bash file ===>"
cat $HOME/exps/hedge/train_detr_dino_polyline_1.sh
echo "script ===>"
cat $script_name

echo "cpu per node: $SLURM_CPUS_ON_NODE"

echo "source $HOME/.bashrc"
source $HOME/.bashrc
conda activate hedge
echo "activate my virtual env: $CONDA_DEFAULT_ENV"

echo "start training"
python $script_name
echo "end training"

echo $(date)