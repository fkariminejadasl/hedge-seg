#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:00:05
#SBATCH -o /home/%u/exps/hedge/semseg_unet/1_%j.out

cd "$HOME/dev/hedge-seg"
echo $(date)
echo $(git log -1 --pretty=%h)

script_name=$HOME/dev/hedge-seg/scripts/train_semseg_unet_resnet18.py
echo "bash file ===>"
cat $HOME/exps/hedge/semseg_unet/1.sh
echo "scripts ===>"
echo "build_pdok_wms_dataset.py:"
cat $HOME/dev/hedge-seg/scripts/data/build_pdok_wms_dataset.py
echo "convert_pdok_polylines_to_semseg.py:"
cat $HOME/dev/hedge-seg/scripts/data/convert_pdok_polylines_to_semseg.py
echo $script_name:
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