#!/bin/bash
#SBATCH --job-name=unet_training
#SBATCH --partition=earth-4
#SBATCH --constraint=rhel8
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1

# Print job info
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "=========================================="

module load USS/2022
module load gcc/9.4.0-pe5.34
module load cuda/11.6.2
module load lsfm-init-miniconda/1.0.0   

cd /cfs/earth/scratch/nogernic/PA2/src/U-net
conda activate unet_gpu        
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" #without this it does not work, does not find libcudart.so so also no GPU

dos2unix train_unet.py

python train_unet.py