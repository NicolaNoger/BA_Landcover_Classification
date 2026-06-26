#!/bin/bash
#SBATCH --job-name=coarse_tiles_predict
#SBATCH --partition=earth-5
#SBATCH --constraint=rhel8
#SBATCH --time=00-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus=1

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

cd /cfs/earth/scratch/nogernic/BA_2026/src/UAV
conda activate unet_gpu
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" #without this it does not find libcudart.so -> no GPU

dos2unix predict_coarse_tiles.py

python predict_coarse_tiles.py

echo "Job finished at: $(date)"
