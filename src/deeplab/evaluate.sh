#!/bin/bash
#SBATCH --job-name=deeplabV3_evaluate
#SBATCH --partition=earth-5
#SBATCH --constraint=rhel8
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1

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

cd /cfs/earth/scratch/nogernic/BA_2026/src/deeplab
conda activate unet_gpu
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

dos2unix evaluate.py

# Usage:
#   sbatch evaluate.sh                          newest run, test split
#   sbatch evaluate.sh --model-dir ../../models/deeplab_20260618_133209
#   sbatch evaluate.sh --num-qualitative 8 --stitch-cols 49
python evaluate.py --arch unet --model-dir /cfs/earth/scratch/nogernic/BA_2026/models/unet_20260510_183058
