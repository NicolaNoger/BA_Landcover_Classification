#!/bin/bash
#SBATCH --job-name=uav_compare
#SBATCH --partition=earth-5
#SBATCH --constraint=rhel8
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1

echo "=========================================="
echo "Job started at: $(date) on $(hostname)  (Job $SLURM_JOB_ID)"
echo "=========================================="

module load USS/2022
module load gcc/9.4.0-pe5.34
module load cuda/11.6.2
module load lsfm-init-miniconda/1.0.0

cd /cfs/earth/scratch/nogernic/BA_2026/src/UAV
conda activate unet_gpu
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

dos2unix compare_models.py

# Ablation comparison: coarse vs RGB+nDSM. With no args it auto-selects the
# newest uav_deeplab_coarse_* and uav_deeplab_rgbndsm_* runs. Extra args pass
# through, e.g.:  sbatch compare_models.sh --num-qualitative 8
python compare_models.py "$@"
