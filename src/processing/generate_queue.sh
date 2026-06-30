#!/bin/bash
#SBATCH --job-name=deeplab_queue
#SBATCH --partition=earth-5
#SBATCH --constraint=rhel8
#SBATCH --time=00-05:00:00    
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8       
#SBATCH --mem=32G                
#SBATCH --gpus=1

# ---------------------------------------------------------------------------
# CONFIGURATION
# Set MODEL_RUN to the timestamp folder name of your trained model.
# Example: unet_20260508_143200
# ---------------------------------------------------------------------------
MODEL_RUN="deeplab_20260511_115648"  


# ---------------------------------------------------------------------------
# Environment (identical to training job)
# ---------------------------------------------------------------------------
module load USS/2022
module load gcc/9.4.0-pe5.34
module load cuda/11.6.2
module load lsfm-init-miniconda/1.0.0

cd /cfs/earth/scratch/nogernic/BA_2026/src/processing
conda activate unet_gpu
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

dos2unix generate_queue.py

# ---------------------------------------------------------------------------
# Run queue generation
# --model  : full path to the SavedModel folder from training
# --threshold : confidence cutoff (0.90 recommended; lower = more blobs)
# --min-blob  : minimum blob size in pixels (10000 recommended)
# ---------------------------------------------------------------------------
MODEL_PATH="/cfs/earth/scratch/nogernic/BA_2026/models/${MODEL_RUN}/final_model"


python generate_queue.py \
    --model     "$MODEL_PATH" \

