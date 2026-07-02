# Multimodal Landcover Segmentation with Assisted Label Refinement

Bachelor thesis project (ZHAW, Applied Digital Life Sciences) on semantic land
cover segmentation of a suburban study area in Wädenswil (Canton Zurich,
Switzerland), combining aerial imagery, LiDAR and UAV data.

The project trains a **DeepLabV3+** model on a 7-channel multimodal raster stack
(NIR, R, G, B, nDSM, LiDAR intensity, NDVI) at 0.1 m resolution over an 8-class
taxonomy, and a second, finer UAV stage at ~0.015 m over a more detailed class
set. A key contribution is an **assisted label-review workflow** that uses the
model's own high-confidence disagreements with the cadastral labels to find and
correct annotation errors.

## Data sources

| Source | Provider | Role |
| --- | --- | --- |
| SWISSIMAGE RS | swisstopo | 4-band (NIR, R, G, B) orthophotos, 0.1 m GSD |
| swissALTI3D | swisstopo | Digital Terrain Model (bare-earth reference for the nDSM) |
| LiDAR point cloud | Geoportal Kanton Zürich | DSM and return-intensity rasters |
| Cadastral data (AV) | Geoportal Kanton Zürich | Reference labels, aggregated to 8 classes |
| UAV survey | Own acquisition (DJI Mavic 3 Pro) | Fine-scale stage, ~0.015 m GSD |

## Repository layout

```text
src/
  processing/        Data preparation and the label-review tool
    lidar_normalize.py     Derive nDSM (DSM - swissALTI3D DTM) and intensity rasters
    create_layer_stack.py  Build the co-registered 7-channel raster stack
    generate_queue.py      Build the label-review queue from model disagreements
    labeling_review.py     Streamlit interface for manual label review
  deeplab/           Main aerial DeepLabV3+ model
    deeplab_v3plus.py, dataloader.py, train_deeplab.py, evaluate.py
  U-Net/             Early U-Net exploration + exploratory data analysis
    U_net.py, train_unet.py, image_snipper.py, data_analysis.ipynb
  UAV/               Fine-scale UAV stage (cascade vs. light configuration)
    train_uav.py, predict_coarse_tiles.py, compare_models.py, uav_image_snipper.py

environment.yml      Conda environment (name: ENV_HPC)
requirements.txt     pip dependencies
```

`.sh` files next to the Python scripts are SLURM job scripts used to run training
and evaluation on the ZHAW HPC cluster.

## Pipeline overview

1. **Preprocessing** — `lidar_normalize.py` derives the nDSM and intensity
   rasters from the classified LAZ files; `create_layer_stack.py` co-registers
   all layers into the 7-channel stack; the image snippers tile the stack into
   512×512 (aerial) / 1024×1024 (UAV) patches.
2. **Training** — `train_deeplab.py` (aerial) and `train_uav.py` (UAV) train the
   segmentation models with on-the-fly z-score normalisation.
3. **Evaluation** — `evaluate.py` computes per-class IoU/F1, confusion matrices
   and qualitative prediction figures.
4. **Label review** — `generate_queue.py` extracts high-confidence
   model/label disagreement regions; `labeling_review.py` presents them in a
   Streamlit UI for accept/keep/both-wrong decisions; the model is then
   re-evaluated against the corrected labels.

## Setup

```bash
# Option A: conda
conda env create -f environment.yml
conda activate ENV_HPC

# Option B: pip
pip install -r requirements.txt
```

## Not included in the repository

The following are excluded via `.gitignore` because of their size and because
they are reproducible from the source data and scripts:

- `data/` — raw and processed imagery, LiDAR and rasters
- `models/` — trained checkpoints and evaluation outputs (~2.6 GB; individual
  checkpoints exceed GitHub's 100 MB file limit)
- `src/**/reports/` — generated analysis figures
