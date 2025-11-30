# Cost Predictor Pipeline (DVC-ready)

This repo contains scripts to create a training pipeline that:
- Generates text embeddings with a Sentence Transformer
- Generates image embeddings with a ViT model
- Merges embeddings and pairs with price
- Trains a PyTorch refiner to produce refined embeddings
- Trains a LightGBM model to predict price

Use `dvc repro` to run the pipeline once you have DVC initialized and the raw data added.

Quick start (PowerShell):

```powershell
python -m pip install -r requirements.txt
dvc init
# Add raw data to DVC
dvc add dataset/train.csv
dvc add -R dataset/images
git add dataset/train.csv.dvc dataset/images.dvc .dvcignore dvc.yaml params.yaml requirements.txt
git commit -m "Add dvc pipeline files"
# Run full pipeline
dvc repro
```

Files of interest:
- `src/text_embeddings/embed.py`
- `src/image_embeddings/embed.py`
- `src/merging_embedding/merge.py`
- `src/refined_embed/train.py`
- `src/lgbm_train/train.py`
- `dvc.yaml`, `params.yaml`
