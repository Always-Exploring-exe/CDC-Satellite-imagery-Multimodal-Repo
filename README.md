# CDC Satellite Imagery House Price Prediction

This repository contains a **two-stage multimodal pipeline** for house price prediction:
- **Tabular XGBoost baseline (Model 1)** on `train_tabular.csv` / `test_tabular.csv`
- **Fusion CNN + MLP model (Model 2)** that combines NAIP satellite imagery with tabular features

The goal is to build a strong, reproducible baseline and then refine it with a multimodal model.

---
#### Very important note :
The naip_images folder has not been provided in this repo here , as files are too large for git , please access them and name them in as properly under ./naip_images/{train/test}_224/{image_id}.tif though this is already handled by the fetching pipeline.

## Repository Structure

### Core Notebooks (Run in Order)

- **`01_train_xgboost_baseline.ipynb`**  
  Trains a 5-fold cross-validated XGBoost model on tabular features to create a baseline prediction. Generates out-of-fold predictions for training data and ensemble predictions for test data. The preprocessing logic is available in `preprocessing.py`. Outputs include:
  - `train_processed_with_residuals.csv` (for fusion training)
  - `test_processed_for_fusion.csv` (for fusion inference)
  - `train_final_fusion.csv` (fusion-ready training data with IDs)
  - `submission_final.csv` (pure XGBoost baseline submission)

- **`02_fetch_satellite_images.ipynb`**  
  Downloads 4-channel NAIP satellite imagery (RGB + NIR) for each house location using the USGS NAIP ImageServer API. Images are saved as TIFF files in:
  - `naip_images/train_640/`
  - `naip_images/test_640/`

- **`03_train_fusion_model.ipynb`**  
  Trains a multimodal fusion model that combines satellite imagery with tabular features. Uses a ResNet50 backbone for image features and an MLP for tabular features, then fuses them to predict log-residuals relative to the XGBoost baseline. Saves the best model weights to `sota_fusion_best.pth`.

- **`04_make_test_predictions.ipynb`**  
  Generates final predictions on the test set by loading the trained fusion model and combining it with XGBoost baseline predictions. Produces the final submission file `final_submission.csv` with predicted house prices.

- **`05_explainability.ipynb`**  
  Provides explainability analysis for the fusion model using Grad-CAM (visualizes which regions of satellite images the model focuses on) and SHAP (quantifies the contribution of each tabular feature). Results are saved to the `explainability_results/` directory.

### Supporting Files

- **`preprocessing.py`**  
  Contains the preprocessing functions used for tabular data transformation. This includes date handling, feature engineering (age, renovation), zipcode conversion, and target transformation. The preprocessing logic is also available in the XGBoost training notebook.

- **`requirements.txt`**  
  Python dependencies needed to run all notebooks.

- **`other_models_and_their_weights/`**  
  Additional experimental fusion models and their pretrained weights:
  - `models_code/cross_attenton.py`: Cross-attention fusion variant that replaces the simple concatenation head with a tabular–image cross-attention block.
  - `models_code/without_4th_channel.py`: RGB-only variant that uses a standard 3-channel ResNet50 backbone (ignores the NIR channel).
  - `models_weights/cross_attention_weights.pth`: Weights trained for the cross-attention model.
  - `models_weights/without_4th_channel_weights.pth`: Weights trained for the RGB-only fusion model.

  To run these alternative models, you can:
  - Start from `03_train_fusion_model.ipynb`, and replace the `MultimodalDataset` and `FusionModel` definitions with the versions from the corresponding file in `models_code/`, and
  - Update the model weight path (e.g., `MODEL_SAVE_PATH` / `torch.load(...)`) to point to the matching file in `other_models_and_their_weights/models_weights/` (`cross_attention_weights.pth` or `without_4th_channel_weights.pth`).

---

## Setup

1. **Create environment** (example with `conda`):

```bash
conda create -n cdc-sat python=3.10 -y
conda activate cdc-sat
pip install -r requirements.txt
```

2. Ensure the following data files are present in the project root:
   - `train_tabular.csv`
   - `test_tabular.csv`

---

## End-to-End Workflow

### Step 1: Train XGBoost Baseline Model

1. Open **`01_train_xgboost_baseline.ipynb`**.
2. Run all cells to:
   - Load and preprocess tabular data (preprocessing code is in `preprocessing.py`)
   - Train 5-fold cross-validated XGBoost model
   - Generate out-of-fold predictions for training data
   - Generate ensemble predictions for test data
   - Create fusion-ready CSV files

**Outputs:**
- `train_processed_with_residuals.csv`
- `test_processed_for_fusion.csv`
- `train_final_fusion.csv`
- `submission_final.csv` (XGBoost baseline)

### Step 2: Fetch Satellite Imagery

1. Open **`02_fetch_satellite_images.ipynb`**.
2. Run all cells to:
   - Download 4-channel NAIP TIFF images (RGB + NIR) for each house location
   - Save images to `naip_images/train_640/` and `naip_images/test_640/`

**Note:** This step may take a while depending on your internet connection and the number of images to download.

### Step 3: Train Multimodal Fusion Model

1. Open **`03_train_fusion_model.ipynb`**.
2. Confirm the paths to:
   - `train_final_fusion.csv`
   - `naip_images/train_640/`
3. Run all cells to train the fusion model.

**Outputs:**
- `sota_fusion_best.pth` (best model weights)

### Step 4: Make Final Test Predictions

1. Ensure you have:
   - `xg_boost_test.csv` (XGBoost predictions on the test set - can be extracted from `test_processed_for_fusion.csv`)
   - `naip_images/test_640/` contains test TIFFs
   - `sota_fusion_best.pth` exists
2. Open **`04_make_test_predictions.ipynb`** and run all cells.

**Outputs:**
- `final_submission.csv` (final competition-style submission)

### Step 5: Explainability Analysis (Optional)

1. Open **`05_explainability.ipynb`**.
2. The notebook automatically loads target IDs from `final_submission.csv` (or you can set `TARGET_IDS` manually).
3. Run all cells to generate:
   - Grad-CAM heatmaps in `explainability_results/`
   - SHAP summaries for tabular features

---

## Notes

- The preprocessing logic is available in both `preprocessing.py` (as a reusable module) and within the XGBoost training notebook.
- The **core logic of all models** (tabular XGBoost and fusion CNN) is preserved; only the **notebook structure and documentation** have been cleaned up for reproducibility.
- `other_fusion_2nd_models_and_weights/` is intentionally untouched and can be considered a sandbox for alternative architectures.
- Each notebook contains a single markdown cell at the top explaining what the code does.
- The naip_images folder has not been provided in this repo here , as files are too large for git , please access them and name them in as properly under ./naip_images/{train/test}_224/{image_id}.tif though this is already handled by the fetching pipeline.
