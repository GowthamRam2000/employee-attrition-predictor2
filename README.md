## Employee Attrition Predictor

Interactive Streamlit experience and training pipeline for forecasting employee attrition using a PyTorch deep neural network plus an enhanced wide & deep ensemble that blends neural and tree-based models.

---

### Project Overview
- End-to-end workspace for exploring the IBM HR attrition dataset, training predictive models, and serving interactive insights.
- Baseline PyTorch classifier calibrated with class balancing techniques.
- Enhanced ensemble (model version 2.0) combining two neural backbones with XGBoost, Random Forest, and Gradient Boosting components.
- Streamlit UI for dashboards, retraining, batch scoring, and what-if analysis.

---

### Repository Layout
- `README.md` – Project documentation, setup, and usage instructions.
- `app.py` – Streamlit application that loads trained artifacts, provides dashboards, and offers scoring utilities.
- `train_model.py` – CLI for training baseline (`--model_version 1.0`) and enhanced (`--model_version 2.0`) models and exporting artifacts.
- `src/models/attrition_model.py` – Baseline PyTorch model definition, training loop, evaluation, and serialization helpers.
- `src/models/enhanced_attrition_model.py` – Wide & deep neural ensemble plus XGBoost/RandomForest/GradientBoosting stacking logic.
- `src/utils/data_processor.py` – Data ingestion, preprocessing, label encoding, scaling, and persistence utilities.
- `src/utils/feature_engineering.py` – Advanced feature engineering used by the enhanced training pipeline.
- `models/` – Saved model weights, preprocessors, metadata, and evaluation metrics (populated after training).
- `assets/` – Placeholder for static assets (logos, images) referenced by the Streamlit app.
- `requirements.txt` – Pinned Python dependencies.
- `WA_Fn-UseC_-HR-Employee-Attrition.csv` – Sample IBM HR dataset for experimentation.

> `__pycache__/` directories are Python bytecode caches and can be ignored.

---

### Data Requirements
- Source data must match the IBM HR attrition schema (`Attrition` target column plus demographic and employment fields).
- CSV (`.csv`) or Excel (`.xlsx`, `.xls`) files are supported by `HRDataProcessor.load_data`.
- Drop-in replacements with new data should maintain equivalent column names or include mappable categories for label encoding.

---

### Dependencies
- Python 3.9 or 3.10 (3.9.17+ recommended for PyTorch wheels).
- Core libraries (see `requirements.txt` for exact versions):
  - `streamlit`, `plotly` – interactive app and visualizations.
  - `pandas`, `numpy`, `scikit-learn` – data wrangling, preprocessing, classical ML utilities.
  - `torch` – baseline and neural ensemble models.
  - `xgboost`, `imbalanced-learn` – gradient boosting and imbalance handling.
  - `matplotlib`, `seaborn`, `shap`, `lime` – optional visual diagnostics.
  - `openpyxl`, `xlrd` – Excel ingestion.

Install everything with:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

### Training Pipelines
Run the unified CLI after downloading the IBM HR dataset into the project root (or provide the absolute path).

```bash
python train_model.py \
  --data WA_Fn-UseC_-HR-Employee-Attrition.csv \
  --epochs 120 \
  --batch_size 32 \
  --use_smote \
  --model_version 2.0 \
  --save_path models/
```

Key arguments:
- `--data` (required): Path to the input dataset.
- `--epochs`: Training epochs (baseline and enhanced neural components). Defaults to 150.
- `--batch_size`: Mini-batch size. Defaults to 32.
- `--use_smote`: Enable SMOTE oversampling for baseline model class balancing.
- `--model_version`: Choose `1.0` for the baseline PyTorch model or `2.0` to add the enhanced ensemble.
- `--save_path`: Directory where artifacts are exported (defaults to `models/`).

What happens during training:
1. `HRDataProcessor` loads and cleans the dataset, handling categorical encodings and scaling.
2. Baseline PyTorch classifier trains with optional SMOTE/class weights, evaluates on held-out data, and calibrates a decision threshold.
3. Metrics, feature importance estimates, and preprocessing objects are saved.
4. When `--model_version 2.0` is selected, additional engineered features are produced, neural wide & deep models are trained alongside tree models, and ensemble weights are learned.
5. All artifacts are written to `save_path` for later inference (see next section).

---

### Exported Artifacts
- **Baseline Core**
  - `attrition_model.pt`, `model_metadata.pkl` – trained PyTorch weights and metadata (input dim, optimized threshold).
  - `training_metrics.pkl` – evaluation dictionary (`accuracy`, `auc`, classification report, confusion matrix, ROC curve).
- **Preprocessing**
  - `scaler.pkl`, `label_encoders.pkl`, `feature_columns.pkl`, `feature_defaults.pkl` – reproducible transformations and defaults.
- **Enhanced Ensemble (v2.0)**
  - `enhanced_deep_model.pt`, `enhanced_wide_model.pt` – neural components.
  - `xgboost_model.pkl`, `rf_model.pkl`, `gb_model.pkl` – gradient boosting, random forest, and XGBoost models.
  - `enhanced_metadata.pkl` – ensemble weights and input dimensionality.
  - `enhanced_training_metrics.pkl` – evaluation results for the enhanced stack.
  - `poly_transformer.pkl`, `top_features.pkl`, `enh_*` preprocessors – engineered feature utilities.
- `model_version.txt` – training summary, timestamps, and key metrics.

Re-run `train_model.py` whenever you refresh the dataset or hyperparameters; the Streamlit app always loads the newest artifacts on startup. Set `ATTRITION_MODELS_DIR=/path/to/models` if you keep the artifacts outside the repository.

---

### Inference & Serving
- **Streamlit UI (interactive inference)**
  ```bash
  streamlit run app.py
  ```
  - Pages: Dashboard KPIs & charts, Train Model (user uploads), Make Predictions (single employee), Batch Scoring (CSV/XLSX plus cost calculator), Analytics / Model Performance.
  - Requires the `models/` directory to be populated; the app displays guidance if artifacts are missing.

- **Programmatic Usage (baseline model)**
  ```python
  import pandas as pd
  from src.models.attrition_model import AttritionPredictor
  from src.utils.data_processor import HRDataProcessor

  df = pd.read_csv("your_employees.csv")

  processor = HRDataProcessor()
  processor.load_preprocessors("models/")

  model = AttritionPredictor()
  model.load_model("models/")

  X, _, feature_names = processor.preprocess_data(df, is_training=False)
  scores = model.predict_proba(X)
  classes = model.predict_classes(X)  # Uses calibrated threshold from metadata
  ```

- **Programmatic Usage (enhanced ensemble)**
  ```python
  from src.models.enhanced_attrition_model import EnhancedAttritionPredictor

  ensemble = EnhancedAttritionPredictor()
  ensemble.load_ensemble("models/")
  enhanced_scores = ensemble.predict_proba(X)  # reuse X from the previous snippet
  ```

- **Batch scoring via Streamlit**
  - Use the *Batch Scoring* page to upload CSV/XLSX files; results can be downloaded along with attrition cost estimates.
  - For automated pipelines, wrap the programmatic usage snippet in your ETL job and write predictions back to storage.

---

### Deployment (Streamlit Cloud or similar)
1. Run `train_model.py` locally so `models/` contains fresh artifacts.
2. Commit/push the repo **including the `models/` folder** (or upload artifacts alongside the code).
3. Configure the deployment command to `streamlit run app.py`.
4. Optional: set `ATTRITION_MODELS_DIR` if artifacts live outside the working directory.

On startup the app checks for required files and prompts you to retrain if they are missing.

---

### Troubleshooting
- **PyTorch wheel issues** – confirm Python/torch compatibility (`python -c "import torch; print(torch.__version__)"`) and reinstall if needed.
- **XGBoost import failures** – reinstall `xgboost==2.0.3`; on macOS you may also need `brew install libomp`.
- **Missing artifacts in Streamlit** – rerun `train_model.py` so `.pt`/`.pkl` files exist in `models/`.
- **Unseen categorical levels** – `HRDataProcessor` maps unseen categories to `-1`; retrain to fully support new categorical values.
