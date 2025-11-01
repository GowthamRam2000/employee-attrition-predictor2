## Employee Attrition Predictor

Interactive Streamlit experience and training pipeline for forecasting employee attrition using deep neural networks plus an extended wide & deep ensemble.

---

### 1. Environment Setup
- Python 3.9 or 3.10 (3.9.17+ recommended)
- Install deps:
  ```bash
  python -m venv .venv
  source .venv/bin/activate

  pip install --upgrade pip
  pip install -r requirements.txt
  ```

---

### 2. Train & Export Artifacts
Run the unified CLI once you have the IBM HR dataset locally:
```bash
python train_model.py \
  --data WA_Fn-UseC_-HR-Employee-Attrition.csv \
  --epochs 120 \
  --batch_size 32 \
  --use_smote \
  --model_version 2.0
```

This command first trains the baseline PyTorch network and, when `--model_version 2.0` is supplied, also fits the enhanced wide & deep ensemble with tree models. Artifacts are written to `models/`:

- Baseline: `attrition_model.pt`, `model_metadata.pkl`, `training_metrics.pkl`
- Preprocessing: `scaler.pkl`, `label_encoders.pkl`, `feature_columns.pkl`, `feature_defaults.pkl`
- Enhanced (2.0): `enhanced_deep_model.pt`, `enhanced_wide_model.pt`, `xgboost_model.pkl`, `rf_model.pkl`, `gb_model.pkl`, `enhanced_metadata.pkl`, `enhanced_training_metrics.pkl`, `poly_transformer.pkl`, `top_features.pkl`

> Re-run the CLI whenever you refresh data or hyperparameters; the Streamlit app automatically loads the freshest artifacts at startup.
> Set `ATTRITION_MODELS_DIR=/path/to/models` if you keep artifacts outside the project root.

---

### 3. Launch Streamlit Locally
```bash
streamlit run app.py
```

Pages include:
- **Dashboard** with KPIs, ROC, confusion matrix, attrition summary
- **Train Model** to retrain using uploaded data
- **Make Predictions** for single-employee scoring & what-if analysis
- **Batch Scoring** for CSV/XLSX uploads plus business cost calculator
- **Analytics / Model Performance** for feature insights and stored metrics

---

### 4. Deploying to Streamlit Cloud (or similar)
1. Run `train_model.py` locally so the `models/` directory contains all artifacts.
2. Commit/push the repo **including the `models/` folder** or upload it alongside the code.
3. Configure the deployment command: `streamlit run app.py`.
4. Optional: set environment variables if using non-default artifact paths—otherwise the defaults work.

On boot the app checks for required files and warns if training needs to be rerun.

---

### 5. Programmatic Usage
```python
from src.models.attrition_model import AttritionPredictor
from src.utils.data_processor import HRDataProcessor

processor = HRDataProcessor()
processor.load_preprocessors("models/")

model = AttritionPredictor()
model.load_model("models/")

X, _, _ = processor.preprocess_data(your_dataframe, is_training=False)
probs = model.predict_proba(X)
classes = model.predict_classes(X)  # uses the calibrated threshold by default
```

Enhanced ensemble:
```python
from src.models.enhanced_attrition_model import EnhancedAttritionPredictor

ensemble = EnhancedAttritionPredictor()
ensemble.load_ensemble("models/")
scores = ensemble.predict_proba(X)
```

---

### 6. Troubleshooting
- **PyTorch wheel issues** – verify Python/torch compatibility (`python -c "import torch; print(torch.__version__)"`), reinstall if needed.
- **XGBoost import failures** – `pip install xgboost==2.0.3` (macOS may also need `brew install libomp`).
- **Missing artifacts in Streamlit** – rerun `train_model.py` so `.pt`/`.pkl` files exist in `models/`.
- **Unseen categorical levels** – the processor maps them to `-1`; retrain to fully support new categories.
