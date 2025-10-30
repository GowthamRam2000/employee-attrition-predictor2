## Employee Attrition Predictor

This project provides an end-to-end workflow for training deep-learning and ensemble models that predict employee attrition, along with a Streamlit dashboard for interactive analytics and scoring.

---

- Python 3.9 or 3.10 (3.9.17+ recommended)


---

### 2. Setup
```bash
python -m venv .venv
source .venv/bin/activate                      

pip install --upgrade pip
pip install -r requirements.txt
```
```bash
python train_model.py \
  --data WA_Fn-UseC_-HR-Employee-Attrition.csv \
  --epochs 100 \
  --batch_size 32 \
  --use_smote
```
Artifacts saved to `models/`:
- `attrition_model.pt` — PyTorch weights
- `model_metadata.pkl`, `training_metrics.pkl`
- `scaler.pkl`, `label_encoders.pkl`, `feature_columns.pkl`, `feature_defaults.pkl`

```bash
python train_enhanced.py
```
Outputs (in addition to baseline preprocessors):
- `enhanced_deep_model.pt`, `enhanced_wide_model.pt`
- `xgboost_model.pkl`, `rf_model.pkl`, `gb_model.pkl`
- `enhanced_metadata.pkl`, `enhanced_training_metrics.pkl`
- `poly_transformer.pkl`, `top_features.pkl`

> The script expects XGBoost to be available. If installation fails, fix the environment and rerun.

---

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Key features:
- **Dashboard:** KPIs, confusion matrix, ROC curves, attrition summary
- **Train Model:** Upload data and retrain directly from the UI
- **Make Predictions:** Single-employee scoring with what-if analysis
- **Batch Scoring:** Upload CSV/XLSX for bulk inference plus business-cost calculator
- **Analytics & Model Performance:** Feature importance, cohort summaries, saved metric review

---

### 5. Using the Models Programmatically
```python
from src.models.attrition_model import AttritionPredictor
from src.utils.data_processor import HRDataProcessor

processor = HRDataProcessor()
processor.load_preprocessors('models/')

model = AttritionPredictor()
model.load_model('models/')

X, _, _ = processor.preprocess_data(your_dataframe, is_training=False)
probabilities = model.predict_proba(X)
```

For the enhanced ensemble:
```python
from src.models.enhanced_attrition_model import EnhancedAttritionPredictor

ensemble = EnhancedAttritionPredictor()
ensemble.load_ensemble('models/')
scores = ensemble.predict_proba(X)
```

---

### 6. Troubleshooting
- **PyTorch segmentation fault:** Make sure the current Python interpreter matches the platform torch wheel (run `python -c "import torch; print(torch.__version__)"`). Reinstall torch manually if needed.
- **XGBoost import errors:** `pip install xgboost==2.0.3` (or brew install libomp on macOS if OpenMP is missing).
- **Streamlit fails to load models:** Ensure training scripts have been executed so `.pt` artifacts exist. The app warns if it only finds legacy `.h5` files.
- **Unknown categorical values at inference time:** The processor maps unseen categories to `-1`; update label encoders by retraining when new labels appear in production data.





