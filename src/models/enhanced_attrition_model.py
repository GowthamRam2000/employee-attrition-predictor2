import os
from dataclasses import dataclass
from typing import Dict, Optional

import joblib
import numpy as np
import torch
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

try:
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE

    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    SMOTE = ADASYN = BorderlineSMOTE = SMOTETomek = None


@dataclass
class _TorchHistory:
    history: dict


class _EnhancedDeepNet(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.LayerNorm(256),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.LayerNorm(128),
            torch.nn.PReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.LayerNorm(64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(64, 32),
            torch.nn.LayerNorm(32),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 16),
            torch.nn.ELU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _EnhancedWideNet(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EnhancedAttritionPredictor:
    def __init__(self, input_dim=None):
        self.models: Dict[str, object] = {}
        self.input_dim = input_dim
        self.history: Optional[_TorchHistory] = None
        self.ensemble_weights: Optional[Dict[str, float]] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_enhanced_model(self, input_dim):
        self.input_dim = input_dim
        self.models["deep"] = _EnhancedDeepNet(input_dim).to(self.device)
        self.models["wide"] = _EnhancedWideNet(input_dim).to(self.device)
        return self.models["deep"]

    def _prepare_tensor(self, X, y=None):
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        if y is None:
            return X_tensor
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(1)
        return X_tensor, y_tensor

    def _train_torch_model(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_tensors,
        *,
        epochs: int,
        optimizer,
        pos_weight: Optional[torch.Tensor] = None,
        patience: int = 20,
        scheduler_patience: int = 10,
    ):
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=scheduler_patience, min_lr=1e-5, verbose=False
        )
        history = {"loss": [], "val_loss": [], "val_auc": []}

        val_X_tensor, val_y_tensor = val_tensors
        val_y_np = val_y_tensor.detach().cpu().numpy().flatten()

        best_epoch = -1
        best_auc = -np.inf
        best_state = None

        for epoch in range(epochs):
            model.train()
            batch_losses = []

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            avg_loss = float(np.mean(batch_losses))

            model.eval()
            with torch.no_grad():
                val_logits = model(val_X_tensor)
                val_loss = float(criterion(val_logits, val_y_tensor).item())
                val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()

            try:
                val_auc = float(roc_auc_score(val_y_np, val_probs))
            except ValueError:
                val_auc = 0.0

            history["loss"].append(avg_loss)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)
            scheduler.step(val_auc)

            if val_auc > best_auc + 1e-4:
                best_auc = val_auc
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            elif epoch - best_epoch >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        return _TorchHistory(history=history)

    def train_ensemble(self, X_train, y_train, X_val=None, y_val=None, epochs=150, batch_size=32):
        if self.input_dim is None:
            self.build_enhanced_model(X_train.shape[1])
        elif "deep" not in self.models or "wide" not in self.models:
            self.build_enhanced_model(self.input_dim)

        if IMBLEARN_AVAILABLE:
            sampler = BorderlineSMOTE(random_state=42, k_neighbors=5)
            X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
        else:
            X_balanced, y_balanced = X_train, y_train

        if X_val is None or y_val is None:
            X_balanced, X_val, y_balanced, y_val = train_test_split(
                X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
            )

        train_X_t, train_y_t = self._prepare_tensor(X_balanced, y_balanced)
        val_X_t, val_y_t = self._prepare_tensor(X_val, y_val)

        train_dataset = torch.utils.data.TensorDataset(train_X_t, train_y_t)
        train_loader_deep = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loader_wide = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size * 2, shuffle=True)

        deep_optimizer = torch.optim.AdamW(
            self.models["deep"].parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
        )
        wide_optimizer = torch.optim.NAdam(self.models["wide"].parameters(), lr=0.001)

        pos_weight = torch.tensor([3.0], device=self.device)

        self.history = self._train_torch_model(
            self.models["deep"],
            train_loader_deep,
            (val_X_t, val_y_t),
            epochs=epochs,
            optimizer=deep_optimizer,
            pos_weight=pos_weight,
            patience=25,
            scheduler_patience=10,
        )

        self._train_torch_model(
            self.models["wide"],
            train_loader_wide,
            (val_X_t, val_y_t),
            epochs=max(epochs // 2, 1),
            optimizer=wide_optimizer,
            pos_weight=None,
            patience=15,
            scheduler_patience=8,
        )

        self.models["xgboost"] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            scale_pos_weight=3,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        self.models["xgboost"].fit(X_balanced, y_balanced)

        self.models["rf"] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.models["rf"].fit(X_balanced, y_balanced)

        self.models["gb"] = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        )
        self.models["gb"].fit(X_balanced, y_balanced)

        self._calculate_ensemble_weights(val_X_t.cpu().numpy(), y_val)
        return self.history

    def _predict_torch(self, model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
        model.eval()
        X_tensor = self._prepare_tensor(X)
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs

    def _calculate_ensemble_weights(self, X_val, y_val):
        weights = {}
        predictions = {}

        for name, model in self.models.items():
            if name in ["deep", "wide"]:
                preds = self._predict_torch(model, X_val)
            else:
                preds = model.predict_proba(X_val)[:, 1]
            predictions[name] = preds

            try:
                auc = roc_auc_score(y_val, preds)
            except ValueError:
                auc = 0.0
            weights[name] = max(auc, 0.0)

        total = sum(weights.values())
        if total == 0:
            self.ensemble_weights = {k: 1 / len(weights) for k in weights}
        else:
            self.ensemble_weights = {k: v / total for k, v in weights.items()}

    def predict_proba(self, X):
        if not self.models:
            raise ValueError("Models not trained. Call train_ensemble or load_ensemble first.")

        predictions = []
        weights = self.ensemble_weights or {}
        default_weight = 1 / len(self.models) if self.models else 0

        for name, model in self.models.items():
            weight = weights.get(name, default_weight)
            if name in ["deep", "wide"]:
                pred = self._predict_torch(model, X)
            else:
                pred = model.predict_proba(X)[:, 1]
            predictions.append(pred * weight)

        return np.sum(predictions, axis=0)

    def predict_classes(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs > threshold).astype(int)

    def optimize_threshold(self, X_val, y_val):
        probs = self.predict_proba(X_val)
        best_threshold = 0.5
        best_f1 = 0.0

        for threshold in np.arange(0.3, 0.7, 0.01):
            preds = (probs > threshold).astype(int)
            report = classification_report(y_val, preds, output_dict=True, zero_division=0)
            f1 = report["weighted avg"]["f1-score"]
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold

    def evaluate(self, X_test, y_test, optimize_threshold=True):
        y_pred_proba = self.predict_proba(X_test)

        if optimize_threshold:
            threshold = self.optimize_threshold(X_test, y_test)
        else:
            threshold = 0.5

        y_pred = (y_pred_proba > threshold).astype(int)

        accuracy = float(np.mean(y_pred == y_test))
        auc_score = float(roc_auc_score(y_test, y_pred_proba))

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        return {
            "accuracy": accuracy,
            "auc": auc_score,
            "classification_report": report,
            "confusion_matrix": cm,
            "roc_curve": (fpr, tpr, thresholds),
            "threshold": threshold,
        }

    def save_ensemble(self, path="models/"):
        os.makedirs(path, exist_ok=True)
        if "deep" not in self.models or "wide" not in self.models:
            raise ValueError("Neural models not trained. Train before saving.")

        torch.save(self.models["deep"].state_dict(), os.path.join(path, "enhanced_deep_model.pt"))
        torch.save(self.models["wide"].state_dict(), os.path.join(path, "enhanced_wide_model.pt"))

        joblib.dump(self.models["xgboost"], os.path.join(path, "xgboost_model.pkl"))
        joblib.dump(self.models["rf"], os.path.join(path, "rf_model.pkl"))
        joblib.dump(self.models["gb"], os.path.join(path, "gb_model.pkl"))

        metadata = {
            "input_dim": self.input_dim,
            "ensemble_weights": self.ensemble_weights,
        }
        joblib.dump(metadata, os.path.join(path, "enhanced_metadata.pkl"))

    def load_ensemble(self, path="models/"):
        metadata = joblib.load(os.path.join(path, "enhanced_metadata.pkl"))
        self.input_dim = metadata["input_dim"]
        self.ensemble_weights = metadata.get("ensemble_weights")

        self.build_enhanced_model(self.input_dim)

        deep_state = torch.load(os.path.join(path, "enhanced_deep_model.pt"), map_location=self.device)
        wide_state = torch.load(os.path.join(path, "enhanced_wide_model.pt"), map_location=self.device)

        self.models["deep"].load_state_dict(deep_state)
        self.models["wide"].load_state_dict(wide_state)

        self.models["xgboost"] = joblib.load(os.path.join(path, "xgboost_model.pkl"))
        self.models["rf"] = joblib.load(os.path.join(path, "rf_model.pkl"))
        self.models["gb"] = joblib.load(os.path.join(path, "gb_model.pkl"))
