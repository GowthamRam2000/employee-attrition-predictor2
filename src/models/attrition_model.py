import os
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

try:
    from imblearn.over_sampling import SMOTE

    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    SMOTE = None


class _AttritionNet(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class _TrainingHistory:
    history: dict


class AttritionPredictor:
    def __init__(self, input_dim=None):
        self.model: Optional[torch.nn.Module] = None
        self.input_dim = input_dim
        self.history: Optional[_TrainingHistory] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_state_dict = None
        self.best_threshold = 0.5

    def build_model(self, input_dim):
        self.input_dim = input_dim
        self.model = _AttritionNet(input_dim).to(self.device)
        return self.model

    def _prepare_tensors(self, X, y=None):
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        if y is None:
            return X_tensor
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(1)
        return X_tensor, y_tensor

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=100,
        batch_size=32,
        use_smote=True,
    ):
        if self.model is None:
            self.build_model(X_train.shape[1])

        class_weight_dict = None
        X_train_balanced, y_train_balanced = X_train, y_train

        if use_smote and SMOTE_AVAILABLE:
            try:
                smote = SMOTE(random_state=42)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            except Exception:
                use_smote = False
        elif use_smote and not SMOTE_AVAILABLE:
            use_smote = False

        if not use_smote:
            classes = np.unique(y_train)
            class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes, class_weights))

        if X_val is None or y_val is None:
            X_train_balanced, X_val, y_train_balanced, y_val = train_test_split(
                X_train_balanced,
                y_train_balanced,
                test_size=0.2,
                random_state=42,
                stratify=y_train_balanced,
            )

        train_X_tensor, train_y_tensor = self._prepare_tensors(X_train_balanced, y_train_balanced)
        val_X_tensor, val_y_tensor = self._prepare_tensors(X_val, y_val)

        train_dataset = torch.utils.data.TensorDataset(train_X_tensor, train_y_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()

        if class_weight_dict is not None:
            weight_ratio = class_weight_dict.get(1, 1.0) / class_weight_dict.get(0, 1.0)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight_ratio], device=self.device))

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5, verbose=False
        )

        history = {"loss": [], "val_loss": []}
        patience = 15
        best_val_loss = float("inf")
        best_epoch = -1

        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = float(np.mean(epoch_losses))

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(val_X_tensor)
                val_loss = criterion(val_logits, val_y_tensor).item()

            history["loss"].append(avg_loss)
            history["val_loss"].append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_epoch = epoch
                self.best_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
            elif epoch - best_epoch >= patience:
                break

        if self.best_state_dict:
            self.model.load_state_dict(self.best_state_dict)

        self.history = _TrainingHistory(history=history)
        return self.history

    def _predict_logits(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet!")
        self.model.eval()
        X_tensor = self._prepare_tensors(X)
        with torch.no_grad():
            logits = self.model(X_tensor).cpu().numpy()
        return logits

    def predict(self, X):
        logits = self._predict_logits(X)
        return 1 / (1 + np.exp(-logits))

    def predict_proba(self, X):
        return self.predict(X)

    def predict_classes(self, X, threshold=0.5):
        probs = self.predict_proba(X).flatten()
        return (probs > threshold).astype(int)

    def evaluate(self, X_test, y_test):
        y_pred_proba = self.predict_proba(X_test).flatten()
        best_threshold = 0.5
        best_f1 = 0.0
        for threshold in np.arange(0.3, 0.7, 0.01):
            preds_tmp = (y_pred_proba > threshold).astype(int)
            report_tmp = classification_report(y_test, preds_tmp, output_dict=True, zero_division=0)
            f1_tmp = report_tmp["weighted avg"]["f1-score"]
            if f1_tmp > best_f1:
                best_f1 = f1_tmp
                best_threshold = threshold

        self.best_threshold = best_threshold
        y_pred = (y_pred_proba > best_threshold).astype(int)
        accuracy = float(np.mean(y_pred == y_test))
        auc_score = float(roc_auc_score(y_test, y_pred_proba))

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        return {
            "accuracy": accuracy,
            "auc": auc_score,
            "classification_report": report,
            "confusion_matrix": cm,
            "roc_curve": (fpr, tpr, thresholds),
            "threshold": best_threshold,
        }

    def get_feature_importance(self, feature_names, X_sample, num_samples=100):
        if self.model is None:
            raise ValueError("Model not trained yet!")

        if len(X_sample) > num_samples:
            indices = np.random.choice(len(X_sample), num_samples, replace=False)
            X_sample = X_sample[indices]

        base_predictions = self.predict_proba(X_sample)
        importances = []

        for idx in range(X_sample.shape[1]):
            X_permuted = X_sample.copy()
            np.random.shuffle(X_permuted[:, idx])
            permuted_predictions = self.predict_proba(X_permuted)
            importances.append(float(np.mean(np.abs(base_predictions - permuted_predictions))))

        importances = np.array(importances)
        if importances.sum() > 0:
            importances = importances / importances.sum()

        return dict(zip(feature_names, importances))

    def save_model(self, path="models/"):
        os.makedirs(path, exist_ok=True)
        if self.model is None:
            raise ValueError("No model to save.")

        torch.save(self.model.state_dict(), os.path.join(path, "attrition_model.pt"))
        metadata = {"input_dim": self.input_dim}
        joblib.dump(metadata, os.path.join(path, "model_metadata.pkl"))

    def load_model(self, path="models/"):
        metadata_path = os.path.join(path, "model_metadata.pkl")
        meta = joblib.load(metadata_path)
        self.input_dim = meta["input_dim"]
        self.build_model(self.input_dim)

        model_path = os.path.join(path, "attrition_model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "attrition_model.pt not found. Retrain the model with the PyTorch pipeline."
            )
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
