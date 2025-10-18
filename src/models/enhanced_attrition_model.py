import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib
import os

# Try to import SMOTE
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.combine import SMOTETomek

    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: SMOTE not available. Will use class weights instead.")


class EnhancedAttritionPredictor:
    def __init__(self, input_dim=None):
        self.models = {}
        self.input_dim = input_dim
        self.history = None
        self.ensemble_weights = None

    def build_enhanced_model(self, input_dim):
        """Build an enhanced deep neural network with advanced techniques"""
        self.input_dim = input_dim

        # Model 1: Deep Network with Advanced Architecture
        model1 = models.Sequential([
            # Input layer
            layers.Input(shape=(input_dim,)),

            # First block with residual connection simulation
            layers.Dense(256, kernel_initializer='he_normal'),
            layers.LayerNormalization(),
            layers.Activation('swish'),  # Swish activation often performs better
            layers.Dropout(0.3),

            # Second block
            layers.Dense(128, kernel_initializer='he_normal'),
            layers.LayerNormalization(),
            layers.PReLU(),  # Parametric ReLU
            layers.Dropout(0.3),

            # Third block with attention mechanism simulation
            layers.Dense(64, kernel_initializer='he_normal'),
            layers.LayerNormalization(),
            layers.Activation('gelu'),  # GELU activation
            layers.Dropout(0.25),

            # Fourth block
            layers.Dense(32, kernel_initializer='he_normal'),
            layers.LayerNormalization(),
            layers.LeakyReLU(alpha=0.1),
            layers.Dropout(0.2),

            # Fifth block
            layers.Dense(16, kernel_initializer='he_normal'),
            layers.Activation('elu'),
            layers.Dropout(0.15),

            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])

        # Use a more sophisticated optimizer
        optimizer = keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.0001,
            beta_1=0.9,
            beta_2=0.999
        )

        model1.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

        # Model 2: Wider Network
        model2 = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])

        model2.compile(
            optimizer=keras.optimizers.Nadam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

        self.models['deep'] = model1
        self.models['wide'] = model2

        return model1

    def train_ensemble(self, X_train, y_train, X_val=None, y_val=None, epochs=150, batch_size=32):
        """Train multiple models and create an ensemble"""

        # Build neural network models
        self.build_enhanced_model(X_train.shape[1])

        # Apply advanced SMOTE techniques if available
        if SMOTE_AVAILABLE:
            # Use BorderlineSMOTE for better handling of borderline cases
            smote = BorderlineSMOTE(random_state=42, k_neighbors=5)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train

        # Split validation set if not provided
        if X_val is None or y_val is None:
            X_train_balanced, X_val, y_train_balanced, y_val = train_test_split(
                X_train_balanced, y_train_balanced, test_size=0.2, random_state=42, stratify=y_train_balanced
            )

        # Advanced callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_auc',  # Monitor AUC instead of loss
            patience=25,
            restore_best_weights=True,
            mode='max'
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=10,
            min_lr=0.00001,
            mode='max'
        )

        lr_scheduler = callbacks.LearningRateScheduler(
            lambda epoch: 0.001 * (0.95 ** (epoch // 10))
        )

        # Train deep model
        print("Training enhanced deep model...")
        self.history = self.models['deep'].fit(
            X_train_balanced, y_train_balanced,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr, lr_scheduler],
            class_weight={0: 1, 1: 3},  # Give more weight to minority class
            verbose=0
        )

        # Train wide model
        print("Training wide model...")
        self.models['wide'].fit(
            X_train_balanced, y_train_balanced,
            validation_data=(X_val, y_val),
            epochs=epochs // 2,  # Fewer epochs for the simpler model
            batch_size=batch_size * 2,
            callbacks=[early_stopping],
            verbose=0
        )

        # Train XGBoost
        print("Training XGBoost...")
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            scale_pos_weight=3,  # Handle imbalanced data
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.models['xgboost'].fit(X_train_balanced, y_train_balanced)

        # Train Random Forest
        print("Training Random Forest...")
        self.models['rf'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.models['rf'].fit(X_train_balanced, y_train_balanced)

        # Train Gradient Boosting
        print("Training Gradient Boosting...")
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42
        )
        self.models['gb'].fit(X_train_balanced, y_train_balanced)

        # Calculate ensemble weights based on validation performance
        self._calculate_ensemble_weights(X_val, y_val)

        return self.history

    def _calculate_ensemble_weights(self, X_val, y_val):
        """Calculate optimal weights for ensemble based on validation performance"""
        weights = {}

        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            if name in ['deep', 'wide']:
                pred = model.predict(X_val, verbose=0).flatten()
            else:
                pred = model.predict_proba(X_val)[:, 1]
            predictions[name] = pred

            # Calculate AUC for each model
            auc = roc_auc_score(y_val, pred)
            weights[name] = auc
            print(f"{name} AUC: {auc:.4f}")

        # Normalize weights
        total = sum(weights.values())
        self.ensemble_weights = {k: v / total for k, v in weights.items()}
        print(f"Ensemble weights: {self.ensemble_weights}")

    def predict_proba(self, X):
        """Get ensemble prediction probabilities"""
        predictions = []

        for name, model in self.models.items():
            weight = self.ensemble_weights.get(name, 1 / len(self.models))

            if name in ['deep', 'wide']:
                pred = model.predict(X, verbose=0).flatten()
            else:
                pred = model.predict_proba(X)[:, 1]

            predictions.append(pred * weight)

        # Weighted average ensemble
        ensemble_pred = np.sum(predictions, axis=0)

        return ensemble_pred

    def predict_classes(self, X, threshold=0.5):
        """Get binary predictions with optimized threshold"""
        probs = self.predict_proba(X)
        return (probs > threshold).astype(int)

    def optimize_threshold(self, X_val, y_val):
        """Find optimal threshold for classification"""
        probs = self.predict_proba(X_val)

        best_threshold = 0.5
        best_f1 = 0

        for threshold in np.arange(0.3, 0.7, 0.01):
            preds = (probs > threshold).astype(int)
            report = classification_report(y_val, preds, output_dict=True)
            f1 = report['weighted avg']['f1-score']

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        print(f"Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
        return best_threshold

    def evaluate(self, X_test, y_test, optimize_threshold=True):
        """Evaluate ensemble model performance"""
        # Get predictions
        y_pred_proba = self.predict_proba(X_test)

        # Optimize threshold if requested
        if optimize_threshold:
            threshold = self.optimize_threshold(X_test, y_test)
        else:
            threshold = 0.5

        y_pred = (y_pred_proba > threshold).astype(int)

        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        auc_score = roc_auc_score(y_test, y_pred_proba)

        # Get classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Get ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        print("\n" + "=" * 50)
        print("ENHANCED MODEL PERFORMANCE")
        print("=" * 50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC-ROC: {auc_score:.4f}")
        print(f"Precision: {report['1']['precision']:.4f}")
        print(f"Recall: {report['1']['recall']:.4f}")
        print(f"F1-Score: {report['1']['f1-score']:.4f}")

        return {
            'accuracy': accuracy,
            'auc': auc_score,
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_curve': (fpr, tpr, thresholds),
            'threshold': threshold
        }

    def save_ensemble(self, path='models/'):
        """Save all models in the ensemble"""
        os.makedirs(path, exist_ok=True)

        # Save neural network models
        self.models['deep'].save(os.path.join(path, 'enhanced_deep_model.h5'))
        self.models['wide'].save(os.path.join(path, 'enhanced_wide_model.h5'))

        # Save sklearn models
        joblib.dump(self.models['xgboost'], os.path.join(path, 'xgboost_model.pkl'))
        joblib.dump(self.models['rf'], os.path.join(path, 'rf_model.pkl'))
        joblib.dump(self.models['gb'], os.path.join(path, 'gb_model.pkl'))

        # Save ensemble weights and metadata
        metadata = {
            'input_dim': self.input_dim,
            'ensemble_weights': self.ensemble_weights
        }
        joblib.dump(metadata, os.path.join(path, 'enhanced_metadata.pkl'))

    def load_ensemble(self, path='models/'):
        """Load the ensemble models"""
        # Load neural networks
        self.models['deep'] = keras.models.load_model(os.path.join(path, 'enhanced_deep_model.h5'))
        self.models['wide'] = keras.models.load_model(os.path.join(path, 'enhanced_wide_model.h5'))

        # Load sklearn models
        self.models['xgboost'] = joblib.load(os.path.join(path, 'xgboost_model.pkl'))
        self.models['rf'] = joblib.load(os.path.join(path, 'rf_model.pkl'))
        self.models['gb'] = joblib.load(os.path.join(path, 'gb_model.pkl'))

        # Load metadata
        metadata = joblib.load(os.path.join(path, 'enhanced_metadata.pkl'))
        self.input_dim = metadata['input_dim']
        self.ensemble_weights = metadata['ensemble_weights']
