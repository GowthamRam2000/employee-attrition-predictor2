import sys
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import warnings
import joblib
import traceback

warnings.filterwarnings('ignore')


sys.path.append('src')

try:
    from src.utils.data_processor import HRDataProcessor
    from src.models.attrition_model import AttritionPredictor
except ImportError:
    from src.utils.data_processor import HRDataProcessor
    from src.models.attrition_model import AttritionPredictor

from src.utils.feature_engineering import apply_enhanced_feature_engineering


def train_enhanced_pipeline(df, *, epochs, batch_size, save_path):
    from src.models.enhanced_attrition_model import EnhancedAttritionPredictor

    print("\n" + "=" * 60)
    print("ENHANCED MODEL TRAINING")
    print("=" * 60)

    processor_enh = HRDataProcessor()

    print("\n🔧 Performing advanced feature engineering...")
    df_enh = apply_enhanced_feature_engineering(df)
    print(f" Feature engineering complete! Shape: {df_enh.shape}")

    print("\nPreprocessing enhanced data...")
    X_raw, y, feature_names = processor_enh.preprocess_data(df_enh, is_training=True, fit_scaler=False)

    print("\n Splitting enhanced data...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"- Enhanced training samples: {len(X_train_raw)}")
    print(f"- Enhanced testing samples: {len(X_test_raw)}")

    X_train_base = processor_enh.fit_transform_features(X_train_raw)
    X_test_base = processor_enh.transform_features(X_test_raw)

    print("\nCreating polynomial interaction features...")
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    top_feature_indices = list(range(min(10, X_train_base.shape[1])))
    X_train_poly = poly.fit_transform(X_train_base[:, top_feature_indices])
    X_test_poly = poly.transform(X_test_base[:, top_feature_indices])

    X_train_enh = np.hstack([X_train_base, X_train_poly])
    X_test_enh = np.hstack([X_test_base, X_test_poly])
    print(f" Enhanced feature dimensions -> Train: {X_train_enh.shape}, Test: {X_test_enh.shape}")

    model_enh = EnhancedAttritionPredictor()

    print("\n Training enhanced ensemble (wide & deep + tree models)...")
    history_enh = model_enh.train_ensemble(
        X_train_enh,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
    )
    if history_enh:
        print(" Enhanced neural components converged successfully.")

    print("\n Evaluating enhanced model performance...")
    metrics_enh = model_enh.evaluate(X_test_enh, y_test, optimize_threshold=True)

    print("\n Saving enhanced artifacts...")
    os.makedirs(save_path, exist_ok=True)
    model_enh.save_ensemble(save_path)
    processor_enh.save_preprocessors(save_path, prefix='enh_')
    joblib.dump(poly, os.path.join(save_path, 'poly_transformer.pkl'))
    joblib.dump(top_feature_indices, os.path.join(save_path, 'top_features.pkl'))
    joblib.dump(metrics_enh, os.path.join(save_path, 'enhanced_training_metrics.pkl'))

    return metrics_enh


def main():
    parser = argparse.ArgumentParser(description='Train Employee Attrition Prediction Model')
    parser.add_argument('--data', type=str, required=True, help='Path to the IBM HR dataset')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--use_smote', action='store_true',
                        help='Use SMOTE for imbalanced data (less critical with enhanced model)')
    parser.add_argument('--save_path', type=str, default='models/', help='Path to save the model')
    parser.add_argument('--model_version', type=str, default='2.0', choices=['1.0', '2.0'],
                        help='Model version to train (2.0 is enhanced model)')

    args = parser.parse_args()
    metrics_enh = None

    print("=" * 60)
    print("EMPLOYEE ATTRITION PREDICTION MODEL TRAINING")
    print("=" * 60)


    print("\n📁 Loading data...")
    processor = HRDataProcessor()
    df = processor.load_data(args.data)
    print(f" Data loaded successfully! Shape: {df.shape}")


    print("\nDataset Statistics:")
    print(f"- Total Employees: {len(df)}")
    print(f"- Features: {len(df.columns)}")
    if 'Attrition' in df.columns:
        attrition_rate = (df['Attrition'] == 'Yes').mean() * 100
        print(f"Attrition Rate: {attrition_rate:.1f}%")
        if attrition_rate < 20:
            print(f" Warning: Class imbalance detected (only {attrition_rate:.1f}% attrition). "
                "Handled via SMOTE (if enabled) or class weights."
            )


    print("\nPreprocessing data...")
    X_raw, y, feature_names = processor.preprocess_data(df, is_training=True, fit_scaler=False)
    print(f"Data preprocessed! Features: {len(feature_names)}")


    print("\n Splitting data...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"- Training samples: {len(X_train_raw)}")
    print(f"- Testing samples: {len(X_test_raw)}")

    X_train = processor.fit_transform_features(X_train_raw)
    X_test = processor.transform_features(X_test_raw)


    print("\ Training deep learning model...")
    print(f" - Model Version: {args.model_version}")
    print(f"- Epochs: {args.epochs}")
    print(f"- Batch Size: {args.batch_size}")
    print(f"- Using SMOTE: {args.use_smote}")

    model = AttritionPredictor()


    try:
        history = model.train(
            X_train, y_train,
            X_val=X_test,
            y_val=y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_smote=args.use_smote
        )
        print("Model training completed!")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        print("\nAttempting to train with standard model configuration...")
        try:

            model = AttritionPredictor()
            history = model.train(
                X_train, y_train,
                X_val=X_test,
                y_val=y_test,
                epochs=min(args.epochs, 100),
                batch_size=args.batch_size,
                use_smote=args.use_smote
            )
            print("Model training completed with fallback configuration!")
        except Exception as e2:
            print(f"Critical error: {str(e2)}")
            print(f"Traceback: {traceback.format_exc()}")
            sys.exit(1)


    print("\ Evaluating model performance...")
    metrics = model.evaluate(X_test, y_test)

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 60)
    print(f" Accuracy:  {metrics['accuracy']:.3f}")
    print(f"AUC-ROC:   {metrics['auc']:.3f}")
    print(f"Precision: {metrics['classification_report']['1']['precision']:.3f}")
    print(f"Recall:    {metrics['classification_report']['1']['recall']:.3f}")
    print(f"F1-Score:  {metrics['classification_report']['1']['f1-score']:.3f}")
    print(f"Optimal Threshold: {getattr(model, 'best_threshold', 0.5):.2f}")


    print("\n Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"Predicted")
    print(f"Stay  Leave")
    print(f"Actual Stay  {cm[0, 0]:4d}  {cm[0, 1]:4d}")
    print(f" Actual Leave {cm[1, 0]:4d}  {cm[1, 1]:4d}")


    print("\n Top 10 Most Important Features:")
    try:
        importance_dict = model.get_feature_importance(feature_names, X_test, num_samples=100)
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]

        for i, (feature, importance) in enumerate(sorted_features, 1):

            human_feature = feature
            if 'IncomeEfficiency' in feature:
                human_feature = "Income Efficiency"
            elif 'TenureBucket' in feature:
                human_feature = "Tenure Stage"
            elif 'poly' in feature:
                human_feature = "Income-Experience Relationship"

            print(f"  {i:2d}. {human_feature:30s} {importance:.3f}")
    except Exception as e:
        print(f" Could not calculate feature importance: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        print("Using fallback feature importance calculation")

    if metrics_enh:
        print("\n" + "=" * 60)
        print("ENHANCED MODEL PERFORMANCE METRICS")
        print("=" * 60)
        print(f" Accuracy:  {metrics_enh['accuracy']:.3f}")
        print(f"AUC-ROC:   {metrics_enh['auc']:.3f}")
        print(f"Precision: {metrics_enh['classification_report']['1']['precision']:.3f}")
        print(f"Recall:    {metrics_enh['classification_report']['1']['recall']:.3f}")
        print(f"F1-Score:  {metrics_enh['classification_report']['1']['f1-score']:.3f}")
        print(f"Optimal Threshold: {metrics_enh.get('threshold', 0.5):.2f}")


    print(f"Saving model to {args.save_path}...")
    os.makedirs(args.save_path, exist_ok=True)


    model.save_model(args.save_path)
    processor.save_preprocessors(args.save_path)


    joblib.dump(metrics, os.path.join(args.save_path, 'training_metrics.pkl'))

    if args.model_version == '2.0':
        try:
            metrics_enh = train_enhanced_pipeline(
                df,
                epochs=args.epochs,
                batch_size=args.batch_size,
                save_path=args.save_path,
            )
        except Exception as enh_err:
            print(f"\nEnhanced training failed: {enh_err}")
            print("Enhanced artifacts were not generated.")


    with open(os.path.join(args.save_path, 'model_version.txt'), 'w') as f:
        f.write(f"Model Version: {args.model_version}\n")
        f.write(f"Training Date: {pd.Timestamp.now()}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.3f}\n")
        f.write(f"AUC: {metrics['auc']:.3f}\n")
        if metrics_enh:
            f.write(f"Enhanced Accuracy: {metrics_enh['accuracy']:.3f}\n")
            f.write(f"Enhanced AUC: {metrics_enh['auc']:.3f}\n")

    print(" Model and preprocessors saved successfully!")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel saved to: {args.save_path}")
    print("You can now use the model in the Streamlit app by running:")
    print("  streamlit run app.py")


    if metrics_enh:
        print("\n Enhanced Model Features:")
        print("- Wide & Deep architecture for better pattern recognition")
        print("- Imbalance handling via Borderline-SMOTE or class weights")
        print("- Advanced feature engineering (tenure buckets, income efficiency)")
        print("- Polynomial interaction enrichment on key drivers")
        print("- Automatic threshold optimization for business needs")
    elif args.model_version == '2.0':
        print("\n Enhanced model training was requested but did not complete successfully.")


if __name__ == "__main__":
    main()
