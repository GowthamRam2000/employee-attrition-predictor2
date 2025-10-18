import sys
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import warnings
import joblib
import traceback

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

try:
    from src.utils.data_processor import HRDataProcessor
    from src.models.attrition_model import AttritionPredictor
except ImportError:
    from src.utils.data_processor import HRDataProcessor
    from src.models.attrition_model import AttritionPredictor


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

    print("=" * 60)
    print("EMPLOYEE ATTRITION PREDICTION MODEL TRAINING")
    print("=" * 60)

    # Load data
    print("\n📁 Loading data...")
    processor = HRDataProcessor()
    df = processor.load_data(args.data)
    print(f"✅ Data loaded successfully! Shape: {df.shape}")

    # Display basic statistics
    print("\n📊 Dataset Statistics:")
    print(f"  - Total Employees: {len(df)}")
    print(f"  - Features: {len(df.columns)}")
    if 'Attrition' in df.columns:
        attrition_rate = (df['Attrition'] == 'Yes').mean() * 100
        print(f"  - Attrition Rate: {attrition_rate:.1f}%")
        if attrition_rate < 20:
            print(
                f"  ⚠️ Warning: Class imbalance detected (only {attrition_rate:.1f}% attrition). "
                "Handled via SMOTE (if enabled) or class weights."
            )

    # Preprocess data
    print("\n🔧 Preprocessing data...")
    X, y, feature_names = processor.preprocess_data(df, is_training=True)
    print(f"✅ Data preprocessed! Features: {len(feature_names)}")

    # Split data
    print("\n📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Testing samples: {len(X_test)}")

    # Create and train model
    print("\n🚀 Training deep learning model...")
    print(f"  - Model Version: {args.model_version}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Using SMOTE: {args.use_smote}")

    model = AttritionPredictor()

    # Train with appropriate configuration
    try:
        history = model.train(
            X_train, y_train,
            X_val=X_test,
            y_val=y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_smote=args.use_smote
        )
        print("✅ Model training completed!")
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        print("\nAttempting to train with standard model configuration...")
        try:
            # Fallback to simpler configuration
            model = AttritionPredictor()
            history = model.train(
                X_train, y_train,
                X_val=X_test,
                y_val=y_test,
                epochs=min(args.epochs, 100),
                batch_size=args.batch_size,
                use_smote=args.use_smote
            )
            print("✅ Model training completed with fallback configuration!")
        except Exception as e2:
            print(f"❌ Critical error: {str(e2)}")
            print(f"Traceback: {traceback.format_exc()}")
            sys.exit(1)

    # Evaluate model
    print("\n📈 Evaluating model performance...")
    metrics = model.evaluate(X_test, y_test)

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 60)
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  AUC-ROC:   {metrics['auc']:.3f}")
    print(f"  Precision: {metrics['classification_report']['1']['precision']:.3f}")
    print(f"  Recall:    {metrics['classification_report']['1']['recall']:.3f}")
    print(f"  F1-Score:  {metrics['classification_report']['1']['f1-score']:.3f}")
    print(f"  Optimal Threshold: {getattr(model, 'best_threshold', 0.5):.2f}")

    # Display confusion matrix
    print("\n📊 Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"              Predicted")
    print(f"              Stay  Leave")
    print(f"  Actual Stay  {cm[0, 0]:4d}  {cm[0, 1]:4d}")
    print(f"  Actual Leave {cm[1, 0]:4d}  {cm[1, 1]:4d}")

    # Get feature importance
    print("\n🎯 Top 10 Most Important Features:")
    try:
        importance_dict = model.get_feature_importance(feature_names, X_test, num_samples=100)
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]

        for i, (feature, importance) in enumerate(sorted_features, 1):
            # Format feature names for readability
            human_feature = feature
            if 'IncomeEfficiency' in feature:
                human_feature = "Income Efficiency"
            elif 'TenureBucket' in feature:
                human_feature = "Tenure Stage"
            elif 'poly' in feature:
                human_feature = "Income-Experience Relationship"

            print(f"  {i:2d}. {human_feature:30s} {importance:.3f}")
    except Exception as e:
        print(f"  ⚠️ Could not calculate feature importance: {str(e)}")
        print(f"  Traceback: {traceback.format_exc()}")
        print("  Using fallback feature importance calculation...")

    # Save model and preprocessors
    print(f"\n💾 Saving model to {args.save_path}...")
    os.makedirs(args.save_path, exist_ok=True)

    # Save the model
    model.save_model(args.save_path)
    processor.save_preprocessors(args.save_path)

    # Save training metrics for future reference
    joblib.dump(metrics, os.path.join(args.save_path, 'training_metrics.pkl'))

    # Save model version information
    with open(os.path.join(args.save_path, 'model_version.txt'), 'w') as f:
        f.write(f"Model Version: {args.model_version}\n")
        f.write(f"Training Date: {pd.Timestamp.now()}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.3f}\n")
        f.write(f"AUC: {metrics['auc']:.3f}\n")

    print("✅ Model and preprocessors saved successfully!")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel saved to: {args.save_path}")
    print("You can now use the model in the Streamlit app by running:")
    print("  streamlit run app.py")

    # Additional information for enhanced model
    if args.model_version == '2.0':
        print("\n🌟 Enhanced Model Features:")
        print("  - Wide & Deep architecture for better pattern recognition")
        print("  - Imbalance handling via SMOTE or class weights")
        print("  - Advanced feature engineering (tenure buckets, income efficiency)")
        print("  - Permutation-based feature importance")
        print("  - Automatic threshold optimization for business needs")


if __name__ == "__main__":
    main()
