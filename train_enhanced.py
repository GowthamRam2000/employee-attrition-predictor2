import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from src.utils.data_processor import HRDataProcessor

print("=" * 60)
print("ENHANCED EMPLOYEE ATTRITION MODEL TRAINING")
print("=" * 60)

# Load data
print("\n📁 Loading data...")
processor = HRDataProcessor()
df = processor.load_data('WA_Fn-UseC_-HR-Employee-Attrition.csv')
print(f"✅ Data loaded! Shape: {df.shape}")

# Enhanced Feature Engineering
print("\n🔧 Performing advanced feature engineering...")

# Create new numerical features
df['YearsPerCompany'] = df['TotalWorkingYears'] / (df['NumCompaniesWorked'] + 1)
df['YearsWithoutPromotion'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']
df['AverageWorkingYears'] = df['TotalWorkingYears'] / df['Age']
df['IncomePerYear'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)
df['SatisfactionScore'] = (df['JobSatisfaction'] + df['EnvironmentSatisfaction'] +
                           df['RelationshipSatisfaction'] + df['WorkLifeBalance']) / 4
df['InvolvementScore'] = df['JobInvolvement'] * df['PerformanceRating']

# Create age groups and encode them as numbers
age_bins = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], labels=False)
df['AgeGroup'] = age_bins

# Create income categories and encode them as numbers
income_bins = pd.qcut(df['MonthlyIncome'], q=5, labels=False)
df['IncomeCategory'] = income_bins

# Overtime and distance interaction
df['OvertimeDistance'] = (df['OverTime'].map({'Yes': 1, 'No': 0}) *
                          df['DistanceFromHome'])

# Career progression ratio
df['CareerProgressionRatio'] = df['JobLevel'] / (df['YearsAtCompany'] + 1)

# Loyalty index
df['LoyaltyIndex'] = df['YearsAtCompany'] / df['TotalWorkingYears'].replace(0, 1)

# Handle any infinite or NaN values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.median(numeric_only=True))

print(f"✅ Feature engineering complete! New shape: {df.shape}")

# Preprocess data
print("\n🔧 Preprocessing data...")
X, y, feature_names = processor.preprocess_data(df, is_training=True)

# Add polynomial features for top features
print("🔧 Creating polynomial features...")
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)

# Select top features for polynomial expansion (to avoid too many features)
top_feature_indices = list(range(min(10, X.shape[1])))  # Top 10 features or all if less than 10
X_top = X[:, top_feature_indices]
X_poly = poly.fit_transform(X_top)

# Combine original and polynomial features
X_enhanced = np.hstack([X, X_poly])
print(f"✅ Enhanced features created! Shape: {X_enhanced.shape}")

# Split data
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_enhanced, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  - Training samples: {len(X_train)}")
print(f"  - Testing samples: {len(X_test)}")

# Check if XGBoost is available
try:
    import xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    print("\n⚠️ XGBoost not installed. Installing now...")
    os.system("pip install xgboost")
    try:
        import xgboost
        XGBOOST_AVAILABLE = True
    except:
        XGBOOST_AVAILABLE = False
        print("❌ Could not install XGBoost. Will use alternative models.")

# Import and train enhanced model
if XGBOOST_AVAILABLE:
    # Copy the enhanced model to src/models if it doesn't exist
    enhanced_model_path = 'src/models/enhanced_attrition_model.py'
    if not os.path.exists(enhanced_model_path):
        print("\n📝 Creating enhanced model file...")
        os.makedirs('src/models', exist_ok=True)
        if os.path.exists('enhanced_attrition_model.py'):
            os.system(f"cp enhanced_attrition_model.py {enhanced_model_path}")

    try:
        from src.models.enhanced_attrition_model import EnhancedAttritionPredictor

        print("\n🚀 Training enhanced ensemble model...")
        print("This will train 5 different models and combine them:")
        print("  1. Enhanced Deep Neural Network")
        print("  2. Wide Neural Network")
        print("  3. XGBoost")
        print("  4. Random Forest")
        print("  5. Gradient Boosting")

        model = EnhancedAttritionPredictor()
        history = model.train_ensemble(
            X_train, y_train,
            epochs=150,
            batch_size=32
        )

        print("\n✅ Ensemble training completed!")

        # Evaluate model
        print("\n📈 Evaluating enhanced model...")
        metrics = model.evaluate(X_test, y_test, optimize_threshold=True)

        # Save model
        print("\n💾 Saving enhanced model...")
        os.makedirs('models', exist_ok=True)
        model.save_ensemble('models/')

        # Also save the enhanced processor
        processor.save_preprocessors('models/', prefix='enh_')

        # Save polynomial transformer
        import joblib
        joblib.dump(poly, 'models/poly_transformer.pkl')
        joblib.dump(top_feature_indices, 'models/top_features.pkl')

        # Save evaluation metrics for dashboard usage
        joblib.dump(metrics, 'models/enhanced_training_metrics.pkl')

    except ImportError:
        print("⚠️ Enhanced model not found, falling back to standard model...")
        XGBOOST_AVAILABLE = False

if not XGBOOST_AVAILABLE:
    # Fallback to regular model with better hyperparameters
    print("\n⚠️ Using fallback enhanced neural network...")
    from src.models.attrition_model import AttritionPredictor

    model = AttritionPredictor()

    # Build a better model
    from tensorflow import keras
    from tensorflow.keras import layers

    enhanced_nn = keras.Sequential([
        layers.Input(shape=(X_enhanced.shape[1],)),
        layers.Dense(256, activation='swish'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='gelu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    enhanced_nn.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC()]
    )

    model.model = enhanced_nn

    # Train with more epochs
    history = model.train(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        use_smote=True
    )

    metrics = model.evaluate(X_test, y_test)
    os.makedirs('models', exist_ok=True)
    model.save_model('models/')
    processor.save_preprocessors('models/')

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print("\nYour enhanced model is ready with improved performance!")
print("Run 'streamlit run app.py' to use the enhanced model.")
