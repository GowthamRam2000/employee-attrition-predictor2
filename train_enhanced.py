import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import warnings

warnings.filterwarnings('ignore')


sys.path.append('src')

from src.utils.data_processor import HRDataProcessor

print("=" * 60)
print("ENHANCED EMPLOYEE ATTRITION MODEL TRAINING")
print("=" * 60)


print("\nLoading data...")
processor = HRDataProcessor()
df = processor.load_data('WA_Fn-UseC_-HR-Employee-Attrition.csv')
print(f"Data loaded! Shape: {df.shape}")


print("\n🔧 Performing advanced feature engineering...")


df['YearsPerCompany'] = df['TotalWorkingYears'] / (df['NumCompaniesWorked'] + 1)
df['YearsWithoutPromotion'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']
df['AverageWorkingYears'] = df['TotalWorkingYears'] / df['Age']
df['IncomePerYear'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)
df['SatisfactionScore'] = (df['JobSatisfaction'] + df['EnvironmentSatisfaction'] +
                           df['RelationshipSatisfaction'] + df['WorkLifeBalance']) / 4
df['InvolvementScore'] = df['JobInvolvement'] * df['PerformanceRating']


age_bins = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], labels=False)
df['AgeGroup'] = age_bins


income_bins = pd.qcut(df['MonthlyIncome'], q=5, labels=False)
df['IncomeCategory'] = income_bins


df['OvertimeDistance'] = (df['OverTime'].map({'Yes': 1, 'No': 0}) *
                          df['DistanceFromHome'])


df['CareerProgressionRatio'] = df['JobLevel'] / (df['YearsAtCompany'] + 1)


df['LoyaltyIndex'] = df['YearsAtCompany'] / df['TotalWorkingYears'].replace(0, 1)


df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.median(numeric_only=True))

print(f" Feature engineering complete! New shape: {df.shape}")


print("\n Preprocessing data...")
X, y, feature_names = processor.preprocess_data(df, is_training=True)


print("Creating polynomial features...")
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)


top_feature_indices = list(range(min(10, X.shape[1])))
X_top = X[:, top_feature_indices]
X_poly = poly.fit_transform(X_top)


X_enhanced = np.hstack([X, X_poly])
print(f"Enhanced features created! Shape: {X_enhanced.shape}")


print("\n Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_enhanced, y, test_size=0.2, random_state=42, stratify=y
)
print(f"- Training samples: {len(X_train)}")
print(f"- Testing samples: {len(X_test)}")


try:
    import xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    print("\n XGBoost not installed. Installing now...")
    os.system("pip install xgboost")
    try:
        import xgboost
        XGBOOST_AVAILABLE = True
    except:
        XGBOOST_AVAILABLE = False
        print(" Could not install XGBoost. Will use alternative models.")


if XGBOOST_AVAILABLE:

    enhanced_model_path = 'src/models/enhanced_attrition_model.py'
    if not os.path.exists(enhanced_model_path):
        print("\n Creating enhanced model file...")
        os.makedirs('src/models', exist_ok=True)
        if os.path.exists('enhanced_attrition_model.py'):
            os.system(f"cp enhanced_attrition_model.py {enhanced_model_path}")

    try:
        from src.models.enhanced_attrition_model import EnhancedAttritionPredictor

        print("\n Training enhanced ensemble model...")
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

        print("\n Ensemble training completed!")


        print("\n Evaluating enhanced model...")
        metrics = model.evaluate(X_test, y_test, optimize_threshold=True)


        print("\n Saving enhanced model...")
        os.makedirs('models', exist_ok=True)
        model.save_ensemble('models/')


        processor.save_preprocessors('models/', prefix='enh_')


        import joblib
        joblib.dump(poly, 'models/poly_transformer.pkl')
        joblib.dump(top_feature_indices, 'models/top_features.pkl')


        joblib.dump(metrics, 'models/enhanced_training_metrics.pkl')

    except ImportError:
        print(" Enhanced model not found, falling back to standard model...")
        XGBOOST_AVAILABLE = False

if not XGBOOST_AVAILABLE:
    print("\n Enhanced ensemble training requires XGBoost. Please install it and retry.")
    sys.exit(1)

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print("\nYour enhanced model is ready with improved performance!")
print("Run 'streamlit run app.py' to use the enhanced model.")
