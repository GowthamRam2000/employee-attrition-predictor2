import numpy as np
import pandas as pd


def apply_enhanced_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional engineered features used by the enhanced attrition model."""
    df = df.copy()

    df['YearsPerCompany'] = df['TotalWorkingYears'] / (df['NumCompaniesWorked'] + 1)
    df['YearsWithoutPromotion'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']
    df['AverageWorkingYears'] = df['TotalWorkingYears'] / df['Age']
    df['IncomePerYear'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)
    df['SatisfactionScore'] = (
        df['JobSatisfaction']
        + df['EnvironmentSatisfaction']
        + df['RelationshipSatisfaction']
        + df['WorkLifeBalance']
    ) / 4
    df['InvolvementScore'] = df['JobInvolvement'] * df['PerformanceRating']

    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], labels=False)
    df['IncomeCategory'] = pd.qcut(df['MonthlyIncome'], q=5, labels=False, duplicates='drop')

    df['OvertimeDistance'] = df['OverTime'].map({'Yes': 1, 'No': 0}) * df['DistanceFromHome']
    df['CareerProgressionRatio'] = df['JobLevel'] / (df['YearsAtCompany'] + 1)
    df['LoyaltyIndex'] = df['YearsAtCompany'] / df['TotalWorkingYears'].replace(0, 1)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))
    return df
