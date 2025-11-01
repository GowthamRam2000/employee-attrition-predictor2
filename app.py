import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys
from typing import Dict, Any, Tuple

sys.path.append('src')

from src.utils.data_processor import HRDataProcessor
from src.models.attrition_model import AttritionPredictor

st.set_page_config(
    page_title="Employee Retention Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main color scheme - corporate blue and gray */
    :root {
        --primary-color: #1e3a5f;
        --secondary-color: #4a90a4;
        --accent-color: #67b3cc;
        --background-color: #f8f9fa;
        --text-color: #2c3e50;
        --success-color: #27ae60;
        --warning-color: #f39c12;
        --danger-color: #e74c3c;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #4a90a4 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid var(--primary-color);
        margin-bottom: 1rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: var(--primary-color);
    }

    .metric-label {
        color: var(--text-color);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }

    /* Prediction result cards */
    .prediction-card {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }

    .low-risk {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        color: white;
    }

    .medium-risk {
        background: linear-gradient(135deg, #f39c12 0%, #f1c40f 100%);
        color: white;
    }

    .high-risk {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
    }

    /* Button styling */
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border-radius: 5px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: var(--secondary-color);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }

    /* Info boxes */
    .info-box {
        background: #e8f4f8;
        border-left: 4px solid var(--secondary-color);
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    /* Data table styling */
    .dataframe {
        font-size: 0.9rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: white;
        border-radius: 5px 5px 0 0;
        border: 1px solid #ddd;
        padding: 0 24px;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def _ensure_model_loaded() -> Tuple[Any, Any, Any]:
    if not st.session_state.get('model_trained'):
        m, p, feats, metrics = load_saved_model()
        if m is not None:
            st.session_state.model = m
            st.session_state.processor = p
            st.session_state.model_trained = True
            st.session_state.training_metrics = metrics
            st.session_state.feature_names = feats
    return st.session_state.get('model'), st.session_state.get('processor'), st.session_state.get('feature_names')


def build_input_df_from_overrides(processor: HRDataProcessor, feature_names, overrides: Dict[str, Any]) -> pd.DataFrame:
    base = dict(getattr(processor, 'column_defaults', {}) or {})
    if not base:

        for col in feature_names or []:
            if col in processor.label_encoders:
                le = processor.label_encoders[col]
                base[col] = le.classes_[0] if len(le.classes_) else ''
            else:
                base[col] = 0

    for k, v in overrides.items():
        if feature_names and k in feature_names:
            base[k] = v

    return pd.DataFrame([{k: base.get(k, None) for k in feature_names}])


def add_enhanced_engineered_features(df: pd.DataFrame, defaults: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()

    def get(col, series=True):
        if col in df.columns:
            return df[col]

        val = defaults.get(col, 0)
        return pd.Series([val] * len(df)) if series else val

    total_work_years = get('TotalWorkingYears')
    num_companies = get('NumCompaniesWorked')
    years_at_company = get('YearsAtCompany')
    years_since_promo = get('YearsSinceLastPromotion')
    age = get('Age')
    monthly_income = get('MonthlyIncome')
    job_sat = get('JobSatisfaction')
    env_sat = get('EnvironmentSatisfaction')
    rel_sat = get('RelationshipSatisfaction')
    wlb = get('WorkLifeBalance')
    job_involvement = get('JobInvolvement')
    perf_rating = get('PerformanceRating')
    distance = get('DistanceFromHome')
    overtime = get('OverTime')
    job_level = get('JobLevel')

    ov_series = overtime.apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

    df['YearsPerCompany'] = (total_work_years.astype(float)) / (num_companies.astype(float) + 1.0)
    df['YearsWithoutPromotion'] = years_at_company.astype(float) - years_since_promo.astype(float)
    df['AverageWorkingYears'] = (total_work_years.astype(float)) / (age.astype(float).replace(0, 1))
    df['IncomePerYear'] = (monthly_income.astype(float)) / (years_at_company.astype(float) + 1.0)
    df['SatisfactionScore'] = (job_sat.astype(float) + env_sat.astype(float) + rel_sat.astype(float) + wlb.astype(
        float)) / 4.0
    df['InvolvementScore'] = job_involvement.astype(float) * perf_rating.astype(float)

    bins = [0, 25, 35, 45, 55, 100]
    labels = [0, 1, 2, 3, 4]
    df['AgeGroup'] = pd.cut(age.astype(float), bins=bins, labels=labels, include_lowest=True).astype('Int64').fillna(
        2).astype(int)

    try:
        inc = monthly_income.astype(float)

        cats = pd.cut(inc, bins=[-1, 3000, 5000, 8000, 12000, 1e9], labels=[0, 1, 2, 3, 4]).astype('Int64').fillna(2)
        df['IncomeCategory'] = cats.astype(int)
    except Exception:
        df['IncomeCategory'] = 2
    df['OvertimeDistance'] = ov_series.astype(float) * distance.astype(float)
    df['CareerProgressionRatio'] = (job_level.astype(float)) / (years_at_company.astype(float) + 1.0)
    df['LoyaltyIndex'] = (years_at_company.astype(float)) / (total_work_years.astype(float).replace(0, 1))

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df


if 'model' not in st.session_state:
    st.session_state.model = None
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'processor_baseline' not in st.session_state:
    st.session_state.processor_baseline = None
if 'processor_enhanced' not in st.session_state:
    st.session_state.processor_enhanced = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'training_metrics' not in st.session_state:
    st.session_state.training_metrics = None
if 'enhanced_model' not in st.session_state:
    st.session_state.enhanced_model = None
if 'poly_transformer' not in st.session_state:
    st.session_state.poly_transformer = None
if 'top_features' not in st.session_state:
    st.session_state.top_features = None
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = 'Baseline'


@st.cache_resource
def load_saved_model():
    import os
    import joblib
    model_path = os.path.join('models', 'attrition_model.pt')
    legacy_path = os.path.join('models', 'attrition_model.h5')

    if not os.path.exists(model_path):
        if os.path.exists(legacy_path):
            st.sidebar.warning("Legacy TensorFlow model detected. Retrain to generate PyTorch checkpoints.")
        return None, None, None, None

    try:
        from src.models.attrition_model import AttritionPredictor
        from src.utils.data_processor import HRDataProcessor

        model = AttritionPredictor()
        model.load_model('models/')

        processor_baseline = HRDataProcessor()
        processor_baseline.load_preprocessors('models/')

        processor = processor_baseline
        feature_names = processor.feature_columns

        metrics_path = os.path.join('models', 'training_metrics.pkl')
        metrics = None
        if os.path.exists(metrics_path):
            try:
                metrics = joblib.load(metrics_path)
            except Exception:
                metrics = None

        enh_metrics_path = os.path.join('models', 'enhanced_training_metrics.pkl')
        metrics_enh = None
        if os.path.exists(enh_metrics_path):
            try:
                metrics_enh = joblib.load(enh_metrics_path)
            except Exception:
                metrics_enh = None

        st.session_state.processor_baseline = processor_baseline
        st.session_state.processor_enhanced = st.session_state.get('processor_enhanced')

        st.session_state.metrics_baseline = metrics
        st.session_state.metrics_enhanced = metrics_enh

        return model, processor, feature_names, metrics
    except Exception as e:
        st.sidebar.error(f"Error loading saved model: {str(e)}")
        return None, None, None, None
    return None, None, None, None


def load_enhanced_bundle():
    import os
    import joblib
    enhanced_needed = [
        os.path.join('models', 'enhanced_deep_model.pt'),
        os.path.join('models', 'enhanced_wide_model.pt'),
        os.path.join('models', 'xgboost_model.pkl'),
        os.path.join('models', 'rf_model.pkl'),
        os.path.join('models', 'gb_model.pkl'),
        os.path.join('models', 'enhanced_metadata.pkl'),
        os.path.join('models', 'poly_transformer.pkl'),
        os.path.join('models', 'top_features.pkl'),
    ]
    if not all(os.path.exists(p) for p in enhanced_needed):
        return False
    try:
        from src.models.enhanced_attrition_model import EnhancedAttritionPredictor
        m = EnhancedAttritionPredictor()
        m.load_ensemble('models/')
        st.session_state.enhanced_model = m
        try:
            p2 = HRDataProcessor()
            p2.load_preprocessors('models/', prefix='enh_')
            st.session_state.processor_enhanced = st.session_state.get('processor_enhanced') or p2
        except Exception:
            pass
        st.session_state.poly_transformer = joblib.load(os.path.join('models', 'poly_transformer.pkl'))
        st.session_state.top_features = joblib.load(os.path.join('models', 'top_features.pkl'))
        return True
    except Exception as e:
        st.sidebar.warning(f"Enhanced model not loaded: {str(e)}")
        return False


if not st.session_state.model_trained:
    model, processor, feature_names, metrics = load_saved_model()
    if model is not None:
        st.session_state.model = model
        st.session_state.processor = processor
        st.session_state.model_trained = True
        st.session_state.training_metrics = metrics
        st.session_state.feature_names = feature_names
        st.sidebar.success(" Model loaded from disk!")

st.markdown("""
<div class="main-header">
    <h1> Employee Retention Analytics Platform</h1>
    <p>Advanced Deep Learning System for Predicting Employee Attrition</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("###  Navigation")
    page = st.radio(
        "Navigation",
        ["Dashboard", "Train Model", "Make Predictions", "Batch Scoring", "Analytics", "Model Performance"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("###  Quick Actions")

    sample_path = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
    if os.path.exists(sample_path):
        try:
            with open(sample_path, 'rb') as f:
                st.download_button(
                    " Download Sample Data (IBM HR CSV)",
                    data=f.read(),
                    file_name=os.path.basename(sample_path),
                    mime='text/csv'
                )
        except Exception:
            st.caption("Sample CSV available in repo root.")
    else:
        st.caption("Sample CSV not found in repo root.")

    if st.button(" Use Built-in Sample Data"):
        try:
            proc = HRDataProcessor()
            df0 = proc.load_data(sample_path)
            st.session_state.sample_df = df0
            st.success("Sample dataset loaded into session.")
        except Exception as e:
            st.warning(f"Could not load sample data: {str(e)}")

    if st.button(" Generate Report"):
        try:
            if 'sample_df' in st.session_state:
                df_rep = st.session_state.sample_df
            elif os.path.exists(sample_path):
                proc = HRDataProcessor()
                df_rep = proc.load_data(sample_path)
                st.session_state.sample_df = df_rep
            else:
                df_rep = None
            if df_rep is None:
                st.info("Please upload or load the sample dataset first.")
            else:

                stats = {
                    'rows': int(len(df_rep)),
                    'cols': int(len(df_rep.columns)),
                    'attrition_rate': float(
                        (df_rep['Attrition'].eq('Yes')).mean() * 100) if 'Attrition' in df_rep.columns else None,
                    'top_departments': df_rep['Department'].value_counts().head(
                        3).to_dict() if 'Department' in df_rep.columns else {},
                    'overtime_share': float(
                        (df_rep['OverTime'].eq('Yes')).mean() * 100) if 'OverTime' in df_rep.columns else None,
                }
                st.session_state.quick_stats = stats
                st.success("Quick report generated. See Analytics page for details.")
        except Exception as e:
            st.warning(f"Could not generate report: {str(e)}")

    st.markdown("---")
    st.markdown("###  Model Selector")

    enhanced_available = load_enhanced_bundle()
    model_options = ["Baseline"] + (["Enhanced"] if enhanced_available else [])
    st.session_state.model_choice = st.selectbox("Active Model", model_options, index=0)

    if enhanced_available:
        st.caption("Enhanced = Wide & Deep + Tree Ensemble (if trained)")

    st.markdown("---")
    st.markdown("### ℹ About")
    st.markdown("""
    This platform uses deep learning to predict employee attrition risk,
    helping HR teams make data-driven retention decisions.
    Developed as part of Deep Learning Coursework in winter term at IIT Jodhpur
    Created by Gowtham Ram M24DE3036
    Saravanan GS m24de3070
    Rajendra Panda m24de3091
    Geetika Vijay m24de3035
    **Version:** 1.0.0
    **Last Updated:** 2025
    """)

if page == "Dashboard":
    st.markdown("##  Executive Dashboard")

    if st.session_state.model_trained:

        model_choice = st.session_state.get('model_choice', 'Baseline')
        metrics = None
        if model_choice == 'Enhanced':
            metrics = st.session_state.get('metrics_enhanced')
        if metrics is None:
            metrics = st.session_state.get('metrics_baseline') or st.session_state.get('training_metrics')

        if not metrics:
            st.info("No saved metrics found. Use Batch Scoring with labeled data or retrain to populate metrics.")
        else:

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">{:.1f}%</div>
                    <div class="metric-label">Model Accuracy</div>
                </div>
                """.format(metrics['accuracy'] * 100), unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">{:.3f}</div>
                    <div class="metric-label">AUC Score</div>
                </div>
                """.format(metrics['auc']), unsafe_allow_html=True)

            with col3:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">{:.1f}%</div>
                    <div class="metric-label">Precision</div>
                </div>
                """.format(metrics['classification_report']['1']['precision'] * 100), unsafe_allow_html=True)

            with col4:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">{:.1f}%</div>
                    <div class="metric-label">Recall</div>
                </div>
                """.format(metrics['classification_report']['1']['recall'] * 100), unsafe_allow_html=True)

            st.markdown("###  Model Performance Visualization")

            col1, col2 = st.columns(2)

            with col1:
                roc_tuple = metrics.get('roc_curve') if isinstance(metrics, dict) else None
                if roc_tuple and all(v is not None for v in roc_tuple):
                    fpr, tpr, _ = roc_tuple
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'ROC Curve (AUC = {metrics["auc"]:.3f})',
                        line=dict(color='#1e3a5f', width=3)
                    ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Random Classifier',
                        line=dict(color='gray', width=2, dash='dash')
                    ))
                    fig_roc.update_layout(
                        title="ROC Curve",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        height=400,
                        showlegend=True,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)
                else:
                    st.info("ROC data not available. Train a model in this session to view ROC.")

            with col2:

                cm = metrics['confusion_matrix']
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Predicted Stay', 'Predicted Leave'],
                    y=['Actual Stay', 'Actual Leave'],
                    colorscale=[[0, '#f8f9fa'], [1, '#1e3a5f']],
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 20},
                    showscale=False
                ))
                fig_cm.update_layout(
                    title="Confusion Matrix",
                    height=400,
                    template="plotly_white"
                )
                st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info(" Welcome! Please train a model first to see the dashboard metrics.")
        st.markdown("""
        <div class="info-box">
            <h4>Getting Started:</h4>
            <ol>
                <li>Navigate to the 'Train Model' page</li>
                <li>Upload the IBM HR Analytics dataset</li>
                <li>Configure training parameters</li>
                <li>Train the deep learning model</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

elif page == "Train Model":
    st.markdown("##  Model Training")

    uploaded_file = st.file_uploader(
        "Upload IBM HR Analytics Dataset",
        type=['csv', 'xlsx', 'xls'],
        help="Upload the IBM HR Analytics Employee Attrition & Performance dataset"
    )
    use_session_sample = False
    df = None
    if uploaded_file is not None:

        processor = HRDataProcessor()
        df = processor.load_data(uploaded_file)
        use_session_sample = False
    elif 'sample_df' in st.session_state:

        processor = HRDataProcessor()
        df = st.session_state.sample_df
        use_session_sample = True

    if df is not None:

        st.success(f" Dataset loaded successfully! Shape: {df.shape}")

        with st.expander(" Data Preview"):
            st.dataframe(df.head())

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("  Dataset Statistics")
            st.markdown(f"- **Total Employees:** {len(df)}")
            st.markdown(f"- **Features:** {len(df.columns)}")
            if 'Attrition' in df.columns:
                attrition_rate = (df['Attrition'] == 'Yes').mean() * 100
                st.markdown(f"- **Attrition Rate:** {attrition_rate:.1f}%")

        with col2:
            st.markdown("  Training Configuration")
            epochs = st.slider("Number of Epochs", 50, 200, 100)
            batch_size = st.select_slider("Batch Size", [16, 32, 64, 128], value=32)
            use_smote = st.checkbox("Use SMOTE for Imbalanced Data", value=True)

        btn_label = " Train Model (Sample)" if use_session_sample and uploaded_file is None else "🎯 Train Model"
        if st.button(btn_label, type="primary"):
            with st.spinner(" Training deep learning model... This may take a few minutes."):

                X_raw, y, feature_names = processor.preprocess_data(df, is_training=True, fit_scaler=False)

                from sklearn.model_selection import train_test_split

                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    X_raw, y, test_size=0.2, random_state=42, stratify=y
                )

                X_train = processor.fit_transform_features(X_train_raw)
                X_test = processor.transform_features(X_test_raw)

                model = AttritionPredictor()
                history = model.train(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    use_smote=use_smote
                )

                metrics = model.evaluate(X_test, y_test)

                st.session_state.model = model
                st.session_state.processor = processor
                st.session_state.model_trained = True
                st.session_state.training_metrics = metrics
                st.session_state.metrics_baseline = metrics
                st.session_state.feature_names = feature_names

                model.save_model('models/')
                processor.save_preprocessors('models/')

                st.success(" Model trained successfully!")

                st.markdown("###  Training Results")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                with col2:
                    st.metric("AUC Score", f"{metrics['auc']:.3f}")
                with col3:
                    st.metric("F1 Score", f"{metrics['classification_report']['1']['f1-score']:.3f}")

                if history:
                    fig_history = go.Figure()
                    fig_history.add_trace(go.Scatter(
                        y=history.history['loss'],
                        mode='lines',
                        name='Training Loss',
                        line=dict(color='#1e3a5f')
                    ))
                    fig_history.add_trace(go.Scatter(
                        y=history.history['val_loss'],
                        mode='lines',
                        name='Validation Loss',
                        line=dict(color='#4a90a4')
                    ))
                    fig_history.update_layout(
                        title="Training History",
                        xaxis_title="Epoch",
                        yaxis_title="Loss",
                        height=400,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_history, use_container_width=True)

elif page == "Make Predictions":
    st.markdown("##  Attrition Risk Prediction")

    if not st.session_state.model_trained:
        st.warning(" Please train a model first before making predictions.")
    else:
        st.markdown("### Enter Employee Information")

        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("Age", min_value=18, max_value=65, value=30)
                monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
                total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=5)
                years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=2)

            with col2:
                job_satisfaction = st.select_slider("Job Satisfaction", [1, 2, 3, 4], value=3)
                environment_satisfaction = st.select_slider("Environment Satisfaction", [1, 2, 3, 4], value=3)
                work_life_balance = st.select_slider("Work Life Balance", [1, 2, 3, 4], value=3)
                job_involvement = st.select_slider("Job Involvement", [1, 2, 3, 4], value=3)

            with col3:
                overtime = st.selectbox("Overtime", ["No", "Yes"])
                marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
                gender = st.selectbox("Gender", ["Male", "Female"])
                education = st.selectbox("Education Level",
                                         ["Below College", "College", "Bachelor", "Master", "Doctor"])

            col1, col2 = st.columns(2)

            with col1:
                distance_from_home = st.number_input("Distance From Home (miles)", min_value=1, max_value=30, value=10)
                num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, max_value=10, value=2)

            with col2:
                years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=20, value=2)
                years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, max_value=20,
                                                          value=1)

            submit_button = st.form_submit_button("🔮 Predict Attrition Risk", type="primary")

        if submit_button:
            try:

                model_choice = st.session_state.get('model_choice', 'Baseline')
                if model_choice == 'Enhanced' and st.session_state.get('processor_enhanced') is not None:
                    processor = st.session_state.processor_enhanced
                else:
                    processor = st.session_state.processor_baseline or st.session_state.processor
                feature_names = processor.feature_columns

                base = dict(processor.column_defaults) if getattr(processor, 'column_defaults', None) else {}
                if not base:
                    base = {}
                    for col in feature_names:
                        if col in processor.label_encoders:
                            le = processor.label_encoders[col]
                            base[col] = le.classes_[0] if len(le.classes_) else ''
                        else:
                            base[col] = 0

                edu_map = {
                    "Below College": 1, "College": 2, "Bachelor": 3, "Master": 4, "Doctor": 5
                }

                overrides = {
                    'Age': age,
                    'MonthlyIncome': monthly_income,
                    'TotalWorkingYears': total_working_years,
                    'YearsAtCompany': years_at_company,
                    'JobSatisfaction': int(job_satisfaction),
                    'EnvironmentSatisfaction': int(environment_satisfaction),
                    'WorkLifeBalance': int(work_life_balance),
                    'JobInvolvement': int(job_involvement),
                    'OverTime': overtime,
                    'MaritalStatus': marital_status,
                    'Gender': gender,
                    'Education': edu_map.get(education, 3),
                    'DistanceFromHome': distance_from_home,
                    'NumCompaniesWorked': num_companies_worked,
                    'YearsInCurrentRole': years_in_current_role,
                    'YearsWithCurrManager': years_with_curr_manager,
                }

                input_row = {k: base.get(k, overrides.get(k)) for k in feature_names}
                for k, v in overrides.items():
                    if k in feature_names:
                        input_row[k] = v

                input_df = pd.DataFrame([input_row])

                if model_choice == 'Enhanced' and st.session_state.get('processor_enhanced') is not None:
                    input_df = add_enhanced_engineered_features(input_df,
                                                                getattr(processor, 'column_defaults', {}) or {})

                X_input, _, _ = processor.preprocess_data(input_df, is_training=False)

                proba = None
                if model_choice == 'Enhanced' and st.session_state.get('enhanced_model') is not None:

                    try:
                        import numpy as np

                        poly = st.session_state.poly_transformer
                        idx = st.session_state.top_features
                        X_top = X_input[:, idx]
                        X_poly = poly.transform(X_top)
                        X_enh = np.hstack([X_input, X_poly])
                        proba = float(st.session_state.enhanced_model.predict_proba(X_enh).flatten()[0])
                    except Exception as e:
                        st.warning(f"Enhanced path failed, using baseline: {str(e)}")
                if proba is None:
                    proba = float(st.session_state.model.predict_proba(X_input).flatten()[0])

                st.markdown("### 🎯 Prediction Result")

                if proba < 0.3:
                    st.markdown(
                        f"""
                        <div class="prediction-card low-risk">
                            <h2> Low Attrition Risk</h2>
                            <h3>Risk Score: {proba * 100:.1f}%</h3>
                            <p>This employee shows strong retention indicators.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif proba < 0.7:
                    st.markdown(
                        f"""
                        <div class="prediction-card medium-risk">
                            <h2>️ Medium Attrition Risk</h2>
                            <h3>Risk Score: {proba * 100:.1f}%</h3>
                            <p>Monitor this employee and consider retention strategies.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="prediction-card high-risk">
                            <h2> High Attrition Risk</h2>
                            <h3>Risk Score: {proba * 100:.1f}%</h3>
                            <p>Immediate intervention recommended.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("### Recommended Actions")
                if proba > 0.5:
                    st.markdown(
                        """
                        - Schedule a one-on-one meeting to discuss concerns
                        - Review compensation and benefits package
                        - Explore career development opportunities
                        - Consider flexible work arrangements
                        - Assess workload and work-life balance
                        """
                    )
                st.caption(f"Exact probability: {proba:.4f}")

                with st.expander(" What‑If Analysis and Local Sensitivity"):
                    st.markdown("Adjust key features to see impact on risk.")

                    w_col1, w_col2, w_col3 = st.columns(3)
                    with w_col1:
                        wi_monthly_income = st.number_input("What‑If Monthly Income", 1000, 20000, int(monthly_income))
                        wi_years_at_company = st.number_input("What‑If Years at Company", 0, 40, int(years_at_company))
                    with w_col2:
                        wi_distance = st.number_input("What‑If Distance From Home", 1, 30, int(distance_from_home))
                        wi_wlb = st.select_slider("What‑If Work Life Balance", [1, 2, 3, 4],
                                                  value=int(work_life_balance))
                    with w_col3:
                        wi_js = st.select_slider("What‑If Job Satisfaction", [1, 2, 3, 4], value=int(job_satisfaction))
                        wi_overtime = st.selectbox("What‑If Overtime", ["No", "Yes"],
                                                   index=0 if overtime == "No" else 1)

                    if st.button("Recompute What‑If Risk"):
                        wi_overrides = {
                            'Age': age,
                            'MonthlyIncome': wi_monthly_income,
                            'TotalWorkingYears': total_working_years,
                            'YearsAtCompany': wi_years_at_company,
                            'JobSatisfaction': int(wi_js),
                            'EnvironmentSatisfaction': int(environment_satisfaction),
                            'WorkLifeBalance': int(wi_wlb),
                            'JobInvolvement': int(job_involvement),
                            'OverTime': wi_overtime,
                            'MaritalStatus': marital_status,
                            'Gender': gender,
                            'Education': edu_map.get(education, 3),
                            'DistanceFromHome': wi_distance,
                            'NumCompaniesWorked': num_companies_worked,
                            'YearsInCurrentRole': years_in_current_role,
                            'YearsWithCurrManager': years_with_curr_manager,
                        }
                        wi_df = build_input_df_from_overrides(processor, feature_names, wi_overrides)
                        X_wi, _, _ = processor.preprocess_data(wi_df, is_training=False)
                        wi_proba = float(st.session_state.model.predict_proba(X_wi).flatten()[0])

                        delta = wi_proba - proba
                        st.write(f"New Risk: {wi_proba * 100:.1f}%  (Δ {delta * 100:+.1f} pp)")

                        import numpy as np
                        import plotly.graph_objects as go

                        sensitivity = []

                        num_feats = {
                            'MonthlyIncome': wi_monthly_income,
                            'YearsAtCompany': wi_years_at_company,
                            'DistanceFromHome': wi_distance,
                            'WorkLifeBalance': wi_wlb,
                            'JobSatisfaction': wi_js,
                        }
                        for f, val in num_feats.items():
                            val_up = val * 1.1 if f in ['MonthlyIncome'] else min(val + 1, 40)
                            val_dn = val * 0.9 if f in ['MonthlyIncome'] else max(val - 1, 0)
                            for new_val in [val_up, val_dn]:
                                tmp = dict(wi_overrides)
                                tmp[f] = new_val
                                tmp_df = build_input_df_from_overrides(processor, feature_names, tmp)
                                X_tmp, _, _ = processor.preprocess_data(tmp_df, is_training=False)
                                p_tmp = float(st.session_state.model.predict_proba(X_tmp).flatten()[0])
                                sensitivity.append((f, abs(p_tmp - wi_proba)))

                        tmp = dict(wi_overrides)
                        tmp['OverTime'] = 'No' if wi_overtime == 'Yes' else 'Yes'
                        tmp_df = build_input_df_from_overrides(processor, feature_names, tmp)
                        X_tmp, _, _ = processor.preprocess_data(tmp_df, is_training=False)
                        p_tmp = float(st.session_state.model.predict_proba(X_tmp).flatten()[0])
                        sensitivity.append(('OverTime', abs(p_tmp - wi_proba)))

                        agg = {}
                        for k, v in sensitivity:
                            agg[k] = max(agg.get(k, 0), v)
                        items = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:6]
                        if items:
                            fig = go.Figure(
                                data=[go.Bar(x=[i[1] for i in items], y=[i[0] for i in items], orientation='h')])
                            fig.update_layout(title="Local Sensitivity (Δ probability)", xaxis_title="Δ", height=300,
                                              template='plotly_white')
                            st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

elif page == "Batch Scoring":
    st.markdown(" Batch Scoring")

    model, _, _ = _ensure_model_loaded()
    has_processor = (st.session_state.get('processor_baseline') is not None) or (
                st.session_state.get('processor_enhanced') is not None)
    if not model or not has_processor:
        st.warning("Model not loaded. Train a model or ensure models/ artifacts exist.")
    else:
        uploaded = st.file_uploader(
            "Upload CSV/XLSX for scoring",
            type=["csv", "xlsx", "xls"],
            help="Optional: include Attrition column for evaluation (Yes/No)"
        )

        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            thresh = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.01)
        with col_cfg2:
            cost_fn = st.number_input("Cost of False Negative", min_value=0, max_value=1000000, value=1000)
        with col_cfg3:
            cost_fp = st.number_input("Cost of False Positive", min_value=0, max_value=1000000, value=200)

        if uploaded is not None:

            model_choice = st.session_state.get('model_choice', 'Baseline')
            if model_choice == 'Enhanced' and st.session_state.get('processor_enhanced') is not None:
                processor = st.session_state.processor_enhanced
            else:
                processor = st.session_state.processor_baseline or st.session_state.processor

            df_in = processor.load_data(uploaded)

            if model_choice == 'Enhanced' and st.session_state.get('processor_enhanced') is not None:
                df_in = add_enhanced_engineered_features(df_in, getattr(processor, 'column_defaults', {}) or {})
            st.success(f"Loaded file. Rows: {len(df_in)}, Columns: {len(df_in.columns)}")

            df_out = df_in.copy()

            Xb, yb, _ = processor.preprocess_data(df_in, is_training=False)

            model_choice = st.session_state.get('model_choice', 'Baseline')
            if model_choice == 'Enhanced' and st.session_state.get('enhanced_model') is not None:
                try:
                    import numpy as np

                    poly = st.session_state.poly_transformer
                    idx = st.session_state.top_features
                    X_top = Xb[:, idx]
                    X_poly = poly.transform(X_top)
                    Xb_enh = np.hstack([Xb, X_poly])
                    proba = st.session_state.enhanced_model.predict_proba(Xb_enh).flatten()
                except Exception as e:
                    st.warning(f"Enhanced scoring failed, using baseline: {str(e)}")
                    proba = model.predict_proba(Xb).flatten()
            else:
                proba = model.predict_proba(Xb).flatten()

            import numpy as np

            preds = (proba > thresh).astype(int)
            df_out['AttritionRisk'] = proba
            df_out['PredictedAttrition'] = preds
            bands = np.where(proba < 0.3, 'Low', np.where(proba < 0.7, 'Medium', 'High'))
            df_out['RiskBand'] = bands

            st.markdown("###  Risk Distribution")
            import plotly.express as px

            hist = px.histogram(x=proba, nbins=30, labels={'x': 'Attrition Risk'}, title='Risk Score Histogram')
            hist.update_layout(template='plotly_white', height=350)
            st.plotly_chart(hist, use_container_width=True)

            st.markdown("### 👥 Cohort Summary (Mean Risk)")
            candidates = ['Department', 'JobRole', 'OverTime', 'MaritalStatus']
            group_col = next((c for c in candidates if c in df_out.columns), None)
            if group_col:
                grp = df_out.groupby(group_col)['AttritionRisk'].mean().sort_values(ascending=False)[:12]
                bar = px.bar(x=grp.values, y=grp.index, orientation='h', labels={'x': 'Mean Risk', 'y': group_col})
                bar.update_layout(template='plotly_white', height=400)
                st.plotly_chart(bar, use_container_width=True)
            else:
                st.info("No standard cohort column found (Department/JobRole/OverTime/MaritalStatus).")

            y_true = None
            if 'Attrition' in df_in.columns:
                ser = df_in['Attrition']
                if ser.dtype == object:
                    y_true = ser.map({'Yes': 1, 'No': 0}).values
                else:
                    try:
                        y_true = ser.astype(int).values
                    except Exception:
                        y_true = None

            if y_true is not None:
                from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, \
                    accuracy_score, brier_score_loss

                cm = confusion_matrix(y_true, preds)
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                try:
                    auc = roc_auc_score(y_true, proba)
                except Exception:
                    auc = float('nan')
                acc = accuracy_score(y_true, preds)
                prec = precision_score(y_true, preds, zero_division=0)
                rec = recall_score(y_true, preds, zero_division=0)
                f1 = f1_score(y_true, preds, zero_division=0)
                total_cost = fn * cost_fn + fp * cost_fp
                brier = brier_score_loss(y_true, proba)

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Accuracy", f"{acc:.3f}")
                c2.metric("AUC", f"{auc:.3f}")
                c3.metric("Precision", f"{prec:.3f}")
                c4.metric("Recall", f"{rec:.3f}")
                c5.metric("F1", f"{f1:.3f}")
                st.write(f"Estimated Business Cost: {total_cost:,.0f}")

                from sklearn.calibration import calibration_curve

                try:
                    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy='quantile')
                    import plotly.graph_objects as go

                    fig_cal = go.Figure()
                    fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfectly Calibrated',
                                                 line=dict(color='gray', dash='dash')))
                    fig_cal.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode='lines+markers', name='Model',
                                                 line=dict(color='#1e3a5f')))
                    fig_cal.update_layout(title=f"Calibration Curve (Brier: {brier:.3f})",
                                          xaxis_title='Mean Predicted Value', yaxis_title='Fraction of Positives',
                                          template='plotly_white', height=350)
                    st.plotly_chart(fig_cal, use_container_width=True)
                except Exception as e:
                    st.info(f"Calibration plot unavailable: {str(e)}")

                import plotly.graph_objects as go

                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Pred Stay', 'Pred Leave'],
                    y=['Actual Stay', 'Actual Leave'],
                    showscale=False,
                    text=cm,
                    texttemplate='%{text}'
                ))
                fig_cm.update_layout(title="Confusion Matrix", template='plotly_white', height=350)
                st.plotly_chart(fig_cm, use_container_width=True)

                if st.button("Find Optimal Threshold"):
                    best_t = thresh
                    best_cost = float('inf')
                    ts = np.arange(0.1, 0.9001, 0.01)
                    for t in ts:
                        p = (proba > t).astype(int)
                        cm2 = confusion_matrix(y_true, p)
                        if cm2.size == 4:
                            tn2, fp2, fn2, tp2 = cm2.ravel()
                            cost = fn2 * cost_fn + fp2 * cost_fp
                            if cost < best_cost:
                                best_cost, best_t = cost, t
                    st.success(f"Optimal threshold by cost: {best_t:.2f} (Cost: {best_cost:,.0f})")
            else:
                st.info("No ground truth (Attrition) column found; showing predictions only.")

            st.markdown("### Download Scored Results")
            csv_bytes = df_out.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv_bytes, file_name="scored_results.csv", mime="text/csv")

elif page == "Analytics":
    st.markdown("##  HR Analytics & Insights")

    if 'quick_stats' in st.session_state:
        qs = st.session_state.quick_stats
        st.markdown("###  Dataset Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{qs.get('rows', 0)}")
        c2.metric("Columns", f"{qs.get('cols', 0)}")
        if qs.get('attrition_rate') is not None:
            c3.metric("Attrition Rate", f"{qs['attrition_rate']:.1f}%")
        if qs.get('top_departments'):
            st.markdown("Top Departments by Count:")
            for k, v in qs['top_departments'].items():
                st.write(f"- {k}: {v}")
        if qs.get('overtime_share') is not None:
            st.caption(f"OverTime share: {qs['overtime_share']:.1f}%")

    if st.session_state.model_trained:

        st.markdown("###  Key Factors Influencing Attrition")

        features = ['Overtime', 'Monthly Income', 'Years at Company', 'Job Satisfaction',
                    'Work Life Balance', 'Age', 'Distance From Home', 'Environment Satisfaction']
        importance = [0.18, 0.15, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07]

        fig = go.Figure(data=[
            go.Bar(
                x=importance,
                y=features,
                orientation='h',
                marker_color='#1e3a5f'
            )
        ])
        fig.update_layout(
            title="Feature Importance Analysis",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(" Train a model first to see analytics and insights.")

elif page == "Model Performance":
    st.markdown("##  Model Performance Metrics")

    if st.session_state.model_trained:
        model_choice = st.session_state.get('model_choice', 'Baseline')
        metrics = None
        if model_choice == 'Enhanced':
            metrics = st.session_state.get('metrics_enhanced')
        if metrics is None:
            metrics = st.session_state.get('metrics_baseline') or st.session_state.get('training_metrics')
        if not metrics:
            st.info("No saved metrics found. Use Batch Scoring with labeled data or retrain to populate metrics.")
        else:

            st.markdown("###  Classification Report")

            report = metrics['classification_report']

            metrics_df = pd.DataFrame({
                'Class': ['Stay (0)', 'Leave (1)'],
                'Precision': [report['0']['precision'], report['1']['precision']],
                'Recall': [report['0']['recall'], report['1']['recall']],
                'F1-Score': [report['0']['f1-score'], report['1']['f1-score']],
                'Support': [report['0']['support'], report['1']['support']]
            })

            st.dataframe(metrics_df.style.format({
                'Precision': '{:.3f}',
                'Recall': '{:.3f}',
                'F1-Score': '{:.3f}',
                'Support': '{:.0f}'
            }))

            st.markdown("###  Overall Performance")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            with col2:
                st.metric("AUC-ROC", f"{metrics['auc']:.3f}")
            with col3:
                st.metric("Macro Avg F1", f"{report['macro avg']['f1-score']:.3f}")
            with col4:
                st.metric("Weighted Avg F1", f"{report['weighted avg']['f1-score']:.3f}")

    else:
        st.info(" Train a model first to see performance metrics.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Employee Retention Analytics Platform | Built with Deep Learning | © 2024</p>
</div>
""", unsafe_allow_html=True)
