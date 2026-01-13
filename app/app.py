# app/app.py
"""
Streamlit Application for Credit Risk Assessment Engine.
AI-Powered Lending Decisions with Full Explainability
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

from src.explain import (
    load_model_and_metadata,
    create_shap_explainer,
    generate_reason_for_decision
)

# =============================================================================
# PAGE CONFIGURATION - Sidebar collapsed by default
# =============================================================================
st.set_page_config(
    page_title="Credit Risk Assessment Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar by default
)

# =============================================================================
# DARK THEME CSS - FIXED FOR VISIBILITY
# =============================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Dark Theme */
    .stApp {
        background: #0a0a0f;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e2e8f0;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding: 1.5rem 2rem;
        max-width: 1300px;
    }
    
    /* Hero Header */
    .hero-box {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4c1d95 100%);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(139, 92, 246, 0.3);
        box-shadow: 0 0 60px rgba(139, 92, 246, 0.2);
    }
    
    .hero-icon {
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0.5rem 0;
        text-shadow: 0 2px 20px rgba(139, 92, 246, 0.5);
    }
    
    .hero-subtitle {
        font-size: 1rem;
        color: #c4b5fd;
        font-weight: 400;
        margin: 0;
    }
    
    /* Section Headers */
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #8b5cf6;
        display: inline-block;
    }
    
    .section-icon {
        margin-right: 8px;
    }
    
    /* Input Cards */
    .input-section {
        background: #111118;
        border: 1px solid #2d2d3a;
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    
    .input-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #a78bfa;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Override Streamlit input styles for dark theme */
    .stNumberInput > label,
    .stSelectbox > label,
    .stSlider > label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }
    
    .stNumberInput input {
        background: #1a1a24 !important;
        border: 1px solid #3f3f50 !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
    }
    
    .stNumberInput input:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2) !important;
    }
    
    div[data-baseweb="select"] {
        background: #1a1a24 !important;
    }
    
    div[data-baseweb="select"] > div {
        background: #1a1a24 !important;
        border: 1px solid #3f3f50 !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: #3f3f50 !important;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #8b5cf6, #a78bfa) !important;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 30px rgba(139, 92, 246, 0.5) !important;
    }
    
    /* Risk Cards */
    .result-card-high {
        background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(239, 68, 68, 0.2);
    }
    
    .result-card-low {
        background: linear-gradient(135deg, #052e16 0%, #14532d 100%);
        border: 2px solid #22c55e;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(34, 197, 94, 0.2);
    }
    
    .risk-status {
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    
    .result-card-high .risk-status { color: #fca5a5; }
    .result-card-low .risk-status { color: #86efac; }
    
    .risk-percent {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0.25rem 0;
        line-height: 1;
    }
    
    .result-card-high .risk-percent { color: #fecaca; }
    .result-card-low .risk-percent { color: #bbf7d0; }
    
    .risk-label {
        font-size: 0.9rem;
        color: #a1a1aa;
        margin-top: 0.25rem;
    }
    
    .confidence-box {
        margin-top: 1.25rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    
    .confidence-label {
        font-size: 0.75rem;
        color: #a1a1aa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .confidence-value {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }
    
    .result-card-high .confidence-value { color: #fca5a5; }
    .result-card-low .confidence-value { color: #86efac; }
    
    /* Factors Card */
    .factors-card {
        background: #111118;
        border: 1px solid #2d2d3a;
        border-radius: 16px;
        padding: 1.5rem;
    }
    
    .factors-title {
        font-size: 1rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 1rem;
    }
    
    .factors-subtitle {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 1rem;
    }
    
    /* Factor Tags */
    .factor-tag {
        display: inline-block;
        padding: 8px 14px;
        border-radius: 8px;
        margin: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .factor-up {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #fca5a5;
    }
    
    .factor-down {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.4);
        color: #86efac;
    }
    
    /* SHAP Plot Section */
    .shap-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .shap-info {
        background: #1e1b4b;
        border: 1px solid #4c3d8f;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
        font-size: 0.85rem;
        color: #c4b5fd;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #111118 !important;
        border: 1px solid #2d2d3a !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #3f3f50, transparent);
        margin: 1.5rem 0;
    }
    
    /* Footer */
    .footer-box {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        border-top: 1px solid #2d2d3a;
    }
    
    .footer-text {
        color: #71717a;
        font-size: 0.85rem;
    }
    
    .footer-highlight {
        color: #a78bfa;
        font-weight: 600;
    }
    
    /* Model Info in Expander */
    .model-info {
        background: #111118;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 0.5rem;
    }
    
    .model-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #2d2d3a;
    }
    
    .model-row:last-child {
        border-bottom: none;
    }
    
    .model-label {
        color: #94a3b8;
        font-size: 0.85rem;
    }
    
    .model-value {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    /* Metrics Grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-top: 1rem;
    }
    
    .metric-box {
        background: #1a1a24;
        border: 1px solid #2d2d3a;
        border-radius: 10px;
        padding: 0.75rem;
        text-align: center;
    }
    
    .metric-num {
        font-size: 1.25rem;
        font-weight: 700;
        color: #a78bfa;
    }
    
    .metric-name {
        font-size: 0.7rem;
        color: #71717a;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.25rem;
    }
    
    /* Hide sidebar toggle on mobile */
    [data-testid="collapsedControl"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD MODEL
# =============================================================================
@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)."""
    model_dir = Path(__file__).parent.parent / 'model'
    try:
        model, feature_names, metadata, encoders = load_model_and_metadata(str(model_dir))
        explainer = create_shap_explainer(model)
        return model, feature_names, metadata, explainer, encoders
    except FileNotFoundError:
        return None, None, None, None, None

model, feature_names, metadata, explainer, encoders = load_model()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def encode_input_value(col_name: str, value: str, encoders: dict) -> int:
    """Encode a categorical value using the trained encoder."""
    if encoders is None or col_name not in encoders:
        return 0
    encoder = encoders[col_name]
    try:
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        return 0
    except Exception:
        return 0

def parse_emp_length(val: str) -> float:
    """Parse employment length string to numeric years."""
    if pd.isna(val) or val == 'Missing':
        return 0
    val = str(val).lower()
    if '10+' in val:
        return 10
    elif '< 1' in val or '<1' in val:
        return 0.5
    try:
        return float(''.join(filter(str.isdigit, val.split()[0])))
    except:
        return 0

def prepare_applicant_data(input_data: dict, feature_names: list, encoders: dict) -> pd.DataFrame:
    """Prepare applicant data for model prediction."""
    applicant_df = pd.DataFrame([{feat: 0 for feat in feature_names}])
    categorical_cols = ['grade', 'sub_grade', 'term', 'emp_length', 'home_ownership', 
                        'verification_status', 'purpose', 'emp_title', 'earliest_cr_line', 'loan_status']
    
    for key, value in input_data.items():
        if key in applicant_df.columns:
            if key in categorical_cols:
                applicant_df[key] = encode_input_value(key, str(value), encoders)
            else:
                applicant_df[key] = value
    return applicant_df

# =============================================================================
# HERO HEADER
# =============================================================================
st.markdown("""
<div class="hero-box">
    <div class="hero-icon">🏦</div>
    <h1 class="hero-title">Credit Risk Assessment Engine</h1>
    <p class="hero-subtitle">AI-Powered Lending Decisions with Full Explainability</p>
</div>
""", unsafe_allow_html=True)

# Check if model is loaded
if model is None:
    st.markdown("""
    <div style="background: #450a0a; border: 1px solid #dc2626; border-radius: 12px; padding: 2rem; text-align: center;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">⚠️</div>
        <h3 style="color: #fecaca; margin: 0.5rem 0;">Model Not Found</h3>
        <p style="color: #fca5a5;">Please train the model first by running:</p>
        <code style="background: #7f1d1d; padding: 0.5rem 1rem; border-radius: 6px; color: #fecaca;">python src/train.py</code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# =============================================================================
# MODEL INFO (Collapsible)
# =============================================================================
with st.expander("📊 Model Information & Performance"):
    if metadata:
        metrics = metadata.get('test_metrics', {})
        st.markdown(f"""
        <div class="model-info">
            <div class="model-row">
                <span class="model-label">Model Type</span>
                <span class="model-value" style="color: #a78bfa;">{metadata.get('model_name', 'XGBoost')}</span>
            </div>
            <div class="model-row">
                <span class="model-label">Training Date</span>
                <span class="model-value">{metadata.get('training_date', 'N/A')[:10]}</span>
            </div>
            <div class="model-row">
                <span class="model-label">Total Features</span>
                <span class="model-value">{len(feature_names)}</span>
            </div>
        </div>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-num">{metrics.get('roc_auc', 0):.2f}</div>
                <div class="metric-name">ROC-AUC</div>
            </div>
            <div class="metric-box">
                <div class="metric-num">{metrics.get('f1_score', 0):.2f}</div>
                <div class="metric-name">F1 Score</div>
            </div>
            <div class="metric-box">
                <div class="metric-num">{metrics.get('precision', 0):.2f}</div>
                <div class="metric-name">Precision</div>
            </div>
            <div class="metric-box">
                <div class="metric-num">{metrics.get('recall', 0):.2f}</div>
                <div class="metric-name">Recall</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# MAIN FORM
# =============================================================================
st.markdown('<h2 class="section-title"><span class="section-icon">📝</span>Loan Application Details</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="input-section"><div class="input-title">💰 Loan Info</div>', unsafe_allow_html=True)
    loan_amnt = st.number_input("Loan Amount ($)", min_value=1000, max_value=100000, value=15000, step=1000)
    term = st.selectbox("Term", [" 36 months", " 60 months"], index=0)
    int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 12.0, 0.5)
    installment = st.number_input("Monthly Installment ($)", min_value=50, max_value=2000, value=500)
    purpose = st.selectbox("Loan Purpose", [
        "debt_consolidation", "credit_card", "home_improvement", 
        "major_purchase", "small_business", "car", "medical", "other"
    ])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-section"><div class="input-title">👤 Applicant</div>', unsafe_allow_html=True)
    annual_inc = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=65000, step=5000)
    emp_length = st.selectbox("Employment Length", [
        "< 1 year", "1 year", "2 years", "3 years", "4 years", 
        "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"
    ], index=5)
    home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    verification_status = st.selectbox("Income Verification", ["Verified", "Source Verified", "Not Verified"])
    grade = st.selectbox("Credit Grade", ["A", "B", "C", "D", "E", "F", "G"], index=2)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="input-section"><div class="input-title">📊 Credit History</div>', unsafe_allow_html=True)
    dti = st.slider("Debt-to-Income (%)", 0.0, 50.0, 18.0, 0.5)
    revol_util = st.slider("Credit Utilization (%)", 0.0, 100.0, 45.0, 1.0)
    open_acc = st.number_input("Open Accounts", min_value=0, max_value=50, value=8)
    total_acc = st.number_input("Total Accounts", min_value=1, max_value=100, value=20)
    delinq_2yrs = st.number_input("Delinquencies (2 yrs)", min_value=0, max_value=20, value=0)
    pub_rec = st.number_input("Public Records", min_value=0, max_value=10, value=0)
    revol_bal = st.number_input("Revolving Balance ($)", min_value=0, max_value=100000, value=15000)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# PREDICTION BUTTON
# =============================================================================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    assess_button = st.button("🔍 Assess Credit Risk", use_container_width=True)

if assess_button:
    # Calculate engineered features
    emp_years = parse_emp_length(emp_length)
    home_scores = {'OWN': 1.0, 'MORTGAGE': 0.7, 'RENT': 0.3, 'OTHER': 0.2}
    stability_index = 0.6 * (min(10, emp_years) / 10) + 0.4 * home_scores.get(home_ownership, 0.2)
    
    input_data = {
        'loan_amnt': float(loan_amnt),
        'term': term,
        'int_rate': float(int_rate),
        'installment': float(installment),
        'grade': grade,
        'emp_length': emp_length,
        'home_ownership': home_ownership,
        'annual_inc': float(annual_inc),
        'verification_status': verification_status,
        'purpose': purpose,
        'dti': float(dti),
        'delinq_2yrs': float(delinq_2yrs),
        'open_acc': float(open_acc),
        'pub_rec': float(pub_rec),
        'revol_bal': float(revol_bal),
        'revol_util': float(revol_util),
        'total_acc': float(total_acc),
        'debt_to_income_ratio': float(loan_amnt) / (float(annual_inc) + 1),
        'credit_utilization': float(revol_util) / 100,
        'stability_index': stability_index
    }
    
    try:
        applicant_df = prepare_applicant_data(input_data, feature_names, encoders)
        prob_default = model.predict_proba(applicant_df)[0, 1]
        confidence = abs(0.5 - prob_default) * 200
        
        # =============================================================================
        # RESULTS
        # =============================================================================
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title"><span class="section-icon">📊</span>Assessment Results</h2>', unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if prob_default > 0.5:
                st.markdown(f"""
                <div class="result-card-high">
                    <div class="risk-status">⚠️ HIGH RISK</div>
                    <div class="risk-percent">{prob_default:.1%}</div>
                    <div class="risk-label">Probability of Default</div>
                    <div class="confidence-box">
                        <div class="confidence-label">Confidence</div>
                        <div class="confidence-value">{confidence:.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-low">
                    <div class="risk-status">✅ LOW RISK</div>
                    <div class="risk-percent">{prob_default:.1%}</div>
                    <div class="risk-label">Probability of Default</div>
                    <div class="confidence-box">
                        <div class="confidence-label">Confidence</div>
                        <div class="confidence-value">{confidence:.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with res_col2:
            # SHAP Analysis
            shap_values = explainer.shap_values(applicant_df)
            base_value = explainer.expected_value
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1]
            
            shap_vals = shap_values[0] if shap_values.ndim > 1 else shap_values
            
            feature_impacts = sorted(
                zip(feature_names, shap_vals),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
            
            st.markdown("""
            <div class="factors-card">
                <div class="factors-title">🧠 Decision Factors (SHAP Analysis)</div>
                <div class="factors-subtitle">Top 5 factors influencing this decision:</div>
            """, unsafe_allow_html=True)
            
            factors_html = '<div style="margin-top: 0.5rem;">'
            for feat, impact in feature_impacts:
                if impact > 0:
                    factors_html += f'<span class="factor-tag factor-up">↑ {feat}: +{impact:.3f}</span>'
                else:
                    factors_html += f'<span class="factor-tag factor-down">↓ {feat}: {impact:.3f}</span>'
            factors_html += '</div></div>'
            st.markdown(factors_html, unsafe_allow_html=True)
        
        # SHAP Feature Impact Chart (replacing force plot which doesn't render well)
        st.markdown('<h2 class="section-title"><span class="section-icon">📈</span>Feature Impact Analysis</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="shap-info">
            <strong>How to read:</strong> 
            🔴 <span style="color: #f87171;">Red bars</span> = increase default risk | 
            � <span style="color: #4ade80;">Green bars</span> = decrease default risk
        </div>
        """, unsafe_allow_html=True)
        
        # Create a proper bar chart showing top 10 feature impacts
        top_n = 10
        sorted_impacts = sorted(
            zip(feature_names, shap_vals),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]
        
        # Prepare data
        features_plot = [x[0] for x in sorted_impacts][::-1]  # Reverse for horizontal bar
        impacts_plot = [x[1] for x in sorted_impacts][::-1]
        colors = ['#ef4444' if x > 0 else '#22c55e' for x in impacts_plot]
        
        # Create figure with proper styling
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0a0a0f')
        ax.set_facecolor('#0a0a0f')
        
        # Create horizontal bar chart
        bars = ax.barh(features_plot, impacts_plot, color=colors, height=0.6, edgecolor='none')
        
        # Add value labels on bars
        for bar, val in zip(bars, impacts_plot):
            x_pos = bar.get_width()
            label = f'{val:+.3f}'
            ha = 'left' if val >= 0 else 'right'
            offset = 0.002 if val >= 0 else -0.002
            ax.text(x_pos + offset, bar.get_y() + bar.get_height()/2, label,
                   va='center', ha=ha, fontsize=9, color='#e2e8f0', fontweight='500')
        
        # Style the chart
        ax.axvline(x=0, color='#4b5563', linewidth=1, linestyle='-')
        ax.set_xlabel('SHAP Value (Impact on Default Probability)', fontsize=10, color='#94a3b8', labelpad=10)
        ax.set_ylabel('')
        
        # Style ticks
        ax.tick_params(axis='y', colors='#e2e8f0', labelsize=9)
        ax.tick_params(axis='x', colors='#94a3b8', labelsize=9)
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Add grid
        ax.xaxis.grid(True, linestyle='--', alpha=0.2, color='#4b5563')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Display the chart
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
        # =====================================================================
        # COMPREHENSIVE DECISION REPORT - Using Native Streamlit Components
        # =====================================================================
        with st.expander("📄 Full Decision Report (Regulatory Documentation)", expanded=False):
            
            # Get current date for the report
            from datetime import datetime
            report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
            
            # Determine risk level and recommendation
            if prob_default > 0.7:
                risk_level = "Very High Risk"
                risk_emoji = "🔴"
                recommendation = "DECLINE - The applicant presents a very high probability of defaulting on this loan. It is strongly recommended to decline this application or require substantial additional collateral and/or a co-signer."
                rec_type = "error"
            elif prob_default > 0.5:
                risk_level = "High Risk"
                risk_emoji = "🟠"
                recommendation = "DECLINE or CONDITIONAL APPROVAL - The applicant shows elevated default risk. Consider declining or approving only with additional requirements such as higher interest rate, reduced loan amount, collateral, or a co-signer."
                rec_type = "error"
            elif prob_default > 0.3:
                risk_level = "Moderate Risk"
                risk_emoji = "🟡"
                recommendation = "CONDITIONAL APPROVAL - The applicant presents moderate risk. Standard approval may be appropriate, but consider risk-adjusted pricing or additional verification of income and employment."
                rec_type = "warning"
            elif prob_default > 0.15:
                risk_level = "Low Risk"
                risk_emoji = "🟢"
                recommendation = "APPROVE - The applicant demonstrates a good credit profile with below-average default probability. Standard loan terms are appropriate."
                rec_type = "success"
            else:
                risk_level = "Very Low Risk"
                risk_emoji = "✅"
                recommendation = "APPROVE - The applicant presents an excellent credit profile with very low default probability. Consider offering preferred rates to secure this quality borrower."
                rec_type = "success"
            
            # ===== REPORT HEADER =====
            st.markdown("---")
            st.markdown(f"### 📋 Credit Risk Assessment Report")
            st.caption(f"Generated on {report_date}")
            st.markdown("---")
            
            # ===== EXECUTIVE SUMMARY =====
            st.markdown("#### 📊 Executive Summary")
            
            sum_col1, sum_col2, sum_col3 = st.columns(3)
            with sum_col1:
                st.metric(label="Risk Classification", value=f"{risk_emoji} {risk_level}")
            with sum_col2:
                st.metric(label="Default Probability", value=f"{prob_default:.1%}")
            with sum_col3:
                st.metric(label="Model Confidence", value=f"{confidence:.1f}%")
            
            st.markdown("")
            
            # ===== RECOMMENDATION =====
            st.markdown("#### 💡 Recommendation")
            if rec_type == "error":
                st.error(recommendation)
            elif rec_type == "warning":
                st.warning(recommendation)
            else:
                st.success(recommendation)
            
            st.markdown("")
            
            # ===== APPLICANT SUMMARY =====
            st.markdown("#### 👤 Applicant Summary")
            
            app_col1, app_col2, app_col3 = st.columns(3)
            with app_col1:
                st.markdown(f"**Loan Amount Requested:** ${input_data.get('loan_amnt', 0):,.0f}")
                st.markdown(f"**Interest Rate:** {input_data.get('int_rate', 0):.1f}%")
            with app_col2:
                st.markdown(f"**Annual Income:** ${input_data.get('annual_inc', 0):,.0f}")
                st.markdown(f"**Credit Grade:** {input_data.get('grade', 'N/A')}")
            with app_col3:
                st.markdown(f"**Debt-to-Income Ratio:** {input_data.get('dti', 0):.1f}%")
                st.markdown(f"**Credit Utilization:** {input_data.get('revol_util', 0):.1f}%")
            
            st.markdown("")
            
            # ===== DETAILED FACTOR ANALYSIS =====
            st.markdown("#### 🔍 Detailed Factor Analysis")
            st.markdown("The following factors had the most significant impact on this credit decision, ranked from most influential to least influential:")
            st.markdown("")
            
            # Feature name translations for clarity
            feature_labels = {
                'loan_amnt': 'Loan Amount Requested',
                'term': 'Loan Term (Duration)',
                'int_rate': 'Interest Rate',
                'installment': 'Monthly Payment Amount',
                'grade': 'Credit Grade',
                'sub_grade': 'Credit Sub-Grade',
                'emp_length': 'Length of Employment',
                'home_ownership': 'Home Ownership Status',
                'annual_inc': 'Annual Income',
                'verification_status': 'Income Verification Status',
                'purpose': 'Loan Purpose',
                'dti': 'Debt-to-Income Ratio',
                'delinq_2yrs': 'Past-Due Accounts (Last 2 Years)',
                'open_acc': 'Number of Open Accounts',
                'pub_rec': 'Public Records (Bankruptcies, Liens, etc.)',
                'revol_bal': 'Total Revolving Credit Balance',
                'revol_util': 'Revolving Credit Utilization',
                'total_acc': 'Total Credit Accounts (Lifetime)',
                'debt_to_income_ratio': 'Calculated Debt-to-Income Ratio',
                'credit_utilization': 'Credit Utilization Percentage',
                'stability_index': 'Financial Stability Score',
                'earliest_cr_line': 'Credit History Start Date',
                'emp_title': 'Job Title/Occupation'
            }
            
            # Prepare detailed factor explanations
            all_impacts = sorted(
                zip(feature_names, shap_vals, applicant_df.values[0]),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            for i, (feat, impact, value) in enumerate(all_impacts[:8], 1):
                label = feature_labels.get(feat, feat.replace('_', ' ').title())
                
                # Format value for display
                if isinstance(value, float):
                    if abs(value) > 1000:
                        display_value = f"${value:,.0f}"
                    elif abs(value) < 1 and value != 0:
                        display_value = f"{value:.2%}"
                    else:
                        display_value = f"{value:.2f}"
                else:
                    display_value = str(value)
                
                if impact > 0:
                    effect_text = "⚠️ **INCREASES DEFAULT RISK**"
                    explanation = "This factor contributes to a higher likelihood of loan default."
                else:
                    effect_text = "✅ **DECREASES DEFAULT RISK**"
                    explanation = "This factor contributes to a lower likelihood of loan default."
                
                st.markdown(f"""
**{i}. {label}**
- Current Value: `{display_value}`
- Effect: {effect_text}
- Impact Strength: `{abs(impact):.4f}`
- *{explanation}*
""")
            
            st.markdown("")
            
            # ===== HOW TO READ THIS REPORT =====
            st.markdown("#### 📖 How to Read This Report")
            st.info("""
**Understanding the Results:**

• **Factors that INCREASE default risk** (marked with ⚠️) are characteristics that historically correlate with higher likelihood of a borrower failing to repay their loan.

• **Factors that DECREASE default risk** (marked with ✅) are characteristics that historically correlate with borrowers successfully repaying their loans.

• **Impact Strength** indicates how much influence each factor has on the decision. Higher numbers mean the factor had more influence on the final prediction.

• **Default Probability** represents the likelihood (as a percentage) that this applicant will fail to repay the loan, based on patterns found in historical lending data.

• **Model Confidence** indicates how certain the model is about this prediction. Higher confidence means the model found clear patterns in the applicant's data.
""")
            
            st.markdown("")
            
            # ===== REGULATORY DISCLOSURE =====
            st.markdown("#### ⚖️ Regulatory Disclosure")
            st.markdown("""
**Model Information:**
This credit risk assessment was generated using an XGBoost (Extreme Gradient Boosting) machine learning model trained on historical lending data. The model uses SHAP (Shapley Additive Explanations) values to provide transparent and interpretable explanations for each decision.

**Equal Credit Opportunity Act (ECOA) Compliance:**
This model does not use protected characteristics such as race, color, religion, national origin, sex, marital status, or age as direct inputs for credit decisions.

**Adverse Action Notice:**
If this application is declined or offered less favorable terms than requested, the applicant is entitled to receive the specific reasons for the decision. The factors listed in the "Detailed Factor Analysis" section above constitute the primary reasons for this credit decision.

**Right to Appeal:**
Applicants have the right to request a review of this decision and to provide additional documentation that may affect the assessment.
""")
            
    except Exception as e:
        st.markdown(f"""
        <div style="background: #450a0a; border: 1px solid #dc2626; border-radius: 12px; padding: 1.5rem;">
            <div style="color: #fecaca; font-weight: 600; margin-bottom: 0.5rem;">⚠️ Error generating prediction</div>
            <div style="color: #fca5a5;">{str(e)}</div>
            <div style="color: #f87171; margin-top: 1rem; font-size: 0.9rem;">
                Run <code style="background: #7f1d1d; padding: 2px 6px; border-radius: 4px;">python src/train.py</code> first.
            </div>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div class="footer-box">
    <p class="footer-text">
        <strong>Credit Risk Assessment Engine v2.0</strong><br>
        Built with <span class="footer-highlight">XGBoost</span> • 
        <span class="footer-highlight">SHAP</span> • 
        <span class="footer-highlight">Streamlit</span>
    </p>
</div>
""", unsafe_allow_html=True)
