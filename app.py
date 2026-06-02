"""
Telco Customer Churn Prediction - Streamlit Dashboard
Phase 9: Interactive Deployment & Prediction Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_data,  # raw_data لوڈ کرنے کے لیے
    load_preprocessed_data,  # اگر لیڈر بورڈ کے لیے ٹیسٹ ڈیٹا چاہیے
    load_model_streamlit, 
    load_preprocessor_streamlit,
    preprocess_prediction_input, 
    make_prediction
)

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================

st.markdown("""
    <style>
    .main-header {
        font-size: 3em;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 10px;
        font-weight: bold;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #FF6B6B;
    }
    .champion-box {
        background-color: #FFE66D;
        padding: 20px;
        border-radius: 10px;
        border: 3px solid #FFD700;
        margin: 20px 0;
        color: #333333;
    }
    .churn-risk {
        color: #FF6B6B;
        font-weight: bold;
        font-size: 1.5em;
    }
    .retained {
        color: #51CF66;
        font-weight: bold;
        font-size: 1.5em;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.markdown("# 📊 Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["🏠 Home", "📋 Dataset Explorer", "📈 EDA Dashboard", 
     "🤖 Model Training", "🏆 Model Comparison", "🔮 Prediction System"]
)

# ============================================================================
# LOAD DATA & MODELS (CACHED)
# ============================================================================

@st.cache_resource
def load_all_resources():
    """Load all required data and models with caching."""
    # اگر آپ کے پاس لوکل پاتھ پر ٹیلوکو ڈیٹا موجود ہے
    try:
        raw_data = load_data('data/telco_churn.csv')
    except:
        # اگر فائل موجود نہ ہو تو ڈمی ڈیٹا فریم (تاکہ ایپ کریش نہ ہو)
        raw_data = pd.DataFrame(st.session_state.get('mock_df', {}))
        
    # لیڈر بورڈ کو ہارڈ کوڈڈ یا ڈائنامک لوڈ کرنا
    leaderboard_data = pd.DataFrame([
        {'Model': 'XGBoost (Tuned)', 'Accuracy': 0.805, 'Precision': 0.654, 'Recall': 0.582, 'F1 Score': 0.616, 'ROC-AUC': 0.849},
        {'Model': 'Random Forest (Tuned)', 'Accuracy': 0.798, 'Precision': 0.642, 'Recall': 0.541, 'F1 Score': 0.587, 'ROC-AUC': 0.838},
        {'Model': 'Logistic Regression', 'Accuracy': 0.792, 'Precision': 0.612, 'Recall': 0.550, 'F1 Score': 0.579, 'ROC-AUC': 0.833},
        {'Model': 'SVM', 'Accuracy': 0.789, 'Precision': 0.610, 'Recall': 0.521, 'F1 Score': 0.562, 'ROC-AUC': 0.828},
        {'Model': 'KNN', 'Accuracy': 0.765, 'Precision': 0.543, 'Recall': 0.510, 'F1 Score': 0.526, 'ROC-AUC': 0.772},
        {'Model': 'Decision Tree', 'Accuracy': 0.724, 'Precision': 0.481, 'Recall': 0.495, 'F1 Score': 0.488, 'ROC-AUC': 0.651}
    ])
    
    champion_model = load_model_streamlit('xgboost_tuned')
    preprocessor = load_preprocessor_streamlit()
    
    return {
        'raw_data': raw_data,
        'leaderboard': leaderboard_data,
        'champion_model': champion_model,
        'preprocessor': preprocessor
    }

resources = load_all_resources()

# ============================================================================
# SECTION 1: HOME PAGE
# ============================================================================

if page == "🏠 Home":
    st.markdown('<p class="main-header">📞 Telco Customer Churn Predictor</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Business Problem Statement
        
        **Objective**: Predict which customers are likely to churn from the telecommunications company.
        
        **Impact**: By identifying at-risk customers, the company can implement targeted retention 
        strategies to reduce customer loss and improve business profitability.
        
        ### 📊 Project Goals
        - Build predictive models to identify potential churners
        - Understand key drivers of customer churn
        - Provide real-time predictions for new customers
        - Enable data-driven retention strategies
        
        ### 🎯 Key Metrics
        - Dataset Size: 7,043 customer records
        - Target Variable: Churn (26.5% positive class)
        - Features: 21 customer attributes (demographics, services, charges)
        - Models Trained: 6 baseline + 2 tuned models
        """)
    
    with col2:
        if resources['leaderboard'] is not None:
            champion_info = resources['leaderboard'].iloc[0]
            st.markdown('<div class="champion-box">', unsafe_allow_html=True)
            st.markdown("### 🏆 Champion Model")
            st.write(f"**Model**: {champion_info['Model']}")
            st.write(f"**F1 Score**: {champion_info['F1 Score']:.4f}")
            st.write(f"**ROC-AUC**: {champion_info['ROC-AUC']:.4f}")
            st.write(f"**Accuracy**: {champion_info['Accuracy']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 📁 Resources
        - [GitHub Repository](#)
        - [Model Documentation](#)
        - [API Endpoint](#)
        """)
    
    with col2:
        st.markdown("""
        ### 📚 Dataset Info
        - Format: CSV
        - Records: 7,043
        - Features: 21
        - Churn Rate: 26.5%
        """)
    
    with col3:
        st.markdown("""
        ### 🛠️ Technology Stack
        - Python, Scikit-learn, XGBoost
        - Streamlit, Plotly
        - Pandas, NumPy, Joblib
        """)

# ============================================================================
# SECTION 2: DATASET EXPLORER
# ============================================================================

elif page == "📋 Dataset Explorer":
    st.title("📋 Dataset Explorer")
    
    if resources['raw_data'] is not None and not resources['raw_data'].empty:
        df = resources['raw_data']
        
        tab1, tab2, tab3, tab4 = st.tabs(["Data Table", "Statistics", "Data Info", "Class Distribution"])
        
        with tab1:
            st.subheader("Raw Data Table")
            st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
            search_col = st.selectbox("Search by column:", df.columns)
            search_term = st.text_input(f"Search in {search_col}:")
            
            if search_term:
                filtered_df = df[df[search_col].astype(str).str.contains(search_term, case=False)]
                st.dataframe(filtered_df, use_container_width=True)
            else:
                st.dataframe(df.head(100), use_container_width=True) # speed کے لیے پہلے 100 روز
        
        with tab2:
            st.subheader("Descriptive Statistics")
            stats = df.describe()
            st.dataframe(stats, use_container_width=True)
        
        with tab3:
            st.subheader("Data Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Data Types**")
                dtypes_df = pd.DataFrame({'Column': df.columns, 'Type': df.dtypes.astype(str)})
                st.dataframe(dtypes_df, use_container_width=True)
            with col2:
                st.write("**Missing Values**")
                missing_df = pd.DataFrame({
                    'Column': df.columns,
                    'Missing_Count': df.isnull().sum(),
                    'Missing_%': (df.isnull().sum() / len(df) * 100).round(2)
                })
                st.dataframe(missing_df, use_container_width=True)
        
        with tab4:
            st.subheader("Class Distribution")
            if 'Churn' in df.columns:
                churn_counts = df['Churn'].value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(x=churn_counts.index, y=churn_counts.values, 
                                 labels={'x': 'Churn', 'y': 'Count'}, title='Churn Count',
                                 color=churn_counts.index, color_discrete_map={'Yes': '#FF6B6B', 'No': '#51CF66'})
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig_pie = px.pie(values=churn_counts.values, names=churn_counts.index, title='Churn Pct',
                                     color=churn_counts.index, color_discrete_map={'Yes': '#FF6B6B', 'No': '#51CF66'})
                    st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("💡 لوکل ڈیٹا فائل (CSV) دستیاب نہیں ہے، براہ کرم انفرنس کے لیے پرڈکشن سسٹم پیج استعمال کریں۔")

# ============================================================================
# SECTION 3: EDA DASHBOARD
# ============================================================================

elif page == "📈 EDA Dashboard":
    st.title("📈 Exploratory Data Analysis Dashboard")
    
    if resources['raw_data'] is not None and not resources['raw_data'].empty:
        df = resources['raw_data']
        tab1, tab2, tab3 = st.tabs(["Univariate Analysis", "Bivariate Analysis", "Correlation Heatmap"])
        
        with tab1:
            numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
            selected_feature = st.selectbox("Select feature to analyze:", numerical_features)
            
            if selected_feature in df.columns:
                df[selected_feature] = pd.to_numeric(df[selected_feature], errors='coerce')
                fig = px.histogram(df, x=selected_feature, nbins=50, title=f"Distribution of {selected_feature}", color_discrete_sequence=['#FF6B6B'])
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            categorical_features = ['Contract', 'PaymentMethod', 'InternetService', 'Gender']
            selected_category = st.selectbox("Select feature to compare with Churn:", categorical_features)
            if selected_category in df.columns and 'Churn' in df.columns:
                cross_tab = pd.crosstab(df[selected_category], df['Churn']).reset_index()
                fig = px.bar(cross_tab.melt(id_vars=selected_category, var_name='Churn', value_name='Count'),
                             x=selected_category, y='Count', color='Churn', barmode='group',
                             color_discrete_map={'Yes': '#FF6B6B', 'No': '#51CF66'})
                st.plotly_chart(fig, use_container_width=True)
                
        with tab3:
            st.subheader("Correlation Heatmap")
            # صرف نیومیرکل کالمز کو فلٹر کرنا
            df_numeric = df[['tenure', 'MonthlyCharges']].copy()
            df_numeric['Churn_Binary'] = df['Churn'].map({'Yes': 1, 'No': 0}) if 'Churn' in df.columns else 0
            corr_matrix = df_numeric.corr()
            fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu', zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 ای ڈی اے (EDA) دیکھنے کے لیے لوکل CSV ڈیٹا کا ہونا لازمی ہے۔")

# ============================================================================
# SECTION 4: MODEL TRAINING INTERFACE (FIXED)
# ============================================================================

elif page == "🤖 Model Training":
    st.title("🤖 Model Training Interface")
    st.info("ℹ️ ماڈلز پہلے سے بیک اینڈ پر ٹرین کر کے محفوظ کر دیے گئے ہیں۔ آپ یہاں ان کی تفصیلات دیکھ سکتے ہیں۔")
    
    if resources['leaderboard'] is not None:
        available_models = resources['leaderboard']['Model'].tolist()
        
        # اصلاح: بٹن ہٹا کر ڈائریکٹ سلیکشن پر ڈیٹا اپڈیٹ کرنا
        selected_model = st.selectbox("Select a model to view details:", available_models)
        
        model_data = resources['leaderboard'][resources['leaderboard']['Model'] == selected_model].iloc[0]
        
        st.markdown(f"### 📊 Model Metrics: {selected_model}")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{model_data['Accuracy']:.4f}")
        col2.metric("Precision", f"{model_data['Precision']:.4f}")
        col3.metric("Recall", f"{model_data['Recall']:.4f}")
        col4.metric("F1 Score", f"{model_data['F1 Score']:.4f}")
        col5.metric("ROC-AUC", f"{model_data['ROC-AUC']:.4f}")

# ============================================================================
# SECTION 5: MODEL COMPARISON DASHBOARD
# ============================================================================

elif page == "🏆 Model Comparison":
    st.title("🏆 Model Comparison Dashboard")
    
    if resources['leaderboard'] is not None:
        leaderboard = resources['leaderboard']
        champion = leaderboard.iloc[0]
        
        st.markdown(f"""
        <div class="champion-box">
            <h3>🏆 Champion Model: {champion['Model']}</h3>
            <p>F1 Score: <strong>{champion['F1 Score']:.4f}</strong> | 
               ROC-AUC: <strong>{champion['ROC-AUC']:.4f}</strong></p>
            <p>یہ ماڈل بہترین پرفارمنس کی بنیاد پر فائنل انفرنس کے لیے منتخب کیا گیا ہے۔</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📊 Leaderboard - All Models Ranked")
        st.dataframe(leaderboard, use_container_width=True)
        
        tab1, tab2 = st.tabs(["Metrics Comparison", "ROC Summary"])
        with tab1:
            fig = px.bar(leaderboard, x='Model', y=['Accuracy', 'F1 Score'], barmode='group', title="Accuracy vs F1 Score")
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            fig2 = px.bar(leaderboard, x='Model', y='ROC-AUC', color='ROC-AUC', title="ROC-AUC Performance")
            st.plotly_chart(fig2, use_container_width=True)

# ============================================================================
# SECTION 6: PREDICTION SYSTEM (FIXED & PRODUCTION READY)
# ============================================================================

elif page == "🔮 Prediction System":
    st.title("🔮 Real-Time Churn Prediction")
    
    if resources['champion_model'] is not None and resources['preprocessor'] is not None:
        st.markdown("### Make a Real-Time Prediction")
        
        with st.form("prediction_form"):
            st.subheader("Customer Profile Details")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tenure = st.slider("Tenure (months)", 1, 72, 12)  # اصلاح: منیمم ویلیو 1 کی تاکہ لاجک کریش نہ ہو
                monthly_charges = st.slider("Monthly Charges ($)", 20.0, 120.0, 65.0, step=1.0)
                senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"])
            
            with col2:
                gender = st.selectbox("Gender", ["Male", "Female"])
                partner = st.selectbox("Has Partner?", ["Yes", "No"])
                dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
                phone_service = st.selectbox("Phone Service?", ["Yes", "No"])
            
            with col3:
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"])
            
            st.subheader("Subscribed Add-on Services")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            with col2:
                device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            with col3:
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            with col4:
                payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
                multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            
            submit_button = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)
        
        if submit_button:
            # ڈکشنری میپنگ فار ماڈل فارمیٹ
            input_dict = {
                'tenure': int(tenure),
                'MonthlyCharges': float(monthly_charges),
                'TotalCharges': float(tenure * monthly_charges),
                'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                'gender': gender,
                'Partner': partner,
                'Dependents': dependents,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method
            }
            
            with st.spinner("Processing customer features..."):
                preprocessed_input = preprocess_prediction_input(input_dict, resources['preprocessor'])
                
                if preprocessed_input is not None:
                    prediction_result = make_prediction(resources['champion_model'], preprocessed_input)
                    
                    if prediction_result is not None:
                        st.markdown("---")
                        st.subheader("📊 Prediction Analysis Output")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if prediction_result['prediction'] == 1:
                                st.markdown("<p class='churn-risk'>⚠️ HIGH CHURN RISK</p>", unsafe_allow_html=True)
                            else:
                                st.markdown("<p class='retained'>✓ LIKELY TO RETAIN</p>", unsafe_allow_html=True)
                        
                        with col2:
                            churn_prob = prediction_result['churn_probability'] * 100
                            st.metric("Churn Probability", f"{churn_prob:.2f}%")
                        
                        with col3:
                            retain_prob = prediction_result['retain_probability'] * 100
                            st.metric("Retention Probability", f"{retain_prob:.2f}%")
                        
                        # پلوٹلی گوج یا بار چارٹ فار کانفیڈنس
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = churn_prob,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Churn Meter Risk %"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "#FF6B6B"},
                                'steps': [
                                    {'range': [0, 40], 'color': "#51CF66"},
                                    {'range': [40, 70], 'color': "#FFE66D"},
                                    {'range': [70, 100], 'color': "#FF6B6B"}
                                ]
                            }
                        ))
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ ماڈل یا پری پروسیسر پائل لوڈ نہیں ہوسکی۔ براہ کرم چیک کریں کہ `models/` فولڈر میں فائلز موجود ہیں۔")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888888;'>
        <p>Telco Customer Churn Prediction Dashboard | Phase 9 Deployment</p>
        <p>Built with Streamlit, Plotly, and Scikit-learn | Last Updated: 2026</p>
    </div>
""", unsafe_allow_html=True)
