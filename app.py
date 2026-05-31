"""
Telco Customer Churn Prediction - Streamlit Dashboard
Phase 9: Interactive Deployment & Prediction Interface

This dashboard provides:
- Home page with project overview
- Dataset explorer
- EDA dashboard with interactive visualizations
- Model training interface
- Model comparison leaderboard
- Real-time prediction system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from datetime import datetime
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_raw_data_streamlit, load_leaderboard_streamlit, 
    load_model_streamlit, load_preprocessor_streamlit,
    preprocess_prediction_input, make_prediction
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
    }
    .churn-risk {
        color: #FF6B6B;
        font-weight: bold;
        font-size: 1.2em;
    }
    .retained {
        color: #51CF66;
        font-weight: bold;
        font-size: 1.2em;
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
    raw_data = load_raw_data_streamlit()
    leaderboard = load_leaderboard_streamlit()
    champion_model = load_model_streamlit('xgboost_tuned')
    preprocessor = load_preprocessor_streamlit()
    
    return {
        'raw_data': raw_data,
        'leaderboard': leaderboard,
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
    
    if resources['raw_data'] is not None:
        df = resources['raw_data']
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Data Table", "Statistics", "Data Info", "Class Distribution"])
        
        with tab1:
            st.subheader("Raw Data Table")
            st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Search/filter functionality
            search_col = st.selectbox("Search by column:", df.columns)
            search_term = st.text_input(f"Search in {search_col}:")
            
            if search_term:
                filtered_df = df[df[search_col].astype(str).str.contains(search_term, case=False)]
                st.dataframe(filtered_df, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
        
        with tab2:
            st.subheader("Descriptive Statistics")
            stats = df.describe()
            st.dataframe(stats, use_container_width=True)
            
            # Download statistics
            csv = stats.to_csv()
            st.download_button(
                label="Download Statistics",
                data=csv,
                file_name="statistics.csv",
                mime="text/csv"
            )
        
        with tab3:
            st.subheader("Data Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Data Types**")
                dtypes_df = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str)
                })
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
                churn_pct = df['Churn'].value_counts(normalize=True) * 100
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        x=churn_counts.index,
                        y=churn_counts.values,
                        labels={'x': 'Churn', 'y': 'Count'},
                        title='Churn Distribution (Count)',
                        color=churn_counts.index,
                        color_discrete_map={'Yes': '#FF6B6B', 'No': '#51CF66'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig_pie = px.pie(
                        values=churn_counts.values,
                        names=churn_counts.index,
                        title='Churn Distribution (%)',
                        color=churn_counts.index,
                        color_discrete_map={'Yes': '#FF6B6B', 'No': '#51CF66'}
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                st.write(f"\nChurn Summary:")
                st.write(f"- No Churn: {churn_counts.get('No', 0)} ({churn_pct.get('No', 0):.2f}%)")
                st.write(f"- Churn: {churn_counts.get('Yes', 0)} ({churn_pct.get('Yes', 0):.2f}%)")


# ============================================================================
# SECTION 3: EDA DASHBOARD
# ============================================================================

elif page == "📈 EDA Dashboard":
    st.title("📈 Exploratory Data Analysis Dashboard")
    
    if resources['raw_data'] is not None:
        df = resources['raw_data']
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["Univariate Analysis", "Bivariate Analysis", "Correlation Heatmap"])
        
        with tab1:
            st.subheader("Univariate Analysis - Feature Distributions")
            
            # Select numerical feature
            numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
            selected_feature = st.selectbox("Select feature to analyze:", numerical_features)
            
            if selected_feature in df.columns:
                fig = px.histogram(
                    df,
                    x=selected_feature,
                    nbins=50,
                    title=f"Distribution of {selected_feature}",
                    labels={selected_feature: selected_feature},
                    color_discrete_sequence=['#FF6B6B']
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                st.write(f"**Statistics for {selected_feature}:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{df[selected_feature].mean():.2f}")
                with col2:
                    st.metric("Median", f"{df[selected_feature].median():.2f}")
                with col3:
                    st.metric("Std Dev", f"{df[selected_feature].std():.2f}")
                with col4:
                    st.metric("Range", f"{df[selected_feature].max() - df[selected_feature].min():.2f}")
        
        with tab2:
            st.subheader("Bivariate Analysis - Churn vs Features")
            
            # Select categorical feature
            categorical_features = ['Contract', 'PaymentMethod', 'InternetService', 'Gender']
            selected_category = st.selectbox("Select feature to compare with Churn:", categorical_features)
            
            if selected_category in df.columns and 'Churn' in df.columns:
                cross_tab = pd.crosstab(df[selected_category], df['Churn'])
                
                fig = px.bar(
                    cross_tab.reset_index().melt(id_vars=selected_category, var_name='Churn', value_name='Count'),
                    x=selected_category,
                    y='Count',
                    color='Churn',
                    title=f"Churn Distribution by {selected_category}",
                    color_discrete_map={'Yes': '#FF6B6B', 'No': '#51CF66'},
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Correlation Heatmap - Numerical Features")
            
            # Select numerical columns
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Add churn as binary
            df_corr = df.copy()
            df_corr['Churn_Binary'] = (df_corr['Churn'] == 'Yes').astype(int)
            
            # Calculate correlation
            corr_matrix = df_corr[numerical_cols + ['Churn_Binary']].corr()
            
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale="RdBu",
                zmin=-1, zmax=1,
                title="Correlation Matrix - Numerical Features"
            )
            st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# SECTION 4: MODEL TRAINING INTERFACE
# ============================================================================

elif page == "🤖 Model Training":
    st.title("🤖 Model Training Interface")
    
    st.warning("ℹ️ Models are pre-trained. This section demonstrates model selection and performance.")
    
    if resources['leaderboard'] is not None:
        # Get list of available models
        available_models = resources['leaderboard']['Model'].tolist()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_model = st.selectbox("Select a model to view:", available_models)
        
        with col2:
            if st.button("📊 View Model Details"):
                model_data = resources['leaderboard'][
                    resources['leaderboard']['Model'] == selected_model
                ].iloc[0]
                
                st.success(f"✓ Model: {selected_model}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{model_data['Accuracy']:.4f}")
                with col2:
                    st.metric("Precision", f"{model_data['Precision']:.4f}")
                with col3:
                    st.metric("Recall", f"{model_data['Recall']:.4f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("F1 Score", f"{model_data['F1 Score']:.4f}")
                with col2:
                    st.metric("ROC-AUC", f"{model_data['ROC-AUC']:.4f}")


# ============================================================================
# SECTION 5: MODEL COMPARISON DASHBOARD
# ============================================================================

elif page == "🏆 Model Comparison":
    st.title("🏆 Model Comparison Dashboard")
    
    if resources['leaderboard'] is not None:
        leaderboard = resources['leaderboard']
        
        # Champion callout
        champion = leaderboard.iloc[0]
        st.markdown(f"""
        <div class="champion-box">
            <h3>🏆 Champion Model: {champion['Model']}</h3>
            <p>F1 Score: <strong>{champion['F1 Score']:.4f}</strong> | 
               ROC-AUC: <strong>{champion['ROC-AUC']:.4f}</strong></p>
            <p>Selected based on highest combined F1 Score and ROC-AUC performance.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display leaderboard
        st.subheader("📊 Leaderboard - All Models Ranked")
        st.dataframe(leaderboard, use_container_width=True)
        
        # Create tabs for visualizations
        tab1, tab2 = st.tabs(["Metrics Comparison", "ROC Curves"])
        
        with tab1:
            st.subheader("Accuracy vs F1 Score")
            
            fig = px.bar(
                leaderboard,
                x='Model',
                y=['Accuracy', 'F1 Score'],
                title="Model Comparison: Accuracy vs F1 Score",
                barmode='group',
                color_discrete_sequence=['#4472C4', '#ED7D31']
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional comparisons
            col1, col2 = st.columns(2)
            
            with col1:
                fig2 = px.bar(
                    leaderboard,
                    x='Model',
                    y='Precision',
                    title="Precision Comparison",
                    color='Precision',
                    color_continuous_scale='Blues'
                )
                fig2.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                fig3 = px.bar(
                    leaderboard,
                    x='Model',
                    y='Recall',
                    title="Recall Comparison",
                    color='Recall',
                    color_continuous_scale='Greens'
                )
                fig3.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig3, use_container_width=True)
        
        with tab2:
            st.subheader("ROC-AUC Comparison")
            
            fig = px.bar(
                leaderboard.sort_values('ROC-AUC', ascending=False),
                x='Model',
                y='ROC-AUC',
                title="ROC-AUC Score Comparison",
                color='ROC-AUC',
                color_continuous_scale='Reds'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# SECTION 6: PREDICTION SYSTEM
# ============================================================================

elif page == "🔮 Prediction System":
    st.title("🔮 Real-Time Churn Prediction")
    
    if resources['champion_model'] is not None and resources['preprocessor'] is not None:
        st.markdown("""
        ### Make a Prediction
        Fill in the customer information below and click "Predict" to get a churn risk assessment.
        """)
        
        # Create form
        with st.form("prediction_form"):
            st.subheader("Customer Information")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tenure = st.slider("Tenure (months)", 0, 72, 12)
                monthly_charges = st.slider("Monthly Charges ($)", 20.0, 120.0, 65.0, step=1.0)
                senior_citizen = st.checkbox("Senior Citizen?")
            
            with col2:
                gender = st.selectbox("Gender", ["Male", "Female"])
                partner = st.selectbox("Has Partner?", ["Yes", "No"])
                dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
                phone_service = st.selectbox("Phone Service?", ["Yes", "No"])
            
            with col3:
                internet_service = st.selectbox("Internet Service", 
                                               ["DSL", "Fiber optic", "No"])
                contract = st.selectbox("Contract Type", 
                                       ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"])
            
            # Additional services
            st.subheader("Additional Services")
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
                payment_method = st.selectbox("Payment Method",
                                             ["Electronic check", "Mailed check", 
                                              "Bank transfer (automatic)", "Credit card (automatic)"])
                multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            
            # Submit button
            submit_button = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)
        
        # Make prediction
        if submit_button:
            # Prepare input
            input_dict = {
                'tenure': tenure,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': tenure * monthly_charges,
                'SeniorCitizen': 1 if senior_citizen else 0,
                'Gender': gender,
                'Partner': partner,
                'Dependents': dependents,
                'PhoneService': phone_service,
                'InternetService': internet_service,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'PaymentMethod': payment_method,
                'MultipleLines': multiple_lines
            }
            
            # Preprocess and predict
            preprocessed_input = preprocess_prediction_input(
                input_dict, resources['preprocessor']
            )
            
            if preprocessed_input is not None:
                prediction_result = make_prediction(
                    resources['champion_model'], preprocessed_input
                )
                
                if prediction_result is not None:
                    st.markdown("---")
                    st.subheader("📊 Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if prediction_result['prediction'] == 1:
                            st.markdown(
                                f"<p class='churn-risk'>⚠️ HIGH CHURN RISK</p>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"<p class='retained'>✓ LIKELY TO RETAIN</p>",
                                unsafe_allow_html=True
                            )
                    
                    with col2:
                        churn_prob = prediction_result['churn_probability'] * 100
                        st.metric("Churn Probability", f"{churn_prob:.2f}%", 
                                 delta=f"{churn_prob:.2f}")
                    
                    with col3:
                        retain_prob = prediction_result['retain_probability'] * 100
                        st.metric("Retention Probability", f"{retain_prob:.2f}%",
                                 delta=f"{retain_prob:.2f}")
                    
                    # Visualization
                    fig = go.Figure(data=[
                        go.Bar(name='Churn Risk', x=['Risk'], y=[churn_prob],
                              marker_color='#FF6B6B'),
                        go.Bar(name='Retention', x=['Risk'], y=[retain_prob],
                              marker_color='#51CF66')
                    ])
                    fig.update_layout(barmode='stack', title='Prediction Confidence',
                                    yaxis_title='Probability (%)')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendation
                    st.markdown("---")
                    if prediction_result['prediction'] == 1:
                        st.warning("""
                        ### 💡 Recommended Action
                        This customer is at high risk of churning. Consider:
                        - Offering retention incentives or discounts
                        - Proactive customer service outreach
                        - Personalized engagement campaigns
                        - Service upgrade recommendations
                        """)
                    else:
                        st.success("""
                        ### 💡 Recommended Action
                        This customer appears satisfied. Continue:
                        - Regular service quality monitoring
                        - Cross-sell/upsell opportunities
                        - Loyalty program engagement
                        """)
            else:
                st.error("❌ Error preprocessing input. Please check your inputs.")
    else:
        st.error("❌ Champion model or preprocessor not found. Please ensure models are trained.")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Telco Customer Churn Prediction Dashboard | Phase 9 Deployment</p>
        <p>Built with Streamlit, Plotly, and Scikit-learn | Last Updated: May 2026</p>
    </div>
""", unsafe_allow_html=True)
