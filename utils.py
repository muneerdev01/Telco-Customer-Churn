"""
Utility functions for Telco Churn Prediction project
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings

warnings.filterwarnings('ignore')


def load_data(filepath='data/telco_churn.csv', url=None):
    """لوکل فائل یا یو آر ایل سے ڈیٹا لوڈ کرنے کا فنکشن"""
    if os.path.exists(filepath):
        print(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
    elif url:
        print(f"File not found locally. Downloading from {url}")
        df = pd.read_csv(url)
    else:
        raise FileNotFoundError(f"File {filepath} not found and no URL provided")
    
    print(f"Dataset shape: {df.shape}")
    return df


def handle_missing_values(df):
    """مسنگ ویلیوز کو بغیر کسی وارننگ کے ہینڈل کرنے کا فنکشن"""
    print("\nHandling missing values...")
    
    # ڈیٹا کی کاپی بنانا تاکہ اصل ڈیٹا متاثر نہ ہو
    df = df.copy()
    
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        median_value = df['TotalCharges'].median()
        df['TotalCharges'] = df['TotalCharges'].fillna(median_value)
        print(f"Filled TotalCharges missing values with median: {median_value}")
        
    return df


def preprocess_data(df):
    """Data Leakage سے پاک ڈیٹا پری پروسیسنگ فنکشن"""
    print("\n" + "="*50)
    print("PREPROCESSING PHASE")
    print("="*50)
    
    df = handle_missing_values(df)
    
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        print("Dropped customerID column")
    
    if 'Churn' in df.columns:
        y = df['Churn'].map({'Yes': 1, 'No': 0})
        X = df.drop('Churn', axis=1)
        print(f"\nTarget variable (Churn) shape: {y.shape}")
    else:
        raise ValueError("Churn column not found in dataset")
    
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nNumerical columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    # One-Hot Encoder کا درست استعمال
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
    X_cat_encoded = encoder.fit_transform(X[categorical_cols])
    encoded_cat_features = encoder.get_feature_names_out(categorical_cols).tolist()
    
    # نیا اسٹرکچرڈ ڈیٹا فریم بنانا
    X_encoded = pd.DataFrame(X_cat_encoded, columns=encoded_cat_features, index=X.index)
    X_encoded[numerical_cols] = X[numerical_cols]
    
    # نکل فیچرز کو اسکیل کرنا
    scaler = StandardScaler()
    X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])
    print(f"Scaled numerical features: {numerical_cols}")
    
    # تمام ضروری کنفیگریشنز کو ایک ڈکشنری میں محفوظ کرنا
    encoders_dict = {
        'scaler': scaler,
        'encoder': encoder,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'final_features_layout': X_encoded.columns.tolist()
    }
    
    return X_encoded, y, encoders_dict


def split_data(X, y, test_size=0.2, random_state=42):
    """ڈیٹا کو ٹرین اور ٹیسٹ سیٹس میں تقسیم کرنا"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n" + "="*50)
    print("DATA SPLIT")
    print("="*50)
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def save_processed_data(X_train, X_test, y_train, y_test, encoders_dict, output_dir='data/processed'):
    """پروسیسڈ ڈیٹا اور انکوڈرز کو محفوظ کرنا"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    # مکمل پری پروسیسر بنڈل محفوظ کرنا
    joblib.dump(encoders_dict, 'models/preprocessor.joblib')
    print(f"\nPreprocessor configuration bundle saved to models/preprocessor.joblib")


def load_preprocessed_data(processed_dir='data/processed'):
    """پروسیسڈ ڈیٹا لوڈ کرنا"""
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).iloc[:, 0]
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).iloc[:, 0]
    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    """6 الگ الگ کلاسیفیکیشن ماڈلز کو ٹرین کرنا"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from xgboost import XGBClassifier
    
    print("\n" + "="*60)
    print("PHASE 5: TRAINING MODELS")
    print("="*60)
    
    svm_base = LinearSVC(max_iter=5000, tol=1e-3, dual=False, random_state=42)
    svm_model = CalibratedClassifierCV(svm_base, cv=3)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': svm_model,
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...", end=" ")
        trained_models[name] = model.fit(X_train, y_train)
        print("✓ Complete")
    
    return trained_models


def evaluate_models(models_dict, X_test, y_test):
    """ماڈلز کی کارکردگی کا جائزہ لینا اور رزلٹ تیار کرنا"""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                 f1_score, roc_auc_score, confusion_matrix)
    
    results = []
    predictions_dict, probabilities_dict, confusion_matrices = {}, {}, {}
    
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        predictions_dict[name] = y_pred
        probabilities_dict[name] = y_pred_proba
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
        })
    
    results_df = pd.DataFrame(results).sort_values('F1 Score', ascending=False)
    return results_df, predictions_dict, probabilities_dict, confusion_matrices


def hyperparameter_tuning(X_train, y_train):
    """Random Forest اور XGBoost کی ہائپر پیرامیٹر ٹیوننگ کرنا"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    
    tuning_results = {}
    
    # Random Forest Tuning
    rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 15], 'min_samples_split': [2, 5]}
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=4, scoring='f1', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    tuning_results['Random Forest'] = {'model': best_rf, 'score': rf_grid.best_score_}
    
    # XGBoost Tuning
    xgb_params = {'learning_rate': [0.05, 0.1], 'max_depth': [3, 5], 'n_estimators': [100, 150]}
    xgb_grid = GridSearchCV(XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0), xgb_params, cv=4, scoring='f1', n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    best_xgb = xgb_grid.best_estimator_
    tuning_results['XGBoost'] = {'model': best_xgb, 'score': xgb_grid.best_score_}
    
    return best_rf, best_xgb, tuning_results


def save_models(models_dict, best_rf, best_xgb, model_dir='models'):
    """تمام ٹرینڈ ماڈلز کو جاب لِب (.joblib) فارمیٹ میں محفوظ کرنا"""
    os.makedirs(model_dir, exist_ok=True)
    
    for name, model in models_dict.items():
        filename = f"{model_dir}/{name.lower().replace(' ', '_')}_baseline.joblib"
        joblib.dump(model, filename)
    
    joblib.dump(best_rf, f"{model_dir}/random_forest_tuned.joblib")
    joblib.dump(best_xgb, f"{model_dir}/xgboost_tuned.joblib")
    print("✓ All models written out successfully using .joblib schema formatting.")


def create_results_leaderboard(results_df, models_dict, best_rf, best_xgb, X_test, y_test):
    """لیڈر بورڈ تیار کرنا اور بہترین ماڈل (Champion) تلاش کرنا"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    tuned_results = []
    for name, model in [('Random Forest (Tuned)', best_rf), ('XGBoost (Tuned)', best_xgb)]:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        tuned_results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
        })
    
    leaderboard = pd.concat([results_df, pd.DataFrame(tuned_results)], ignore_index=True)
    leaderboard = leaderboard.sort_values('F1 Score', ascending=False).reset_index(drop=True)
    return leaderboard, leaderboard.iloc[0]['Model']


# ============================================================================
# STREAMLIT کے لیے درست کیے گئے فنکشنز (REPAIRED MODULES)
# ============================================================================

def load_model_streamlit(model_name='xgboost_tuned'):
    """اسٹریم لِٹ کے لیے درست ایکسٹینشن کے ساتھ ماڈل لوڈ کرنا"""
    model_path = f'models/{model_name}.joblib'
    try:
        if os.path.exists(model_path):
            return joblib.load(model_path)
        raise FileNotFoundError(f"Model not found at {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def load_preprocessor_streamlit():
    """پری پروسیسر بنڈل کو لوڈ کرنا"""
    try:
        return joblib.load('models/preprocessor.joblib')
    except Exception as e:
        print(f"Error loading preprocessor config bundle: {e}")
        return None


def preprocess_prediction_input(input_dict, preprocessor_bundle):
    """
    صارف کے لائیو ان پٹ کو بغیر کسی کالم ڈرفٹ (Column Drift) کے 
    ماڈل کے لیے تیار کرنا۔
    """
    try:
        df_input = pd.DataFrame([input_dict])
        
        # بنڈل سے تمام آبجیکٹس نکالنا
        scaler = preprocessor_bundle['scaler']
        encoder = preprocessor_bundle['encoder']
        numerical_cols = preprocessor_bundle['numerical_cols']
        categorical_cols = preprocessor_bundle['categorical_cols']
        final_features_layout = preprocessor_bundle['final_features_layout']
        
        # محفوظ شدہ انکوڈر کے ذریعے کیٹیگریز تبدیل کرنا (محفوظ طریقہ)
        cat_encoded = encoder.transform(df_input[categorical_cols])
        encoded_cat_features = encoder.get_feature_names_out(categorical_cols).tolist()
        
        df_encoded = pd.DataFrame(cat_encoded, columns=encoded_cat_features, index=df_input.index)
        df_encoded[numerical_cols] = df_input[numerical_cols]
        
        # نیومیرکل ڈیٹا کو اسکیل کرنا
        df_encoded[numerical_cols] = scaler.transform(df_encoded[numerical_cols])
        
        # کالمز کی ترتیب کو بالکل ٹریننگ ڈیٹا کی طرح ترتیب دینا
        df_encoded = df_encoded[final_features_layout]
        return df_encoded
        
    except Exception as e:
        print(f"Inference input processing exception triggered: {e}")
        return None


def make_prediction(model, preprocessed_input):
    """فائنل پیشگوئی اور پروببیلیٹی (Probability) معلوم کرنا"""
    try:
        prediction = model.predict(preprocessed_input)[0]
        probability = model.predict_proba(preprocessed_input)[0]
        
        return {
            'prediction': int(prediction),
            'churn_probability': float(probability[1]),
            'retain_probability': float(probability[0])
        }
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None
