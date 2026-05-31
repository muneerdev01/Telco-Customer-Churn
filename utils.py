"""
Utility functions for Telco Churn Prediction project
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings

warnings.filterwarnings('ignore')


def load_data(filepath='data/telco_churn.csv', url=None):
    """
    Load the Telco Customer Churn dataset from local file or URL.
    
    Parameters:
    -----------
    filepath : str
        Local file path to the CSV
    url : str
        URL to download the dataset if local file not found
        
    Returns:
    --------
    pd.DataFrame : Dataset
    """
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
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame : Dataframe with missing values handled
    """
    print("\nHandling missing values...")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values found:\n{missing[missing > 0]}")
        
        # Handle TotalCharges specifically
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            median_value = df['TotalCharges'].median()
            df['TotalCharges'].fillna(median_value, inplace=True)
            print(f"Filled TotalCharges missing values with median: {median_value}")
    
    return df


def preprocess_data(df):
    """
    Perform complete data preprocessing.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw input dataframe
        
    Returns:
    --------
    tuple : (X_processed, y_processed, scaler, encoders_dict)
    """
    print("\n" + "="*50)
    print("PREPROCESSING PHASE")
    print("="*50)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Drop customerID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        print("Dropped customerID column")
    
    # Separate target and features
    if 'Churn' in df.columns:
        y = df['Churn'].map({'Yes': 1, 'No': 0})
        X = df.drop('Churn', axis=1)
        print(f"\nTarget variable (Churn) shape: {y.shape}")
    else:
        raise ValueError("Churn column not found in dataset")
    
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nNumerical columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    # One-Hot Encoding for categorical features
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    print(f"\nAfter One-Hot Encoding - Shape: {X_encoded.shape}")
    
    # Save encoded column names for future input preprocessing
    encoded_columns = X_encoded.columns.tolist()
    
    # Scale numerical features
    scaler = StandardScaler()
    X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])
    print(f"Scaled numerical features: {numerical_cols}")
    
    encoders_dict = {
        'scaler': scaler,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'encoded_columns': encoded_columns
    }
    
    return X_encoded, y, encoders_dict


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    test_size : float
        Proportion of test set
    random_state : int
        Random seed
        
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n" + "="*50)
    print("DATA SPLIT")
    print("="*50)
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Class distribution in training set:\n{y_train.value_counts()}")
    print(f"Class distribution in test set:\n{y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test


def save_processed_data(X_train, X_test, y_train, y_test, encoders_dict, output_dir='data/processed'):
    """
    Save processed data and encoders.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and test features
    y_train, y_test : pd.Series
        Training and test targets
    encoders_dict : dict
        Dictionary containing scaler and encoder info
    output_dir : str
        Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Save processed data
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    # Save encoders and scaler
    joblib.dump(encoders_dict['scaler'], 'models/scaler.joblib')
    joblib.dump(encoders_dict, 'models/encoders.joblib')
    
    print(f"\n" + "="*50)
    print("DATA SAVED")
    print("="*50)
    print(f"Training data saved to {output_dir}/X_train.csv")
    print(f"Test data saved to {output_dir}/X_test.csv")
    print(f"Scaler saved to models/scaler.joblib")
    print(f"Encoders saved to models/encoders.joblib")


def get_class_distribution(y):
    """
    Get class distribution information.
    
    Parameters:
    -----------
    y : pd.Series
        Target variable
        
    Returns:
    --------
    dict : Class distribution statistics
    """
    counts = y.value_counts()
    percentages = y.value_counts(normalize=True) * 100
    
    return {
        'counts': counts,
        'percentages': percentages,
        'imbalance_ratio': counts.max() / counts.min()
    }


# ============================================================================
# MODEL TRAINING FUNCTIONS (Phase 5, 6, 7)
# ============================================================================

def load_preprocessed_data(processed_dir='data/processed'):
    """
    Load preprocessed training and test data.
    
    Parameters:
    -----------
    processed_dir : str
        Directory containing processed data
        
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test)
    """
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).iloc[:, 0]
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).iloc[:, 0]
    
    print(f"Loaded preprocessed data:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    """
    Train 6 classification models with a faster SVM approximation.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
        
    Returns:
    --------
    dict : Trained models
    """
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
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False, verbosity=0)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"\nTraining {name}...", end=" ")
        trained_models[name] = model.fit(X_train, y_train)
        print("✓ Complete")
    
    print("\n✓ All 6 models trained successfully!")
    return trained_models


def evaluate_models(models_dict, X_test, y_test):
    """
    Evaluate models and calculate metrics.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of trained models
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
        
    Returns:
    --------
    pd.DataFrame : Evaluation results
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                f1_score, roc_auc_score, confusion_matrix)
    
    print("\n" + "="*60)
    print("PHASE 6: MODEL EVALUATION")
    print("="*60)
    
    results = []
    predictions_dict = {}
    probabilities_dict = {}
    confusion_matrices = {}
    
    for name, model in models_dict.items():
        print(f"\nEvaluating {name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Store for visualization
        predictions_dict[name] = y_pred
        probabilities_dict[name] = y_pred_proba
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC-AUC': roc_auc
        })
        
        print(f"  Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
    
    results_df = pd.DataFrame(results).sort_values('F1 Score', ascending=False)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\n{results_df.to_string(index=False)}")
    
    return results_df, predictions_dict, probabilities_dict, confusion_matrices


def hyperparameter_tuning(X_train, y_train):
    """
    Perform hyperparameter tuning for Random Forest and XGBoost.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
        
    Returns:
    --------
    tuple : (best_rf_model, best_xgb_model, tuning_results)
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    
    print("\n" + "="*60)
    print("PHASE 7: HYPERPARAMETER TUNING")
    print("="*60)
    
    tuning_results = {}
    
    # Random Forest Tuning
    print("\n--- Tuning Random Forest ---")
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15],
        'min_samples_split': [2, 5]
    }
    
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params,
        cv=4,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    
    print(f"Best RF params: {rf_grid.best_params_}")
    print(f"Best CV F1 Score: {rf_grid.best_score_:.4f}")
    
    tuning_results['Random Forest'] = {
        'best_params': rf_grid.best_params_,
        'best_score': rf_grid.best_score_,
        'model': best_rf
    }
    
    # XGBoost Tuning
    print("\n--- Tuning XGBoost ---")
    xgb_params = {
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'n_estimators': [100, 150],
        'subsample': [0.8, 1.0]
    }
    
    xgb_grid = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False, verbosity=0),
        xgb_params,
        cv=4,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    xgb_grid.fit(X_train, y_train)
    best_xgb = xgb_grid.best_estimator_
    
    print(f"Best XGB params: {xgb_grid.best_params_}")
    print(f"Best CV F1 Score: {xgb_grid.best_score_:.4f}")
    
    tuning_results['XGBoost'] = {
        'best_params': xgb_grid.best_params_,
        'best_score': xgb_grid.best_score_,
        'model': best_xgb
    }
    
    print("\n✓ Hyperparameter tuning completed!")
    
    return best_rf, best_xgb, tuning_results


def save_models(models_dict, best_rf, best_xgb, model_dir='models'):
    """
    Save all trained models as pickle files.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of baseline models
    best_rf : RandomForestClassifier
        Best tuned Random Forest
    best_xgb : XGBClassifier
        Best tuned XGBoost
    model_dir : str
        Directory to save models
    """
    os.makedirs(model_dir, exist_ok=True)
    
    print("\n--- Saving Models ---")
    
    # Save baseline models
    for name, model in models_dict.items():
        filename = f"{model_dir}/{name.lower().replace(' ', '_')}_baseline.pkl"
        joblib.dump(model, filename)
        print(f"✓ Saved: {filename}")
    
    # Save tuned models
    joblib.dump(best_rf, f"{model_dir}/random_forest_tuned.pkl")
    print(f"✓ Saved: {model_dir}/random_forest_tuned.pkl")
    
    joblib.dump(best_xgb, f"{model_dir}/xgboost_tuned.pkl")
    print(f"✓ Saved: {model_dir}/xgboost_tuned.pkl")


def create_results_leaderboard(results_df, models_dict, best_rf, best_xgb, 
                               X_test, y_test):
    """
    Create comprehensive results leaderboard.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Baseline evaluation results
    models_dict : dict
        Baseline trained models
    best_rf : RandomForestClassifier
        Best tuned Random Forest
    best_xgb : XGBClassifier
        Best tuned XGBoost
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
        
    Returns:
    --------
    pd.DataFrame : Comprehensive leaderboard
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                f1_score, roc_auc_score)
    
    # Evaluate tuned models
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
    
    # Combine baseline and tuned results
    leaderboard = pd.concat([results_df, pd.DataFrame(tuned_results)], ignore_index=True)
    leaderboard = leaderboard.sort_values('F1 Score', ascending=False).reset_index(drop=True)
    
    # Identify champion model
    champion_model = leaderboard.iloc[0]['Model']
    champion_score = leaderboard.iloc[0]['F1 Score']
    
    print("\n" + "="*60)
    print("LEADERBOARD - ALL MODELS")
    print("="*60)
    print(f"\n{leaderboard.to_string(index=False)}")
    print(f"\n{'🏆 CHAMPION MODEL 🏆':^60}")
    print(f"{champion_model} with F1 Score: {champion_score:.4f}")
    
    return leaderboard, champion_model


# ============================================================================
# STREAMLIT-SPECIFIC FUNCTIONS (Phase 9: Deployment)
# ============================================================================

def load_model_streamlit(model_name='xgboost_tuned'):
    """
    Load a trained model for Streamlit app with error handling.
    
    Parameters:
    -----------
    model_name : str
        Name of the model file (without .pkl extension)
        
    Returns:
    --------
    model : Trained model object or None if not found
    """
    model_path = f'models/{model_name}.pkl'
    
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return model
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def load_preprocessor_streamlit():
    """
    Load preprocessor objects (scaler and encoders) for Streamlit app.
    
    Returns:
    --------
    dict : Scaler and encoder information
    """
    try:
        scaler = joblib.load('models/scaler.joblib')
        encoders = joblib.load('models/encoders.joblib')
        return {'scaler': scaler, 'encoders': encoders}
    except Exception as e:
        print(f"Error loading preprocessor: {e}")
        return None


def load_raw_data_streamlit(filepath='WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    """
    Load raw dataset for Streamlit dashboard.
    
    Parameters:
    -----------
    filepath : str
        Path to raw CSV file
        
    Returns:
    --------
    pd.DataFrame : Raw dataset
    """
    try:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            return df
        else:
            raise FileNotFoundError(f"Data file not found at {filepath}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def load_leaderboard_streamlit(filepath='models/leaderboard.csv'):
    """
    Load model leaderboard for comparison dashboard.
    
    Parameters:
    -----------
    filepath : str
        Path to leaderboard CSV
        
    Returns:
    --------
    pd.DataFrame : Leaderboard with all model metrics
    """
    try:
        if os.path.exists(filepath):
            leaderboard = pd.read_csv(filepath)
            return leaderboard
        else:
            raise FileNotFoundError(f"Leaderboard not found at {filepath}")
    except Exception as e:
        print(f"Error loading leaderboard: {e}")
        return None


def preprocess_prediction_input(input_dict, preprocessor):
    """
    Preprocess user input for model prediction.
    
    Parameters:
    -----------
    input_dict : dict
        User input with feature values
    preprocessor : dict
        Scaler and encoders
        
    Returns:
    --------
    pd.DataFrame : Preprocessed input ready for prediction
    """
    try:
        df = pd.DataFrame([input_dict])
        
        # Get encoder info
        encoders = preprocessor['encoders']
        scaler = preprocessor['scaler']
        numerical_cols = encoders['numerical_cols']
        categorical_cols = encoders['categorical_cols']
        
        # One-hot encode categorical features
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Ensure all columns match training data
        expected_cols = encoders['encoded_columns']
        for col in expected_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Keep only expected columns in same order
        df_encoded = df_encoded[expected_cols]
        
        # Scale numerical features
        df_encoded[numerical_cols] = scaler.transform(df_encoded[numerical_cols])
        
        return df_encoded
    except Exception as e:
        print(f"Error preprocessing input: {e}")
        return None


def make_prediction(model, preprocessed_input):
    """
    Make prediction using loaded model.
    
    Parameters:
    -----------
    model : Trained model object
        Loaded model
    preprocessed_input : pd.DataFrame
        Preprocessed input features
        
    Returns:
    --------
    dict : Prediction (0/1) and probability
    """
    try:
        prediction = model.predict(preprocessed_input)[0]
        probability = model.predict_proba(preprocessed_input)[0]
        
        return {
            'prediction': prediction,
            'churn_probability': probability[1],
            'retain_probability': probability[0]
        }
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None
