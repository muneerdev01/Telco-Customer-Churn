# Telco Customer Churn Prediction Project

A comprehensive machine learning project for predicting customer churn in telecommunications using data preprocessing, exploratory data analysis, and supervised learning techniques.

## 📁 Project Structure

```
telco-churn/
├── data/
│   ├── raw/                              # Original dataset directory
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/                        # Preprocessed data (generated)
│       ├── X_train.csv                  # Training features
│       ├── X_test.csv                   # Test features
│       ├── y_train.csv                  # Training targets
│       └── y_test.csv                   # Test targets
├── models/                              # Saved preprocessing artifacts
│   ├── scaler.joblib                    # StandardScaler for feature scaling
│   ├── encoders.joblib                  # Encoding information
│   └── metadata.json                    # Preprocessing metadata
├── visuals/                             # EDA visualizations (PNG files)
│   ├── 01_univariate_analysis.png
│   ├── 02_churn_distribution.png
│   ├── 03_bivariate_analysis.png
│   ├── 04_correlation_heatmap.png
│   └── 05_train_test_split.png
├── notebooks/                           # Jupyter notebooks
│   └── 01_EDA_Preprocessing.ipynb       # Complete workflow notebook
├── utils.py                             # Reusable utility functions
├── main.py                              # Main preprocessing script
├── requirements.txt                     # Python dependencies
├── .gitignore                           # Git ignore patterns
└── README.md                            # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone or download the project**
   ```bash
   cd telco-churn
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place the dataset**
   - Copy `WA_Fn-UseC_-Telco-Customer-Churn.csv` to the `data/raw/` directory

### Running the Project

#### Option 1: Using Jupyter Notebook (Recommended)
```bash
jupyter notebook notebooks/01_EDA_Preprocessing.ipynb
```
Run cells sequentially to execute the complete workflow with visualizations.

#### Option 2: Using Python Script
```bash
python main.py
```
Executes the entire preprocessing pipeline and saves artifacts.

## 📊 Dataset Overview

- **Records**: 7,043 customer accounts
- **Features**: 21 attributes (demographic, account, and service information)
- **Target**: Churn (Yes/No)
- **Churn Rate**: ~26.5%

### Key Features
- **Demographics**: Age, Gender, Senior Citizen Status
- **Account Info**: Tenure, Contract Type, Billing Method
- **Services**: Internet, Phone, Security, Backup, Device Protection
- **Charges**: Monthly and Total Charges

## 🔄 Data Processing Pipeline

### Phase 1: Data Loading
- Load raw CSV file
- Display basic statistics and data types
- Identify missing values

### Phase 2: EDA (Exploratory Data Analysis)
- **Univariate**: Tenure, Monthly Charges, Senior Citizen distributions
- **Bivariate**: Churn relationships with Contract, Internet Service, Payment Method
- **Correlation**: Heatmap of numerical features with target
- **Class Balance**: Churn distribution analysis

### Phase 3: Preprocessing
- Handle missing values in TotalCharges (convert to numeric, fill with median)
- Drop non-predictive customerID
- Encode target variable (Yes=1, No=0)

### Phase 4: Feature Engineering
- **One-Hot Encoding**: 10 categorical features → 43 features
- **Label Encoding**: Applied to ordinal features if present
- **Feature Scaling**: StandardScaler on numerical features (tenure, MonthlyCharges, TotalCharges)

### Phase 5: Data Splitting
- Train-Test Split: 80% training, 20% test
- Stratification: Maintains class distribution
- Random State: 42 (for reproducibility)

### Phase 6: Artifact Saving
- Processed datasets (CSV)
- Scaler object (joblib)
- Encoder information (joblib)
- Metadata (JSON)

## 📈 Outputs

### Data Files
- `X_train.csv` (5,634 × 43): Training features
- `X_test.csv` (1,409 × 43): Test features
- `y_train.csv` (5,634 × 1): Training targets
- `y_test.csv` (1,409 × 1): Test targets

### Visualizations
1. **Univariate Analysis**: Feature distributions
2. **Churn Distribution**: Class imbalance (pie and bar charts)
3. **Bivariate Analysis**: Churn vs categorical features
4. **Correlation Heatmap**: Feature relationships
5. **Train-Test Split**: Class distribution in splits

### Preprocessing Artifacts
- `scaler.joblib`: Fitted StandardScaler
- `encoders.joblib`: Categorical encoder information
- `metadata.json`: Complete preprocessing metadata

## 🛠️ Utils Functions

The `utils.py` file contains reusable functions:

```python
# Load dataset
load_data(filepath, url)

# Handle missing values
handle_missing_values(df)

# Complete preprocessing pipeline
preprocess_data(df)

# Split data
split_data(X, y, test_size=0.2, random_state=42)

# Save artifacts
save_processed_data(X_train, X_test, y_train, y_test, encoders_dict)

# Get class distribution
get_class_distribution(y)
```

## 📋 Requirements

See `requirements.txt` for complete dependencies:
- pandas: Data manipulation
- numpy: Numerical computing
- matplotlib: Visualization
- seaborn: Statistical visualization
- scikit-learn: Machine learning
- joblib: Serialization
- jupyter: Interactive notebooks
- python-dotenv: Environment variables

## 🔍 Key Findings

1. **Tenure Impact**: Longer tenure reduces churn significantly
2. **Contract Type**: Month-to-month contracts have 42% churn vs 11% for two-year contracts
3. **Internet Service**: Fiber optic customers have highest churn (42%)
4. **Payment Method**: Electronic check users have higher churn rates
5. **Tech Support**: Customers with tech support have lower churn

## ✅ Next Steps

1. **Model Training**
   - Logistic Regression (baseline)
   - Random Forest
   - XGBoost
   - Neural Networks

2. **Hyperparameter Tuning**
   - Grid/Random Search
   - Cross-validation
   - ROC-AUC optimization

3. **Handling Class Imbalance**
   - SMOTE
   - Class weights
   - Cost-sensitive learning

4. **Model Evaluation**
   - Confusion Matrix
   - ROC-AUC Curve
   - Feature Importance
   - Prediction Confidence

5. **Deployment**
   - Model versioning
   - Prediction API
   - Monitoring & Maintenance

## 📝 Notes

- **Reproducibility**: Random state set to 42 for consistent results
- **Data Leakage Prevention**: Scaler fitted only on training data
- **Stratification**: Maintains churn distribution in train/test sets
- **Feature Scaling**: Essential for distance-based and gradient-based algorithms

## 🤝 Contributing

To improve this project:
1. Enhance EDA visualizations
2. Add advanced preprocessing techniques
3. Implement feature selection methods
4. Create model training notebooks
5. Add hyperparameter tuning

## 📄 License

This project is provided as-is for educational and commercial purposes.

## 🙋 Support

For issues or questions:
- Review the Jupyter notebook: `notebooks/01_EDA_Preprocessing.ipynb`
- Check the main script: `main.py`
- Inspect utility functions: `utils.py`

---

**Last Updated**: May 2026  
**Version**: 1.0  
**Status**: Production Ready
"# Telco-Customer-Churn" 
