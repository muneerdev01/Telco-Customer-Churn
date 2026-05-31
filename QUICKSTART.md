# 🚀 Quick Start Guide - Streamlit Dashboard

## Prerequisites
- Python 3.8+
- Trained models in `models/` directory
- Preprocessed data in `data/processed/` directory

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Files
Ensure you have:
```
models/
├── xgboost_tuned.pkl
├── random_forest_tuned.pkl
├── logistic_regression_baseline.pkl
├── decision_tree_baseline.pkl
├── knn_baseline.pkl
├── svm_baseline.pkl
├── xgboost_baseline.pkl
├── scaler.joblib
├── encoders.joblib
├── leaderboard.csv
└── metadata.json

data/processed/
├── X_train.csv
├── X_test.csv
├── y_train.csv
└── y_test.csv

WA_Fn-UseC_-Telco-Customer-Churn.csv (in project root)
```

## Running the Dashboard

### Local Development
```bash
streamlit run app.py
```

The app opens at: `http://localhost:8501`

### Production Mode
```bash
streamlit run app.py --logger.level=error
```

## Dashboard Sections

1. **🏠 Home** - Project overview
2. **📋 Dataset Explorer** - Data inspection
3. **📈 EDA Dashboard** - Interactive analysis
4. **🤖 Model Training** - Model selection
5. **🏆 Model Comparison** - Leaderboard & metrics
6. **🔮 Prediction System** - Real-time predictions

## Troubleshooting

### Models Not Loading
```bash
# Verify model files exist
ls -la models/*.pkl
ls -la models/*.joblib

# Retrain if needed
jupyter notebook notebooks/02_Model_Training.ipynb
```

### Data Not Found
```bash
# Check data exists
ls -la data/processed/
ls -la WA_Fn-UseC_-Telco-Customer-Churn.csv

# Run preprocessing if needed
jupyter notebook notebooks/01_EDA_Preprocessing.ipynb
```

### Port Already in Use
```bash
# Use different port
streamlit run app.py --server.port 8502
```

## Performance Tips

1. **Clear cache** if experiencing issues:
   ```bash
   rm -rf ~/.streamlit/cache
   ```

2. **Use smaller subsets** for testing:
   - Edit utils.py to sample data

3. **Monitor memory** in production:
   ```bash
   streamlit run app.py --logger.level=debug
   ```

## Deployment Options

- **Streamlit Cloud**: [streamlit.io/cloud](https://streamlit.io/cloud)
- **Heroku**: See DEPLOYMENT_README.md
- **Docker**: `docker build -t telco-churn . && docker run -p 8501:8501 telco-churn`
- **AWS**: EC2 with streamlit running on port 80

## Next Steps

1. ✅ Install dependencies
2. ✅ Train models (if not done)
3. ✅ Run app locally
4. ✅ Test predictions
5. ✅ Deploy to cloud

Good luck! 🎉
