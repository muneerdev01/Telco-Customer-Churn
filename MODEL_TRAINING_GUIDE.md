# Model Training & Evaluation - Phase 5, 6, and 7 Summary

## Project: Telco Customer Churn Prediction
## Notebook: `notebooks/02_Model_Training.ipynb`

---

## Overview

This notebook implements comprehensive machine learning model development including training, evaluation, and hyperparameter optimization for the Telco Customer Churn dataset.

## Phases Implemented

### Phase 5: Model Training
**Objective**: Train 6 different classification models with default hyperparameters

#### Models Trained:
1. **Logistic Regression**
   - Linear, interpretable baseline model
   - Parameters: max_iter=1000, random_state=42

2. **Decision Tree**
   - Tree-based, captures non-linear relationships
   - Parameters: random_state=42

3. **Random Forest**
   - Ensemble of decision trees
   - Parameters: n_estimators=100, random_state=42

4. **K-Nearest Neighbors (KNN)**
   - Instance-based, similarity-based classification
   - Parameters: n_neighbors=5

5. **Support Vector Machine (SVM)**
   - Margin-based classifier, good for binary classification
   - Parameters: kernel='rbf', probability=True, random_state=42

6. **XGBoost**
   - Gradient boosting ensemble, state-of-the-art performance
   - Parameters: random_state=42, eval_metric='logloss'

#### Training Approach:
- Trained on preprocessed, scaled training data (5,634 samples × 43 features)
- All models use consistent random_state=42 for reproducibility
- Probability estimates enabled for ROC-AUC calculation

---

### Phase 6: Model Evaluation
**Objective**: Evaluate all models on test set with comprehensive metrics

#### Evaluation Metrics:
1. **Accuracy**: Overall correctness of predictions
   - Formula: (TP + TN) / Total
   - Note: Can be misleading with imbalanced data

2. **Precision**: False positive rate control
   - Formula: TP / (TP + FP)
   - Importance: Minimize false alarms for churn predictions

3. **Recall**: False negative rate control
   - Formula: TP / (TP + FN)
   - Importance: Catch as many churners as possible

4. **F1 Score**: Harmonic mean of Precision and Recall
   - Formula: 2 × (Precision × Recall) / (Precision + Recall)
   - Best for: Imbalanced datasets (primary metric)

5. **ROC-AUC**: Discrimination ability of model
   - Range: 0.5 (random) to 1.0 (perfect)
   - Best for: Comparing models across probability thresholds

#### Test Set Details:
- Size: 1,409 samples (20% of total data)
- Class Distribution:
  - No Churn: ~1,038 samples (73.5%)
  - Churn: ~371 samples (26.5%)
- Stratification: Maintained same distribution as training set

#### Evaluation Outputs:
- Results DataFrame with all metrics sorted by F1 Score
- Confusion matrices for error analysis
- Prediction probabilities for ROC and PR curves

---

### Phase 7: Hyperparameter Tuning
**Objective**: Optimize hyperparameters for best-performing models

#### Tuning Strategy:
- **Method**: GridSearchCV with 5-fold Cross-Validation
- **Optimization Metric**: F1 Score (better for imbalanced data)
- **Cross-Validation**: 5-fold to estimate generalization performance
- **Parallel Processing**: n_jobs=-1 (use all CPU cores)

#### Random Forest Tuning:
**Parameter Grid**:
- `n_estimators`: [50, 100, 200] - Number of trees
- `max_depth`: [5, 10, 15] - Maximum tree depth
- `min_samples_split`: [2, 5, 10] - Minimum samples to split

**Total Combinations**: 3 × 3 × 3 = 27
**CV Folds**: 5
**Total Models**: 27 × 5 = 135 model trainings

#### XGBoost Tuning:
**Parameter Grid**:
- `learning_rate`: [0.01, 0.05, 0.1] - Shrinkage (lower = slower learning)
- `max_depth`: [3, 5, 7] - Maximum tree depth
- `n_estimators`: [50, 100, 150] - Number of boosting rounds
- `subsample`: [0.7, 0.8, 1.0] - Fraction of samples per tree

**Total Combinations**: 3 × 3 × 3 × 3 = 81
**CV Folds**: 5
**Total Models**: 81 × 5 = 405 model trainings

#### Outputs:
- Best parameters for each model
- Cross-validation F1 scores
- Tuned model objects saved for evaluation

---

## Visualizations Generated

### 1. Accuracy vs F1 Score Comparison
- **File**: `visuals/06_accuracy_vs_f1_comparison.png`
- **Type**: Grouped bar chart
- **Shows**: Side-by-side comparison of Accuracy and F1 Score for all models
- **Purpose**: Identify best performers considering class imbalance
- **Layout**: All models on x-axis, metrics on y-axis

### 2. Superimposed ROC Curves
- **File**: `visuals/07_roc_curves_comparison.png`
- **Type**: Multi-line plot
- **Shows**: ROC curves for all 6 baseline + 2 tuned models (8 total)
- **Metrics**: ROC-AUC values displayed in legend
- **Purpose**: Compare model discrimination ability across probability thresholds
- **Features**: 
  - Random classifier line (AUC=0.5) for reference
  - Solid lines for baseline, dashed for tuned models
  - Color-coded for distinction

### 3. Confusion Matrix Grid (2×3 Layout)
- **File**: `visuals/08_confusion_matrices_grid.png`
- **Type**: 6-panel heatmap grid
- **Shows**: Confusion matrices for all 6 baseline models
- **Cells**: TN, FP, FN, TP for each model
- **Purpose**: Analyze error patterns and trade-offs
- **Color Scale**: Blue intensity indicates frequency

### 4. Precision-Recall Curves (Top 3 Models)
- **File**: `visuals/09_precision_recall_top3.png`
- **Type**: Multi-line plot
- **Shows**: PR curves for top 3 models by F1 Score
- **Metrics**: AUPRC (Area Under PR Curve) in legend
- **Purpose**: Evaluate performance on imbalanced data
- **Advantage**: Better than ROC for imbalanced datasets

### 5. Feature Importance (Champion Model)
- **File**: `visuals/10_feature_importance_champion.png`
- **Type**: Horizontal bar chart
- **Shows**: Top 15 most important features
- **Ranking**: Sorted by importance score (descending)
- **Purpose**: Understand model decision-making
- **Method**: 
  - tree_importance for tree-based models
  - permutation_importance for linear models

---

## Leaderboard & Results

### Output Files:
- **CSV**: `models/leaderboard.csv`
- **Format**: 8 rows (6 baseline + 2 tuned), 6 columns (Model, Accuracy, Precision, Recall, F1 Score, ROC-AUC)
- **Sorted By**: F1 Score (descending)
- **Champion**: Model with highest combined F1 and ROC-AUC score

### Leaderboard Columns:
1. **Model**: Model name and tuning status
2. **Accuracy**: Overall correctness
3. **Precision**: False positive control
4. **Recall**: False negative control
5. **F1 Score**: Primary ranking metric
6. **ROC-AUC**: Secondary ranking metric

---

## Model Serialization

### Files Created:

**Baseline Models** (in `models/` directory):
```
logistic_regression_baseline.pkl
decision_tree_baseline.pkl
random_forest_baseline.pkl
knn_baseline.pkl
svm_baseline.pkl
xgboost_baseline.pkl
```

**Tuned Models**:
```
random_forest_tuned.pkl
xgboost_tuned.pkl
```

### Serialization Method:
- **Library**: joblib
- **Advantages**: Handles Python objects, efficient for large models
- **Size**: Varies by model (typically 100KB - 10MB)

### Loading Models:
```python
import joblib
model = joblib.load('models/model_name.pkl')
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)
```

---

## Performance Summary

### Expected Results:
- **Best Performer**: Usually ensemble models (Random Forest, XGBoost)
- **Tuned vs Baseline**: Tuned models typically 2-5% improvement in F1 Score
- **Benchmark**: 
  - Random Classifier: AUC = 0.5, F1 ~ 0.3 (imbalanced)
  - Good Model: AUC > 0.8, F1 > 0.65
  - Excellent Model: AUC > 0.85, F1 > 0.70

### Class Imbalance Considerations:
- Churn rate: 26.5% (minority class)
- F1 Score preferred over Accuracy
- ROC-AUC robust to class imbalance
- Recall important (catch churners) vs Precision (avoid false alarms)

---

## Utility Functions Used

### From `utils.py`:
1. `load_preprocessed_data()` - Load train/test data
2. `train_models()` - Train 6 baseline models
3. `evaluate_models()` - Evaluate with all metrics
4. `hyperparameter_tuning()` - GridSearchCV optimization
5. `save_models()` - Serialize all models
6. `create_results_leaderboard()` - Combine all results

---

## Notebook Structure

### Cells (9 main sections):
1. **Imports & Setup** - Libraries, paths, configuration
2. **Load Data** - Preprocessed train/test datasets
3. **Phase 5** - Train 6 baseline models
4. **Phase 6** - Evaluate all models
5. **Phase 7** - Hyperparameter tuning (GridSearchCV)
6. **Leaderboard** - Combine and rank all results
7. **Visualizations** - Generate 5 comprehensive plots
8. **Save Models** - Serialize all trained models
9. **Summary** - Recommendations and next steps

---

## Key Insights & Recommendations

### Insights:
1. **Model Performance**: Compare baseline vs tuned to assess tuning impact
2. **Trade-offs**: Precision vs Recall for business use case
3. **Feature Importance**: Which features drive churn decisions?
4. **Generalization**: CV scores vs test set performance

### Deployment Recommendations:
1. **Champion Model**: Use highest F1 + ROC-AUC model
2. **Threshold Selection**: Optimize based on business costs
3. **Ensemble Predictions**: Combine top 3 models for robustness
4. **Monitoring**: Track performance degradation over time
5. **Retraining**: Update quarterly with new data

---

## Next Steps in ML Pipeline

1. **Feature Engineering** - Create additional features
2. **Class Imbalance Handling** - SMOTE, class weights, thresholding
3. **Ensemble Methods** - Voting classifier, stacking
4. **Cross-validation** - Nested CV for hyperparameter selection
5. **Deployment** - Model serving, API endpoints, monitoring
6. **Feedback Loop** - Collect predictions, retrain periodically

---

## Technical Details

### Environment:
- Python 3.8+
- scikit-learn 1.3.0+
- xgboost, pandas, numpy
- matplotlib, seaborn for visualization

### Computational Requirements:
- Training time: ~5-10 minutes (depends on hardware)
- Tuning time: ~30-60 minutes (GridSearchCV, 5-fold CV)
- Evaluation time: <1 minute

### Data Pipeline:
```
Raw Data → Preprocessing → EDA → Feature Engineering → Scaling → 
Train/Test Split → Model Training → Evaluation → Tuning → 
Deployment
```

---

## File References

**Notebook**: `notebooks/02_Model_Training.ipynb`
**Utils Module**: `utils.py` (with functions defined above)
**Output Directory**: `models/` and `visuals/`
**Data**: `data/processed/` (from notebook 01)

---

**Last Updated**: May 2026
**Version**: 1.0
**Status**: Complete & Production Ready
