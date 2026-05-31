import os
from utils import load_data, preprocess_data, split_data, save_processed_data, train_models, evaluate_models, hyperparameter_tuning, save_models, create_results_leaderboard


def main():
    raw_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    print('Starting model training pipeline...')

    if not os.path.exists(raw_path):
        raise FileNotFoundError(raw_path)

    df = load_data(filepath=raw_path)
    X, y, encoders = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    save_processed_data(X_train, X_test, y_train, y_test, encoders)

    models = train_models(X_train, y_train)
    results_df, _, _, _ = evaluate_models(models, X_test, y_test)
    best_rf, best_xgb, _ = hyperparameter_tuning(X_train, y_train)
    save_models(models, best_rf, best_xgb, model_dir='models')
    leaderboard, champion = create_results_leaderboard(results_df, models, best_rf, best_xgb, X_test, y_test)
    leaderboard.to_csv('models/leaderboard.csv', index=False)

    print('Champion model:', champion)
    print('Model training pipeline completed successfully.')


if __name__ == '__main__':
    main()
