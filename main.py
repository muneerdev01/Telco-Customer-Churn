"""
Telco Customer Churn - Main Preprocessing and EDA Script
Phase 3 & 4: EDA Visualizations and Data Preprocessing
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import (
    load_data, preprocess_data, split_data, 
    save_processed_data, get_class_distribution
)

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create necessary directories
os.makedirs('visuals', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)


def generate_eda_visualizations(df, y):
    """
    Generate comprehensive EDA visualizations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed features (without target)
    y : pd.Series
        Target variable
    """
    print("\n" + "="*50)
    print("EDA VISUALIZATIONS PHASE")
    print("="*50)
    
    # 1. Univariate Analysis
    print("\n--- Generating Univariate Visualizations ---")
    
    # Tenure distribution
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    if 'tenure' in df.columns:
        df['tenure'].hist(bins=50, edgecolor='black', ax=ax)
        ax.set_title('Distribution of Tenure (months)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Tenure')
        ax.set_ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('visuals/01_tenure_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 01_tenure_distribution.png")
    
    # Monthly Charges distribution
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    if 'MonthlyCharges' in df.columns:
        df['MonthlyCharges'].hist(bins=50, edgecolor='black', ax=ax, color='coral')
        ax.set_title('Distribution of Monthly Charges', fontsize=14, fontweight='bold')
        ax.set_xlabel('Monthly Charges ($)')
        ax.set_ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('visuals/02_monthly_charges_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 02_monthly_charges_distribution.png")
    
    # Senior Citizen distribution
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    if 'SeniorCitizen' in df.columns:
        senior_counts = df['SeniorCitizen'].value_counts()
        senior_counts.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
        ax.set_title('Distribution of Senior Citizens', fontsize=14, fontweight='bold')
        ax.set_xlabel('Senior Citizen (0=No, 1=Yes)')
        ax.set_ylabel('Count')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        plt.tight_layout()
        plt.savefig('visuals/03_senior_citizen_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 03_senior_citizen_distribution.png")
    
    # 2. Bivariate Analysis
    print("\n--- Generating Bivariate Visualizations ---")
    
    # Churn distribution (Class Imbalance)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    churn_dist = y.value_counts()
    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(['No Churn', 'Churn'], churn_dist.values, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_title('Churn Distribution (Class Imbalance)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count')
    
    # Add percentages on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{churn_dist.values[i]}\n({churn_dist.values[i]/len(y)*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visuals/04_churn_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 04_churn_distribution.png")
    
    # Print class distribution
    dist_info = get_class_distribution(y)
    print(f"\nClass Distribution:\n{dist_info['counts']}")
    print(f"\nPercentages:\n{dist_info['percentages']}")
    print(f"Imbalance Ratio: {dist_info['imbalance_ratio']:.2f}")
    
    # 3. Correlation Heatmap
    print("\n--- Generating Correlation Heatmap ---")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if len(numerical_cols) > 0:
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create correlation matrix
        corr_data = df[numerical_cols].copy()
        corr_data['Churn'] = y
        corr_matrix = corr_data.corr()
        
        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Correlation Matrix - Numerical Features & Churn', 
                    fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('visuals/05_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 05_correlation_heatmap.png")
    
    print("\n✓ All EDA visualizations generated successfully!")


def main():
    """Main execution function"""
    
    print("="*60)
    print("TELCO CUSTOMER CHURN PREDICTION - PREPROCESSING & EDA")
    print("="*60)
    
    # Step 1: Load Data
    print("\nStep 1: Loading Data...")
    csv_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    # Check if file exists in current directory or data directory
    if not os.path.exists(csv_path):
        if os.path.exists(os.path.join('data', csv_path)):
            csv_path = os.path.join('data', csv_path)
        else:
            raise FileNotFoundError(f"CSV file not found. Please place {csv_path} in the project root or data/ directory")
    
    df = pd.read_csv(csv_path)
    print(f"✓ Data loaded successfully!")
    print(f"  Dataset shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Step 2: Preprocess Data
    print("\n\nStep 2: Preprocessing Data...")
    X_processed, y_processed, encoders_dict = preprocess_data(df)
    print("✓ Data preprocessing completed!")
    
    # Step 3: Generate EDA Visualizations (before split to use full data)
    print("\n\nStep 3: Generating EDA Visualizations...")
    generate_eda_visualizations(X_processed, y_processed)
    
    # Step 4: Split Data
    print("\n\nStep 4: Splitting Data...")
    X_train, X_test, y_train, y_test = split_data(X_processed, y_processed)
    print("✓ Data split completed!")
    
    # Step 5: Save Processed Data and Models
    print("\n\nStep 5: Saving Processed Data...")
    save_processed_data(X_train, X_test, y_train, y_test, encoders_dict)
    print("✓ Data saved successfully!")
    
    print("\n" + "="*60)
    print("PREPROCESSING AND EDA COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("  - data/X_train.csv")
    print("  - data/X_test.csv")
    print("  - data/y_train.csv")
    print("  - data/y_test.csv")
    print("  - models/scaler.joblib")
    print("  - models/encoders.joblib")
    print("  - visuals/*.png (EDA plots)")
    

if __name__ == "__main__":
    main()
