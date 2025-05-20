"""
Prediction module for cell analysis.

This module provides functions for cell classification and regression analysis
using Random Forest models. It encapsulates functionality from the original 
classification and regression prediction scripts.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    roc_curve, auc, mean_squared_error, r2_score
)
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

import shap


def classification_model_build(data_path, output_dir=None, test_size=0.3, random_state=42):
    """
    Build and train a classification model using Random Forest for cell status prediction.
    
    Args:
        data_path: Path to the Excel file containing the data
        output_dir: Directory to save model and results (default: None, creates directory next to data file)
        test_size: Test set proportion (default: 0.3)
        random_state: Random state for reproducibility (default: 42)
        
    Returns:
        tuple: (output_directory, X_train, X_test, y_train, y_test)
    """
    # Check and create output directory
    if not data_path or not os.path.exists(data_path):
        raise ValueError(f"Data file not found: {data_path}")
    
    if output_dir is None:
        data_directory = os.path.dirname(data_path)
        output_dir = os.path.join(data_directory, 'classification_results')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    try:
        data = pd.read_excel(data_path, sheet_name=0)
    except Exception as e:
        raise ValueError(f"Error reading data file: {e}")
    
    # Define feature columns 
    columns = [
        'P_Mean', 'P_Std', 'S_Mean', 'S_Std', 'dop_Mean', 'dop_Std',
        'x0SConst_Mean', 'x0SConst_Std', 'x90SConst_Mean', 'x90SConst_Std'
    ]
    
    # Rename columns to numerical for simplicity
    rename_dict = {old: str(i+1) for i, old in enumerate(columns)}
    
    # Extract features and target
    X = data[columns].rename(columns=rename_dict)
    y = data['Fit_Status']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print("Original training set distribution:", dict(pd.Series(y_train).value_counts()))
    
    # Apply undersampling to balance classes
    rus = RandomUnderSampler(sampling_strategy='not minority', random_state=random_state)
    X_train, y_train = rus.fit_resample(X_train, y_train)
    print("Resampled training set distribution:", dict(pd.Series(y_train).value_counts()))
    
    # Define model and parameter grid
    model = RandomForestClassifier(random_state=random_state)
    pipeline = Pipeline(steps=[('model', model)])
    param_grid = {
        'model__n_estimators': [5, 10, 20, 30, 40, 50, 60, 100, 150, 200],
        'model__max_depth': [None, 1, 2, 5, 10, 20, 30],
        'model__min_samples_split': [1, 2, 5, 10, 15, 20]
    }
    
    # Perform grid search for hyperparameter tuning
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_
    
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    
    # Evaluate on test set
    y_pred_test = best_pipeline.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    report_test = classification_report(y_test, y_pred_test, output_dict=True)
    
    # Store metrics
    model_scores = {
        'Test': {
            'Accuracy': accuracy_test,
            'Precision': report_test['weighted avg']['precision'],
            'Recall': report_test['weighted avg']['recall'],
            'F1': report_test['weighted avg']['f1-score']
        }
    }
    
    # Print evaluation metrics
    for key, value in model_scores['Test'].items():
        print(f"{key}: {value:.4f}")
    
    # Generate confusion matrix
    cm_test = confusion_matrix(y_test, y_pred_test)
    print("Confusion Matrix (Test Set):")
    print(cm_test)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_test.pdf"), dpi=600)
    plt.close()
    
    # Generate ROC curve
    fpr_test, tpr_test, _ = roc_curve(y_test, best_pipeline.predict_proba(X_test)[:, 1])
    roc_auc_test = auc(fpr_test, tpr_test)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_test, tpr_test, color='blue', lw=2, label=f'ROC (AUC = {roc_auc_test:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False positive rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve (Test Set)")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve_test.pdf"), dpi=600)
    plt.close()
    
    # Save test set predictions
    predictions_test_df = pd.DataFrame({'True': y_test, 'Predicted': y_pred_test})
    predictions_test_df.to_excel(os.path.join(output_dir, "predictions_test.xlsx"), index=False)
    
    # Evaluate on training set
    y_pred_train = best_pipeline.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    report_train = classification_report(y_train, y_pred_train, output_dict=True)
    
    # Store training metrics
    model_scores['Train'] = {
        'Accuracy': accuracy_train,
        'Precision': report_train['weighted avg']['precision'],
        'Recall': report_train['weighted avg']['recall'],
        'F1': report_train['weighted avg']['f1-score']
    }
    
    # Save all evaluation metrics
    model_scores_df = pd.DataFrame(model_scores).T
    model_scores_df.to_excel(os.path.join(output_dir, "model_scores.xlsx"))
    
    # Save the model
    joblib.dump(best_pipeline, os.path.join(output_dir, 'classification_model.pkl'))
    
    return output_dir, X_train, X_test, y_train, y_test


def classification_shap_analysis(model_path, data_path, output_dir=None):
    """
    Perform SHAP analysis on a trained classification model.
    
    Args:
        model_path: Path to the saved classification model
        data_path: Path to the data file used for analysis
        output_dir: Directory to save analysis results (default: None, uses directory of model)
        
    Returns:
        str: Path to the output directory
    """
    # Check model path
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(model_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = joblib.load(model_path)
    
    # Get the actual model from pipeline if needed
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        model = model.named_steps['model']
    
    # Load data
    data = pd.read_excel(data_path, sheet_name=0)
    
    # Define feature columns
    columns = [
        'P_Mean', 'P_Std', 'S_Mean', 'S_Std', 'dop_Mean', 'dop_Std',
        'x0SConst_Mean', 'x0SConst_Std', 'x90SConst_Mean', 'x90SConst_Std'
    ]
    
    # Rename columns to numerical for simplicity
    rename_dict = {old: str(i+1) for i, old in enumerate(columns)}
    
    # Extract features and rename
    X = data[columns].rename(columns=rename_dict)
    
    # Split data for analysis (using a fixed split for demonstration)
    X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
    
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    # Create summary bar plot
    plt.figure(figsize=(8, 6))
    plt.title('Feature Importance (SHAP)', fontsize=16)
    plt.ylabel('Features', fontsize=14)
    shap.summary_plot(shap_values[:, :, 1], X_test, feature_names=X.columns, 
                      plot_type="bar", alpha=0.5, show=False, plot_size=(8, 6))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_bar.pdf"), 
                bbox_inches='tight', dpi=600)
    plt.close()
    
    # Save SHAP values to Excel
    try:
        shap_values_1_df = pd.DataFrame(shap_values[:, :, 1].values, columns=X_test.columns)
        shap_values_0_df = pd.DataFrame(shap_values[:, :, 0].values, columns=X_test.columns)
        
        with pd.ExcelWriter(os.path.join(output_dir, "shap_values.xlsx")) as writer:
            shap_values_1_df.to_excel(writer, sheet_name='SHAP_Values_Class1', index=False)
            shap_values_0_df.to_excel(writer, sheet_name='SHAP_Values_Class0', index=False)
    except Exception as e:
        print(f"Warning: Could not save SHAP values to Excel: {e}")
    
    # Create summary plot
    shap.summary_plot(shap_values[:, :, 1], X_test, feature_names=X.columns, 
                      show=False, plot_size=(8, 6))
    plt.gcf().axes[0].set_title('SHAP Summary Plot', fontsize=16)
    plt.gcf().axes[0].set_ylabel('Features', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.pdf"), 
                bbox_inches='tight', dpi=600)
    plt.close()
    
    return output_dir


def regression_model_build(data_path, target_column='kd', output_dir=None, 
                           filter_column='Fit_Status', filter_value=0, 
                           test_size=0.05, random_state=42):
    """
    Build and train a regression model using Random Forest for predicting 
    continuous values like kd (dissociation constant).
    
    Args:
        data_path: Path to the Excel file containing the data
        target_column: Name of the target column to predict (default: 'kd')
        output_dir: Directory to save model and results (default: None, creates directory next to data file)
        filter_column: Column to use for filtering rows (default: 'Fit_Status')
        filter_value: Value to filter out from the filter_column (default: 0)
        test_size: Test set proportion (default: 0.05)
        random_state: Random state for reproducibility (default: 42)
        
    Returns:
        tuple: (output_directory, X_train, X_test, y_train, y_test, X, y)
    """
    # Check and create output directory
    if not data_path or not os.path.exists(data_path):
        raise ValueError(f"Data file not found: {data_path}")
    
    if output_dir is None:
        data_directory = os.path.dirname(data_path)
        output_dir = os.path.join(data_directory, 'regression_results')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    try:
        data = pd.read_excel(data_path)
    except Exception as e:
        raise ValueError(f"Error reading data file: {e}")
    
    # Filter data if needed
    if filter_column is not None and filter_column in data.columns:
        data = data[data[filter_column] != filter_value]
    
    # Define feature columns
    columns = [
        'P_Mean', 'P_Std', 'S_Mean', 'S_Std', 'dop_Mean', 'dop_Std',
        'x0SConst_Mean', 'x0SConst_Std', 'x90SConst_Mean', 'x90SConst_Std'
    ]
    
    # Rename columns to numerical for simplicity
    rename_dict = {old: str(i+1) for i, old in enumerate(columns)}
    
    # Extract features and target
    X = data[columns].rename(columns=rename_dict)
    y = data[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Define model and pipeline
    model = RandomForestRegressor(random_state=random_state)
    pipeline = Pipeline(steps=[('model', model)])
    
    # Define parameter grid for tuning
    param_grid = {
        'model__n_estimators': [1, 2, 3, 5, 10, 15, 20, 25, 30],
        'model__max_depth': [None, 1, 5, 10, 20, 30],
        'model__min_samples_split': [1, 2, 3, 5, 10, 15, 20]
    }
    
    # Perform grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, 
        scoring='neg_mean_squared_error', 
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_
    
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    
    # Make predictions
    y_train_pred = best_pipeline.predict(X_train)
    y_test_pred = best_pipeline.predict(X_test)
    y_all_pred = best_pipeline.predict(X)
    
    # Evaluate models
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)
    
    all_mse = mean_squared_error(y, y_all_pred)
    all_rmse = np.sqrt(all_mse)
    all_r2 = r2_score(y, y_all_pred)
    
    # Print evaluation metrics
    print(f"Training set: RMSE = {train_rmse:.4f}, R² = {train_r2:.4f}")
    print(f"Test set: RMSE = {test_rmse:.4f}, R² = {test_r2:.4f}")
    print(f"All data: RMSE = {all_rmse:.4f}, R² = {all_r2:.4f}")
    
    # Create scatter plot of predicted vs actual values
    plt.figure(figsize=(4, 3))
    plt.scatter(y.values, y_all_pred, color='blue', alpha=1, s=5, label='Data')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.title(f'True vs Predicted Values (R² = {all_r2:.4f})')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'true_vs_predicted_scatter.pdf'), dpi=600)
    plt.close()
    
    # Save results to Excel
    train_results = pd.DataFrame({
        'Train_True': y_train.values,
        'Train_Predicted': y_train_pred
    })
    test_results = pd.DataFrame({
        'Test_True': y_test.values,
        'Test_Predicted': y_test_pred
    })
    all_results = pd.DataFrame({
        'True': y.values,
        'Predicted': y_all_pred
    })
    
    # Save results
    train_results.to_excel(os.path.join(output_dir, 'train_true_vs_predicted.xlsx'), index=False)
    test_results.to_excel(os.path.join(output_dir, 'test_true_vs_predicted.xlsx'), index=False)
    all_results.to_excel(os.path.join(output_dir, 'all_true_vs_predicted.xlsx'), index=False)
    
    # Store evaluation metrics
    metrics = {
        'Train_MSE': train_mse,
        'Train_RMSE': train_rmse,
        'Train_R2': train_r2,
        'Test_MSE': test_mse,
        'Test_RMSE': test_rmse,
        'Test_R2': test_r2,
        'All_MSE': all_mse,
        'All_RMSE': all_rmse,
        'All_R2': all_r2
    }
    
    # Save metrics to text file
    with open(os.path.join(output_dir, 'evaluation_metrics.txt'), 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.10f}\n")
    
    # Also save as DataFrame
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_excel(os.path.join(output_dir, 'evaluation_metrics.xlsx'), index=False)
    
    # Save the model
    joblib.dump(best_pipeline, os.path.join(output_dir, 'regression_model.pkl'))
    
    return output_dir, X_train, X_test, y_train, y_test, X, y


def regression_shap_analysis(model_path, data_path, output_dir=None,
                             filter_column='Fit_Status', filter_value=0):
    """
    Perform SHAP analysis on a trained regression model.
    
    Args:
        model_path: Path to the saved regression model
        data_path: Path to the data file used for analysis
        output_dir: Directory to save analysis results (default: None, uses directory of model)
        filter_column: Column to use for filtering rows (default: 'Fit_Status')
        filter_value: Value to filter out from the filter_column (default: 0)
        
    Returns:
        str: Path to the output directory
    """
    # Check model path
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(model_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = joblib.load(model_path)
    
    # Get the actual model from pipeline if needed
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        model = model.named_steps['model']
    
    # Load data
    data = pd.read_excel(data_path)
    
    # Filter data if needed
    if filter_column is not None and filter_column in data.columns:
        data = data[data[filter_column] != filter_value]
    
    # Define feature columns
    columns = [
        'P_Mean', 'P_Std', 'S_Mean', 'S_Std', 'dop_Mean', 'dop_Std',
        'x0SConst_Mean', 'x0SConst_Std', 'x90SConst_Mean', 'x90SConst_Std'
    ]
    
    # Rename columns to numerical for simplicity
    rename_dict = {old: str(i+1) for i, old in enumerate(columns)}
    
    # Extract features and rename
    X = data[columns].rename(columns=rename_dict)
    
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    
    # Save SHAP values to Excel
    shap_values_df = pd.DataFrame(shap_values.values, columns=X.columns)
    shap_values_df.to_excel(os.path.join(output_dir, "shap_values_matrix.xlsx"), index=False)
    
    # Create summary plot
    plt.figure(figsize=(8, 6))
    plt.title('Feature Importance (SHAP)', fontsize=16)
    plt.xlabel('SHAP value', fontsize=16)
    plt.ylabel('Features', fontsize=16)
    shap.summary_plot(shap_values, X, feature_names=X.columns, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.pdf"), bbox_inches='tight', dpi=600)
    plt.close()
    
    # Create bar plot
    plt.figure(figsize=(10, 5))
    plt.title('Feature Importance (SHAP)', fontsize=16)
    plt.xlabel('Mean absolute SHAP value', fontsize=16)
    plt.ylabel('Features', fontsize=16)
    shap.summary_plot(shap_values, X, feature_names=X.columns, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_bar.pdf"), bbox_inches='tight', dpi=600)
    plt.close()
    
    return output_dir
