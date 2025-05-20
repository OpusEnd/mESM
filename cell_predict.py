#!/usr/bin/env python
"""
Command-line script for cell classification and regression analysis.

This script provides a command-line interface to the predict.py module,
allowing users to build classification and regression models for cell analysis,
as well as perform SHAP analysis on the trained models.
"""

import argparse
import os
import sys
from datetime import datetime

# Import local module
from src.predict import (
    classification_model_build, classification_shap_analysis,
    regression_model_build, regression_shap_analysis
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build classification and regression models for cell analysis",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Classification command
    classify_parser = subparsers.add_parser(
        "classify",
        help="Build a classification model for cell status"
    )
    classify_parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to Excel file containing cell data"
    )
    classify_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save model and results (default: creates directory next to data file)"
    )
    classify_parser.add_argument(
        "--test_size",
        type=float,
        default=0.3,
        help="Proportion of data to use for testing (default: 0.3)"
    )
    
    # SHAP analysis for classification command
    classify_shap_parser = subparsers.add_parser(
        "classify_shap",
        help="Perform SHAP analysis on a trained classification model"
    )
    classify_shap_parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved classification model (.pkl file)"
    )
    classify_shap_parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to Excel file containing cell data for analysis"
    )
    classify_shap_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save analysis results (default: uses directory of model)"
    )
    
    # Regression command
    regress_parser = subparsers.add_parser(
        "regress",
        help="Build a regression model for predicting continuous values (e.g., kd)"
    )
    regress_parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to Excel file containing cell data"
    )
    regress_parser.add_argument(
        "--target_column",
        type=str,
        default="kd",
        help="Name of the target column to predict (default: 'kd')"
    )
    regress_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save model and results (default: creates directory next to data file)"
    )
    regress_parser.add_argument(
        "--filter_column",
        type=str,
        default="Fit_Status",
        help="Column to use for filtering rows (default: 'Fit_Status')"
    )
    regress_parser.add_argument(
        "--filter_value",
        type=int,
        default=0,
        help="Value to filter out from the filter_column (default: 0)"
    )
    regress_parser.add_argument(
        "--test_size",
        type=float,
        default=0.05,
        help="Proportion of data to use for testing (default: 0.05)"
    )
    
    # SHAP analysis for regression command
    regress_shap_parser = subparsers.add_parser(
        "regress_shap",
        help="Perform SHAP analysis on a trained regression model"
    )
    regress_shap_parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved regression model (.pkl file)"
    )
    regress_shap_parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to Excel file containing cell data for analysis"
    )
    regress_shap_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save analysis results (default: uses directory of model)"
    )
    regress_shap_parser.add_argument(
        "--filter_column",
        type=str,
        default="Fit_Status",
        help="Column to use for filtering rows (default: 'Fit_Status')"
    )
    regress_shap_parser.add_argument(
        "--filter_value",
        type=int,
        default=0,
        help="Value to filter out from the filter_column (default: 0)"
    )
    
    if len(sys.argv) == 1:
        # No arguments provided, print help
        parser.print_help()
        sys.exit(1)
        
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.command == "classify":
        print("Building classification model...")
        
        # Run classification model building
        output_dir, X_train, X_test, y_train, y_test = classification_model_build(
            data_path=args.data_path,
            output_dir=args.output_dir,
            test_size=args.test_size
        )
        
        print(f"Classification model built successfully. Results saved to: {output_dir}")
        print(f"To perform SHAP analysis on this model, run:")
        print(f"python cell_predict.py classify_shap --model_path {os.path.join(output_dir, 'classification_model.pkl')} --data_path {args.data_path}")
        
    elif args.command == "classify_shap":
        print("Performing SHAP analysis on classification model...")
        
        # Run SHAP analysis for classification
        output_dir = classification_shap_analysis(
            model_path=args.model_path,
            data_path=args.data_path,
            output_dir=args.output_dir
        )
        
        print(f"SHAP analysis completed. Results saved to: {output_dir}")
        
    elif args.command == "regress":
        print("Building regression model...")
        
        # Run regression model building
        output_dir, X_train, X_test, y_train, y_test, X, y = regression_model_build(
            data_path=args.data_path,
            target_column=args.target_column,
            output_dir=args.output_dir,
            filter_column=args.filter_column,
            filter_value=args.filter_value,
            test_size=args.test_size
        )
        
        print(f"Regression model built successfully. Results saved to: {output_dir}")
        print(f"To perform SHAP analysis on this model, run:")
        print(f"python cell_predict.py regress_shap --model_path {os.path.join(output_dir, 'regression_model.pkl')} --data_path {args.data_path}")
        
    elif args.command == "regress_shap":
        print("Performing SHAP analysis on regression model...")
        
        # Run SHAP analysis for regression
        output_dir = regression_shap_analysis(
            model_path=args.model_path,
            data_path=args.data_path,
            output_dir=args.output_dir,
            filter_column=args.filter_column,
            filter_value=args.filter_value
        )
        
        print(f"SHAP analysis completed. Results saved to: {output_dir}")


if __name__ == "__main__":
    # Set matplotlib parameters for consistent output
    import matplotlib as mpl
    
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['pdf.fonttype'] = 42  # For editable text in PDFs
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['legend.fontsize'] = 8
    
    main()
