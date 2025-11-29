#!/usr/bin/env python
"""
Pharmaceutical Drug Sales Forecasting - Main Execution Script

This script runs the complete forecasting pipeline for predicting
pharmaceutical drug sales post-generic entry.

Scenarios:
    1. Scenario 1: Forecast 24 months post-generic entry from pre-entry data only
    2. Scenario 2: Forecast months 6-23 using first 6 months of post-entry data

Usage:
    python main.py [--data-path DATA_PATH] [--output-path OUTPUT_PATH]
"""

import argparse
import os
import sys
import warnings
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import DataPreprocessor
from src.eda import ExploratoryAnalysis
from src.feature_engineering import FeatureEngineer
from src.models import Scenario1Model, Scenario2Model
from src.evaluation import ForecastEvaluator
from src.insights import InsightsGenerator

warnings.filterwarnings('ignore')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Pharmaceutical Drug Sales Forecasting'
    )
    parser.add_argument(
        '--data-path', 
        type=str, 
        default='data/',
        help='Path to data directory'
    )
    parser.add_argument(
        '--output-path', 
        type=str, 
        default='outputs/',
        help='Path to output directory'
    )
    parser.add_argument(
        '--skip-eda',
        action='store_true',
        help='Skip exploratory data analysis'
    )
    parser.add_argument(
        '--scenario',
        type=int,
        choices=[1, 2],
        default=None,
        help='Run specific scenario (1 or 2). If not specified, runs both.'
    )
    
    return parser.parse_args()


def run_preprocessing(data_path: str) -> pd.DataFrame:
    """
    Run data preprocessing pipeline.
    
    Args:
        data_path: Path to data directory
        
    Returns:
        Preprocessed DataFrame
    """
    print("\n" + "="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)
    
    preprocessor = DataPreprocessor(data_path=data_path)
    df = preprocessor.preprocess_pipeline()
    
    print(f"\nPreprocessed data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df


def run_eda(df: pd.DataFrame, output_path: str) -> dict:
    """
    Run exploratory data analysis.
    
    Args:
        df: Preprocessed DataFrame
        output_path: Path for outputs
        
    Returns:
        Dictionary of summary statistics
    """
    print("\n" + "="*60)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    eda = ExploratoryAnalysis(output_path=output_path)
    stats = eda.run_full_eda(df)
    
    return stats


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run feature engineering pipeline.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        Engineered DataFrame
    """
    print("\n" + "="*60)
    print("STEP 3: FEATURE ENGINEERING")
    print("="*60)
    
    fe = FeatureEngineer()
    df_engineered = fe.run_feature_engineering_pipeline(df)
    
    return df_engineered


def run_scenario1(df: pd.DataFrame, output_path: str) -> dict:
    """
    Run Scenario 1: Forecast 24 months from pre-entry data.
    
    Args:
        df: Engineered DataFrame
        output_path: Path for outputs
        
    Returns:
        Dictionary of predictions and evaluations
    """
    print("\n" + "="*60)
    print("STEP 4A: SCENARIO 1 - FORECASTING FROM PRE-ENTRY DATA")
    print("="*60)
    
    # Train model
    model = Scenario1Model(model_type='ensemble')
    model.fit(df)
    
    # Make predictions for all drugs
    predictions = {}
    for drug_id in df['drug_id'].unique():
        pred = model.predict(df, drug_id=drug_id, n_months=24)
        predictions[drug_id] = pred
    
    # Evaluate predictions
    evaluator = ForecastEvaluator()
    
    # Get pre-entry averages for normalization
    pre_entry = df[df['months_postgx'] < 0]
    pre_entry_avgs = pre_entry.groupby('drug_id')['volume'].mean().to_dict()
    
    results = evaluator.evaluate_scenario1(df, predictions, pre_entry_avgs)
    
    # Save predictions
    pred_list = []
    for drug_id, pred in predictions.items():
        for month, vol in enumerate(pred):
            pred_list.append({
                'drug_id': drug_id,
                'months_postgx': month,
                'predicted_volume': vol,
                'scenario': 1
            })
    
    pred_df = pd.DataFrame(pred_list)
    pred_df.to_csv(f'{output_path}scenario1_predictions.csv', index=False)
    
    print(f"\nScenario 1 predictions saved to: {output_path}scenario1_predictions.csv")
    
    # Generate evaluation plots for sample drugs
    sample_drugs = list(predictions.keys())[:5]
    for drug_id in sample_drugs:
        actual_drug = df[(df['drug_id'] == drug_id) & 
                        (df['months_postgx'] >= 0) & 
                        (df['months_postgx'] < 24)].sort_values('months_postgx')
        if len(actual_drug) > 0:
            evaluator.plot_prediction_vs_actual(
                actual_drug['volume'].values,
                predictions[drug_id],
                drug_id,
                scenario=1,
                save_path=f'{output_path}scenario1_drug_{drug_id}_prediction.png'
            )
    
    # Plot error distribution
    evaluator.plot_error_distribution(results, scenario=1, 
                                       save_path=f'{output_path}scenario1_error_distribution.png')
    
    return {'predictions': predictions, 'results': results}


def run_scenario2(df: pd.DataFrame, output_path: str) -> dict:
    """
    Run Scenario 2: Forecast months 6-23 using first 6 months.
    
    Args:
        df: Engineered DataFrame
        output_path: Path for outputs
        
    Returns:
        Dictionary of predictions and evaluations
    """
    print("\n" + "="*60)
    print("STEP 4B: SCENARIO 2 - FORECASTING FROM FIRST 6 MONTHS")
    print("="*60)
    
    # Train model
    model = Scenario2Model(model_type='hybrid')
    model.fit(df)
    
    # Make predictions for all drugs
    predictions = {}
    post_entry = df[df['months_postgx'] >= 0]
    
    for drug_id in df['drug_id'].unique():
        # Get first 6 months of post-entry data
        drug_early = post_entry[(post_entry['drug_id'] == drug_id) & 
                                (post_entry['months_postgx'] < 6)].sort_values('months_postgx')
        
        if len(drug_early) >= 3:
            early_volumes = drug_early['volume'].values
            pred = model.predict(df, drug_id=drug_id, early_volumes=early_volumes)
            predictions[drug_id] = pred
    
    # Evaluate predictions
    evaluator = ForecastEvaluator()
    
    # Get pre-entry averages for normalization
    pre_entry = df[df['months_postgx'] < 0]
    pre_entry_avgs = pre_entry.groupby('drug_id')['volume'].mean().to_dict()
    
    results = evaluator.evaluate_scenario2(df, predictions, pre_entry_avgs)
    
    # Save predictions
    pred_list = []
    for drug_id, pred in predictions.items():
        for i, vol in enumerate(pred):
            pred_list.append({
                'drug_id': drug_id,
                'months_postgx': i + 6,  # Months 6-23
                'predicted_volume': vol,
                'scenario': 2
            })
    
    pred_df = pd.DataFrame(pred_list)
    pred_df.to_csv(f'{output_path}scenario2_predictions.csv', index=False)
    
    print(f"\nScenario 2 predictions saved to: {output_path}scenario2_predictions.csv")
    
    # Generate evaluation plots for sample drugs
    sample_drugs = list(predictions.keys())[:5]
    for drug_id in sample_drugs:
        actual_drug = df[(df['drug_id'] == drug_id) & 
                        (df['months_postgx'] >= 6) & 
                        (df['months_postgx'] < 24)].sort_values('months_postgx')
        if len(actual_drug) > 0:
            evaluator.plot_prediction_vs_actual(
                actual_drug['volume'].values,
                predictions[drug_id],
                drug_id,
                scenario=2,
                save_path=f'{output_path}scenario2_drug_{drug_id}_prediction.png'
            )
    
    # Plot error distribution
    evaluator.plot_error_distribution(results, scenario=2, 
                                       save_path=f'{output_path}scenario2_error_distribution.png')
    
    return {'predictions': predictions, 'results': results}


def run_insights(df: pd.DataFrame, output_path: str) -> dict:
    """
    Generate business insights.
    
    Args:
        df: Engineered DataFrame
        output_path: Path for outputs
        
    Returns:
        Dictionary of insights
    """
    print("\n" + "="*60)
    print("STEP 5: GENERATING BUSINESS INSIGHTS")
    print("="*60)
    
    insights_gen = InsightsGenerator(output_path=output_path)
    insights = insights_gen.run_full_insights_analysis(df)
    
    return insights


def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#    PHARMACEUTICAL DRUG SALES FORECASTING PIPELINE" + " "*8 + "#")
    print("#" + " "*58 + "#")
    print("#"*60)
    
    # Step 1: Preprocessing
    df = run_preprocessing(args.data_path)
    
    # Step 2: EDA (optional)
    if not args.skip_eda:
        run_eda(df, args.output_path)
    
    # Step 3: Feature Engineering
    df_engineered = run_feature_engineering(df)
    
    # Step 4: Model Training and Prediction
    results = {}
    
    if args.scenario is None or args.scenario == 1:
        results['scenario1'] = run_scenario1(df_engineered, args.output_path)
    
    if args.scenario is None or args.scenario == 2:
        results['scenario2'] = run_scenario2(df_engineered, args.output_path)
    
    # Step 5: Insights
    insights = run_insights(df_engineered, args.output_path)
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"\nAll outputs saved to: {args.output_path}")
    print("\nGenerated files:")
    for f in os.listdir(args.output_path):
        print(f"  - {f}")
    
    return results


if __name__ == '__main__':
    main()
