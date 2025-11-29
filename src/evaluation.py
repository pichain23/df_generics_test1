"""
Evaluation Module

This module provides evaluation metrics for pharmaceutical drug sales
forecasting models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class ForecastEvaluator:
    """
    Evaluates forecast accuracy for pharmaceutical drug sales predictions.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.evaluation_results = {}
        
    def calculate_absolute_error(self, actual: np.ndarray, 
                                  predicted: np.ndarray) -> np.ndarray:
        """
        Calculate absolute error for each time period.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Array of absolute errors
        """
        return np.abs(actual - predicted)
    
    def calculate_normalized_error(self, actual: np.ndarray,
                                    predicted: np.ndarray,
                                    pre_entry_avg: float) -> np.ndarray:
        """
        Calculate absolute error normalized by average pre-entry monthly volumes.
        
        Normalized Error = |Actual - Predicted| / Pre-Entry Average
        
        Args:
            actual: Actual values
            predicted: Predicted values
            pre_entry_avg: Average pre-entry monthly volume
            
        Returns:
            Array of normalized errors
        """
        if pre_entry_avg <= 0:
            return np.zeros_like(actual)
        
        absolute_error = self.calculate_absolute_error(actual, predicted)
        return absolute_error / pre_entry_avg
    
    def calculate_monthly_normalized_error(self, df_actual: pd.DataFrame,
                                           df_predicted: pd.DataFrame,
                                           pre_entry_avgs: Dict[int, float]) -> pd.DataFrame:
        """
        Calculate monthly normalized error for all drugs.
        
        Args:
            df_actual: DataFrame with actual values
            df_predicted: DataFrame with predicted values
            pre_entry_avgs: Dictionary of pre-entry averages by drug_id
            
        Returns:
            DataFrame with monthly normalized errors
        """
        errors = []
        
        for drug_id in df_actual['drug_id'].unique():
            actual_drug = df_actual[df_actual['drug_id'] == drug_id].sort_values('months_postgx')
            pred_drug = df_predicted[df_predicted['drug_id'] == drug_id].sort_values('months_postgx')
            
            pre_avg = pre_entry_avgs.get(drug_id, 1)
            
            for month in actual_drug['months_postgx'].unique():
                actual_vol = actual_drug[actual_drug['months_postgx'] == month]['volume'].values
                pred_vol = pred_drug[pred_drug['months_postgx'] == month]['predicted_volume'].values
                
                if len(actual_vol) > 0 and len(pred_vol) > 0:
                    norm_error = abs(actual_vol[0] - pred_vol[0]) / pre_avg if pre_avg > 0 else 0
                    
                    errors.append({
                        'drug_id': drug_id,
                        'months_postgx': month,
                        'actual': actual_vol[0],
                        'predicted': pred_vol[0],
                        'absolute_error': abs(actual_vol[0] - pred_vol[0]),
                        'normalized_error': norm_error
                    })
        
        return pd.DataFrame(errors)
    
    def calculate_accumulated_error(self, actual: np.ndarray,
                                     predicted: np.ndarray,
                                     pre_entry_avg: float) -> float:
        """
        Calculate accumulated error normalized by pre-entry average.
        
        Accumulated Error = Σ|Actual - Predicted| / (N × Pre-Entry Average)
        
        Args:
            actual: Actual values
            predicted: Predicted values
            pre_entry_avg: Average pre-entry monthly volume
            
        Returns:
            Accumulated normalized error
        """
        if pre_entry_avg <= 0 or len(actual) == 0:
            return 0
        
        total_error = np.sum(np.abs(actual - predicted))
        return total_error / (len(actual) * pre_entry_avg)
    
    def calculate_mape(self, actual: np.ndarray,
                       predicted: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            MAPE value
        """
        # Avoid division by zero
        mask = actual != 0
        if not np.any(mask):
            return 0
        
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    
    def calculate_rmse(self, actual: np.ndarray,
                       predicted: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            RMSE value
        """
        return np.sqrt(np.mean((actual - predicted) ** 2))
    
    def calculate_r2(self, actual: np.ndarray,
                     predicted: np.ndarray) -> float:
        """
        Calculate R-squared (coefficient of determination).
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            R² value
        """
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        
        if ss_tot == 0:
            return 0
        
        return 1 - (ss_res / ss_tot)
    
    def evaluate_forecast(self, actual: np.ndarray,
                          predicted: np.ndarray,
                          pre_entry_avg: float,
                          drug_id: Optional[int] = None) -> Dict:
        """
        Comprehensive forecast evaluation for a single drug.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            pre_entry_avg: Average pre-entry monthly volume
            drug_id: Optional drug identifier
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'drug_id': drug_id,
            'n_months': len(actual),
            'mape': self.calculate_mape(actual, predicted),
            'rmse': self.calculate_rmse(actual, predicted),
            'r2': self.calculate_r2(actual, predicted),
            'accumulated_normalized_error': self.calculate_accumulated_error(
                actual, predicted, pre_entry_avg
            ),
            'mean_absolute_error': np.mean(np.abs(actual - predicted)),
            'mean_normalized_error': np.mean(
                self.calculate_normalized_error(actual, predicted, pre_entry_avg)
            )
        }
        
        return metrics
    
    def evaluate_scenario1(self, df_actual: pd.DataFrame,
                           predictions: Dict[int, np.ndarray],
                           pre_entry_avgs: Dict[int, float]) -> pd.DataFrame:
        """
        Evaluate Scenario 1 forecasts (24 months from pre-entry data).
        
        Args:
            df_actual: DataFrame with actual post-entry volumes
            predictions: Dictionary of predictions by drug_id
            pre_entry_avgs: Dictionary of pre-entry averages by drug_id
            
        Returns:
            DataFrame with evaluation metrics per drug
        """
        results = []
        
        for drug_id, pred in predictions.items():
            actual_drug = df_actual[
                (df_actual['drug_id'] == drug_id) & 
                (df_actual['months_postgx'] >= 0) &
                (df_actual['months_postgx'] < 24)
            ].sort_values('months_postgx')
            
            if len(actual_drug) > 0:
                actual = actual_drug['volume'].values
                predicted = pred[:len(actual)]
                pre_avg = pre_entry_avgs.get(drug_id, 1)
                
                metrics = self.evaluate_forecast(actual, predicted, pre_avg, drug_id)
                metrics['scenario'] = 1
                results.append(metrics)
        
        results_df = pd.DataFrame(results)
        self.evaluation_results['scenario1'] = results_df
        
        print("\n=== Scenario 1 Evaluation Summary ===")
        print(f"Number of drugs evaluated: {len(results_df)}")
        print(f"Mean MAPE: {results_df['mape'].mean():.2f}%")
        print(f"Mean RMSE: {results_df['rmse'].mean():.2f}")
        print(f"Mean R²: {results_df['r2'].mean():.4f}")
        print(f"Mean Accumulated Normalized Error: {results_df['accumulated_normalized_error'].mean():.4f}")
        
        return results_df
    
    def evaluate_scenario2(self, df_actual: pd.DataFrame,
                           predictions: Dict[int, np.ndarray],
                           pre_entry_avgs: Dict[int, float]) -> pd.DataFrame:
        """
        Evaluate Scenario 2 forecasts (months 6-23 from first 6 months).
        
        Args:
            df_actual: DataFrame with actual volumes
            predictions: Dictionary of predictions by drug_id (for months 6-23)
            pre_entry_avgs: Dictionary of pre-entry averages by drug_id
            
        Returns:
            DataFrame with evaluation metrics per drug
        """
        results = []
        
        for drug_id, pred in predictions.items():
            actual_drug = df_actual[
                (df_actual['drug_id'] == drug_id) & 
                (df_actual['months_postgx'] >= 6) &
                (df_actual['months_postgx'] < 24)
            ].sort_values('months_postgx')
            
            if len(actual_drug) > 0:
                actual = actual_drug['volume'].values
                predicted = pred[:len(actual)]
                pre_avg = pre_entry_avgs.get(drug_id, 1)
                
                metrics = self.evaluate_forecast(actual, predicted, pre_avg, drug_id)
                metrics['scenario'] = 2
                results.append(metrics)
        
        results_df = pd.DataFrame(results)
        self.evaluation_results['scenario2'] = results_df
        
        print("\n=== Scenario 2 Evaluation Summary ===")
        print(f"Number of drugs evaluated: {len(results_df)}")
        print(f"Mean MAPE: {results_df['mape'].mean():.2f}%")
        print(f"Mean RMSE: {results_df['rmse'].mean():.2f}")
        print(f"Mean R²: {results_df['r2'].mean():.4f}")
        print(f"Mean Accumulated Normalized Error: {results_df['accumulated_normalized_error'].mean():.4f}")
        
        return results_df
    
    def plot_prediction_vs_actual(self, actual: np.ndarray,
                                   predicted: np.ndarray,
                                   drug_id: int,
                                   scenario: int,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot predicted vs actual values.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            drug_id: Drug identifier
            scenario: Scenario number (1 or 2)
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Time series plot
        months = np.arange(len(actual))
        if scenario == 2:
            months = months + 6  # Adjust for scenario 2
        
        axes[0].plot(months, actual, 'b-o', label='Actual', linewidth=2, markersize=6)
        axes[0].plot(months, predicted[:len(actual)], 'r--s', label='Predicted', 
                    linewidth=2, markersize=6)
        axes[0].fill_between(months, actual, predicted[:len(actual)], alpha=0.3)
        axes[0].set_xlabel('Months Post-Generic Entry', fontsize=12)
        axes[0].set_ylabel('Volume', fontsize=12)
        axes[0].set_title(f'Drug {drug_id} - Scenario {scenario}: Actual vs Predicted', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[1].scatter(actual, predicted[:len(actual)], alpha=0.6, s=60)
        max_val = max(max(actual), max(predicted[:len(actual)]))
        axes[1].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Fit')
        axes[1].set_xlabel('Actual Volume', fontsize=12)
        axes[1].set_ylabel('Predicted Volume', fontsize=12)
        axes[1].set_title('Prediction Accuracy', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_error_distribution(self, results_df: pd.DataFrame,
                                 scenario: int,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of evaluation metrics.
        
        Args:
            results_df: DataFrame with evaluation results
            scenario: Scenario number
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # MAPE distribution
        axes[0, 0].hist(results_df['mape'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 0].axvline(results_df['mape'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {results_df["mape"].mean():.2f}%')
        axes[0, 0].set_xlabel('MAPE (%)', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Distribution of MAPE', fontsize=14)
        axes[0, 0].legend()
        
        # RMSE distribution
        axes[0, 1].hist(results_df['rmse'], bins=20, edgecolor='black', alpha=0.7, color='coral')
        axes[0, 1].axvline(results_df['rmse'].mean(), color='red', linestyle='--',
                          label=f'Mean: {results_df["rmse"].mean():.2f}')
        axes[0, 1].set_xlabel('RMSE', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title('Distribution of RMSE', fontsize=14)
        axes[0, 1].legend()
        
        # R² distribution
        axes[1, 0].hist(results_df['r2'], bins=20, edgecolor='black', alpha=0.7, color='green')
        axes[1, 0].axvline(results_df['r2'].mean(), color='red', linestyle='--',
                          label=f'Mean: {results_df["r2"].mean():.4f}')
        axes[1, 0].set_xlabel('R²', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Distribution of R²', fontsize=14)
        axes[1, 0].legend()
        
        # Normalized error distribution
        axes[1, 1].hist(results_df['accumulated_normalized_error'], bins=20, 
                       edgecolor='black', alpha=0.7, color='purple')
        axes[1, 1].axvline(results_df['accumulated_normalized_error'].mean(), 
                          color='red', linestyle='--',
                          label=f'Mean: {results_df["accumulated_normalized_error"].mean():.4f}')
        axes[1, 1].set_xlabel('Accumulated Normalized Error', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title('Distribution of Accumulated Normalized Error', fontsize=14)
        axes[1, 1].legend()
        
        plt.suptitle(f'Scenario {scenario} Error Distribution', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_evaluation_report(self, output_path: str = 'outputs/') -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_path: Path to save report
            
        Returns:
            Dictionary containing summary statistics
        """
        report = {}
        
        for scenario, results_df in self.evaluation_results.items():
            if len(results_df) > 0:
                report[scenario] = {
                    'n_drugs': len(results_df),
                    'metrics': {
                        'mape': {
                            'mean': results_df['mape'].mean(),
                            'std': results_df['mape'].std(),
                            'min': results_df['mape'].min(),
                            'max': results_df['mape'].max()
                        },
                        'rmse': {
                            'mean': results_df['rmse'].mean(),
                            'std': results_df['rmse'].std(),
                            'min': results_df['rmse'].min(),
                            'max': results_df['rmse'].max()
                        },
                        'r2': {
                            'mean': results_df['r2'].mean(),
                            'std': results_df['r2'].std(),
                            'min': results_df['r2'].min(),
                            'max': results_df['r2'].max()
                        },
                        'accumulated_normalized_error': {
                            'mean': results_df['accumulated_normalized_error'].mean(),
                            'std': results_df['accumulated_normalized_error'].std(),
                            'min': results_df['accumulated_normalized_error'].min(),
                            'max': results_df['accumulated_normalized_error'].max()
                        }
                    }
                }
                
                # Save results to CSV
                results_df.to_csv(f'{output_path}{scenario}_results.csv', index=False)
        
        print("\n=== Evaluation Report Generated ===")
        for scenario, data in report.items():
            print(f"\n{scenario.upper()}:")
            print(f"  Drugs evaluated: {data['n_drugs']}")
            for metric, values in data['metrics'].items():
                print(f"  {metric}: mean={values['mean']:.4f}, std={values['std']:.4f}")
        
        return report
