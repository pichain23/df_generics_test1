"""
Data Preprocessing Module

This module handles loading, cleaning, and preprocessing of pharmaceutical
drug sales data for forecasting models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Handles all data loading and preprocessing tasks for pharmaceutical 
    drug sales forecasting.
    """
    
    def __init__(self, data_path: str = 'data/'):
        """
        Initialize the preprocessor.
        
        Args:
            data_path: Path to the directory containing data files.
        """
        self.data_path = data_path
        self.df_volume = None
        self.df_generics = None
        self.df_medicine_info = None
        self.scaler = StandardScaler()
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all required datasets.
        
        Returns:
            Tuple of DataFrames: (df_volume, df_generics, df_medicine_info)
        """
        try:
            self.df_volume = pd.read_csv(f'{self.data_path}df_volume.csv')
            self.df_generics = pd.read_csv(f'{self.data_path}df_generics.csv')
            self.df_medicine_info = pd.read_csv(f'{self.data_path}df_medicine_info.csv')
            
            print(f"Loaded df_volume: {self.df_volume.shape}")
            print(f"Loaded df_generics: {self.df_generics.shape}")
            print(f"Loaded df_medicine_info: {self.df_medicine_info.shape}")
            
            return self.df_volume, self.df_generics, self.df_medicine_info
            
        except FileNotFoundError as e:
            print(f"Data files not found. Creating sample data for demonstration.")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create sample data for demonstration when actual data is not available.
        
        Returns:
            Tuple of sample DataFrames
        """
        np.random.seed(42)
        
        # Sample df_volume data
        n_drugs = 50
        n_months = 48  # 24 pre-entry + 24 post-entry
        
        volume_data = []
        for drug_id in range(1, n_drugs + 1):
            brand_name = f"Drug_{drug_id}"
            country = np.random.choice(['USA', 'UK', 'Germany', 'France', 'Japan'])
            
            # Pre-generic entry volumes (stable with some variance)
            pre_entry_base = np.random.uniform(10000, 100000)
            
            for month in range(-24, 24):  # months_postgx
                if month < 0:
                    # Pre-entry: relatively stable volumes
                    volume = pre_entry_base * (1 + np.random.uniform(-0.1, 0.1))
                else:
                    # Post-entry: declining volumes (erosion effect)
                    erosion_rate = np.random.uniform(0.02, 0.08)
                    volume = pre_entry_base * (1 - erosion_rate) ** (month + 1)
                    volume *= (1 + np.random.uniform(-0.15, 0.15))
                
                volume_data.append({
                    'drug_id': drug_id,
                    'brand_name': brand_name,
                    'country': country,
                    'months_postgx': month,
                    'volume': max(0, volume)
                })
        
        self.df_volume = pd.DataFrame(volume_data)
        
        # Sample df_generics data
        generics_data = []
        for drug_id in range(1, n_drugs + 1):
            for month in range(0, 24):
                n_gxs = min(month // 3 + 1, 8)  # Number of generic entrants
                generics_data.append({
                    'drug_id': drug_id,
                    'months_postgx': month,
                    'n_gxs': n_gxs,
                    'gx_market_share': min(0.1 * n_gxs, 0.7)
                })
        
        self.df_generics = pd.DataFrame(generics_data)
        
        # Sample df_medicine_info data
        therapeutic_areas = ['Oncology', 'Cardiology', 'Neurology', 'Immunology', 'Respiratory']
        medicine_data = []
        for drug_id in range(1, n_drugs + 1):
            medicine_data.append({
                'drug_id': drug_id,
                'brand_name': f"Drug_{drug_id}",
                'therapeutic_area': np.random.choice(therapeutic_areas),
                'hospital_rate': np.random.uniform(0.1, 0.9),
                'biological': np.random.choice([0, 1], p=[0.7, 0.3]),
                'small_molecule': np.random.choice([0, 1], p=[0.3, 0.7])
            })
        
        self.df_medicine_info = pd.DataFrame(medicine_data)
        
        # Save sample data
        self.df_volume.to_csv(f'{self.data_path}df_volume.csv', index=False)
        self.df_generics.to_csv(f'{self.data_path}df_generics.csv', index=False)
        self.df_medicine_info.to_csv(f'{self.data_path}df_medicine_info.csv', index=False)
        
        print("Created and saved sample data for demonstration.")
        
        return self.df_volume, self.df_generics, self.df_medicine_info
    
    def clean_data(self, df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
        """
        Clean a dataframe by handling missing values and duplicates.
        
        Args:
            df: Input DataFrame
            drop_duplicates: Whether to drop duplicate rows
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Drop duplicates
        if drop_duplicates:
            initial_len = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            print(f"Removed {initial_len - len(df_clean)} duplicate rows")
        
        return df_clean
    
    def handle_missing_values(self, df: pd.DataFrame, 
                               strategy: str = 'mean',
                               numeric_only: bool = True) -> pd.DataFrame:
        """
        Handle missing values in the dataframe.
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values ('mean', 'median', 'zero', 'ffill')
            numeric_only: Only apply to numeric columns
            
        Returns:
            DataFrame with handled missing values
        """
        df_filled = df.copy()
        
        # Get numeric columns
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        
        missing_before = df_filled.isnull().sum().sum()
        
        if strategy == 'mean':
            df_filled[numeric_cols] = df_filled[numeric_cols].fillna(
                df_filled[numeric_cols].mean()
            )
        elif strategy == 'median':
            df_filled[numeric_cols] = df_filled[numeric_cols].fillna(
                df_filled[numeric_cols].median()
            )
        elif strategy == 'zero':
            df_filled[numeric_cols] = df_filled[numeric_cols].fillna(0)
        elif strategy == 'ffill':
            df_filled = df_filled.ffill()
        
        # Fill remaining categorical missing values with mode
        if not numeric_only:
            for col in df_filled.select_dtypes(include=['object']).columns:
                df_filled[col] = df_filled[col].fillna(df_filled[col].mode().iloc[0] 
                                                        if not df_filled[col].mode().empty 
                                                        else 'Unknown')
        
        missing_after = df_filled.isnull().sum().sum()
        print(f"Handled {missing_before - missing_after} missing values using {strategy} strategy")
        
        return df_filled
    
    def scale_features(self, df: pd.DataFrame, 
                        columns: list,
                        scaler_type: str = 'standard') -> Tuple[pd.DataFrame, object]:
        """
        Scale specified features in the dataframe.
        
        Args:
            df: Input DataFrame
            columns: List of column names to scale
            scaler_type: Type of scaler ('standard' or 'minmax')
            
        Returns:
            Tuple of (scaled DataFrame, fitted scaler)
        """
        df_scaled = df.copy()
        
        if scaler_type == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        # Only scale columns that exist
        existing_cols = [col for col in columns if col in df_scaled.columns]
        
        if existing_cols:
            df_scaled[existing_cols] = scaler.fit_transform(df_scaled[existing_cols])
            print(f"Scaled {len(existing_cols)} columns using {scaler_type} scaler")
        
        return df_scaled, scaler
    
    def handle_imbalance(self, df: pd.DataFrame, 
                          target_col: str,
                          method: str = 'oversample') -> pd.DataFrame:
        """
        Handle data imbalance in the dataset.
        
        Args:
            df: Input DataFrame
            target_col: Column to balance on
            method: Method for handling imbalance ('oversample', 'undersample', 'smote')
            
        Returns:
            Balanced DataFrame
        """
        df_balanced = df.copy()
        
        if target_col not in df_balanced.columns:
            print(f"Target column {target_col} not found")
            return df_balanced
        
        value_counts = df_balanced[target_col].value_counts()
        
        if method == 'oversample':
            max_count = value_counts.max()
            dfs = []
            for value in value_counts.index:
                subset = df_balanced[df_balanced[target_col] == value]
                if len(subset) < max_count:
                    subset = subset.sample(n=max_count, replace=True, random_state=42)
                dfs.append(subset)
            df_balanced = pd.concat(dfs, ignore_index=True)
            
        elif method == 'undersample':
            min_count = value_counts.min()
            dfs = []
            for value in value_counts.index:
                subset = df_balanced[df_balanced[target_col] == value]
                if len(subset) > min_count:
                    subset = subset.sample(n=min_count, random_state=42)
                dfs.append(subset)
            df_balanced = pd.concat(dfs, ignore_index=True)
        
        print(f"Applied {method} to balance data on {target_col}")
        
        return df_balanced
    
    def merge_datasets(self) -> pd.DataFrame:
        """
        Merge all datasets into a single comprehensive dataframe.
        
        Returns:
            Merged DataFrame
        """
        if any(df is None for df in [self.df_volume, self.df_generics, self.df_medicine_info]):
            print("Please load data first using load_data()")
            return None
        
        # Merge volume with medicine info
        df_merged = self.df_volume.merge(
            self.df_medicine_info,
            on='drug_id',
            how='left',
            suffixes=('', '_info')
        )
        
        # Merge with generics data
        df_merged = df_merged.merge(
            self.df_generics,
            on=['drug_id', 'months_postgx'],
            how='left'
        )
        
        # Fill missing n_gxs for pre-generic entry months
        df_merged['n_gxs'] = df_merged['n_gxs'].fillna(0)
        if 'gx_market_share' in df_merged.columns:
            df_merged['gx_market_share'] = df_merged['gx_market_share'].fillna(0)
        
        print(f"Merged dataset shape: {df_merged.shape}")
        
        return df_merged
    
    def preprocess_pipeline(self) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline.
        
        Returns:
            Fully preprocessed DataFrame
        """
        # Load data
        self.load_data()
        
        # Clean each dataset
        self.df_volume = self.clean_data(self.df_volume)
        self.df_volume = self.handle_missing_values(self.df_volume)
        
        self.df_generics = self.clean_data(self.df_generics)
        self.df_generics = self.handle_missing_values(self.df_generics)
        
        self.df_medicine_info = self.clean_data(self.df_medicine_info)
        self.df_medicine_info = self.handle_missing_values(self.df_medicine_info, numeric_only=False)
        
        # Merge datasets
        df_merged = self.merge_datasets()
        
        return df_merged


def get_train_test_split(df: pd.DataFrame, 
                          test_months: int = 6,
                          target_col: str = 'volume') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets based on time.
    
    Args:
        df: Input DataFrame
        test_months: Number of months to use for testing
        target_col: Target column name
        
    Returns:
        Tuple of (train_df, test_df)
    """
    max_month = df['months_postgx'].max()
    train_cutoff = max_month - test_months
    
    train_df = df[df['months_postgx'] <= train_cutoff]
    test_df = df[df['months_postgx'] > train_cutoff]
    
    print(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows")
    
    return train_df, test_df
