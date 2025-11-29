"""
Feature Engineering Module

This module handles feature engineering for pharmaceutical drug sales forecasting,
including erosion bucket derivation and aggregated metrics computation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import LabelEncoder


class FeatureEngineer:
    """
    Handles feature engineering for pharmaceutical drug sales forecasting.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.label_encoders = {}
        self.erosion_buckets = None
        self.feature_stats = {}
    
    def calculate_erosion(self, df: pd.DataFrame, 
                          pre_entry_window: int = 12) -> pd.DataFrame:
        """
        Calculate erosion metrics for each drug.
        
        Erosion is calculated as:
        erosion = 1 - (post_entry_volume / pre_entry_avg_volume)
        
        Args:
            df: DataFrame with volume and months_postgx columns
            pre_entry_window: Number of pre-entry months to use for baseline
            
        Returns:
            DataFrame with erosion metrics
        """
        df_with_erosion = df.copy()
        
        # Calculate pre-entry average volume per drug
        pre_entry = df[df['months_postgx'] < 0]
        
        # Use last N months before entry if available
        if pre_entry_window:
            pre_entry = pre_entry[pre_entry['months_postgx'] >= -pre_entry_window]
        
        pre_avg = pre_entry.groupby('drug_id')['volume'].mean().reset_index()
        pre_avg.columns = ['drug_id', 'pre_entry_avg_volume']
        
        # Merge pre-entry average
        df_with_erosion = df_with_erosion.merge(pre_avg, on='drug_id', how='left')
        
        # Calculate erosion (only for post-entry months)
        df_with_erosion['erosion'] = np.where(
            (df_with_erosion['months_postgx'] >= 0) & (df_with_erosion['pre_entry_avg_volume'] > 0),
            1 - (df_with_erosion['volume'] / df_with_erosion['pre_entry_avg_volume']),
            np.nan
        )
        
        # Clip erosion between 0 and 1
        df_with_erosion['erosion'] = df_with_erosion['erosion'].clip(0, 1)
        
        # Calculate cumulative erosion
        df_with_erosion = df_with_erosion.sort_values(['drug_id', 'months_postgx'])
        df_with_erosion['cumulative_erosion'] = df_with_erosion.groupby('drug_id')['erosion'].cumsum()
        
        print(f"Calculated erosion metrics for {df_with_erosion['drug_id'].nunique()} drugs")
        
        return df_with_erosion
    
    def derive_erosion_buckets(self, df: pd.DataFrame, 
                                n_buckets: int = 5,
                                evaluation_month: int = 12) -> pd.DataFrame:
        """
        Derive erosion buckets from volume trends.
        
        Buckets are determined based on erosion levels at a specific evaluation month:
        - Bucket 1: Low erosion (< 20%)
        - Bucket 2: Low-Medium erosion (20-40%)
        - Bucket 3: Medium erosion (40-60%)
        - Bucket 4: Medium-High erosion (60-80%)
        - Bucket 5: High erosion (> 80%)
        
        Args:
            df: DataFrame with erosion metrics
            n_buckets: Number of buckets to create
            evaluation_month: Month at which to evaluate erosion for bucketing
            
        Returns:
            DataFrame with erosion bucket assignments
        """
        df_bucketed = df.copy()
        
        # Get erosion at evaluation month
        eval_data = df_bucketed[df_bucketed['months_postgx'] == evaluation_month]
        
        if len(eval_data) == 0:
            # Use maximum available month
            max_month = df_bucketed[df_bucketed['months_postgx'] >= 0]['months_postgx'].max()
            eval_data = df_bucketed[df_bucketed['months_postgx'] == max_month]
            print(f"Using month {max_month} for bucket evaluation (month {evaluation_month} not available)")
        
        # Calculate bucket thresholds
        bucket_thresholds = np.linspace(0, 1, n_buckets + 1)
        bucket_labels = [f'Bucket_{i+1}' for i in range(n_buckets)]
        
        # Assign buckets based on erosion level
        drug_buckets = eval_data.groupby('drug_id')['erosion'].mean().reset_index()
        drug_buckets['erosion_bucket'] = pd.cut(
            drug_buckets['erosion'],
            bins=bucket_thresholds,
            labels=bucket_labels,
            include_lowest=True
        )
        
        # Handle NaN erosion values
        drug_buckets['erosion_bucket'] = drug_buckets['erosion_bucket'].fillna('Bucket_3')
        
        # Merge buckets back to main dataframe
        df_bucketed = df_bucketed.merge(
            drug_buckets[['drug_id', 'erosion_bucket']],
            on='drug_id',
            how='left'
        )
        
        # Store bucket statistics
        self.erosion_buckets = drug_buckets
        
        bucket_counts = df_bucketed.groupby('erosion_bucket', observed=False)['drug_id'].nunique()
        print(f"Erosion bucket distribution:\n{bucket_counts}")
        
        return df_bucketed
    
    def compute_mean_erosion(self, df: pd.DataFrame, 
                              group_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compute mean erosion and other aggregated metrics (Equation 1.1).
        
        Mean Erosion = (1/N) * Σ(erosion_i)
        
        Args:
            df: DataFrame with erosion metrics
            group_cols: Columns to group by for aggregation
            
        Returns:
            DataFrame with aggregated erosion metrics
        """
        if group_cols is None:
            group_cols = ['drug_id']
        
        # Ensure we only look at post-entry data
        post_entry = df[df['months_postgx'] >= 0].copy()
        
        # Compute aggregated metrics
        agg_metrics = post_entry.groupby(group_cols).agg({
            'erosion': ['mean', 'std', 'min', 'max'],
            'volume': ['mean', 'std', 'sum'],
            'months_postgx': 'count'
        }).reset_index()
        
        # Flatten column names
        agg_metrics.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                                for col in agg_metrics.columns]
        
        # Rename columns for clarity
        rename_map = {
            'erosion_mean': 'mean_erosion',
            'erosion_std': 'erosion_std',
            'erosion_min': 'min_erosion',
            'erosion_max': 'max_erosion',
            'volume_mean': 'mean_volume',
            'volume_std': 'volume_std',
            'volume_sum': 'total_volume',
            'months_postgx_count': 'n_months'
        }
        agg_metrics = agg_metrics.rename(columns=rename_map)
        
        self.feature_stats['mean_erosion'] = agg_metrics['mean_erosion'].describe()
        
        print(f"Computed aggregated metrics for {len(agg_metrics)} groups")
        
        return agg_metrics
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features for forecasting.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional time features
        """
        df_time = df.copy()
        
        # Months from entry
        df_time['months_from_entry'] = df_time['months_postgx'].abs()
        
        # Time phase indicators
        df_time['pre_entry'] = (df_time['months_postgx'] < 0).astype(int)
        df_time['early_post_entry'] = ((df_time['months_postgx'] >= 0) & 
                                        (df_time['months_postgx'] < 6)).astype(int)
        df_time['mid_post_entry'] = ((df_time['months_postgx'] >= 6) & 
                                      (df_time['months_postgx'] < 12)).astype(int)
        df_time['late_post_entry'] = (df_time['months_postgx'] >= 12).astype(int)
        
        # Quarter indicators (post-entry)
        df_time['post_entry_quarter'] = np.where(
            df_time['months_postgx'] >= 0,
            (df_time['months_postgx'] // 3) + 1,
            0
        )
        
        # Year indicator (post-entry)
        df_time['post_entry_year'] = np.where(
            df_time['months_postgx'] >= 0,
            (df_time['months_postgx'] // 12) + 1,
            0
        )
        
        print("Created time-based features")
        
        return df_time
    
    def create_lag_features(self, df: pd.DataFrame, 
                             lag_periods: List[int] = [1, 3, 6, 12],
                             target_col: str = 'volume') -> pd.DataFrame:
        """
        Create lagged features for time series forecasting.
        
        Args:
            df: Input DataFrame
            lag_periods: List of lag periods to create
            target_col: Target column to create lags for
            
        Returns:
            DataFrame with lag features
        """
        df_lagged = df.copy()
        df_lagged = df_lagged.sort_values(['drug_id', 'months_postgx'])
        
        for lag in lag_periods:
            df_lagged[f'{target_col}_lag_{lag}'] = df_lagged.groupby('drug_id')[target_col].shift(lag)
        
        # Create rolling features
        df_lagged[f'{target_col}_rolling_mean_3'] = df_lagged.groupby('drug_id')[target_col].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df_lagged[f'{target_col}_rolling_mean_6'] = df_lagged.groupby('drug_id')[target_col].transform(
            lambda x: x.rolling(window=6, min_periods=1).mean()
        )
        
        # Create momentum features
        df_lagged[f'{target_col}_momentum_3'] = df_lagged[target_col] - df_lagged[f'{target_col}_lag_3']
        df_lagged[f'{target_col}_momentum_6'] = df_lagged[target_col] - df_lagged[f'{target_col}_lag_6']
        
        print(f"Created lag features with periods: {lag_periods}")
        
        return df_lagged
    
    def encode_categorical_features(self, df: pd.DataFrame,
                                     columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.
        
        Args:
            df: Input DataFrame
            columns: List of columns to encode (None = auto-detect)
            
        Returns:
            DataFrame with encoded features
        """
        df_encoded = df.copy()
        
        if columns is None:
            columns = df_encoded.select_dtypes(include=['object']).columns.tolist()
        
        for col in columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                # Handle NaN values
                df_encoded[f'{col}_encoded'] = df_encoded[col].fillna('Unknown')
                df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[f'{col}_encoded'])
                self.label_encoders[col] = le
        
        print(f"Encoded {len(columns)} categorical features")
        
        return df_encoded
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        df_interact = df.copy()
        
        # Volume-based interactions
        if 'volume' in df_interact.columns and 'n_gxs' in df_interact.columns:
            df_interact['volume_per_gx'] = np.where(
                df_interact['n_gxs'] > 0,
                df_interact['volume'] / df_interact['n_gxs'],
                df_interact['volume']
            )
        
        # Hospital rate interactions
        if 'hospital_rate' in df_interact.columns and 'volume' in df_interact.columns:
            df_interact['hospital_volume'] = df_interact['hospital_rate'] * df_interact['volume']
        
        # Biological/Small molecule interactions
        if 'biological' in df_interact.columns and 'erosion' in df_interact.columns:
            df_interact['bio_erosion'] = df_interact['biological'] * df_interact['erosion']
        
        if 'small_molecule' in df_interact.columns and 'erosion' in df_interact.columns:
            df_interact['sm_erosion'] = df_interact['small_molecule'] * df_interact['erosion']
        
        # Time-based interactions
        if 'months_postgx' in df_interact.columns and 'n_gxs' in df_interact.columns:
            df_interact['time_gx_interaction'] = df_interact['months_postgx'] * df_interact['n_gxs']
        
        print("Created interaction features")
        
        return df_interact
    
    def prepare_features_for_modeling(self, df: pd.DataFrame,
                                       target_col: str = 'volume') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare final feature matrix for modeling.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of (feature_df, target_series)
        """
        # Define feature columns (exclude identifiers and target)
        exclude_cols = ['drug_id', 'brand_name', 'brand_name_info', target_col, 
                        'country', 'therapeutic_area']
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols 
                       and df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy() if target_col in df.columns else None
        
        # Fill any remaining NaN values
        X = X.fillna(0)
        
        print(f"Prepared {len(feature_cols)} features for modeling")
        print(f"Feature matrix shape: {X.shape}")
        
        return X, y
    
    def run_feature_engineering_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Fully engineered DataFrame
        """
        print("\n" + "="*50)
        print("RUNNING FEATURE ENGINEERING PIPELINE")
        print("="*50 + "\n")
        
        # Step 1: Calculate erosion
        df_engineered = self.calculate_erosion(df)
        
        # Step 2: Derive erosion buckets
        df_engineered = self.derive_erosion_buckets(df_engineered)
        
        # Step 3: Create time features
        df_engineered = self.create_time_features(df_engineered)
        
        # Step 4: Create lag features
        df_engineered = self.create_lag_features(df_engineered)
        
        # Step 5: Encode categorical features
        df_engineered = self.encode_categorical_features(df_engineered)
        
        # Step 6: Create interaction features
        df_engineered = self.create_interaction_features(df_engineered)
        
        print(f"\nFinal engineered dataset shape: {df_engineered.shape}")
        
        return df_engineered
