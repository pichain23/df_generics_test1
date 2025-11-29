"""
Model Building Module

This module contains time-series forecasting models for pharmaceutical
drug sales prediction in both Scenario 1 and Scenario 2.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

# Suppress specific warnings from statsmodels and sklearn during model fitting
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')


class BaseForecaster:
    """Base class for forecasting models."""
    
    def __init__(self, name: str = "BaseForecaster"):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseForecaster':
        """Fit the model."""
        raise NotImplementedError
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        raise NotImplementedError
        
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'name': self.name,
            'is_fitted': self.is_fitted
        }


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA-based forecaster for time series data.
    """
    
    def __init__(self, order: Tuple[int, int, int] = (2, 1, 2)):
        """
        Initialize ARIMA forecaster.
        
        Args:
            order: ARIMA order (p, d, q)
        """
        super().__init__(name="ARIMA")
        self.order = order
        self.fitted_models = {}
        
    def fit(self, df: pd.DataFrame, 
            target_col: str = 'volume',
            group_col: str = 'drug_id') -> 'ARIMAForecaster':
        """
        Fit ARIMA models for each group (drug).
        
        Args:
            df: Training DataFrame
            target_col: Target column name
            group_col: Column to group by
            
        Returns:
            self
        """
        groups = df[group_col].unique()
        
        for group_id in groups:
            try:
                group_data = df[df[group_col] == group_id].sort_values('months_postgx')
                series = group_data[target_col].values
                
                if len(series) >= 12:  # Minimum data points needed
                    model = ARIMA(series, order=self.order)
                    self.fitted_models[group_id] = model.fit()
            except Exception:
                continue
        
        self.is_fitted = True
        print(f"Fitted ARIMA models for {len(self.fitted_models)} groups")
        
        return self
    
    def predict(self, df: pd.DataFrame,
                n_periods: int = 24,
                group_col: str = 'drug_id') -> Dict[int, np.ndarray]:
        """
        Make forecasts for each group.
        
        Args:
            df: DataFrame for reference
            n_periods: Number of periods to forecast
            group_col: Column to group by
            
        Returns:
            Dictionary of forecasts per group
        """
        predictions = {}
        
        for group_id, model in self.fitted_models.items():
            try:
                forecast = model.forecast(steps=n_periods)
                predictions[group_id] = forecast
            except Exception:
                continue
        
        return predictions


class EnsembleForecaster(BaseForecaster):
    """
    Ensemble forecaster combining multiple ML models.
    """
    
    def __init__(self, models: Optional[List] = None,
                 rf_n_estimators: int = 100,
                 rf_max_depth: int = 10,
                 gb_n_estimators: int = 100,
                 gb_max_depth: int = 5,
                 ridge_alpha: float = 1.0,
                 elasticnet_alpha: float = 0.5,
                 elasticnet_l1_ratio: float = 0.5,
                 random_state: int = 42):
        """
        Initialize ensemble forecaster.
        
        Args:
            models: List of models to ensemble
            rf_n_estimators: Number of trees for Random Forest
            rf_max_depth: Max depth for Random Forest
            gb_n_estimators: Number of trees for Gradient Boosting
            gb_max_depth: Max depth for Gradient Boosting
            ridge_alpha: Regularization strength for Ridge
            elasticnet_alpha: Regularization strength for ElasticNet
            elasticnet_l1_ratio: L1 ratio for ElasticNet
            random_state: Random seed for reproducibility
        """
        super().__init__(name="Ensemble")
        
        if models is None:
            self.models = [
                ('rf', RandomForestRegressor(n_estimators=rf_n_estimators, 
                                             max_depth=rf_max_depth, 
                                             random_state=random_state)),
                ('gb', GradientBoostingRegressor(n_estimators=gb_n_estimators, 
                                                 max_depth=gb_max_depth, 
                                                 random_state=random_state)),
                ('ridge', Ridge(alpha=ridge_alpha)),
                ('elasticnet', ElasticNet(alpha=elasticnet_alpha, 
                                          l1_ratio=elasticnet_l1_ratio, 
                                          random_state=random_state))
            ]
        else:
            self.models = models
            
        self.fitted_models = []
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnsembleForecaster':
        """
        Fit all ensemble models.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            self
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        self.fitted_models = []
        for name, model in self.models:
            try:
                fitted = model.fit(X_scaled, y)
                self.fitted_models.append((name, fitted))
                print(f"Fitted {name}")
            except Exception as e:
                print(f"Failed to fit {name}: {e}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions (average of all models).
        
        Args:
            X: Feature matrix
            
        Returns:
            Ensemble predictions
        """
        X_scaled = self.scaler.transform(X)
        
        predictions = []
        for name, model in self.fitted_models:
            try:
                pred = model.predict(X_scaled)
                predictions.append(pred)
            except Exception:
                continue
        
        if predictions:
            return np.mean(predictions, axis=0)
        else:
            return np.zeros(len(X))


class Scenario1Model:
    """
    Model for Scenario 1: Forecast 24 months post-generic entry from pre-entry data.
    
    This model uses historical pre-entry data and drug characteristics to 
    forecast volume for 24 months after generic entry.
    """
    
    def __init__(self, model_type: str = 'ensemble'):
        """
        Initialize Scenario 1 model.
        
        Args:
            model_type: Type of model ('arima', 'ensemble', 'hybrid')
        """
        self.model_type = model_type
        self.ensemble = EnsembleForecaster()
        self.arima = ARIMAForecaster()
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.pre_entry_stats = {}
        
    def _prepare_pre_entry_features(self, df: pd.DataFrame, 
                                     drug_id: int) -> pd.DataFrame:
        """
        Prepare features from pre-entry data for a specific drug.
        
        Args:
            df: Full DataFrame
            drug_id: Drug ID to prepare features for
            
        Returns:
            Feature DataFrame
        """
        pre_entry = df[(df['drug_id'] == drug_id) & (df['months_postgx'] < 0)]
        
        if len(pre_entry) == 0:
            return None
        
        features = {
            'drug_id': drug_id,
            'pre_entry_mean_volume': pre_entry['volume'].mean(),
            'pre_entry_std_volume': pre_entry['volume'].std(),
            'pre_entry_max_volume': pre_entry['volume'].max(),
            'pre_entry_min_volume': pre_entry['volume'].min(),
            'pre_entry_trend': self._calculate_trend(pre_entry['volume'].values)
        }
        
        # Add drug characteristics if available
        for col in ['hospital_rate', 'biological', 'small_molecule']:
            if col in pre_entry.columns:
                features[col] = pre_entry[col].iloc[0]
        
        return pd.DataFrame([features])
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate linear trend of values."""
        if len(values) < 2:
            return 0
        x = np.arange(len(values))
        try:
            slope = np.polyfit(x, values, 1)[0]
            return slope
        except Exception:
            return 0
    
    def fit(self, df: pd.DataFrame) -> 'Scenario1Model':
        """
        Fit the Scenario 1 model.
        
        Args:
            df: Full training DataFrame
            
        Returns:
            self
        """
        print("\n" + "="*50)
        print("TRAINING SCENARIO 1 MODEL")
        print("="*50)
        
        # Separate pre and post entry data
        pre_entry = df[df['months_postgx'] < 0]
        post_entry = df[df['months_postgx'] >= 0]
        
        # Calculate pre-entry statistics for each drug
        self.pre_entry_stats = pre_entry.groupby('drug_id')['volume'].agg(
            ['mean', 'std', 'max', 'min']
        ).to_dict('index')
        
        if self.model_type in ['ensemble', 'hybrid']:
            # Prepare features and targets for ensemble model
            feature_list = []
            target_list = []
            
            for drug_id in df['drug_id'].unique():
                features = self._prepare_pre_entry_features(df, drug_id)
                if features is not None:
                    drug_post = post_entry[post_entry['drug_id'] == drug_id]
                    if len(drug_post) > 0:
                        # Add target (average post-entry volume)
                        for _, row in drug_post.iterrows():
                            feat_row = features.copy()
                            feat_row['months_postgx'] = row['months_postgx']
                            feature_list.append(feat_row)
                            target_list.append(row['volume'])
            
            if feature_list:
                X = pd.concat(feature_list, ignore_index=True)
                y = pd.Series(target_list)
                
                # Remove non-numeric columns
                X_numeric = X.select_dtypes(include=[np.number])
                X_numeric = X_numeric.fillna(0)
                
                self.ensemble.fit(X_numeric, y)
        
        if self.model_type in ['arima', 'hybrid']:
            # Fit ARIMA on historical data
            self.arima.fit(df, target_col='volume', group_col='drug_id')
        
        self.is_fitted = True
        print("Scenario 1 model training complete")
        
        return self
    
    def predict(self, df: pd.DataFrame, 
                drug_id: int,
                n_months: int = 24) -> np.ndarray:
        """
        Predict 24 months post-generic entry for a drug.
        
        Args:
            df: Reference DataFrame
            drug_id: Drug to predict for
            n_months: Number of months to predict
            
        Returns:
            Array of predicted volumes
        """
        predictions = np.zeros(n_months)
        
        if self.model_type in ['ensemble', 'hybrid']:
            features = self._prepare_pre_entry_features(df, drug_id)
            if features is not None:
                for month in range(n_months):
                    feat_row = features.copy()
                    feat_row['months_postgx'] = month
                    X_numeric = feat_row.select_dtypes(include=[np.number])
                    X_numeric = X_numeric.fillna(0)
                    pred = self.ensemble.predict(X_numeric)
                    predictions[month] = max(0, pred[0])
        
        if self.model_type == 'arima':
            arima_preds = self.arima.predict(df, n_periods=n_months)
            if drug_id in arima_preds:
                predictions = np.maximum(0, arima_preds[drug_id])
        
        if self.model_type == 'hybrid':
            # Combine ensemble and ARIMA predictions
            arima_preds = self.arima.predict(df, n_periods=n_months)
            if drug_id in arima_preds:
                arima_pred = np.maximum(0, arima_preds[drug_id])
                predictions = 0.6 * predictions + 0.4 * arima_pred
        
        return predictions


class Scenario2Model:
    """
    Model for Scenario 2: Forecast months 6-23 using first 6 months of post-entry data.
    
    This model leverages early post-entry observations to refine forecasts.
    """
    
    def __init__(self, model_type: str = 'hybrid'):
        """
        Initialize Scenario 2 model.
        
        Args:
            model_type: Type of model ('ensemble', 'trend', 'hybrid')
        """
        self.model_type = model_type
        self.ensemble = EnsembleForecaster()
        self.is_fitted = False
        self.early_erosion_patterns = {}
        
    def _calculate_early_erosion_rate(self, volumes: np.ndarray, 
                                       pre_avg: float) -> float:
        """
        Calculate erosion rate from first 6 months.
        
        Args:
            volumes: First 6 months of post-entry volumes
            pre_avg: Pre-entry average volume
            
        Returns:
            Estimated erosion rate
        """
        if pre_avg <= 0 or len(volumes) == 0:
            return 0
        
        # Calculate average monthly erosion
        erosion = 1 - (np.mean(volumes) / pre_avg)
        return max(0, min(1, erosion))
    
    def _extrapolate_trend(self, early_volumes: np.ndarray, 
                            n_months: int = 18) -> np.ndarray:
        """
        Extrapolate volume trend from early months.
        
        Args:
            early_volumes: First 6 months of volumes
            n_months: Number of months to extrapolate
            
        Returns:
            Extrapolated volumes
        """
        if len(early_volumes) < 2:
            return np.full(n_months, early_volumes[-1] if len(early_volumes) > 0 else 0)
        
        # Fit exponential decay
        x = np.arange(len(early_volumes))
        try:
            # Log transform for exponential fit
            log_volumes = np.log(np.maximum(early_volumes, 1))
            coeffs = np.polyfit(x, log_volumes, 1)
            
            # Extrapolate
            future_x = np.arange(len(early_volumes), len(early_volumes) + n_months)
            predictions = np.exp(coeffs[0] * future_x + coeffs[1])
            
            return np.maximum(0, predictions)
        except Exception:
            # Fallback to linear extrapolation
            last_value = early_volumes[-1]
            trend = (early_volumes[-1] - early_volumes[0]) / len(early_volumes)
            return np.maximum(0, [last_value + trend * i for i in range(1, n_months + 1)])
    
    def fit(self, df: pd.DataFrame) -> 'Scenario2Model':
        """
        Fit the Scenario 2 model.
        
        Args:
            df: Full training DataFrame
            
        Returns:
            self
        """
        print("\n" + "="*50)
        print("TRAINING SCENARIO 2 MODEL")
        print("="*50)
        
        # Get pre-entry statistics
        pre_entry = df[df['months_postgx'] < 0]
        pre_avg = pre_entry.groupby('drug_id')['volume'].mean()
        
        # Analyze early post-entry patterns (months 0-5)
        early_post = df[(df['months_postgx'] >= 0) & (df['months_postgx'] < 6)]
        late_post = df[(df['months_postgx'] >= 6) & (df['months_postgx'] < 24)]
        
        # Build features from early post-entry data
        feature_list = []
        target_list = []
        
        for drug_id in df['drug_id'].unique():
            drug_early = early_post[early_post['drug_id'] == drug_id].sort_values('months_postgx')
            drug_late = late_post[late_post['drug_id'] == drug_id].sort_values('months_postgx')
            
            if len(drug_early) >= 3 and len(drug_late) > 0:
                early_volumes = drug_early['volume'].values
                
                features = {
                    'drug_id': drug_id,
                    'early_mean_volume': np.mean(early_volumes),
                    'early_std_volume': np.std(early_volumes),
                    'early_min_volume': np.min(early_volumes),
                    'early_max_volume': np.max(early_volumes),
                    'early_trend': (early_volumes[-1] - early_volumes[0]) / len(early_volumes),
                    'early_erosion_rate': self._calculate_early_erosion_rate(
                        early_volumes, pre_avg.get(drug_id, 1)
                    ),
                    'pre_entry_avg': pre_avg.get(drug_id, 0)
                }
                
                # Add drug characteristics
                for col in ['hospital_rate', 'biological', 'small_molecule', 'n_gxs']:
                    if col in drug_early.columns:
                        features[col] = drug_early[col].iloc[-1]
                
                # Create training samples for each future month
                for _, row in drug_late.iterrows():
                    feat_row = features.copy()
                    feat_row['target_month'] = row['months_postgx']
                    feat_row['months_ahead'] = row['months_postgx'] - 5
                    feature_list.append(feat_row)
                    target_list.append(row['volume'])
        
        if feature_list:
            X = pd.DataFrame(feature_list)
            y = pd.Series(target_list)
            
            X_numeric = X.select_dtypes(include=[np.number])
            X_numeric = X_numeric.fillna(0)
            
            self.ensemble.fit(X_numeric, y)
        
        self.is_fitted = True
        print("Scenario 2 model training complete")
        
        return self
    
    def predict(self, df: pd.DataFrame,
                drug_id: int,
                early_volumes: np.ndarray) -> np.ndarray:
        """
        Predict months 6-23 using first 6 months of data.
        
        Args:
            df: Reference DataFrame
            drug_id: Drug to predict for
            early_volumes: First 6 months of post-entry volumes
            
        Returns:
            Array of predicted volumes for months 6-23
        """
        n_months = 18  # Months 6-23
        predictions = np.zeros(n_months)
        
        # Get pre-entry average
        pre_entry = df[(df['drug_id'] == drug_id) & (df['months_postgx'] < 0)]
        pre_avg = pre_entry['volume'].mean() if len(pre_entry) > 0 else 1
        
        if self.model_type in ['ensemble', 'hybrid']:
            features = {
                'drug_id': drug_id,
                'early_mean_volume': np.mean(early_volumes),
                'early_std_volume': np.std(early_volumes),
                'early_min_volume': np.min(early_volumes),
                'early_max_volume': np.max(early_volumes),
                'early_trend': (early_volumes[-1] - early_volumes[0]) / len(early_volumes),
                'early_erosion_rate': self._calculate_early_erosion_rate(early_volumes, pre_avg),
                'pre_entry_avg': pre_avg
            }
            
            # Get drug characteristics
            drug_data = df[df['drug_id'] == drug_id]
            for col in ['hospital_rate', 'biological', 'small_molecule', 'n_gxs']:
                if col in drug_data.columns:
                    features[col] = drug_data[col].iloc[-1] if len(drug_data) > 0 else 0
            
            for i, month in enumerate(range(6, 24)):
                feat_row = features.copy()
                feat_row['target_month'] = month
                feat_row['months_ahead'] = month - 5
                
                X = pd.DataFrame([feat_row])
                X_numeric = X.select_dtypes(include=[np.number])
                X_numeric = X_numeric.fillna(0)
                
                pred = self.ensemble.predict(X_numeric)
                predictions[i] = max(0, pred[0])
        
        if self.model_type == 'trend':
            predictions = self._extrapolate_trend(early_volumes, n_months)
        
        if self.model_type == 'hybrid':
            # Combine ensemble with trend extrapolation
            trend_pred = self._extrapolate_trend(early_volumes, n_months)
            predictions = 0.7 * predictions + 0.3 * trend_pred
        
        return predictions


class LSTMForecaster:
    """
    LSTM-based forecaster for time series data.
    Note: Requires TensorFlow to be installed.
    """
    
    def __init__(self, sequence_length: int = 12, 
                 units: int = 50,
                 epochs: int = 50):
        """
        Initialize LSTM forecaster.
        
        Args:
            sequence_length: Length of input sequences
            units: Number of LSTM units
            epochs: Training epochs
        """
        self.sequence_length = sequence_length
        self.units = units
        self.epochs = epochs
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.target_col_idx = 0  # Track target column index for inverse transform
        
    def _build_model(self, n_features: int) -> None:
        """Build the LSTM model architecture."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            self.model = Sequential([
                LSTM(self.units, activation='relu', 
                     input_shape=(self.sequence_length, n_features),
                     return_sequences=True),
                Dropout(0.2),
                LSTM(self.units // 2, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            
            self.model.compile(optimizer='adam', loss='mse')
            
        except ImportError:
            print("TensorFlow not available. Using fallback model.")
            self.model = None
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length, self.target_col_idx])
        return np.array(X), np.array(y)
    
    def fit(self, df: pd.DataFrame,
            target_col: str = 'volume',
            feature_cols: Optional[List[str]] = None) -> 'LSTMForecaster':
        """
        Fit the LSTM model.
        
        Args:
            df: Training DataFrame
            target_col: Target column name
            feature_cols: Feature columns to use
            
        Returns:
            self
        """
        if feature_cols is None:
            feature_cols = [target_col]
        
        # Track target column index for proper inverse transform
        self.target_col_idx = feature_cols.index(target_col) if target_col in feature_cols else 0
        
        # Prepare data
        data = df[feature_cols].values
        data_scaled = self.scaler.fit_transform(data)
        
        X, y = self._create_sequences(data_scaled)
        
        if len(X) == 0:
            print("Not enough data for LSTM training")
            return self
        
        self._build_model(n_features=len(feature_cols))
        
        if self.model is not None:
            self.model.fit(X, y, epochs=self.epochs, batch_size=32, 
                          verbose=0, validation_split=0.2)
            self.is_fitted = True
            print("LSTM model training complete")
        
        return self
    
    def predict(self, initial_sequence: np.ndarray, 
                n_periods: int) -> np.ndarray:
        """
        Make multi-step predictions.
        
        Args:
            initial_sequence: Initial sequence for prediction
            n_periods: Number of periods to predict
            
        Returns:
            Array of predictions
        """
        if self.model is None or not self.is_fitted:
            return np.zeros(n_periods)
        
        predictions = []
        current_seq = initial_sequence.copy()
        
        for _ in range(n_periods):
            pred = self.model.predict(current_seq.reshape(1, self.sequence_length, -1), 
                                      verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence
            current_seq = np.roll(current_seq, -1, axis=0)
            current_seq[-1, self.target_col_idx] = pred[0, 0]
        
        # Inverse transform using target column position
        predictions = np.array(predictions).reshape(-1, 1)
        n_features = self.scaler.n_features_in_
        full_array = np.zeros((len(predictions), n_features))
        full_array[:, self.target_col_idx] = predictions.flatten()
        
        inverse_transformed = self.scaler.inverse_transform(full_array)
        result = inverse_transformed[:, self.target_col_idx]
        
        return np.maximum(0, result)
