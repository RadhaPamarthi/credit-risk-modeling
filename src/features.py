"""
Feature Engineering Module
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

from .config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates engineered features for credit risk modeling."""
    
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Loan affordability
        df['loan_to_income'] = df['loan_amount'] / df['income'].clip(lower=1)
        
        # Monthly payment estimate
        monthly_rate = df['apr'] / 100 / 12
        df['monthly_payment_est'] = df['loan_amount'] * (
            monthly_rate * (1 + monthly_rate)**df['term']
        ) / ((1 + monthly_rate)**df['term'] - 1)
        df['monthly_payment_est'] = df['monthly_payment_est'].fillna(df['loan_amount'] / df['term'])
        
        # Payment-to-income ratio
        df['payment_to_income'] = df['monthly_payment_est'] / (df['income'].clip(lower=1) / 12)
        
        # Debt burden score (composite)
        df['debt_burden_score'] = (
            df['debt_to_income'] * 0.4 +
            df['utilization_rate'] * 0.3 +
            df['payment_to_income'].clip(upper=1) * 0.3
        )
        
        # Risk flags
        df['high_utilization'] = (df['utilization_rate'] > self.config.features.high_utilization_threshold).astype(int)
        df['high_inquiries'] = (df['inquiries_last_6m'] > self.config.features.high_inquiries_threshold).astype(int)
        
        return df


class CategoricalEncoder:
    """Label encodes categorical variables."""
    
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.encoders = {}
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'CategoricalEncoder':
        for col in self.config.features.categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self.encoders[col] = le
        
        self._is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise ValueError("Encoder not fitted")
        
        df = df.copy()
        for col, encoder in self.encoders.items():
            if col in df.columns:
                df[f'{col}_encoded'] = df[col].astype(str).apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
    
    def get_encoders(self) -> Dict[str, LabelEncoder]:
        return self.encoders


class FeatureSelector:
    """Selects and scales features for modeling."""
    
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.scaler = None
        self._is_fitted = False
    
    def get_feature_matrix(
        self, 
        df: pd.DataFrame, 
        scale: bool = False
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        
        feature_cols = self.config.features.model_features
        target_col = self.config.data.target_column
        weight_col = self.config.data.weight_column
        
        missing_features = [f for f in feature_cols if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        weights = df[weight_col].copy()
        
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        if scale:
            if self.scaler is None:
                self.scaler = StandardScaler()
                X = pd.DataFrame(
                    self.scaler.fit_transform(X),
                    columns=feature_cols,
                    index=X.index
                )
                self._is_fitted = True
            else:
                X = pd.DataFrame(
                    self.scaler.transform(X),
                    columns=feature_cols,
                    index=X.index
                )
        
        return X, y, weights
    
    def get_scaler(self):
        return self.scaler


class FeaturePipeline:
    """Complete feature engineering pipeline."""
    
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.categorical_encoder = CategoricalEncoder(config)
        self.feature_selector = FeatureSelector(config)
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'FeaturePipeline':
        df = self.feature_engineer.create_features(df)
        self.categorical_encoder.fit(df)
        self._is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise ValueError("Pipeline not fitted")
        
        df = self.feature_engineer.create_features(df)
        df = self.categorical_encoder.transform(df)
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
    
    def get_artifacts(self) -> Dict:
        return {
            'encoders': self.categorical_encoder.get_encoders(),
            'scaler': self.feature_selector.get_scaler()
        }