"""
Data Preprocessing Module
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from .config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and validates credit risk data."""
    
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.df = None
        
    def load(self, path: Optional[str] = None) -> pd.DataFrame:
        data_path = path or self.config.data.data_path
        logger.info(f"Loading data from {data_path}")
        
        self.df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(self.df):,} records with {len(self.df.columns)} columns")
        
        self._validate_columns()
        return self.df
    
    def _validate_columns(self):
        required = [
            self.config.data.target_column,
            self.config.data.weight_column,
            self.config.data.vintage_column
        ]
        
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def get_data_summary(self) -> Dict:
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        target = self.config.data.target_column
        
        return {
            'n_records': len(self.df),
            'n_columns': len(self.df.columns),
            'target_distribution': self.df[target].value_counts().to_dict(),
            'default_rate': float(self.df[target].mean()),
            'vintage_range': {
                'min': self.df[self.config.data.vintage_column].min(),
                'max': self.df[self.config.data.vintage_column].max()
            }
        }


class SpecialValueHandler:
    """
    Handles special values that indicate missing data.
    Creates missing indicators and imputes with median.
    """
    
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.medians = {}
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'SpecialValueHandler':
        logger.info("Fitting SpecialValueHandler")
        
        for col, special_val in self.config.data.special_values.items():
            if col in df.columns:
                valid_data = df[df[col] != special_val][col]
                self.medians[col] = float(valid_data.median())
                
                n_special = (df[col] == special_val).sum()
                logger.info(f"  {col}: {n_special:,} special values, median={self.medians[col]:.2f}")
        
        self._is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise ValueError("Handler not fitted. Call fit() first.")
        
        df = df.copy()
        
        for col, special_val in self.config.data.special_values.items():
            if col in df.columns:
                df[f'{col}_missing'] = (df[col] == special_val).astype(int)
                df[col] = df[col].replace(special_val, self.medians[col])
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


class DataSplitter:
    """Vintage-based out-of-time data splitter."""
    
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
    
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = df.copy()
        df['vintage_int'] = df[self.config.data.vintage_column].astype(int)
        
        train_mask = df['vintage_int'] <= self.config.validation.train_vintage_end
        test_mask = df['vintage_int'] >= self.config.validation.test_vintage_start
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        logger.info(f"Train: {len(train_df):,} records (vintage <= {self.config.validation.train_vintage_end})")
        logger.info(f"Test: {len(test_df):,} records (vintage >= {self.config.validation.test_vintage_start})")
        
        target = self.config.data.target_column
        logger.info(f"Train default rate: {train_df[target].mean()*100:.2f}%")
        logger.info(f"Test default rate: {test_df[target].mean()*100:.2f}%")
        
        return train_df, test_df


def create_preprocessing_pipeline(config: Config = DEFAULT_CONFIG):
    return (
        DataLoader(config),
        SpecialValueHandler(config),
        DataSplitter(config)
    )