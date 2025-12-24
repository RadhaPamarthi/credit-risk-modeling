"""
Model Training Module
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import pickle
import logging

from .config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class CreditRiskModel:
    """Base class for credit risk models."""
    
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.model = None
        self._is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        raise NotImplementedError
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        instance = cls()
        with open(path, 'rb') as f:
            instance.model = pickle.load(f)
        instance._is_fitted = True
        return instance


class XGBoostModel(CreditRiskModel):
    """XGBoost classifier with class imbalance handling."""
    
    def __init__(self, config: Config = DEFAULT_CONFIG, **kwargs):
        super().__init__(config)
        self.params = {**config.model.xgb_params, **kwargs}
    
    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        scale_pos_weight = (y == 0).sum() / (y == 1).sum()
        logger.info(f"Class imbalance ratio: {scale_pos_weight:.1f}:1")
        
        self.model = xgb.XGBClassifier(
            **self.params,
            scale_pos_weight=scale_pos_weight,
            random_state=self.config.model.random_state
        )
        
        logger.info("Training XGBoost model")
        self.model.fit(X, y, sample_weight=sample_weight, verbose=False)
        self._is_fitted = True
        
        return self
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance


class LogisticRegressionModel(CreditRiskModel):
    """Logistic Regression baseline model."""
    
    def __init__(self, config: Config = DEFAULT_CONFIG, **kwargs):
        super().__init__(config)
        self.params = {**config.model.lr_params, **kwargs}
    
    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        self.model = LogisticRegression(
            **self.params,
            random_state=self.config.model.random_state
        )
        
        logger.info("Training Logistic Regression model")
        self.model.fit(X, y, sample_weight=sample_weight)
        self._is_fitted = True
        
        return self
    
    def get_coefficients(self, feature_names: list) -> pd.DataFrame:
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        
        coef = pd.DataFrame({
            'feature': feature_names,
            'coefficient': self.model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        
        return coef


class GradientBoostingModel(CreditRiskModel):
    """Gradient Boosting classifier."""
    
    def __init__(self, config: Config = DEFAULT_CONFIG, **kwargs):
        super().__init__(config)
        self.params = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'min_samples_leaf': 50,
            **kwargs
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None):
        self.model = GradientBoostingClassifier(
            **self.params,
            random_state=self.config.model.random_state
        )
        
        logger.info("Training Gradient Boosting model")
        self.model.fit(X, y, sample_weight=sample_weight)
        self._is_fitted = True
        
        return self


class ModelTrainer:
    """Trains and compares multiple models."""
    
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.models = {}
    
    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_train_scaled: pd.DataFrame,
        sample_weight: Optional[pd.Series] = None
    ) -> Dict[str, CreditRiskModel]:
        
        xgb_model = XGBoostModel(self.config)
        xgb_model.fit(X_train, y_train, sample_weight)
        self.models['xgboost'] = xgb_model
        
        lr_model = LogisticRegressionModel(self.config)
        lr_model.fit(X_train_scaled, y_train, sample_weight)
        self.models['logistic_regression'] = lr_model
        
        return self.models
    
    def get_best_model(self, metric_scores: Dict[str, float]) -> str:
        return max(metric_scores, key=metric_scores.get)