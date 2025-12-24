"""
Configuration Management
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
import json
import yaml
import os


def load_yaml_config(config_path: str = "config/config.yaml") -> Dict:
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@dataclass
class DataConfig:
    data_path: str = "data/credit_risk_data_enhanced.csv"
    target_column: str = "default_12m"
    weight_column: str = "sample_weight"
    vintage_column: str = "vintage"
    
    special_values: Dict[str, Any] = field(default_factory=lambda: {
        'fico_score': 99999,
        'income': -1,
        'inquiries_last_6m': 99
    })
    
    leakage_features: List[str] = field(default_factory=lambda: [
        'days_past_due_current',
        'total_payments_to_date',
        'months_on_book'
    ])
    
    id_columns: List[str] = field(default_factory=lambda: [
        'loan_id', 'origination_date', 'vintage'
    ])
    
    @classmethod
    def from_yaml(cls, yaml_config: Dict) -> 'DataConfig':
        data_section = yaml_config.get('data', {})
        return cls(
            data_path=data_section.get('input_path', cls.data_path),
            target_column=data_section.get('target_column', cls.target_column),
            weight_column=data_section.get('weight_column', cls.weight_column),
            vintage_column=data_section.get('vintage_column', cls.vintage_column),
            special_values=data_section.get('special_values', {}),
            leakage_features=yaml_config.get('features', {}).get('leakage_features', []),
            id_columns=data_section.get('id_columns', [])
        )


@dataclass
class FeatureConfig:
    core_features: List[str] = field(default_factory=lambda: [
        'fico_score', 'income', 'debt_to_income', 'utilization_rate',
        'inquiries_last_6m', 'num_open_trades', 'loan_amount', 'term', 'apr',
        'employment_length', 'age'
    ])
    
    categorical_features: List[str] = field(default_factory=lambda: [
        'product_type', 'state'
    ])
    
    high_utilization_threshold: float = 0.70
    high_inquiries_threshold: int = 3
    
    model_features: List[str] = field(default_factory=lambda: [
        'fico_score', 'income', 'debt_to_income', 'utilization_rate',
        'inquiries_last_6m', 'num_open_trades', 'loan_amount', 'term', 'apr',
        'employment_length', 'age',
        'fico_score_missing', 'income_missing', 'inquiries_last_6m_missing',
        'loan_to_income', 'payment_to_income', 'debt_burden_score',
        'high_utilization', 'high_inquiries',
        'product_type_encoded', 'state_encoded'
    ])
    
    @classmethod
    def from_yaml(cls, yaml_config: Dict) -> 'FeatureConfig':
        feat_section = yaml_config.get('features', {})
        thresholds = feat_section.get('thresholds', {})
        return cls(
            core_features=feat_section.get('core_features', cls().core_features),
            categorical_features=feat_section.get('categorical_features', cls().categorical_features),
            high_utilization_threshold=thresholds.get('high_utilization', 0.70),
            high_inquiries_threshold=thresholds.get('high_inquiries', 3)
        )


@dataclass
class ValidationConfig:
    train_vintage_end: int = 202212
    test_vintage_start: int = 202401
    n_folds: int = 5
    random_state: int = 42
    
    @classmethod
    def from_yaml(cls, yaml_config: Dict) -> 'ValidationConfig':
        val_section = yaml_config.get('validation', {})
        return cls(
            train_vintage_end=val_section.get('train_vintage_end', 202212),
            test_vintage_start=val_section.get('test_vintage_start', 202401),
            n_folds=val_section.get('n_folds', 5),
            random_state=val_section.get('random_state', 42)
        )


@dataclass
class ModelConfig:
    random_state: int = 42
    
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.05,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 50,
        'reg_lambda': 10,
        'reg_alpha': 1,
        'gamma': 1,
        'eval_metric': 'auc',
        'use_label_encoder': False
    })
    
    lr_params: Dict[str, Any] = field(default_factory=lambda: {
        'class_weight': 'balanced',
        'max_iter': 1000,
        'solver': 'lbfgs'
    })
    
    @classmethod
    def from_yaml(cls, yaml_config: Dict) -> 'ModelConfig':
        model_section = yaml_config.get('model', {})
        val_section = yaml_config.get('validation', {})
        return cls(
            random_state=val_section.get('random_state', 42),
            xgb_params=model_section.get('xgboost', cls().xgb_params),
            lr_params=model_section.get('logistic_regression', cls().lr_params)
        )


@dataclass 
class OutputConfig:
    model_dir: str = "outputs/models"
    plots_dir: str = "outputs/plots"
    scoring_dir: str = "outputs/scoring"
    
    @classmethod
    def from_yaml(cls, yaml_config: Dict) -> 'OutputConfig':
        output_section = yaml_config.get('output', {})
        return cls(
            model_dir=output_section.get('model_dir', 'outputs/models'),
            plots_dir=output_section.get('plots_dir', 'outputs/plots'),
            scoring_dir=output_section.get('scoring_dir', 'outputs/scoring')
        )


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str = "config/config.yaml") -> 'Config':
        yaml_config = load_yaml_config(config_path)
        if not yaml_config:
            return cls()
        
        return cls(
            data=DataConfig.from_yaml(yaml_config),
            features=FeatureConfig.from_yaml(yaml_config),
            validation=ValidationConfig.from_yaml(yaml_config),
            model=ModelConfig.from_yaml(yaml_config),
            output=OutputConfig.from_yaml(yaml_config)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'data': {
                'data_path': self.data.data_path,
                'target_column': self.data.target_column,
                'special_values': self.data.special_values,
                'leakage_features': self.data.leakage_features
            },
            'features': {
                'core_features': self.features.core_features,
                'model_features': self.features.model_features,
                'high_utilization_threshold': self.features.high_utilization_threshold,
                'high_inquiries_threshold': self.features.high_inquiries_threshold
            },
            'validation': {
                'train_vintage_end': self.validation.train_vintage_end,
                'test_vintage_start': self.validation.test_vintage_start,
                'n_folds': self.validation.n_folds,
                'random_state': self.validation.random_state
            },
            'model': {
                'xgb_params': self.model.xgb_params,
                'lr_params': self.model.lr_params
            }
        }
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            json.load(f)
        return cls()


DEFAULT_CONFIG = Config()