"""
Production Scoring Pipeline
"""
import os
import yaml
import pickle
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ScoringConfig:
    model_path: str
    scaler_path: str
    encoders_path: str
    feature_columns: List[str]
    special_values: Dict[str, int]
    medians: Dict[str, float]
    decision_thresholds: Dict[str, float]
    batch_size: int = 10000
    output_format: str = "parquet"
    date_filter_enabled: bool = False
    start_vintage: Optional[int] = None
    end_vintage: Optional[int] = None


class ConfigLoader:
    """Loads configuration from YAML with environment support."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.env = os.environ.get('ENV', 'dev')
        self.config = self._load_config()
        logger.info(f"Environment: {self.env.upper()}")
    
    def _load_config(self) -> Dict:
        logger.info(f"Loading configuration from {self.config_path}")
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        config = self._substitute_env_vars(config)
        return config
    
    def _substitute_env_vars(self, obj):
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            var_name = obj[2:-1]
            return os.environ.get(var_name, obj)
        return obj
    
    def get(self, *keys, default=None):
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        if isinstance(value, dict) and self.env in value:
            return value[self.env]
        return value
    
    def get_storage_config(self) -> Dict:
        storage = self.config.get('storage', {})
        return storage.get(self.env, storage.get('dev', {}))
    
    def get_scoring_config(self) -> ScoringConfig:
        storage = self.get_storage_config()
        model_dir = storage.get('model_dir', 'outputs/models')
        
        if model_dir.startswith('s3://'):
            local_model_dir = 'outputs/models'
        else:
            local_model_dir = model_dir
        
        date_filter = self.get('scoring', 'date_filter', default={})
        
        return ScoringConfig(
            model_path=os.path.join(local_model_dir, self.get('storage', 'artifacts', 'xgb_model')),
            scaler_path=os.path.join(local_model_dir, self.get('storage', 'artifacts', 'scaler')),
            encoders_path=os.path.join(local_model_dir, self.get('storage', 'artifacts', 'encoders')),
            feature_columns=self._build_feature_list(),
            special_values=self.get('data', 'special_values', default={}),
            medians={},
            decision_thresholds=self.get('scoring', 'decision_thresholds', default={}),
            batch_size=self.get('scoring', 'batch_size', default=10000),
            output_format=self.get('scoring', 'output_format', default='parquet'),
            date_filter_enabled=date_filter.get('enabled', False),
            start_vintage=date_filter.get('start_vintage'),
            end_vintage=date_filter.get('end_vintage')
        )
    
    def _build_feature_list(self) -> List[str]:
        core = self.get('features', 'core_features', default=[])
        categoricals = [f"{c}_encoded" for c in self.get('features', 'categorical_features', default=[])]
        missing_indicators = [f"{col}_missing" for col in self.get('data', 'special_values', default={}).keys()]
        engineered = ['loan_to_income', 'payment_to_income', 'debt_burden_score', 'high_utilization', 'high_inquiries']
        return core + missing_indicators + engineered + categoricals


class ModelLoader:
    """Loads trained model artifacts."""
    
    def __init__(self, config: ScoringConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.encoders = None
        self.metadata = None
    
    def load_all(self) -> 'ModelLoader':
        logger.info("Loading model artifacts")
        
        with open(self.config.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(self.config.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(self.config.encoders_path, 'rb') as f:
            self.encoders = pickle.load(f)
        
        metadata_path = self.config.model_path.replace('xgb_model.pkl', 'pipeline_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        return self


class FeatureTransformer:
    """Transforms raw input data into model-ready features."""
    
    def __init__(self, config: ScoringConfig, encoders: Dict, metadata: Dict):
        self.config = config
        self.encoders = encoders
        self.metadata = metadata
        self.medians = metadata.get('medians') or metadata.get('preprocessing', {}).get('medians', {})
        self.high_util_threshold = 0.70
        self.high_inq_threshold = 3
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Transforming {len(df):,} records")
        df = df.copy()
        df = self._handle_special_values(df)
        df = self._engineer_features(df)
        df = self._encode_categoricals(df)
        df = self._select_features(df)
        return df
    
    def _handle_special_values(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, special_val in self.config.special_values.items():
            if col in df.columns:
                df[f'{col}_missing'] = (df[col] == special_val).astype(int)
                median = self.medians.get(col, df[df[col] != special_val][col].median())
                df[col] = df[col].replace(special_val, median)
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['loan_to_income'] = df['loan_amount'] / df['income'].clip(lower=1)
        
        monthly_rate = df['apr'] / 100 / 12
        df['monthly_payment_est'] = df['loan_amount'] * (
            monthly_rate * (1 + monthly_rate)**df['term']
        ) / ((1 + monthly_rate)**df['term'] - 1)
        df['monthly_payment_est'] = df['monthly_payment_est'].fillna(df['loan_amount'] / df['term'])
        
        df['payment_to_income'] = df['monthly_payment_est'] / (df['income'].clip(lower=1) / 12)
        
        df['debt_burden_score'] = (
            df['debt_to_income'] * 0.4 +
            df['utilization_rate'] * 0.3 +
            df['payment_to_income'].clip(upper=1) * 0.3
        )
        
        df['high_utilization'] = (df['utilization_rate'] > self.high_util_threshold).astype(int)
        df['high_inquiries'] = (df['inquiries_last_6m'] > self.high_inq_threshold).astype(int)
        
        return df
    
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, encoder in self.encoders.items():
            if col in df.columns:
                df[f'{col}_encoded'] = df[col].astype(str).apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )
        return df
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = self.config.feature_columns
        missing = [f for f in feature_cols if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        return df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)


class ProductionScorer:
    """Production scoring pipeline."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_loader = ConfigLoader(config_path)
        self.scoring_config = self.config_loader.get_scoring_config()
        self.model_loader = ModelLoader(self.scoring_config)
        self.model_loader.load_all()
        self.transformer = FeatureTransformer(
            self.scoring_config,
            self.model_loader.encoders,
            self.model_loader.metadata or {}
        )
    
    def score(self, input_data: pd.DataFrame, include_features: bool = False) -> pd.DataFrame:
        logger.info(f"Scoring {len(input_data):,} records")
        start_time = datetime.now()
        
        id_cols = ['loan_id'] if 'loan_id' in input_data.columns else []
        ids = input_data[id_cols].copy() if id_cols else None
        
        X = self.transformer.transform(input_data)
        probabilities = self.model_loader.model.predict_proba(X)[:, 1]
        results = self._build_results(ids, probabilities, X if include_features else None)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Scoring complete: {len(results):,} records in {elapsed:.2f}s")
        
        return results
    
    def _build_results(self, ids: Optional[pd.DataFrame], probabilities: np.ndarray, features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        results = ids.copy() if ids is not None else pd.DataFrame()
        
        results['pd_score'] = probabilities
        results['risk_decile'] = pd.qcut(probabilities, 10, labels=range(1, 11), duplicates='drop').astype(int)
        
        thresholds = self.scoring_config.decision_thresholds
        approve_below = thresholds.get('approve_below', 0.03)
        decline_above = thresholds.get('decline_above', 0.15)
        
        results['credit_decision'] = np.select(
            [probabilities < approve_below, probabilities > decline_above],
            ['APPROVE', 'DECLINE'],
            default='REVIEW'
        )
        
        results['scored_at'] = datetime.now().isoformat()
        
        if self.model_loader.metadata:
            results['model_version'] = self.model_loader.metadata.get('model_version', 'v1.0')
        
        if features is not None:
            results = pd.concat([results, features], axis=1)
        
        return results
    
    def _filter_by_vintage(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.scoring_config.date_filter_enabled:
            return df
        
        if 'vintage' not in df.columns:
            logger.warning("No vintage column found, skipping date filter")
            return df
        
        df['vintage_int'] = df['vintage'].astype(int)
        original_count = len(df)
        
        if self.scoring_config.start_vintage:
            df = df[df['vintage_int'] >= self.scoring_config.start_vintage]
        
        if self.scoring_config.end_vintage:
            df = df[df['vintage_int'] <= self.scoring_config.end_vintage]
        
        logger.info(f"Date filter: {original_count:,} -> {len(df):,} records "
                   f"(vintage {self.scoring_config.start_vintage} to {self.scoring_config.end_vintage})")
        
        return df
    
    def score_batch(self, input_path: str, output_path: str, batch_size: Optional[int] = None) -> Dict:
        batch_size = batch_size or self.scoring_config.batch_size
        
        logger.info(f"Batch scoring: {input_path} -> {output_path}")
        
        if input_path.endswith('.parquet'):
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)
        
        df = self._filter_by_vintage(df)
        
        total_records = len(df)
        logger.info(f"Total records: {total_records:,}")
        
        if total_records == 0:
            logger.warning("No records to score after filtering")
            return {'total_scored': 0, 'approve_count': 0, 'decline_count': 0, 'review_count': 0, 'avg_pd_score': 0, 'output_path': output_path}
        
        results = []
        for i in range(0, total_records, batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_results = self.score(batch)
            results.append(batch_results)
        
        final_results = pd.concat(results, ignore_index=True)
        
        if output_path.endswith('.parquet'):
            final_results.to_parquet(output_path, index=False)
        else:
            final_results.to_csv(output_path, index=False)
        
        logger.info(f"Results saved to {output_path}")
        
        summary = {
            'total_scored': len(final_results),
            'approve_count': int((final_results['credit_decision'] == 'APPROVE').sum()),
            'decline_count': int((final_results['credit_decision'] == 'DECLINE').sum()),
            'review_count': int((final_results['credit_decision'] == 'REVIEW').sum()),
            'avg_pd_score': float(final_results['pd_score'].mean()),
            'output_path': output_path
        }
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Production Credit Risk Scoring')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=None)
    
    args = parser.parse_args()
    
    scorer = ProductionScorer(args.config)
    summary = scorer.score_batch(args.input, args.output, args.batch_size)
    
    print("\nScoring Complete")
    print("-" * 40)
    print(f"Total Scored:  {summary['total_scored']:,}")
    print(f"Approved:      {summary['approve_count']:,}")
    print(f"Declined:      {summary['decline_count']:,}")
    print(f"Manual Review: {summary['review_count']:,}")
    print(f"Average PD:    {summary['avg_pd_score']:.4f}")
    print(f"Output:        {summary['output_path']}")


if __name__ == '__main__':
    main()