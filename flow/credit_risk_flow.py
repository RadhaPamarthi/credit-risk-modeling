"""
Credit Risk Modeling - Metaflow Pipeline
"""
from metaflow import FlowSpec, step, Parameter, current, card
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CreditRiskFlow(FlowSpec):
    """End-to-end ML pipeline for credit risk prediction."""
    
    data_path = Parameter('data_path', default='data/credit_risk_data_enhanced.csv')
    train_vintage_end = Parameter('train_vintage_end', default=202212, type=int)
    test_vintage_start = Parameter('test_vintage_start', default=202401, type=int)
    random_state = Parameter('random_state', default=42, type=int)
    output_dir = Parameter('output_dir', default='outputs')
    
    SPECIAL_VALUES = {
        'fico_score': 99999,
        'income': -1,
        'inquiries_last_6m': 99
    }
    
    LEAKAGE_FEATURES = [
        'days_past_due_current',
        'total_payments_to_date',
        'months_on_book'
    ]
    
    FEATURE_COLS = [
        'fico_score', 'income', 'debt_to_income', 'utilization_rate',
        'inquiries_last_6m', 'num_open_trades', 'loan_amount', 'term', 'apr',
        'employment_length', 'age',
        'fico_score_missing', 'income_missing', 'inquiries_last_6m_missing',
        'loan_to_income', 'payment_to_income', 'debt_burden_score',
        'high_utilization', 'high_inquiries',
        'product_type_encoded', 'state_encoded'
    ]
    
    @card
    @step
    def start(self):
        """Load and validate data."""
        print("Step 1: Data Loading")
        
        self.df = pd.read_csv(self.data_path)
        
        required_cols = ['default_12m', 'sample_weight', 'vintage']
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self.n_records = len(self.df)
        self.n_features = len(self.df.columns)
        self.default_rate = float(self.df['default_12m'].mean())
        
        print(f"Loaded {self.n_records:,} records, {self.n_features} columns")
        print(f"Default rate: {self.default_rate:.2%}")
        
        self.run_timestamp = datetime.now().isoformat()
        self.next(self.preprocess)
    
    @card
    @step
    def preprocess(self):
        """Handle missing values and data quality issues."""
        print("Step 2: Preprocessing")
        
        self.medians = {}
        self.missing_stats = {}
        
        for col, special_val in self.SPECIAL_VALUES.items():
            if col in self.df.columns:
                missing_col = f'{col}_missing'
                self.df[missing_col] = (self.df[col] == special_val).astype(int)
                
                valid_data = self.df[self.df[col] != special_val][col]
                self.medians[col] = float(valid_data.median())
                
                n_missing = self.df[missing_col].sum()
                self.df[col] = self.df[col].replace(special_val, self.medians[col])
                
                self.missing_stats[col] = {
                    'n_missing': int(n_missing),
                    'pct_missing': float(n_missing / len(self.df) * 100),
                    'median': self.medians[col]
                }
                
                print(f"{col}: {n_missing:,} missing, imputed with {self.medians[col]:.0f}")
        
        self.next(self.feature_engineering)
    
    @card
    @step
    def feature_engineering(self):
        """Create engineered features."""
        print("Step 3: Feature Engineering")
        
        self.df['loan_to_income'] = self.df['loan_amount'] / self.df['income'].clip(lower=1)
        
        monthly_rate = self.df['apr'] / 100 / 12
        self.df['monthly_payment_est'] = self.df['loan_amount'] * (
            monthly_rate * (1 + monthly_rate)**self.df['term']
        ) / ((1 + monthly_rate)**self.df['term'] - 1)
        self.df['monthly_payment_est'] = self.df['monthly_payment_est'].fillna(
            self.df['loan_amount'] / self.df['term']
        )
        
        self.df['payment_to_income'] = self.df['monthly_payment_est'] / (self.df['income'].clip(lower=1) / 12)
        
        self.df['debt_burden_score'] = (
            self.df['debt_to_income'] * 0.4 +
            self.df['utilization_rate'] * 0.3 +
            self.df['payment_to_income'].clip(upper=1) * 0.3
        )
        
        self.df['high_utilization'] = (self.df['utilization_rate'] > 0.70).astype(int)
        self.df['high_inquiries'] = (self.df['inquiries_last_6m'] > 3).astype(int)
        
        from sklearn.preprocessing import LabelEncoder
        self.encoders = {}
        
        for col in ['product_type', 'state']:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.encoders[col] = le
        
        self.next(self.split_data)
    
    @card
    @step
    def split_data(self):
        """Vintage-based train/test split."""
        print("Step 4: Data Splitting")
        
        self.df['vintage_int'] = self.df['vintage'].astype(int)
        
        train_mask = self.df['vintage_int'] <= self.train_vintage_end
        test_mask = self.df['vintage_int'] >= self.test_vintage_start
        
        self.train_df = self.df[train_mask].copy()
        self.test_df = self.df[test_mask].copy()
        
        self.train_size = len(self.train_df)
        self.test_size = len(self.test_df)
        self.train_default_rate = float(self.train_df['default_12m'].mean())
        self.test_default_rate = float(self.test_df['default_12m'].mean())
        
        print(f"Train: {self.train_size:,} records (vintage <= {self.train_vintage_end})")
        print(f"Test: {self.test_size:,} records (vintage >= {self.test_vintage_start})")
        
        self.next(self.train_models)
    
    @card
    @step
    def train_models(self):
        """Train XGBoost and Logistic Regression models."""
        print("Step 5: Model Training")
        
        import xgboost as xgb
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        X_train = self.train_df[self.FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_train = self.train_df['default_12m']
        w_train = self.train_df['sample_weight']
        
        X_test = self.test_df[self.FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
        self.y_test = self.test_df['default_12m']
        
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7, min_child_weight=50,
            scale_pos_weight=scale_pos_weight,
            reg_lambda=10, reg_alpha=1, gamma=1,
            random_state=self.random_state,
            eval_metric='auc', use_label_encoder=False
        )
        self.xgb_model.fit(X_train, y_train, sample_weight=w_train, verbose=False)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.lr_model = LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=self.random_state
        )
        self.lr_model.fit(X_train_scaled, y_train, sample_weight=w_train)
        
        self.y_pred_train_xgb = self.xgb_model.predict_proba(X_train)[:, 1]
        self.y_pred_test_xgb = self.xgb_model.predict_proba(X_test)[:, 1]
        self.y_pred_train_lr = self.lr_model.predict_proba(X_train_scaled)[:, 1]
        self.y_pred_test_lr = self.lr_model.predict_proba(X_test_scaled)[:, 1]
        
        self.y_train = y_train
        self.X_test = X_test
        
        self.next(self.evaluate)
    
    @card
    @step
    def evaluate(self):
        """Evaluate models and calculate metrics."""
        print("Step 6: Evaluation")
        
        from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
        
        def calc_ks(y_true, y_pred):
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            return float(max(tpr - fpr))
        
        self.metrics = {
            'xgboost': {
                'train_auc': float(roc_auc_score(self.y_train, self.y_pred_train_xgb)),
                'test_auc': float(roc_auc_score(self.y_test, self.y_pred_test_xgb)),
                'test_ks': calc_ks(self.y_test, self.y_pred_test_xgb),
                'test_auc_pr': float(average_precision_score(self.y_test, self.y_pred_test_xgb))
            },
            'logistic_regression': {
                'train_auc': float(roc_auc_score(self.y_train, self.y_pred_train_lr)),
                'test_auc': float(roc_auc_score(self.y_test, self.y_pred_test_lr)),
                'test_ks': calc_ks(self.y_test, self.y_pred_test_lr),
                'test_auc_pr': float(average_precision_score(self.y_test, self.y_pred_test_lr))
            }
        }
        
        print(f"{'Model':<25} {'Train AUC':>12} {'Test AUC':>12} {'Test KS':>10}")
        print("-" * 60)
        for model_name, m in self.metrics.items():
            print(f"{model_name:<25} {m['train_auc']:>12.4f} {m['test_auc']:>12.4f} {m['test_ks']:>10.4f}")
        
        self.feature_importance = pd.DataFrame({
            'feature': self.FEATURE_COLS,
            'importance': self.xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.best_model = max(self.metrics, key=lambda x: self.metrics[x]['test_auc'])
        self.best_auc = self.metrics[self.best_model]['test_auc']
        
        self.next(self.end)
    
    @card
    @step
    def end(self):
        """Save artifacts and complete pipeline."""
        print("Step 7: Saving Artifacts")
        
        model_dir = f'{self.output_dir}/models'
        os.makedirs(model_dir, exist_ok=True)
        
        with open(f'{model_dir}/xgb_model.pkl', 'wb') as f:
            pickle.dump(self.xgb_model, f)
        
        with open(f'{model_dir}/lr_model.pkl', 'wb') as f:
            pickle.dump(self.lr_model, f)
        
        with open(f'{model_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(f'{model_dir}/encoders.pkl', 'wb') as f:
            pickle.dump(self.encoders, f)
        
        metadata = {
            'run_id': current.run_id,
            'timestamp': self.run_timestamp,
            'parameters': {
                'data_path': self.data_path,
                'train_vintage_end': self.train_vintage_end,
                'test_vintage_start': self.test_vintage_start,
                'random_state': self.random_state
            },
            'data_stats': {
                'n_records': self.n_records,
                'train_size': self.train_size,
                'test_size': self.test_size,
                'train_default_rate': self.train_default_rate,
                'test_default_rate': self.test_default_rate
            },
            'medians': self.medians,
            'preprocessing': {
                'medians': self.medians,
                'missing_stats': self.missing_stats
            },
            'feature_columns': self.FEATURE_COLS,
            'leakage_features_removed': self.LEAKAGE_FEATURES,
            'metrics': self.metrics,
            'best_model': self.best_model,
            'best_auc': self.best_auc,
            'feature_importance': self.feature_importance.head(10).to_dict('records')
        }
        
        with open(f'{model_dir}/pipeline_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nPipeline Complete")
        print(f"Best Model: {self.best_model}, Test AUC: {self.best_auc:.4f}")
        print(f"Artifacts saved to: {model_dir}/")


if __name__ == '__main__':
    CreditRiskFlow()