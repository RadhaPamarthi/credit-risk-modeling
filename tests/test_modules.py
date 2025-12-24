"""
Unit Tests for Credit Risk Modeling
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


class TestSpecialValueHandling:
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'fico_score': [700, 99999, 650, 720, 99999],
            'income': [50000, 60000, -1, 70000, 80000],
            'inquiries_last_6m': [1, 2, 99, 3, 0],
            'default_12m': [0, 1, 0, 0, 1]
        })
    
    def test_identify_special_values(self, sample_df):
        special_values = {
            'fico_score': 99999,
            'income': -1,
            'inquiries_last_6m': 99
        }
        
        for col, val in special_values.items():
            count = (sample_df[col] == val).sum()
            assert count > 0
    
    def test_median_calculation_excludes_special(self, sample_df):
        valid_fico = sample_df[sample_df['fico_score'] != 99999]['fico_score']
        median_fico = valid_fico.median()
        assert median_fico == 700.0
    
    def test_missing_indicator_creation(self, sample_df):
        sample_df['fico_score_missing'] = (sample_df['fico_score'] == 99999).astype(int)
        assert sample_df['fico_score_missing'].sum() == 2
        assert sample_df['fico_score_missing'].iloc[1] == 1
        assert sample_df['fico_score_missing'].iloc[4] == 1


class TestFeatureEngineering:
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'loan_amount': [10000, 20000, 15000],
            'income': [50000, 60000, 45000],
            'apr': [10.0, 12.0, 15.0],
            'term': [36, 60, 48],
            'debt_to_income': [0.3, 0.4, 0.5],
            'utilization_rate': [0.5, 0.8, 0.6],
            'inquiries_last_6m': [2, 5, 1]
        })
    
    def test_loan_to_income_calculation(self, sample_df):
        sample_df['loan_to_income'] = sample_df['loan_amount'] / sample_df['income']
        assert abs(sample_df['loan_to_income'].iloc[0] - 0.2) < 0.01
        assert abs(sample_df['loan_to_income'].iloc[1] - 0.333) < 0.01
    
    def test_high_utilization_flag(self, sample_df):
        sample_df['high_utilization'] = (sample_df['utilization_rate'] > 0.70).astype(int)
        assert sample_df['high_utilization'].tolist() == [0, 1, 0]
    
    def test_high_inquiries_flag(self, sample_df):
        sample_df['high_inquiries'] = (sample_df['inquiries_last_6m'] > 3).astype(int)
        assert sample_df['high_inquiries'].tolist() == [0, 1, 0]
    
    def test_debt_burden_score_range(self, sample_df):
        sample_df['payment_to_income'] = 0.2
        sample_df['debt_burden_score'] = (
            sample_df['debt_to_income'] * 0.4 +
            sample_df['utilization_rate'] * 0.3 +
            sample_df['payment_to_income'] * 0.3
        )
        assert all(sample_df['debt_burden_score'] >= 0)
        assert all(sample_df['debt_burden_score'] <= 1)


class TestEvaluationMetrics:
    
    def test_ks_statistic_perfect_separation(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        ks = max(tpr - fpr)
        assert ks == 1.0
    
    def test_ks_statistic_random(self):
        np.random.seed(42)
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.random.rand(6)
        
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        ks = max(tpr - fpr)
        assert 0 <= ks <= 1
    
    def test_gini_calculation(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.4, 0.6, 0.9])
        
        auc = roc_auc_score(y_true, y_pred)
        gini = 2 * auc - 1
        assert gini > 0


class TestDataSplitting:
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'vintage': ['202201', '202202', '202212', '202401', '202402'],
            'vintage_int': [202201, 202202, 202212, 202401, 202402],
            'default_12m': [0, 1, 0, 0, 1],
            'sample_weight': [1.0, 1.0, 1.0, 1.0, 1.0]
        })
    
    def test_vintage_split(self, sample_df):
        train_vintage_end = 202212
        test_vintage_start = 202401
        
        train_df = sample_df[sample_df['vintage_int'] <= train_vintage_end]
        test_df = sample_df[sample_df['vintage_int'] >= test_vintage_start]
        
        assert len(train_df) == 3
        assert len(test_df) == 2
        assert len(set(train_df.index) & set(test_df.index)) == 0


class TestLeakageDetection:
    
    def test_dpd_is_leakage(self):
        df = pd.DataFrame({
            'days_past_due_current': [0, 0, 0, 30, 60, 90],
            'default_12m': [0, 0, 0, 1, 1, 1]
        })
        
        auc = roc_auc_score(df['default_12m'], df['days_past_due_current'])
        assert auc == 1.0
    
    def test_valid_feature_not_perfect(self):
        df = pd.DataFrame({
            'fico_score': [650, 700, 750, 600, 720, 680, 710, 620],
            'default_12m': [1, 1, 0, 1, 0, 0, 0, 0]
        })
        
        auc = roc_auc_score(df['default_12m'], -df['fico_score'])
        assert auc < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])