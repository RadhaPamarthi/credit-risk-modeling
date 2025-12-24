"""
Model Evaluation Module
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import logging
import os

from .config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def calculate_ks_statistic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return float(max(tpr - fpr))


def calculate_gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    auc = roc_auc_score(y_true, y_pred)
    return 2 * auc - 1


class ModelEvaluator:
    """Calculates credit risk model evaluation metrics."""
    
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str = "test") -> Dict[str, float]:
        metrics = {
            'auc_roc': roc_auc_score(y_true, y_pred),
            'auc_pr': average_precision_score(y_true, y_pred),
            'ks_statistic': calculate_ks_statistic(y_true, y_pred),
            'gini': calculate_gini(y_true, y_pred),
            'brier_score': brier_score_loss(y_true, y_pred)
        }
        
        logger.info(f"{dataset_name.upper()} Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def compare_models(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
        results = []
        for model_name, y_pred in predictions.items():
            metrics = self.evaluate(y_true, y_pred, model_name)
            metrics['model'] = model_name
            results.append(metrics)
        
        df = pd.DataFrame(results).set_index('model')
        return df
    
    def calculate_overfit(self, train_auc: float, test_auc: float) -> Dict[str, float]:
        gap = train_auc - test_auc
        return {
            'train_auc': train_auc,
            'test_auc': test_auc,
            'overfit_gap': gap,
            'pct_drop': gap / train_auc * 100,
            'is_acceptable': gap < 0.10
        }


class EvaluationPlotter:
    """Creates evaluation visualizations."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 5)):
        self.figsize = figsize
    
    def plot_roc_curves(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray], save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=self.figsize)
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
        
        for (model_name, y_pred), color in zip(predictions.items(), colors):
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{model_name} (AUC={auc:.4f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve Comparison')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray], save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=self.figsize)
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
        
        for (model_name, y_pred), color in zip(predictions.items(), colors):
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            ap = average_precision_score(y_true, y_pred)
            ax.plot(recall, precision, color=color, linewidth=2, label=f'{model_name} (AP={ap:.4f})')
        
        baseline = y_true.mean()
        ax.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline ({baseline:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve Comparison')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 15, save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_features = importance_df.head(top_n).sort_values('importance')
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_features)))
        ax.barh(top_features['feature'], top_features['importance'], color=colors)
        
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.grid(True, alpha=0.3, axis='x')
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_calibration(self, y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10, save_path: Optional[str] = None):
        fig, ax = plt.subplots(figsize=self.figsize)
        
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred, n_bins=n_bins)
        
        ax.plot(mean_predicted_value, fraction_of_positives, marker='o', linewidth=2, label='Model')
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def create_evaluation_report(
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_predictions: Dict[str, np.ndarray],
    test_predictions: Dict[str, np.ndarray],
    feature_importance: pd.DataFrame,
    output_dir: str = "outputs"
) -> Dict:
    
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator = ModelEvaluator()
    plotter = EvaluationPlotter()
    
    train_metrics = {}
    test_metrics = {}
    
    for model_name in train_predictions:
        train_metrics[model_name] = evaluator.evaluate(y_train, train_predictions[model_name], f"train_{model_name}")
        test_metrics[model_name] = evaluator.evaluate(y_test, test_predictions[model_name], f"test_{model_name}")
    
    plotter.plot_roc_curves(y_test, test_predictions, save_path=f"{output_dir}/roc_curves.png")
    plotter.plot_precision_recall(y_test, test_predictions, save_path=f"{output_dir}/pr_curves.png")
    plotter.plot_feature_importance(feature_importance, save_path=f"{output_dir}/feature_importance.png")
    
    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'best_model': max(test_metrics, key=lambda x: test_metrics[x]['auc_roc']),
        'feature_importance': feature_importance.head(10).to_dict('records')
    }