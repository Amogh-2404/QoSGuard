"""
QoSGuard Model Training
Training script for anomaly detection models with MLflow tracking.
"""

import os
import sys
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, List
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import shap
import structlog

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from data_pipeline.src.data_loader import DataLoader as QoSDataLoader
from app.models.ml_models import PyTorchMLP

logger = structlog.get_logger()


class ModelTrainer:
    """Trainer for anomaly detection models."""
    
    def __init__(self, data_path: str = "./data_pipeline/data", 
                 model_registry_path: str = "./models/registry"):
        self.data_path = data_path
        self.model_registry_path = Path(model_registry_path)
        self.model_registry_path.mkdir(parents=True, exist_ok=True)
        
        self.data_loader = QoSDataLoader(data_path)
        self.models = {}
        self.metrics = {}
        
    def train_all_models(self, experiment_name: str = "qosguard_training") -> Dict[str, Any]:
        """Train all models and return results."""
        logger.info("Starting model training", experiment=experiment_name)
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data")
        df = self.data_loader.load_unsw_nb15_sample()
        df_processed, artifacts = self.data_loader.preprocess_data(df)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_loader.split_data(df_processed)
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self.data_loader.handle_class_imbalance(
            X_train, y_train, method='undersample'
        )
        
        # Save preprocessing artifacts
        self._save_preprocessing_artifacts(artifacts, X_train.columns.tolist())
        
        # Train models
        results = {}
        
        # 1. Logistic Regression
        logger.info("Training Logistic Regression")
        lr_results = self._train_logistic_regression(
            X_train_balanced, y_train_balanced, X_val, y_val, X_test, y_test
        )
        results['logistic_regression'] = lr_results
        
        # 2. LightGBM
        logger.info("Training LightGBM")
        lgb_results = self._train_lightgbm(
            X_train_balanced, y_train_balanced, X_val, y_val, X_test, y_test
        )
        results['lightgbm'] = lgb_results
        
        # 3. PyTorch MLP
        logger.info("Training PyTorch MLP")
        mlp_results = self._train_pytorch_mlp(
            X_train_balanced, y_train_balanced, X_val, y_val, X_test, y_test
        )
        results['pytorch_mlp'] = mlp_results
        
        # Generate model comparison report
        self._generate_model_report(results)
        
        logger.info("Model training completed", results=list(results.keys()))
        return results
    
    def _train_logistic_regression(self, X_train, y_train, X_val, y_val, X_test, y_test) -> Dict:
        """Train Logistic Regression model."""
        with mlflow.start_run(run_name="logistic_regression"):
            # Train model
            lr = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
            
            # K-fold cross validation
            cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='roc_auc')
            mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_roc_auc_std", cv_scores.std())
            
            # Fit model
            lr.fit(X_train, y_train)
            
            # Calibrate probabilities
            calibrated_lr = CalibratedClassifierCV(lr, method='sigmoid', cv=3)
            calibrated_lr.fit(X_train, y_train)
            
            # Evaluate
            metrics = self._evaluate_model(calibrated_lr, X_val, y_val, X_test, y_test, "logistic_regression")
            
            # Log model
            mlflow.sklearn.log_model(calibrated_lr, "model")
            
            # Save model
            model_path = self.model_registry_path / "logistic_regression.pkl"
            joblib.dump(calibrated_lr, model_path)
            
            # Generate SHAP explanation
            self._generate_shap_explanation(calibrated_lr, X_test.head(100), "logistic_regression")
            
            self.models['logistic_regression'] = calibrated_lr
            return metrics
    
    def _train_lightgbm(self, X_train, y_train, X_val, y_val, X_test, y_test) -> Dict:
        """Train LightGBM model."""
        with mlflow.start_run(run_name="lightgbm"):
            # Parameters
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42,
                'class_weight': 'balanced'
            }
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train with early stopping
            model = lgb.train(
                params,
                train_data,
                valid_sets=[train_data, val_data],
                num_boost_round=1000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(0)  # Silent
                ]
            )
            
            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)
            mlflow.log_param("num_boost_rounds", model.num_trees())
            
            # Evaluate
            metrics = self._evaluate_model(model, X_val, y_val, X_test, y_test, "lightgbm")
            
            # Log model
            mlflow.lightgbm.log_model(model, "model")
            
            # Save model
            model_path = self.model_registry_path / "lightgbm.pkl"
            joblib.dump(model, model_path)
            
            # Generate SHAP explanation
            self._generate_shap_explanation(model, X_test.head(100), "lightgbm")
            
            self.models['lightgbm'] = model
            return metrics
    
    def _train_pytorch_mlp(self, X_train, y_train, X_val, y_val, X_test, y_test) -> Dict:
        """Train PyTorch MLP model."""
        with mlflow.start_run(run_name="pytorch_mlp"):
            # Parameters
            input_size = X_train.shape[1]
            hidden_sizes = [64, 32]
            dropout = 0.2
            lr = 0.001
            batch_size = 256
            epochs = 100
            
            # Log parameters
            mlflow.log_param("input_size", input_size)
            mlflow.log_param("hidden_sizes", hidden_sizes)
            mlflow.log_param("dropout", dropout)
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("epochs", epochs)
            
            # Create model
            model = PyTorchMLP(input_size, hidden_sizes, dropout)
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train.values)
            y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
            X_val_tensor = torch.FloatTensor(X_val.values)
            y_val_tensor = torch.FloatTensor(y_val.values).unsqueeze(1)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Loss function and optimizer
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            # Training loop
            model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                # Validation
                if epoch % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val_tensor)
                        val_loss = criterion(val_outputs, y_val_tensor)
                        mlflow.log_metric("val_loss", val_loss.item(), step=epoch)
                    model.train()
                
                mlflow.log_metric("train_loss", total_loss / len(train_loader), step=epoch)
            
            # Switch to evaluation mode
            model.eval()
            
            # Evaluate
            metrics = self._evaluate_model(model, X_val, y_val, X_test, y_test, "pytorch_mlp")
            
            # Log model
            mlflow.pytorch.log_model(model, "model")
            
            # Save model
            model_path = self.model_registry_path / "pytorch_mlp.pth"
            torch.save(model.state_dict(), model_path)
            
            self.models['pytorch_mlp'] = model
            return metrics
    
    def _evaluate_model(self, model, X_val, y_val, X_test, y_test, model_name: str) -> Dict:
        """Evaluate model and log metrics."""
        metrics = {}
        
        # Get predictions
        if model_name == "pytorch_mlp":
            # PyTorch model
            with torch.no_grad():
                val_proba = model(torch.FloatTensor(X_val.values)).numpy().flatten()
                test_proba = model(torch.FloatTensor(X_test.values)).numpy().flatten()
        elif model_name == "lightgbm":
            # LightGBM model
            val_proba = model.predict(X_val)
            test_proba = model.predict(X_test)
        else:
            # Sklearn model
            val_proba = model.predict_proba(X_val)[:, 1]
            test_proba = model.predict_proba(X_test)[:, 1]
        
        # Convert probabilities to binary predictions
        val_pred = (val_proba > 0.5).astype(int)
        test_pred = (test_proba > 0.5).astype(int)
        
        # Calculate metrics for validation set
        val_metrics = {
            'val_accuracy': accuracy_score(y_val, val_pred),
            'val_precision': precision_score(y_val, val_pred, average='binary'),
            'val_recall': recall_score(y_val, val_pred, average='binary'),
            'val_f1': f1_score(y_val, val_pred, average='binary'),
            'val_roc_auc': roc_auc_score(y_val, val_proba),
            'val_pr_auc': average_precision_score(y_val, val_proba)
        }
        
        # Calculate metrics for test set
        test_metrics = {
            'test_accuracy': accuracy_score(y_test, test_pred),
            'test_precision': precision_score(y_test, test_pred, average='binary'),
            'test_recall': recall_score(y_test, test_pred, average='binary'),
            'test_f1': f1_score(y_test, test_pred, average='binary'),
            'test_roc_auc': roc_auc_score(y_test, test_proba),
            'test_pr_auc': average_precision_score(y_test, test_proba)
        }
        
        # Combine metrics
        metrics.update(val_metrics)
        metrics.update(test_metrics)
        
        # Log metrics to MLflow
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        
        # Calculate recall at different FPR thresholds
        fpr, tpr, _ = roc_curve(y_test, test_proba)
        for target_fpr in [0.01, 0.05, 0.1]:
            idx = np.argmin(np.abs(fpr - target_fpr))
            recall_at_fpr = tpr[idx]
            metrics[f'recall_at_fpr_{target_fpr}'] = recall_at_fpr
            mlflow.log_metric(f'recall_at_fpr_{target_fpr}', recall_at_fpr)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, test_pred)
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        })
        
        logger.info(f"Model evaluation completed: {model_name}",
                   test_roc_auc=metrics['test_roc_auc'],
                   test_pr_auc=metrics['test_pr_auc'],
                   test_f1=metrics['test_f1'])
        
        return metrics
    
    def _generate_shap_explanation(self, model, X_sample, model_name: str):
        """Generate SHAP explanations for model."""
        try:
            if model_name == "pytorch_mlp":
                # Skip SHAP for PyTorch for now (requires custom wrapper)
                logger.info("Skipping SHAP for PyTorch model")
                return
            
            # Create SHAP explainer
            explainer = shap.Explainer(model)
            shap_values = explainer(X_sample)
            
            # Create summary plot
            import matplotlib.pyplot as plt
            shap.summary_plot(shap_values, X_sample, show=False)
            plot_path = self.model_registry_path / f"{model_name}_shap_summary.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log plot to MLflow
            mlflow.log_artifact(str(plot_path))
            
            logger.info(f"SHAP explanation generated for {model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to generate SHAP explanation for {model_name}", error=str(e))
    
    def _save_preprocessing_artifacts(self, artifacts: Dict, feature_names: List[str]):
        """Save preprocessing artifacts."""
        # Save scaler
        scaler_path = self.model_registry_path / "scaler.pkl"
        joblib.dump(artifacts['scaler'], scaler_path)
        
        # Save label encoders
        encoders_path = self.model_registry_path / "encoders.pkl"
        joblib.dump(artifacts['label_encoders'], encoders_path)
        
        # Save feature names
        features_path = self.model_registry_path / "feature_names.pkl"
        joblib.dump(feature_names, features_path)
        
        logger.info("Preprocessing artifacts saved")
    
    def _generate_model_report(self, results: Dict[str, Dict]):
        """Generate model comparison report."""
        report_lines = ["# QoSGuard Model Training Report\n"]
        report_lines.append(f"Training completed at: {datetime.now().isoformat()}\n")
        report_lines.append("## Model Performance Comparison\n")
        
        # Create comparison table
        header = "| Model | Test ROC-AUC | Test PR-AUC | Test F1 | Test Recall | Test Precision |"
        separator = "|-------|--------------|-------------|---------|-------------|----------------|"
        report_lines.extend([header, separator])
        
        for model_name, metrics in results.items():
            row = (f"| {model_name} | "
                  f"{metrics['test_roc_auc']:.4f} | "
                  f"{metrics['test_pr_auc']:.4f} | "
                  f"{metrics['test_f1']:.4f} | "
                  f"{metrics['test_recall']:.4f} | "
                  f"{metrics['test_precision']:.4f} |")
            report_lines.append(row)
        
        # Write report
        report_path = self.model_registry_path / "training_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info("Model training report generated", path=str(report_path))


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train QoSGuard anomaly detection models")
    parser.add_argument("--data-path", default="./data_pipeline/data", help="Path to data directory")
    parser.add_argument("--model-path", default="./models/registry", help="Path to model registry")
    parser.add_argument("--experiment", default="qosguard_training", help="MLflow experiment name")
    
    args = parser.parse_args()
    
    # Setup logging
    import structlog
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Train models
    trainer = ModelTrainer(args.data_path, args.model_path)
    results = trainer.train_all_models(args.experiment)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Models saved to: {args.model_path}")
    print(f"MLflow tracking: Check your MLflow UI")
    print("="*80)


if __name__ == "__main__":
    main()
