"""
AutoML Engine - Automatic model training and comparison
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import mlflow
import mlflow.sklearn
import shap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import warnings
warnings.filterwarnings('ignore')


class AutoMLEngine:
    """
    Automatic Machine Learning Engine
    Trains multiple models and compares performance
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
        use_mlflow: bool = True,
        experiment_name: str = "AutoML_Experiments"
    ):
        """
        Initialize AutoML Engine

        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
            use_mlflow: Whether to track experiments with MLflow
            experiment_name: Name of MLflow experiment
        """
        self.df = df.copy()
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.use_mlflow = use_mlflow
        self.experiment_name = experiment_name

        self.problem_type = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = None
        self.scaler = None
        self.feature_names = None

        self.models = {}
        self.results = {}

        # Setup MLflow
        if self.use_mlflow:
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment(self.experiment_name)

    def detect_problem_type(self) -> str:
        """
        Detect if it's a classification or regression problem

        Returns:
            'classification' or 'regression'
        """
        target = self.df[self.target_column]

        # Check if target is numeric
        if pd.api.types.is_numeric_dtype(target):
            unique_ratio = target.nunique() / len(target)

            # If less than 5% unique values, it's likely classification
            if unique_ratio < 0.05 or target.nunique() <= 10:
                self.problem_type = "classification"
            else:
                self.problem_type = "regression"
        else:
            # Non-numeric is always classification
            self.problem_type = "classification"

        return self.problem_type

    def prepare_data(self) -> Dict[str, Any]:
        """
        Prepare data for training (encoding, scaling, splitting)

        Returns:
            Dictionary with preparation info
        """
        # Separate features and target
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        # Store original feature names
        self.feature_names = list(X.columns)

        # Encode categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # Encode target if classification with strings
        if self.problem_type == "classification" and not pd.api.types.is_numeric_dtype(y):
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=y if self.problem_type == "classification" else None
        )

        # Scale features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        return {
            "n_samples": len(self.df),
            "n_features": len(self.feature_names),
            "n_train": len(self.X_train),
            "n_test": len(self.X_test),
            "problem_type": self.problem_type,
            "feature_names": self.feature_names,
            "target_classes": list(self.label_encoder.classes_) if self.label_encoder else None
        }

    def get_models(self) -> Dict[str, Any]:
        """
        Get models based on problem type

        Returns:
            Dictionary of model name -> model instance
        """
        if self.problem_type == "classification":
            return {
                "Logistic Regression": LogisticRegression(random_state=self.random_state, max_iter=1000),
                "Random Forest": RandomForestClassifier(random_state=self.random_state, n_estimators=100),
                "Gradient Boosting": GradientBoostingClassifier(random_state=self.random_state),
                "XGBoost": XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                "LightGBM": LGBMClassifier(random_state=self.random_state, verbose=-1)
            }
        else:  # regression
            return {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(random_state=self.random_state),
                "Random Forest": RandomForestRegressor(random_state=self.random_state, n_estimators=100),
                "Gradient Boosting": GradientBoostingRegressor(random_state=self.random_state),
                "XGBoost": XGBRegressor(random_state=self.random_state),
                "LightGBM": LGBMRegressor(random_state=self.random_state, verbose=-1)
            }

    def train_models(self) -> Dict[str, Any]:
        """
        Train all models and collect results

        Returns:
            Dictionary with training results for all models
        """
        self.models = self.get_models()
        self.results = {}

        for name, model in self.models.items():
            # Start MLflow run for each model
            if self.use_mlflow:
                with mlflow.start_run(run_name=name):
                    result = self._train_single_model(name, model)
                    self.results[name] = result
            else:
                result = self._train_single_model(name, model)
                self.results[name] = result

        return self.results

    def _train_single_model(self, name: str, model) -> Dict[str, Any]:
        """Train a single model and log to MLflow"""
        try:
            # Log parameters
            if self.use_mlflow:
                mlflow.log_param("model_type", name)
                mlflow.log_param("problem_type", self.problem_type)
                mlflow.log_param("test_size", self.test_size)
                mlflow.log_param("n_features", len(self.feature_names))
                mlflow.log_param("target_column", self.target_column)

            # Train model
            model.fit(self.X_train, self.y_train)

            # Make predictions
            y_pred = model.predict(self.X_test)

            # Calculate metrics
            if self.problem_type == "classification":
                metrics = self._calculate_classification_metrics(y_pred)
            else:
                metrics = self._calculate_regression_metrics(y_pred)

            # Cross-validation score
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=5,
                scoring='accuracy' if self.problem_type == "classification" else 'r2'
            )

            # Feature importance (if available)
            feature_importance = self._get_feature_importance(model)

            # Log metrics to MLflow
            if self.use_mlflow:
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                mlflow.log_metric("cv_score_mean", float(cv_scores.mean()))
                mlflow.log_metric("cv_score_std", float(cv_scores.std()))

                # Log model
                mlflow.sklearn.log_model(model, "model")

                # Log feature importance as artifact
                if feature_importance:
                    import json
                    with open("feature_importance.json", "w") as f:
                        json.dump(feature_importance, f, indent=2)
                    mlflow.log_artifact("feature_importance.json")

            return {
                "model": model,
                "metrics": metrics,
                "cv_score_mean": float(cv_scores.mean()),
                "cv_score_std": float(cv_scores.std()),
                "feature_importance": feature_importance,
                "status": "success"
            }

        except Exception as e:
            if self.use_mlflow:
                mlflow.log_param("error", str(e))
            return {
                "status": "error",
                "error": str(e)
            }

    def _calculate_classification_metrics(self, y_pred) -> Dict[str, float]:
        """Calculate classification metrics"""
        return {
            "accuracy": float(accuracy_score(self.y_test, y_pred)),
            "precision": float(precision_score(self.y_test, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(self.y_test, y_pred, average='weighted', zero_division=0)),
            "f1_score": float(f1_score(self.y_test, y_pred, average='weighted', zero_division=0))
        }

    def _calculate_regression_metrics(self, y_pred) -> Dict[str, float]:
        """Calculate regression metrics"""
        return {
            "r2_score": float(r2_score(self.y_test, y_pred)),
            "mse": float(mean_squared_error(self.y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(self.y_test, y_pred))),
            "mae": float(mean_absolute_error(self.y_test, y_pred))
        }

    def _get_feature_importance(self, model) -> Optional[List[Dict[str, Any]]]:
        """Get feature importance from model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = [
                    {
                        "feature": self.feature_names[i],
                        "importance": float(importances[i])
                    }
                    for i in range(len(self.feature_names))
                ]
                # Sort by importance
                feature_importance.sort(key=lambda x: x['importance'], reverse=True)
                return feature_importance[:10]  # Top 10 features

            elif hasattr(model, 'coef_'):
                # For linear models
                coefs = np.abs(model.coef_)
                if len(coefs.shape) > 1:
                    coefs = coefs.mean(axis=0)

                feature_importance = [
                    {
                        "feature": self.feature_names[i],
                        "importance": float(coefs[i])
                    }
                    for i in range(len(self.feature_names))
                ]
                feature_importance.sort(key=lambda x: x['importance'], reverse=True)
                return feature_importance[:10]

        except Exception as e:
            return None

        return None

    def compute_shap_values(self, model_name: str = None, max_samples: int = 100) -> Optional[Dict[str, Any]]:
        """
        Compute SHAP values for model interpretability

        Args:
            model_name: Name of model to explain (uses best model if None)
            max_samples: Maximum number of samples to use for SHAP computation

        Returns:
            Dictionary with SHAP values and explanations
        """
        try:
            # Get model to explain
            if model_name is None:
                model_name, model_info = self.get_best_model()
            else:
                model_info = self.results.get(model_name)

            if not model_info or model_info.get('status') != 'success':
                return None

            model = model_info['model']

            # Use subset of training data for SHAP background
            n_samples = min(max_samples, len(self.X_train))
            background_data = self.X_train[:n_samples]

            # Choose appropriate SHAP explainer
            if hasattr(model, 'predict_proba'):
                # Tree-based models
                if 'Forest' in model_name or 'Boosting' in model_name or 'XGB' in model_name or 'LightGBM' in model_name:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(background_data)
                else:
                    # Linear models
                    explainer = shap.KernelExplainer(model.predict_proba, background_data)
                    shap_values = explainer.shap_values(background_data)
            else:
                # Regression models
                if 'Forest' in model_name or 'Boosting' in model_name or 'XGB' in model_name or 'LightGBM' in model_name:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(background_data)
                else:
                    # Linear regression models
                    explainer = shap.LinearExplainer(model, background_data)
                    shap_values = explainer.shap_values(background_data)

            # Handle multi-output SHAP values (classification)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Use first class for summary

            # Calculate mean absolute SHAP values for feature importance
            mean_shap = np.abs(shap_values).mean(axis=0)

            # Create feature importance based on SHAP
            shap_importance = [
                {
                    "feature": self.feature_names[i],
                    "shap_importance": float(mean_shap[i])
                }
                for i in range(len(self.feature_names))
            ]
            shap_importance.sort(key=lambda x: x['shap_importance'], reverse=True)

            return {
                "model_name": model_name,
                "shap_feature_importance": shap_importance[:10],  # Top 10 features
                "n_samples_used": n_samples,
                "explainer_type": type(explainer).__name__
            }

        except Exception as e:
            return {
                "error": f"Failed to compute SHAP values: {str(e)}"
            }

    def get_best_model(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get the best performing model

        Returns:
            Tuple of (model_name, model_results)
        """
        if not self.results:
            return None, None

        # Filter out failed models
        successful_models = {
            name: result for name, result in self.results.items()
            if result.get('status') == 'success'
        }

        if not successful_models:
            return None, None

        # Get best based on primary metric
        if self.problem_type == "classification":
            metric_key = "accuracy"
        else:
            metric_key = "r2_score"

        best_model_name = max(
            successful_models.keys(),
            key=lambda name: successful_models[name]['metrics'][metric_key]
        )

        return best_model_name, successful_models[best_model_name]

    def get_summary(self) -> Dict[str, Any]:
        """
        Get training summary

        Returns:
            Summary dictionary with all results
        """
        best_model_name, best_model_info = self.get_best_model()

        # Prepare results for JSON serialization (exclude model objects)
        results_summary = {}
        for name, result in self.results.items():
            if result.get('status') == 'success':
                results_summary[name] = {
                    "metrics": result['metrics'],
                    "cv_score_mean": result['cv_score_mean'],
                    "cv_score_std": result['cv_score_std'],
                    "feature_importance": result['feature_importance']
                }
            else:
                results_summary[name] = {
                    "status": "error",
                    "error": result.get('error')
                }

        return {
            "problem_type": self.problem_type,
            "n_models_trained": len(self.results),
            "best_model": best_model_name,
            "best_model_score": best_model_info['metrics'] if best_model_info else None,
            "all_results": results_summary,
            "data_info": {
                "n_samples": len(self.df),
                "n_features": len(self.feature_names),
                "feature_names": self.feature_names,
                "target_column": self.target_column
            }
        }
