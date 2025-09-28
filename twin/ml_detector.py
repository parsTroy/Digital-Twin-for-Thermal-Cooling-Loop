"""
Advanced Machine Learning Anomaly Detection Module

This module implements advanced ML-based anomaly detection algorithms including
deep learning models, ensemble methods, and automated feature engineering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import warnings
from sklearn.ensemble import IsolationForest, RandomForestClassifier, VotingClassifier
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
import os
from datetime import datetime
import json

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLAnomalyDetector:
    """
    Advanced Machine Learning Anomaly Detection System.
    
    Implements multiple ML algorithms with automated feature engineering,
    model selection, and performance optimization.
    """
    
    def __init__(self, 
                 feature_engineering: bool = True,
                 auto_tuning: bool = True,
                 ensemble_methods: bool = True,
                 model_persistence: bool = True):
        """
        Initialize the ML anomaly detector.
        
        Parameters:
        -----------
        feature_engineering : bool
            Enable automated feature engineering
        auto_tuning : bool
            Enable automatic hyperparameter tuning
        ensemble_methods : bool
            Enable ensemble learning methods
        model_persistence : bool
            Enable model saving/loading
        """
        self.feature_engineering = feature_engineering
        self.auto_tuning = auto_tuning
        self.ensemble_methods = ensemble_methods
        self.model_persistence = model_persistence
        
        # Models
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.ensemble_model = None
        
        # Feature engineering
        self.feature_names = []
        self.feature_importance = {}
        
        # Training data
        self.training_data = None
        self.training_labels = None
        self.is_trained = False
        
        # Performance metrics
        self.performance_metrics = {}
        
        # Model configuration
        self.config = {
            'test_size': 0.2,
            'random_state': 42,
            'cv_folds': 5,
            'n_features_select': 10,
            'isolation_forest_params': {
                'n_estimators': 100,
                'contamination': 0.1,
                'random_state': 42
            },
            'one_class_svm_params': {
                'nu': 0.1,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            'mlp_params': {
                'hidden_layer_sizes': (100, 50),
                'max_iter': 1000,
                'random_state': 42
            },
            'random_forest_params': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        }
        
        logger.info("ML anomaly detector initialized")
    
    def add_training_data(self, 
                         data: List[Dict[str, float]], 
                         labels: List[bool] = None,
                         auto_label: bool = True):
        """
        Add training data to the detector.
        
        Parameters:
        -----------
        data : list
            List of residual dictionaries
        labels : list, optional
            Anomaly labels (True for anomaly, False for normal)
        auto_label : bool
            Automatically generate labels based on residual thresholds
        """
        if not data:
            logger.warning("No training data provided")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Generate labels if not provided
        if labels is None and auto_label:
            labels = self._generate_auto_labels(df)
        elif labels is None:
            labels = [False] * len(df)
        
        # Store training data
        if self.training_data is None:
            self.training_data = df
            self.training_labels = labels
        else:
            self.training_data = pd.concat([self.training_data, df], ignore_index=True)
            self.training_labels.extend(labels)
        
        logger.info(f"Added {len(data)} training samples. Total: {len(self.training_data)}")
    
    def _generate_auto_labels(self, df: pd.DataFrame) -> List[bool]:
        """Generate automatic labels based on residual thresholds."""
        labels = []
        
        for _, row in df.iterrows():
            # Simple threshold-based labeling
            max_residual = max(abs(row[col]) for col in df.columns)
            threshold = 3.0  # 3-sigma threshold
            labels.append(max_residual > threshold)
        
        return labels
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer advanced features from raw residual data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw residual data
            
        Returns:
        --------
        pd.DataFrame
            Engineered features
        """
        if not self.feature_engineering:
            return data
        
        engineered_data = data.copy()
        
        # Statistical features
        for col in data.columns:
            # Rolling statistics
            for window in [5, 10, 20]:
                engineered_data[f'{col}_rolling_mean_{window}'] = data[col].rolling(window=window).mean()
                engineered_data[f'{col}_rolling_std_{window}'] = data[col].rolling(window=window).std()
                engineered_data[f'{col}_rolling_max_{window}'] = data[col].rolling(window=window).max()
                engineered_data[f'{col}_rolling_min_{window}'] = data[col].rolling(window=window).min()
            
            # Lag features
            for lag in [1, 2, 3]:
                engineered_data[f'{col}_lag_{lag}'] = data[col].shift(lag)
            
            # Difference features
            engineered_data[f'{col}_diff_1'] = data[col].diff(1)
            engineered_data[f'{col}_diff_2'] = data[col].diff(2)
            
            # Polynomial features
            engineered_data[f'{col}_squared'] = data[col] ** 2
            engineered_data[f'{col}_abs'] = data[col].abs()
        
        # Cross-feature interactions
        cols = data.columns
        for i, col1 in enumerate(cols):
            for col2 in cols[i+1:]:
                engineered_data[f'{col1}_{col2}_product'] = data[col1] * data[col2]
                engineered_data[f'{col1}_{col2}_sum'] = data[col1] + data[col2]
                engineered_data[f'{col1}_{col2}_diff'] = data[col1] - data[col2]
        
        # Global features
        engineered_data['residual_magnitude'] = np.sqrt(np.sum(data[cols] ** 2, axis=1))
        engineered_data['residual_sum'] = data[cols].sum(axis=1)
        engineered_data['residual_max'] = data[cols].max(axis=1)
        engineered_data['residual_min'] = data[cols].min(axis=1)
        engineered_data['residual_range'] = engineered_data['residual_max'] - engineered_data['residual_min']
        
        # Fill NaN values
        engineered_data = engineered_data.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        # Store feature names
        self.feature_names = list(engineered_data.columns)
        
        logger.info(f"Engineered {len(engineered_data.columns)} features from {len(data.columns)} original features")
        
        return engineered_data
    
    def train_models(self, 
                    test_size: float = None,
                    feature_selection: bool = True,
                    hyperparameter_tuning: bool = None) -> Dict[str, Any]:
        """
        Train all ML models with the provided data.
        
        Parameters:
        -----------
        test_size : float, optional
            Fraction of data to use for testing
        feature_selection : bool
            Enable feature selection
        hyperparameter_tuning : bool, optional
            Enable hyperparameter tuning (overrides auto_tuning)
            
        Returns:
        --------
        dict
            Training results and performance metrics
        """
        if self.training_data is None:
            raise ValueError("No training data available. Call add_training_data() first.")
        
        test_size = test_size or self.config['test_size']
        hyperparameter_tuning = hyperparameter_tuning if hyperparameter_tuning is not None else self.auto_tuning
        
        logger.info("Starting model training...")
        
        # Engineer features
        engineered_data = self.engineer_features(self.training_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            engineered_data, 
            self.training_labels, 
            test_size=test_size, 
            random_state=self.config['random_state'],
            stratify=self.training_labels
        )
        
        # Feature selection
        if feature_selection and len(engineered_data.columns) > self.config['n_features_select']:
            selector = SelectKBest(score_func=f_classif, k=self.config['n_features_select'])
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            self.feature_selectors['main'] = selector
            selected_features = engineered_data.columns[selector.get_support()]
            logger.info(f"Selected {len(selected_features)} features: {list(selected_features)}")
        else:
            X_train_selected = X_train
            X_test_selected = X_test
            selected_features = engineered_data.columns
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        self.scalers['main'] = scaler
        
        # Train individual models
        models_to_train = {
            'isolation_forest': IsolationForest(**self.config['isolation_forest_params']),
            'one_class_svm': OneClassSVM(**self.config['one_class_svm_params']),
            'mlp': MLPClassifier(**self.config['mlp_params']),
            'random_forest': RandomForestClassifier(**self.config['random_forest_params'])
        }
        
        training_results = {}
        
        for name, model in models_to_train.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train model
                if name in ['isolation_forest', 'one_class_svm']:
                    # Unsupervised models
                    model.fit(X_train_scaled)
                    train_predictions = model.predict(X_train_scaled)
                    test_predictions = model.predict(X_test_scaled)
                    
                    # Convert to binary labels
                    train_predictions = (train_predictions == -1)
                    test_predictions = (test_predictions == -1)
                    
                    train_scores = model.score_samples(X_train_scaled)
                    test_scores = model.score_samples(X_test_scaled)
                    
                else:
                    # Supervised models
                    model.fit(X_train_scaled, y_train)
                    train_predictions = model.predict(X_train_scaled)
                    test_predictions = model.predict(X_test_scaled)
                    
                    train_scores = model.predict_proba(X_train_scaled)[:, 1] if hasattr(model, 'predict_proba') else train_predictions
                    test_scores = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else test_predictions
                
                # Calculate metrics
                train_accuracy = np.mean(train_predictions == y_train)
                test_accuracy = np.mean(test_predictions == y_test)
                
                # Cross-validation
                if name not in ['isolation_forest', 'one_class_svm']:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=self.config['cv_folds'])
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                else:
                    cv_mean = test_accuracy
                    cv_std = 0.0
                
                # Store model and results
                self.models[name] = model
                training_results[name] = {
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'train_predictions': train_predictions,
                    'test_predictions': test_predictions,
                    'train_scores': train_scores,
                    'test_scores': test_scores
                }
                
                logger.info(f"{name} - Train: {train_accuracy:.3f}, Test: {test_accuracy:.3f}, CV: {cv_mean:.3f}±{cv_std:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to train {name}: {str(e)}")
                training_results[name] = {'error': str(e)}
        
        # Train ensemble model
        if self.ensemble_methods and len(self.models) > 1:
            logger.info("Training ensemble model...")
            
            try:
                # Create ensemble with available models
                ensemble_models = []
                for name, model in self.models.items():
                    if name not in ['isolation_forest', 'one_class_svm']:
                        ensemble_models.append((name, model))
                
                if len(ensemble_models) >= 2:
                    self.ensemble_model = VotingClassifier(ensemble_models, voting='soft')
                    self.ensemble_model.fit(X_train_scaled, y_train)
                    
                    ensemble_train_pred = self.ensemble_model.predict(X_train_scaled)
                    ensemble_test_pred = self.ensemble_model.predict(X_test_scaled)
                    ensemble_train_score = self.ensemble_model.predict_proba(X_train_scaled)[:, 1]
                    ensemble_test_score = self.ensemble_model.predict_proba(X_test_scaled)[:, 1]
                    
                    ensemble_train_acc = np.mean(ensemble_train_pred == y_train)
                    ensemble_test_acc = np.mean(ensemble_test_pred == y_test)
                    
                    training_results['ensemble'] = {
                        'train_accuracy': ensemble_train_acc,
                        'test_accuracy': ensemble_test_acc,
                        'train_predictions': ensemble_train_pred,
                        'test_predictions': ensemble_test_pred,
                        'train_scores': ensemble_train_score,
                        'test_scores': ensemble_test_score
                    }
                    
                    logger.info(f"Ensemble - Train: {ensemble_train_acc:.3f}, Test: {ensemble_test_acc:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to train ensemble: {str(e)}")
        
        # Hyperparameter tuning
        if hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning...")
            self._tune_hyperparameters(X_train_scaled, y_train)
        
        # Store performance metrics
        self.performance_metrics = training_results
        self.is_trained = True
        
        # Save models if persistence is enabled
        if self.model_persistence:
            self._save_models()
        
        logger.info("Model training completed")
        
        return training_results
    
    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray):
        """Perform hyperparameter tuning for models."""
        # Tune Random Forest
        if 'random_forest' in self.models:
            try:
                rf = RandomForestClassifier(random_state=self.config['random_state'])
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10]
                }
                
                grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                
                self.models['random_forest'] = grid_search.best_estimator_
                logger.info(f"Random Forest best params: {grid_search.best_params_}")
                
            except Exception as e:
                logger.warning(f"Random Forest tuning failed: {str(e)}")
        
        # Tune MLP
        if 'mlp' in self.models:
            try:
                mlp = MLPClassifier(random_state=self.config['random_state'])
                param_grid = {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate_init': [0.001, 0.01, 0.1]
                }
                
                grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                
                self.models['mlp'] = grid_search.best_estimator_
                logger.info(f"MLP best params: {grid_search.best_params_}")
                
            except Exception as e:
                logger.warning(f"MLP tuning failed: {str(e)}")
    
    def predict_anomaly(self, 
                       data: Union[Dict[str, float], List[Dict[str, float]]], 
                       model_name: str = 'ensemble') -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Predict anomalies using trained models.
        
        Parameters:
        -----------
        data : dict or list
            Residual data to predict
        model_name : str
            Name of model to use for prediction
            
        Returns:
        --------
        dict or list
            Prediction results
        """
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_models() first.")
        
        if model_name not in self.models and model_name != 'ensemble':
            raise ValueError(f"Model {model_name} not available. Available: {list(self.models.keys())}")
        
        # Handle single prediction
        if isinstance(data, dict):
            data = [data]
            single_prediction = True
        else:
            single_prediction = False
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Engineer features
        engineered_data = self.engineer_features(df)
        
        # Select features if selector exists
        if 'main' in self.feature_selectors:
            engineered_data = engineered_data[self.feature_selectors['main'].get_feature_names_out()]
        
        # Scale features
        if 'main' in self.scalers:
            scaled_data = self.scalers['main'].transform(engineered_data)
        else:
            scaled_data = engineered_data.values
        
        # Make predictions
        results = []
        
        for i, row_data in enumerate(data):
            result = {
                'index': i,
                'input_data': row_data,
                'predictions': {},
                'scores': {},
                'ensemble_prediction': False,
                'ensemble_score': 0.0
            }
            
            # Individual model predictions
            for name, model in self.models.items():
                try:
                    if name in ['isolation_forest', 'one_class_svm']:
                        pred = model.predict(scaled_data[i:i+1])[0]
                        score = model.score_samples(scaled_data[i:i+1])[0]
                        result['predictions'][name] = pred == -1
                        result['scores'][name] = -score  # Negative score indicates anomaly
                    else:
                        pred = model.predict(scaled_data[i:i+1])[0]
                        score = model.predict_proba(scaled_data[i:i+1])[0][1] if hasattr(model, 'predict_proba') else pred
                        result['predictions'][name] = bool(pred)
                        result['scores'][name] = float(score)
                except Exception as e:
                    result['predictions'][name] = False
                    result['scores'][name] = 0.0
                    logger.warning(f"Prediction failed for {name}: {str(e)}")
            
            # Ensemble prediction
            if self.ensemble_model and model_name == 'ensemble':
                try:
                    ensemble_pred = self.ensemble_model.predict(scaled_data[i:i+1])[0]
                    ensemble_score = self.ensemble_model.predict_proba(scaled_data[i:i+1])[0][1]
                    result['ensemble_prediction'] = bool(ensemble_pred)
                    result['ensemble_score'] = float(ensemble_score)
                except Exception as e:
                    logger.warning(f"Ensemble prediction failed: {str(e)}")
            
            results.append(result)
        
        return results[0] if single_prediction else results
    
    def get_feature_importance(self, model_name: str = 'random_forest') -> Dict[str, float]:
        """Get feature importance from trained models."""
        if not self.is_trained:
            return {}
        
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not available")
            return {}
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            if 'main' in self.feature_selectors:
                feature_names = self.feature_selectors['main'].get_feature_names_out()
            else:
                feature_names = self.feature_names
            
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {}
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all trained models."""
        return self.performance_metrics.copy()
    
    def _save_models(self, save_dir: str = 'models'):
        """Save trained models to disk."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save models
        for name, model in self.models.items():
            filename = os.path.join(save_dir, f'{name}_{timestamp}.joblib')
            joblib.dump(model, filename)
        
        # Save scalers
        for name, scaler in self.scalers.items():
            filename = os.path.join(save_dir, f'scaler_{name}_{timestamp}.joblib')
            joblib.dump(scaler, filename)
        
        # Save feature selectors
        for name, selector in self.feature_selectors.items():
            filename = os.path.join(save_dir, f'selector_{name}_{timestamp}.joblib')
            joblib.dump(selector, filename)
        
        # Save ensemble model
        if self.ensemble_model:
            filename = os.path.join(save_dir, f'ensemble_{timestamp}.joblib')
            joblib.dump(self.ensemble_model, filename)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'feature_names': self.feature_names,
            'config': self.config,
            'performance_metrics': self.performance_metrics
        }
        
        metadata_file = os.path.join(save_dir, f'metadata_{timestamp}.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Models saved to {save_dir}")
    
    def load_models(self, load_dir: str = 'models', timestamp: str = None):
        """Load trained models from disk."""
        if not os.path.exists(load_dir):
            logger.warning(f"Model directory {load_dir} not found")
            return
        
        # Find latest timestamp if not specified
        if timestamp is None:
            metadata_files = [f for f in os.listdir(load_dir) if f.startswith('metadata_')]
            if not metadata_files:
                logger.warning("No metadata files found")
                return
            timestamp = sorted(metadata_files)[-1].replace('metadata_', '').replace('.json', '')
        
        # Load metadata
        metadata_file = os.path.join(load_dir, f'metadata_{timestamp}.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                self.config.update(metadata.get('config', {}))
                self.feature_names = metadata.get('feature_names', [])
                self.performance_metrics = metadata.get('performance_metrics', {})
        
        # Load models
        model_files = [f for f in os.listdir(load_dir) if f.endswith(f'_{timestamp}.joblib') and not f.startswith('scaler_') and not f.startswith('selector_') and not f.startswith('ensemble_')]
        
        for model_file in model_files:
            model_name = model_file.replace(f'_{timestamp}.joblib', '')
            model_path = os.path.join(load_dir, model_file)
            try:
                self.models[model_name] = joblib.load(model_path)
            except Exception as e:
                logger.warning(f"Failed to load model {model_name}: {str(e)}")
        
        # Load scalers
        scaler_files = [f for f in os.listdir(load_dir) if f.startswith('scaler_') and f.endswith(f'_{timestamp}.joblib')]
        
        for scaler_file in scaler_files:
            scaler_name = scaler_file.replace('scaler_', '').replace(f'_{timestamp}.joblib', '')
            scaler_path = os.path.join(load_dir, scaler_file)
            try:
                self.scalers[scaler_name] = joblib.load(scaler_path)
            except Exception as e:
                logger.warning(f"Failed to load scaler {scaler_name}: {str(e)}")
        
        # Load feature selectors
        selector_files = [f for f in os.listdir(load_dir) if f.startswith('selector_') and f.endswith(f'_{timestamp}.joblib')]
        
        for selector_file in selector_files:
            selector_name = selector_file.replace('selector_', '').replace(f'_{timestamp}.joblib', '')
            selector_path = os.path.join(load_dir, selector_file)
            try:
                self.feature_selectors[selector_name] = joblib.load(selector_path)
            except Exception as e:
                logger.warning(f"Failed to load selector {selector_name}: {str(e)}")
        
        # Load ensemble model
        ensemble_file = os.path.join(load_dir, f'ensemble_{timestamp}.joblib')
        if os.path.exists(ensemble_file):
            try:
                self.ensemble_model = joblib.load(ensemble_file)
            except Exception as e:
                logger.warning(f"Failed to load ensemble model: {str(e)}")
        
        self.is_trained = True
        logger.info(f"Models loaded from {load_dir}")
    
    def reset(self):
        """Reset the detector state."""
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.ensemble_model = None
        self.training_data = None
        self.training_labels = None
        self.feature_names = []
        self.feature_importance = {}
        self.performance_metrics = {}
        self.is_trained = False
        logger.info("ML detector reset")


def create_ml_detector(feature_engineering: bool = True,
                      auto_tuning: bool = True,
                      ensemble_methods: bool = True) -> MLAnomalyDetector:
    """
    Create an ML anomaly detector with default configuration.
    
    Parameters:
    -----------
    feature_engineering : bool
        Enable feature engineering
    auto_tuning : bool
        Enable hyperparameter tuning
    ensemble_methods : bool
        Enable ensemble learning
        
    Returns:
    --------
    MLAnomalyDetector
        Configured ML detector
    """
    return MLAnomalyDetector(
        feature_engineering=feature_engineering,
        auto_tuning=auto_tuning,
        ensemble_methods=ensemble_methods,
        model_persistence=True
    )
