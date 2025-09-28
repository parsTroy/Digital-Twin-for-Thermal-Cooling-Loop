"""
Unit tests for the ML anomaly detector module.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os
import tempfile
import shutil

# Add the parent directory to the path to import the twin module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from twin.ml_detector import MLAnomalyDetector, create_ml_detector


class TestMLAnomalyDetector(unittest.TestCase):
    """Test cases for the MLAnomalyDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = MLAnomalyDetector(
            feature_engineering=True,
            auto_tuning=False,  # Disable for faster tests
            ensemble_methods=True,
            model_persistence=False  # Disable for tests
        )
        
        # Generate training data
        np.random.seed(42)
        self.training_data = []
        self.training_labels = []
        
        # Normal data
        for _ in range(100):
            self.training_data.append({
                'T_hot': np.random.normal(0, 1.0),
                'T_cold': np.random.normal(0, 0.8),
                'm_dot': np.random.normal(0, 0.1)
            })
            self.training_labels.append(False)
        
        # Anomalous data
        for _ in range(20):
            self.training_data.append({
                'T_hot': np.random.normal(0, 5.0),  # Higher variance
                'T_cold': np.random.normal(0, 4.0),
                'm_dot': np.random.normal(0, 0.5)
            })
            self.training_labels.append(True)
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertIsInstance(self.detector, MLAnomalyDetector)
        self.assertTrue(self.detector.feature_engineering)
        self.assertFalse(self.detector.auto_tuning)
        self.assertTrue(self.detector.ensemble_methods)
        self.assertFalse(self.detector.is_trained)
    
    def test_add_training_data(self):
        """Test adding training data."""
        # Test with labels
        self.detector.add_training_data(self.training_data[:50], self.training_labels[:50])
        
        self.assertIsNotNone(self.detector.training_data)
        self.assertIsNotNone(self.detector.training_labels)
        self.assertEqual(len(self.detector.training_data), 50)
        self.assertEqual(len(self.detector.training_labels), 50)
        
        # Test auto-labeling
        detector2 = MLAnomalyDetector()
        detector2.add_training_data(self.training_data[:30], auto_label=True)
        
        self.assertIsNotNone(detector2.training_data)
        self.assertIsNotNone(detector2.training_labels)
        self.assertEqual(len(detector2.training_labels), 30)
    
    def test_feature_engineering(self):
        """Test feature engineering functionality."""
        # Create simple data
        data = pd.DataFrame({
            'T_hot': [1.0, 2.0, 3.0, 4.0, 5.0],
            'T_cold': [0.5, 1.0, 1.5, 2.0, 2.5],
            'm_dot': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        # Test feature engineering
        engineered = self.detector.engineer_features(data)
        
        self.assertIsInstance(engineered, pd.DataFrame)
        self.assertGreater(len(engineered.columns), len(data.columns))
        self.assertIn('T_hot_rolling_mean_5', engineered.columns)
        self.assertIn('T_hot_squared', engineered.columns)
        self.assertIn('residual_magnitude', engineered.columns)
        
        # Test without feature engineering
        detector_no_fe = MLAnomalyDetector(feature_engineering=False)
        engineered_no_fe = detector_no_fe.engineer_features(data)
        
        self.assertEqual(len(engineered_no_fe.columns), len(data.columns))
    
    def test_train_models(self):
        """Test model training."""
        # Add training data
        self.detector.add_training_data(self.training_data, self.training_labels)
        
        # Train models
        results = self.detector.train_models()
        
        self.assertTrue(self.detector.is_trained)
        self.assertIsInstance(results, dict)
        
        # Check that models were trained
        expected_models = ['isolation_forest', 'one_class_svm', 'mlp', 'random_forest']
        for model_name in expected_models:
            self.assertIn(model_name, self.detector.models)
            self.assertIn(model_name, results)
        
        # Check ensemble model
        if self.detector.ensemble_methods:
            self.assertIsNotNone(self.detector.ensemble_model)
            self.assertIn('ensemble', results)
        
        # Check scalers and selectors
        self.assertIn('main', self.detector.scalers)
        if len(self.detector.training_data.columns) > self.detector.config['n_features_select']:
            self.assertIn('main', self.detector.feature_selectors)
    
    def test_predict_anomaly_single(self):
        """Test single anomaly prediction."""
        # Train models first
        self.detector.add_training_data(self.training_data, self.training_labels)
        self.detector.train_models()
        
        # Test normal data
        normal_data = {'T_hot': 0.5, 'T_cold': 0.3, 'm_dot': 0.05}
        result = self.detector.predict_anomaly(normal_data)
        
        self.assertIn('index', result)
        self.assertIn('input_data', result)
        self.assertIn('predictions', result)
        self.assertIn('scores', result)
        self.assertIn('ensemble_prediction', result)
        self.assertIn('ensemble_score', result)
        
        # Test anomalous data
        anomalous_data = {'T_hot': 5.0, 'T_cold': 4.0, 'm_dot': 0.5}
        result = self.detector.predict_anomaly(anomalous_data)
        
        self.assertIsInstance(result['ensemble_prediction'], bool)
        self.assertIsInstance(result['ensemble_score'], float)
    
    def test_predict_anomaly_batch(self):
        """Test batch anomaly prediction."""
        # Train models first
        self.detector.add_training_data(self.training_data, self.training_labels)
        self.detector.train_models()
        
        # Test batch prediction
        batch_data = [
            {'T_hot': 0.5, 'T_cold': 0.3, 'm_dot': 0.05},
            {'T_hot': 5.0, 'T_cold': 4.0, 'm_dot': 0.5},
            {'T_hot': 1.0, 'T_cold': 0.8, 'm_dot': 0.1}
        ]
        
        results = self.detector.predict_anomaly(batch_data)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        
        for i, result in enumerate(results):
            self.assertEqual(result['index'], i)
            self.assertIn('predictions', result)
            self.assertIn('scores', result)
    
    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        # Train models first
        self.detector.add_training_data(self.training_data, self.training_labels)
        self.detector.train_models()
        
        # Get feature importance
        importance = self.detector.get_feature_importance('random_forest')
        
        self.assertIsInstance(importance, dict)
        self.assertGreater(len(importance), 0)
        
        # Check that importance values are sorted
        importance_values = list(importance.values())
        self.assertEqual(importance_values, sorted(importance_values, reverse=True))
    
    def test_get_model_performance(self):
        """Test model performance retrieval."""
        # Train models first
        self.detector.add_training_data(self.training_data, self.training_labels)
        self.detector.train_models()
        
        # Get performance metrics
        performance = self.detector.get_model_performance()
        
        self.assertIsInstance(performance, dict)
        
        # Check that performance metrics exist
        expected_models = ['isolation_forest', 'one_class_svm', 'mlp', 'random_forest']
        for model_name in expected_models:
            if model_name in performance:
                model_perf = performance[model_name]
                self.assertIn('train_accuracy', model_perf)
                self.assertIn('test_accuracy', model_perf)
                self.assertIn('cv_mean', model_perf)
    
    def test_model_persistence(self):
        """Test model saving and loading."""
        # Create detector with persistence enabled
        detector = MLAnomalyDetector(model_persistence=True)
        
        # Train models
        detector.add_training_data(self.training_data, self.training_labels)
        detector.train_models()
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save models
            detector._save_models(temp_dir)
            
            # Check that files were created
            files = os.listdir(temp_dir)
            self.assertGreater(len(files), 0)
            
            # Create new detector and load models
            new_detector = MLAnomalyDetector(model_persistence=True)
            new_detector.load_models(temp_dir)
            
            # Check that models were loaded
            self.assertTrue(new_detector.is_trained)
            self.assertGreater(len(new_detector.models), 0)
            
            # Test prediction with loaded models
            test_data = {'T_hot': 1.0, 'T_cold': 0.8, 'm_dot': 0.1}
            result = new_detector.predict_anomaly(test_data)
            
            self.assertIn('ensemble_prediction', result)
    
    def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning functionality."""
        # Create detector with tuning enabled
        detector = MLAnomalyDetector(auto_tuning=True)
        
        # Add training data
        detector.add_training_data(self.training_data, self.training_labels)
        
        # Train with tuning
        results = detector.train_models(hyperparameter_tuning=True)
        
        self.assertTrue(detector.is_trained)
        self.assertIn('random_forest', detector.models)
        self.assertIn('mlp', detector.models)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty data
        detector = MLAnomalyDetector()
        detector.add_training_data([])
        
        with self.assertRaises(ValueError):
            detector.train_models()
        
        # Test with insufficient data
        detector.add_training_data(self.training_data[:5])
        
        with self.assertRaises(ValueError):
            detector.train_models()
        
        # Test prediction without training
        detector = MLAnomalyDetector()
        
        with self.assertRaises(ValueError):
            detector.predict_anomaly({'T_hot': 1.0, 'T_cold': 0.8, 'm_dot': 0.1})
        
        # Test with invalid model name
        detector.add_training_data(self.training_data, self.training_labels)
        detector.train_models()
        
        with self.assertRaises(ValueError):
            detector.predict_anomaly({'T_hot': 1.0, 'T_cold': 0.8, 'm_dot': 0.1}, model_name='invalid_model')
    
    def test_reset(self):
        """Test detector reset functionality."""
        # Train models
        self.detector.add_training_data(self.training_data, self.training_labels)
        self.detector.train_models()
        
        # Reset detector
        self.detector.reset()
        
        self.assertFalse(self.detector.is_trained)
        self.assertEqual(len(self.detector.models), 0)
        self.assertEqual(len(self.detector.scalers), 0)
        self.assertEqual(len(self.detector.feature_selectors), 0)
        self.assertIsNone(self.detector.ensemble_model)
        self.assertEqual(len(self.detector.feature_names), 0)
    
    def test_different_configurations(self):
        """Test different detector configurations."""
        # Test without feature engineering
        detector_no_fe = MLAnomalyDetector(feature_engineering=False)
        detector_no_fe.add_training_data(self.training_data, self.training_labels)
        results_no_fe = detector_no_fe.train_models()
        
        self.assertTrue(detector_no_fe.is_trained)
        self.assertIn('random_forest', results_no_fe)
        
        # Test without ensemble methods
        detector_no_ensemble = MLAnomalyDetector(ensemble_methods=False)
        detector_no_ensemble.add_training_data(self.training_data, self.training_labels)
        results_no_ensemble = detector_no_ensemble.train_models()
        
        self.assertTrue(detector_no_ensemble.is_trained)
        self.assertIsNone(detector_no_ensemble.ensemble_model)
        self.assertNotIn('ensemble', results_no_ensemble)
    
    def test_feature_engineering_edge_cases(self):
        """Test feature engineering with edge cases."""
        # Test with single column
        single_col_data = pd.DataFrame({'T_hot': [1.0, 2.0, 3.0, 4.0, 5.0]})
        engineered = self.detector.engineer_features(single_col_data)
        
        self.assertIsInstance(engineered, pd.DataFrame)
        self.assertGreater(len(engineered.columns), 1)
        
        # Test with NaN values
        nan_data = pd.DataFrame({
            'T_hot': [1.0, np.nan, 3.0, np.nan, 5.0],
            'T_cold': [0.5, 1.0, np.nan, 2.0, 2.5]
        })
        engineered_nan = self.detector.engineer_features(nan_data)
        
        self.assertIsInstance(engineered_nan, pd.DataFrame)
        self.assertFalse(engineered_nan.isnull().any().any())


class TestMLDetectorFactory(unittest.TestCase):
    """Test cases for the ML detector factory function."""
    
    def test_create_ml_detector(self):
        """Test ML detector creation with factory function."""
        detector = create_ml_detector()
        
        self.assertIsInstance(detector, MLAnomalyDetector)
        self.assertTrue(detector.feature_engineering)
        self.assertTrue(detector.auto_tuning)
        self.assertTrue(detector.ensemble_methods)
        self.assertTrue(detector.model_persistence)
    
    def test_create_ml_detector_custom_config(self):
        """Test ML detector creation with custom configuration."""
        detector = create_ml_detector(
            feature_engineering=False,
            auto_tuning=False,
            ensemble_methods=False
        )
        
        self.assertIsInstance(detector, MLAnomalyDetector)
        self.assertFalse(detector.feature_engineering)
        self.assertFalse(detector.auto_tuning)
        self.assertFalse(detector.ensemble_methods)


if __name__ == '__main__':
    unittest.main()
