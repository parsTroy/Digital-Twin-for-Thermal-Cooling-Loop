"""
Unit tests for the enhanced anomaly detector module.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os
import time
import threading

# Add the parent directory to the path to import the twin module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from twin.enhanced_detector import EnhancedAnomalyDetector, create_enhanced_detector
from twin.detector import DetectionMethod, AnomalyType


class TestEnhancedAnomalyDetector(unittest.TestCase):
    """Test cases for the EnhancedAnomalyDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = EnhancedAnomalyDetector(
            traditional_methods=[DetectionMethod.RESIDUAL_THRESHOLD, DetectionMethod.ROLLING_Z_SCORE],
            ml_enabled=True,
            adaptive_thresholds=True,
            ensemble_voting=True
        )
        
        # Generate test data
        np.random.seed(42)
        self.normal_data = []
        self.anomalous_data = []
        
        # Normal residuals
        for _ in range(50):
            self.normal_data.append({
                'T_hot': np.random.normal(0, 1.0),
                'T_cold': np.random.normal(0, 0.8),
                'm_dot': np.random.normal(0, 0.1)
            })
        
        # Anomalous residuals
        for _ in range(20):
            self.anomalous_data.append({
                'T_hot': np.random.normal(0, 5.0),
                'T_cold': np.random.normal(0, 4.0),
                'm_dot': np.random.normal(0, 0.5)
            })
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertIsInstance(self.detector, EnhancedAnomalyDetector)
        self.assertTrue(self.detector.ml_enabled)
        self.assertTrue(self.detector.adaptive_thresholds)
        self.assertTrue(self.detector.ensemble_voting)
        self.assertIsNotNone(self.detector.traditional_detector)
        self.assertIsNotNone(self.detector.ml_detector)
    
    def test_add_residuals(self):
        """Test adding residuals to the detector."""
        residuals = {'T_hot': 1.0, 'T_cold': 0.5, 'm_dot': 0.1}
        
        self.detector.add_residuals(residuals, 0.0)
        
        self.assertEqual(len(self.detector.residual_history), 1)
        self.assertEqual(self.detector.residual_history[0]['residuals'], residuals)
        self.assertEqual(self.detector.residual_history[0]['timestamp'], 0.0)
        
        # Check that traditional detector was updated
        self.assertEqual(len(self.detector.traditional_detector.residual_history), 1)
    
    def test_detect_anomalies_traditional_only(self):
        """Test anomaly detection using traditional methods only."""
        # Add some normal data first
        for i, data in enumerate(self.normal_data[:10]):
            self.detector.add_residuals(data, float(i))
        
        # Test normal residual
        normal_residual = {'T_hot': 0.5, 'T_cold': 0.3, 'm_dot': 0.05}
        result = self.detector.detect_anomalies(normal_residual, use_ensemble=False)
        
        self.assertIn('timestamp', result)
        self.assertIn('residuals', result)
        self.assertIn('traditional_results', result)
        self.assertIn('final_decision', result)
        
        # Test anomalous residual
        anomalous_residual = {'T_hot': 10.0, 'T_cold': 8.0, 'm_dot': 1.0}
        result = self.detector.detect_anomalies(anomalous_residual, use_ensemble=False)
        
        self.assertIn('final_decision', result)
        self.assertIn('is_anomaly', result['final_decision'])
        self.assertIn('confidence', result['final_decision'])
        self.assertIn('severity', result['final_decision'])
    
    def test_detect_anomalies_with_ml(self):
        """Test anomaly detection with ML methods."""
        # Train ML models first
        training_data = self.normal_data + self.anomalous_data
        training_results = self.detector.train_ml_models(training_data)
        
        self.assertIsInstance(training_results, dict)
        
        # Add some data for traditional detector
        for i, data in enumerate(self.normal_data[:10]):
            self.detector.add_residuals(data, float(i))
        
        # Test detection with ML
        test_residual = {'T_hot': 1.0, 'T_cold': 0.8, 'm_dot': 0.1}
        result = self.detector.detect_anomalies(test_residual, use_ensemble=True)
        
        self.assertIn('traditional_results', result)
        self.assertIn('ml_results', result)
        self.assertIn('ensemble_result', result)
        self.assertIn('final_decision', result)
        
        # Check ensemble result
        if result['ensemble_result']:
            self.assertIn('is_anomaly', result['ensemble_result'])
            self.assertIn('confidence', result['ensemble_result'])
            self.assertIn('method_used', result['ensemble_result'])
    
    def test_ensemble_decision(self):
        """Test ensemble decision making."""
        # Mock traditional and ML results
        traditional_results = {
            'overall_anomaly': True,
            'overall_severity': AnomalyType.WARNING,
            'anomaly_scores': {'residual_threshold': 1.5}
        }
        
        ml_results = {
            'ensemble_prediction': True,
            'ensemble_score': 0.8
        }
        
        # Test ensemble decision
        ensemble_result = self.detector._make_ensemble_decision(traditional_results, ml_results)
        
        self.assertIn('is_anomaly', ensemble_result)
        self.assertIn('confidence', ensemble_result)
        self.assertIn('severity', ensemble_result)
        self.assertIn('method_used', ensemble_result)
        self.assertIn('details', ensemble_result)
        
        # Check details
        details = ensemble_result['details']
        self.assertIn('traditional_vote', details)
        self.assertIn('ml_vote', details)
        self.assertIn('weighted_score', details)
        self.assertIn('method_agreement', details)
    
    def test_train_ml_models(self):
        """Test ML model training."""
        # Test with provided data
        training_data = self.normal_data + self.anomalous_data
        results = self.detector.train_ml_models(training_data)
        
        self.assertIsInstance(results, dict)
        self.assertIn('random_forest', results)
        self.assertIn('mlp', results)
        
        # Test with historical data
        for i, data in enumerate(training_data):
            self.detector.add_residuals(data, float(i))
        
        results_historical = self.detector.train_ml_models()
        
        self.assertIsInstance(results_historical, dict)
    
    def test_adaptive_learning(self):
        """Test adaptive learning functionality."""
        # Start adaptive learning
        self.detector.start_adaptive_learning()
        
        self.assertIsNotNone(self.detector.adaptive_thread)
        self.assertTrue(self.detector.adaptive_thread.is_alive())
        
        # Let it run briefly
        time.sleep(0.1)
        
        # Stop adaptive learning
        self.detector.stop_adaptive_learning()
        
        # Check that thread stopped
        self.assertFalse(self.detector.adaptive_thread.is_alive())
    
    def test_get_performance_metrics(self):
        """Test performance metrics retrieval."""
        # Add some data and anomalies
        for i, data in enumerate(self.normal_data[:20]):
            self.detector.add_residuals(data, float(i))
        
        # Add some anomalies
        for i, data in enumerate(self.anomalous_data[:5]):
            result = self.detector.detect_anomalies(data, float(i + 20))
            if result['final_decision']['is_anomaly']:
                self.detector.anomaly_history.append(result)
        
        # Get performance metrics
        metrics = self.detector.get_performance_metrics()
        
        self.assertIn('traditional_performance', metrics)
        self.assertIn('ml_performance', metrics)
        self.assertIn('ensemble_weights', metrics)
        self.assertIn('config', metrics)
        self.assertIn('total_anomalies', metrics)
        self.assertIn('total_residuals', metrics)
    
    def test_export_detection_data(self):
        """Test detection data export."""
        # Add some data and anomalies
        for i, data in enumerate(self.normal_data[:10]):
            self.detector.add_residuals(data, float(i))
        
        # Add some anomalies
        for i, data in enumerate(self.anomalous_data[:3]):
            result = self.detector.detect_anomalies(data, float(i + 10))
            if result['final_decision']['is_anomaly']:
                self.detector.anomaly_history.append(result)
        
        # Export data
        filename = self.detector.export_detection_data()
        
        self.assertTrue(os.path.exists(filename))
        
        # Check file content
        df = pd.read_csv(filename)
        self.assertGreater(len(df), 0)
        
        # Check required columns
        required_columns = [
            'timestamp', 'is_anomaly', 'confidence', 'severity', 'method_used'
        ]
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Clean up
        os.remove(filename)
    
    def test_reset(self):
        """Test detector reset functionality."""
        # Add some data
        for i, data in enumerate(self.normal_data[:10]):
            self.detector.add_residuals(data, float(i))
        
        # Reset detector
        self.detector.reset()
        
        self.assertEqual(len(self.detector.residual_history), 0)
        self.assertEqual(len(self.detector.anomaly_history), 0)
        self.assertEqual(len(self.detector.performance_history), 0)
        self.assertEqual(len(self.detector.ensemble_weights), 0)
        self.assertEqual(len(self.detector.method_performance), 0)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty residuals
        empty_residuals = {}
        result = self.detector.detect_anomalies(empty_residuals)
        
        self.assertIn('final_decision', result)
        self.assertFalse(result['final_decision']['is_anomaly'])
        
        # Test ML training with insufficient data
        detector_no_ml = EnhancedAnomalyDetector(ml_enabled=False)
        result = detector_no_ml.train_ml_models()
        
        self.assertIn('error', result)
        
        # Test ensemble decision with missing ML results
        traditional_results = {'overall_anomaly': True, 'overall_severity': AnomalyType.WARNING}
        ml_results = {}
        
        ensemble_result = self.detector._make_ensemble_decision(traditional_results, ml_results)
        
        self.assertIn('is_anomaly', ensemble_result)
        self.assertIn('confidence', ensemble_result)
    
    def test_adaptive_thresholds(self):
        """Test adaptive threshold functionality."""
        # Add some anomalies to trigger adaptive behavior
        for i in range(20):
            anomalous_data = {'T_hot': 10.0, 'T_cold': 8.0, 'm_dot': 1.0}
            result = self.detector.detect_anomalies(anomalous_data, float(i))
            if result['final_decision']['is_anomaly']:
                self.detector.anomaly_history.append(result)
        
        # Test adaptive threshold update
        original_threshold = self.detector.config['ensemble_threshold']
        self.detector._update_adaptive_thresholds()
        
        # Threshold should be adjusted based on performance
        new_threshold = self.detector.config['ensemble_threshold']
        self.assertIsInstance(new_threshold, float)
        self.assertGreaterEqual(new_threshold, 0.0)
        self.assertLessEqual(new_threshold, 1.0)
    
    def test_ensemble_weights_update(self):
        """Test ensemble weights update functionality."""
        # Mock training results
        training_results = {
            'traditional_results': {'test_accuracy': 0.8},
            'ensemble': {'test_accuracy': 0.9}
        }
        
        # Update weights
        self.detector._update_ensemble_weights(training_results)
        
        # Check that weights were updated
        self.assertIn('traditional', self.detector.ensemble_weights)
        self.assertIn('ml', self.detector.ensemble_weights)
        
        # Weights should be normalized
        total_weight = (self.detector.ensemble_weights['traditional'] + 
                       self.detector.ensemble_weights['ml'])
        self.assertAlmostEqual(total_weight, 1.0, places=2)


class TestEnhancedDetectorFactory(unittest.TestCase):
    """Test cases for the enhanced detector factory function."""
    
    def test_create_enhanced_detector(self):
        """Test enhanced detector creation with factory function."""
        detector = create_enhanced_detector()
        
        self.assertIsInstance(detector, EnhancedAnomalyDetector)
        self.assertTrue(detector.ml_enabled)
        self.assertTrue(detector.adaptive_thresholds)
        self.assertTrue(detector.ensemble_voting)
    
    def test_create_enhanced_detector_custom_config(self):
        """Test enhanced detector creation with custom configuration."""
        detector = create_enhanced_detector(
            traditional_methods=[DetectionMethod.RESIDUAL_THRESHOLD],
            ml_enabled=False,
            adaptive_thresholds=False,
            ensemble_voting=False
        )
        
        self.assertIsInstance(detector, EnhancedAnomalyDetector)
        self.assertFalse(detector.ml_enabled)
        self.assertFalse(detector.adaptive_thresholds)
        self.assertFalse(detector.ensemble_voting)
        self.assertIsNone(detector.ml_detector)


if __name__ == '__main__':
    unittest.main()
