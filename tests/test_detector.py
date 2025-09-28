"""
Unit tests for the anomaly detector module.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add the parent directory to the path to import the twin module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from twin.detector import (
    AnomalyDetector, 
    DetectionMethod, 
    AnomalyType,
    create_default_detector,
    create_advanced_detector
)


class TestAnomalyDetector(unittest.TestCase):
    """Test cases for the AnomalyDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = AnomalyDetector(
            detection_methods=[
                DetectionMethod.RESIDUAL_THRESHOLD,
                DetectionMethod.ROLLING_Z_SCORE,
                DetectionMethod.ISOLATION_FOREST
            ],
            window_size=50
        )
        
        # Generate training data
        np.random.seed(42)
        self.training_data = []
        for _ in range(100):
            self.training_data.append({
                'T_hot': np.random.normal(0, 1.0),
                'T_cold': np.random.normal(0, 0.8),
                'm_dot': np.random.normal(0, 0.1)
            })
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertIsInstance(self.detector, AnomalyDetector)
        self.assertEqual(len(self.detector.detection_methods), 3)
        self.assertEqual(self.detector.window_size, 50)
        self.assertFalse(self.detector.is_trained)
    
    def test_add_residuals(self):
        """Test adding residuals to history."""
        residuals = {'T_hot': 1.0, 'T_cold': 0.5, 'm_dot': 0.1}
        
        self.detector.add_residuals(residuals, 0.0)
        
        self.assertEqual(len(self.detector.residual_history), 1)
        self.assertEqual(self.detector.residual_history[0]['residuals'], residuals)
        self.assertEqual(self.detector.residual_history[0]['timestamp'], 0.0)
    
    def test_residual_threshold_detection(self):
        """Test residual threshold detection."""
        # Test normal residuals
        normal_residuals = {'T_hot': 1.0, 'T_cold': 0.5, 'm_dot': 0.1}
        result = self.detector._residual_threshold_detection(normal_residuals)
        
        self.assertFalse(result['is_anomaly'])
        self.assertEqual(result['anomaly_type'], AnomalyType.NORMAL)
        
        # Test anomalous residuals
        anomalous_residuals = {'T_hot': 10.0, 'T_cold': 5.0, 'm_dot': 1.0}
        result = self.detector._residual_threshold_detection(anomalous_residuals)
        
        self.assertTrue(result['is_anomaly'])
        self.assertIn(result['anomaly_type'], [AnomalyType.WARNING, AnomalyType.CRITICAL, AnomalyType.FAULT])
    
    def test_rolling_z_score_detection(self):
        """Test rolling z-score detection."""
        # Add some normal residuals first
        for i in range(30):
            residuals = {
                'T_hot': np.random.normal(0, 1.0),
                'T_cold': np.random.normal(0, 0.8),
                'm_dot': np.random.normal(0, 0.1)
            }
            self.detector.add_residuals(residuals, float(i))
        
        # Test normal residual
        normal_residuals = {'T_hot': 0.5, 'T_cold': 0.3, 'm_dot': 0.05}
        result = self.detector._rolling_z_score_detection(normal_residuals, 30.0)
        
        self.assertFalse(result['is_anomaly'])
        self.assertEqual(result['anomaly_type'], AnomalyType.NORMAL)
        
        # Test anomalous residual
        anomalous_residuals = {'T_hot': 5.0, 'T_cold': 4.0, 'm_dot': 0.5}
        result = self.detector._rolling_z_score_detection(anomalous_residuals, 31.0)
        
        self.assertTrue(result['is_anomaly'])
        self.assertIn(result['anomaly_type'], [AnomalyType.WARNING, AnomalyType.CRITICAL, AnomalyType.FAULT])
    
    def test_detect_anomalies(self):
        """Test comprehensive anomaly detection."""
        # Train detector first
        self.detector.train(self.training_data)
        
        # Test normal residuals
        normal_residuals = {'T_hot': 0.5, 'T_cold': 0.3, 'm_dot': 0.05}
        result = self.detector.detect_anomalies(normal_residuals, 0.0)
        
        self.assertIn('timestamp', result)
        self.assertIn('residuals', result)
        self.assertIn('anomaly_scores', result)
        self.assertIn('anomaly_types', result)
        self.assertIn('overall_anomaly', result)
        self.assertIn('overall_severity', result)
        self.assertIn('detection_methods', result)
        
        # Test anomalous residuals
        anomalous_residuals = {'T_hot': 10.0, 'T_cold': 8.0, 'm_dot': 1.0}
        result = self.detector.detect_anomalies(anomalous_residuals, 1.0)
        
        # Should detect anomaly with residual threshold method
        self.assertTrue(result['overall_anomaly'])
        self.assertIn('residual_threshold', result['anomaly_scores'])
    
    def test_training(self):
        """Test detector training."""
        self.assertFalse(self.detector.is_trained)
        
        # Train detector
        self.detector.train(self.training_data)
        
        self.assertTrue(self.detector.is_trained)
        self.assertIn('T_hot', self.detector.baseline_stats)
        self.assertIn('T_cold', self.detector.baseline_stats)
        self.assertIn('m_dot', self.detector.baseline_stats)
        
        # Check baseline statistics
        for sensor in ['T_hot', 'T_cold', 'm_dot']:
            stats = self.detector.baseline_stats[sensor]
            self.assertIn('mean', stats)
            self.assertIn('std', stats)
            self.assertIn('min', stats)
            self.assertIn('max', stats)
    
    def test_ml_models_training(self):
        """Test ML models training."""
        # Create detector with ML methods
        detector = AnomalyDetector(
            detection_methods=[
                DetectionMethod.ISOLATION_FOREST,
                DetectionMethod.ONE_CLASS_SVM
            ]
        )
        
        # Train detector
        detector.train(self.training_data)
        
        self.assertTrue(detector.is_trained)
        self.assertIn('isolation_forest', detector.models)
        self.assertIn('one_class_svm', detector.models)
        self.assertIn('one_class_svm', detector.scalers)
    
    def test_anomaly_summary(self):
        """Test anomaly summary generation."""
        # Add some anomalies
        for i in range(10):
            residuals = {'T_hot': 10.0, 'T_cold': 8.0, 'm_dot': 1.0}
            self.detector.detect_anomalies(residuals, float(i))
        
        summary = self.detector.get_anomaly_summary()
        
        self.assertIn('total_anomalies', summary)
        self.assertIn('anomaly_types', summary)
        self.assertIn('recent_anomalies', summary)
        self.assertIn('detection_methods_used', summary)
        
        self.assertGreater(summary['total_anomalies'], 0)
    
    def test_config_update(self):
        """Test configuration update."""
        original_threshold = self.detector.config['residual_threshold']
        
        # Update configuration
        new_config = {'residual_threshold': 10.0}
        self.detector.update_config(new_config)
        
        self.assertEqual(self.detector.config['residual_threshold'], 10.0)
        self.assertNotEqual(self.detector.config['residual_threshold'], original_threshold)
    
    def test_reset(self):
        """Test detector reset."""
        # Train detector and add some data
        self.detector.train(self.training_data)
        self.detector.add_residuals({'T_hot': 1.0, 'T_cold': 0.5, 'm_dot': 0.1})
        
        # Reset detector
        self.detector.reset()
        
        self.assertFalse(self.detector.is_trained)
        self.assertEqual(len(self.detector.baseline_stats), 0)
        self.assertEqual(len(self.detector.models), 0)
        self.assertEqual(len(self.detector.residual_history), 0)
        self.assertEqual(len(self.detector.anomaly_history), 0)
    
    def test_detection_methods_coverage(self):
        """Test that all detection methods are covered."""
        methods = [
            DetectionMethod.RESIDUAL_THRESHOLD,
            DetectionMethod.Z_SCORE,
            DetectionMethod.ROLLING_Z_SCORE,
            DetectionMethod.ISOLATION_FOREST,
            DetectionMethod.ONE_CLASS_SVM,
            DetectionMethod.STATISTICAL_PROCESS_CONTROL,
            DetectionMethod.CUSUM
        ]
        
        for method in methods:
            detector = AnomalyDetector(detection_methods=[method])
            
            # Test that method can be called without error
            residuals = {'T_hot': 1.0, 'T_cold': 0.5, 'm_dot': 0.1}
            
            try:
                result = detector._run_detection_method(method, residuals, 0.0)
                self.assertIn('is_anomaly', result)
                self.assertIn('score', result)
                self.assertIn('anomaly_type', result)
            except Exception as e:
                self.fail(f"Detection method {method.value} failed: {str(e)}")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty residuals
        empty_residuals = {}
        result = self.detector.detect_anomalies(empty_residuals)
        self.assertFalse(result['overall_anomaly'])
        
        # Test with insufficient history for rolling methods
        detector = AnomalyDetector(
            detection_methods=[DetectionMethod.ROLLING_Z_SCORE],
            window_size=100
        )
        
        residuals = {'T_hot': 1.0, 'T_cold': 0.5, 'm_dot': 0.1}
        result = detector.detect_anomalies(residuals)
        
        # Should not detect anomaly with insufficient history
        self.assertFalse(result['overall_anomaly'])
        
        # Test with invalid detection method
        with self.assertRaises(ValueError):
            detector._run_detection_method("invalid_method", residuals, 0.0)


class TestDetectorFactories(unittest.TestCase):
    """Test cases for detector factory functions."""
    
    def test_create_default_detector(self):
        """Test default detector creation."""
        detector = create_default_detector()
        
        self.assertIsInstance(detector, AnomalyDetector)
        self.assertEqual(len(detector.detection_methods), 3)
        self.assertIn(DetectionMethod.RESIDUAL_THRESHOLD, detector.detection_methods)
        self.assertIn(DetectionMethod.ROLLING_Z_SCORE, detector.detection_methods)
        self.assertIn(DetectionMethod.ISOLATION_FOREST, detector.detection_methods)
    
    def test_create_advanced_detector(self):
        """Test advanced detector creation."""
        detector = create_advanced_detector()
        
        self.assertIsInstance(detector, AnomalyDetector)
        self.assertEqual(len(detector.detection_methods), 7)
        self.assertEqual(detector.window_size, 200)
        self.assertEqual(detector.threshold_multiplier, 1.5)


if __name__ == '__main__':
    unittest.main()
