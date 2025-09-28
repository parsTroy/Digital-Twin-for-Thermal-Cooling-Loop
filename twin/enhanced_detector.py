"""
Enhanced Anomaly Detection System

This module integrates traditional statistical methods with advanced machine learning
algorithms to provide a comprehensive anomaly detection solution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import threading
import queue
import time

from .detector import AnomalyDetector, DetectionMethod, AnomalyType
from .ml_detector import MLAnomalyDetector, create_ml_detector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedAnomalyDetector:
    """
    Enhanced anomaly detection system combining traditional and ML methods.
    
    Provides a unified interface for both statistical and machine learning
    based anomaly detection with automatic model selection and adaptation.
    """
    
    def __init__(self, 
                 traditional_methods: List[DetectionMethod] = None,
                 ml_enabled: bool = True,
                 adaptive_thresholds: bool = True,
                 ensemble_voting: bool = True,
                 update_rate: float = 1.0):
        """
        Initialize the enhanced anomaly detector.
        
        Parameters:
        -----------
        traditional_methods : list
            Traditional detection methods to use
        ml_enabled : bool
            Enable machine learning methods
        adaptive_thresholds : bool
            Enable adaptive threshold adjustment
        ensemble_voting : bool
            Enable ensemble voting between methods
        update_rate : float
            Update rate for adaptive features
        """
        self.ml_enabled = ml_enabled
        self.adaptive_thresholds = adaptive_thresholds
        self.ensemble_voting = ensemble_voting
        self.update_rate = update_rate
        
        # Traditional detector
        self.traditional_detector = AnomalyDetector(
            detection_methods=traditional_methods or [
                DetectionMethod.RESIDUAL_THRESHOLD,
                DetectionMethod.ROLLING_Z_SCORE,
                DetectionMethod.ISOLATION_FOREST
            ],
            window_size=100
        )
        
        # ML detector
        self.ml_detector = create_ml_detector() if ml_enabled else None
        
        # Ensemble configuration
        self.ensemble_weights = {}
        self.method_performance = {}
        self.adaptive_thresholds_config = {}
        
        # Data storage
        self.residual_history = []
        self.anomaly_history = []
        self.performance_history = []
        
        # Adaptive features
        self.adaptive_thread = None
        self.stop_adaptive = threading.Event()
        self.adaptive_queue = queue.Queue()
        
        # Configuration
        self.config = {
            'min_training_samples': 100,
            'retrain_interval': 3600,  # 1 hour
            'performance_window': 1000,
            'adaptive_learning_rate': 0.01,
            'ensemble_threshold': 0.5,
            'confidence_threshold': 0.7
        }
        
        logger.info("Enhanced anomaly detector initialized")
    
    def add_residuals(self, residuals: Dict[str, float], timestamp: float = None):
        """Add new residual data for analysis."""
        if timestamp is None:
            timestamp = len(self.residual_history)
        
        # Add to traditional detector
        self.traditional_detector.add_residuals(residuals, timestamp)
        
        # Add to ML detector if enabled
        if self.ml_enabled and self.ml_detector:
            self.ml_detector.add_training_data([residuals], auto_label=True)
        
        # Store in history
        self.residual_history.append({
            'timestamp': timestamp,
            'residuals': residuals.copy()
        })
        
        # Limit history size
        if len(self.residual_history) > self.config['performance_window']:
            self.residual_history = self.residual_history[-self.config['performance_window']:]
    
    def detect_anomalies(self, 
                        residuals: Dict[str, float], 
                        timestamp: float = None,
                        use_ensemble: bool = True) -> Dict[str, Any]:
        """
        Detect anomalies using enhanced methods.
        
        Parameters:
        -----------
        residuals : dict
            Residual data to analyze
        timestamp : float, optional
            Timestamp for the analysis
        use_ensemble : bool
            Use ensemble voting for final decision
            
        Returns:
        --------
        dict
            Enhanced detection results
        """
        if timestamp is None:
            timestamp = len(self.residual_history)
        
        # Add residuals to history
        self.add_residuals(residuals, timestamp)
        
        # Initialize results
        results = {
            'timestamp': timestamp,
            'residuals': residuals,
            'traditional_results': {},
            'ml_results': {},
            'ensemble_result': {},
            'final_decision': {
                'is_anomaly': False,
                'confidence': 0.0,
                'severity': AnomalyType.NORMAL,
                'method_used': 'none',
                'details': {}
            }
        }
        
        # Traditional detection
        traditional_results = self.traditional_detector.detect_anomalies(residuals, timestamp)
        results['traditional_results'] = traditional_results
        
        # ML detection
        if self.ml_enabled and self.ml_detector and self.ml_detector.is_trained:
            try:
                ml_results = self.ml_detector.predict_anomaly(residuals, model_name='ensemble')
                results['ml_results'] = ml_results
            except Exception as e:
                logger.warning(f"ML detection failed: {str(e)}")
                results['ml_results'] = {'error': str(e)}
        
        # Ensemble decision
        if use_ensemble and self.ensemble_voting:
            ensemble_result = self._make_ensemble_decision(traditional_results, results.get('ml_results', {}))
            results['ensemble_result'] = ensemble_result
            results['final_decision'] = ensemble_result
        else:
            # Use traditional results as fallback
            results['final_decision'] = {
                'is_anomaly': traditional_results.get('overall_anomaly', False),
                'confidence': self._calculate_confidence(traditional_results),
                'severity': traditional_results.get('overall_severity', AnomalyType.NORMAL),
                'method_used': 'traditional',
                'details': traditional_results
            }
        
        # Store anomaly history
        if results['final_decision']['is_anomaly']:
            self.anomaly_history.append(results)
            if len(self.anomaly_history) > self.config['performance_window']:
                self.anomaly_history = self.anomaly_history[-self.config['performance_window']:]
        
        return results
    
    def _make_ensemble_decision(self, 
                               traditional_results: Dict[str, Any], 
                               ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Make ensemble decision combining traditional and ML results."""
        # Initialize ensemble result
        ensemble_result = {
            'is_anomaly': False,
            'confidence': 0.0,
            'severity': AnomalyType.NORMAL,
            'method_used': 'ensemble',
            'details': {
                'traditional_vote': 0.0,
                'ml_vote': 0.0,
                'weighted_score': 0.0,
                'method_agreement': False
            }
        }
        
        # Traditional vote
        traditional_vote = 0.0
        if traditional_results.get('overall_anomaly', False):
            traditional_vote = 1.0
            # Weight by severity
            severity = traditional_results.get('overall_severity', AnomalyType.NORMAL)
            if severity == AnomalyType.CRITICAL:
                traditional_vote = 1.0
            elif severity == AnomalyType.WARNING:
                traditional_vote = 0.7
            elif severity == AnomalyType.FAULT:
                traditional_vote = 0.5
        
        # ML vote
        ml_vote = 0.0
        if ml_results and 'ensemble_prediction' in ml_results:
            ml_vote = float(ml_results.get('ensemble_score', 0.0))
        
        # Calculate ensemble weights
        traditional_weight = self.ensemble_weights.get('traditional', 0.5)
        ml_weight = self.ensemble_weights.get('ml', 0.5)
        
        # Normalize weights
        total_weight = traditional_weight + ml_weight
        if total_weight > 0:
            traditional_weight /= total_weight
            ml_weight /= total_weight
        
        # Calculate weighted score
        weighted_score = (traditional_vote * traditional_weight + ml_vote * ml_weight)
        
        # Make decision
        ensemble_result['is_anomaly'] = weighted_score > self.config['ensemble_threshold']
        ensemble_result['confidence'] = weighted_score
        ensemble_result['details']['traditional_vote'] = traditional_vote
        ensemble_result['details']['ml_vote'] = ml_vote
        ensemble_result['details']['weighted_score'] = weighted_score
        ensemble_result['details']['method_agreement'] = abs(traditional_vote - ml_vote) < 0.3
        
        # Determine severity
        if ensemble_result['is_anomaly']:
            if weighted_score > 0.8:
                ensemble_result['severity'] = AnomalyType.CRITICAL
            elif weighted_score > 0.6:
                ensemble_result['severity'] = AnomalyType.WARNING
            else:
                ensemble_result['severity'] = AnomalyType.FAULT
        
        return ensemble_result
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence score from detection results."""
        if not results.get('overall_anomaly', False):
            return 0.0
        
        # Get anomaly scores
        anomaly_scores = results.get('anomaly_scores', {})
        if not anomaly_scores:
            return 0.5
        
        # Calculate average confidence
        scores = list(anomaly_scores.values())
        return np.mean(scores) if scores else 0.0
    
    def train_ml_models(self, training_data: List[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Train ML models with provided or historical data.
        
        Parameters:
        -----------
        training_data : list, optional
            Training data. If None, uses historical data
            
        Returns:
        --------
        dict
            Training results
        """
        if not self.ml_enabled or not self.ml_detector:
            return {'error': 'ML detection not enabled'}
        
        # Use provided data or historical data
        if training_data is None:
            if len(self.residual_history) < self.config['min_training_samples']:
                return {'error': f'Insufficient training data. Need {self.config["min_training_samples"]}, have {len(self.residual_history)}'}
            training_data = [data['residuals'] for data in self.residual_history]
        
        # Train ML models
        self.ml_detector.add_training_data(training_data, auto_label=True)
        training_results = self.ml_detector.train_models()
        
        # Update ensemble weights based on performance
        self._update_ensemble_weights(training_results)
        
        logger.info("ML models trained successfully")
        return training_results
    
    def _update_ensemble_weights(self, training_results: Dict[str, Any]):
        """Update ensemble weights based on model performance."""
        # Calculate performance scores
        traditional_performance = 0.5  # Default
        ml_performance = 0.5  # Default
        
        # Get traditional performance (simplified)
        if 'traditional_results' in training_results:
            traditional_performance = training_results['traditional_results'].get('test_accuracy', 0.5)
        
        # Get ML performance
        if 'ensemble' in training_results:
            ml_performance = training_results['ensemble'].get('test_accuracy', 0.5)
        elif 'random_forest' in training_results:
            ml_performance = training_results['random_forest'].get('test_accuracy', 0.5)
        
        # Update weights with learning rate
        lr = self.config['adaptive_learning_rate']
        
        self.ensemble_weights['traditional'] = (
            self.ensemble_weights.get('traditional', 0.5) * (1 - lr) + 
            traditional_performance * lr
        )
        
        self.ensemble_weights['ml'] = (
            self.ensemble_weights.get('ml', 0.5) * (1 - lr) + 
            ml_performance * lr
        )
        
        # Normalize weights
        total_weight = self.ensemble_weights['traditional'] + self.ensemble_weights['ml']
        if total_weight > 0:
            self.ensemble_weights['traditional'] /= total_weight
            self.ensemble_weights['ml'] /= total_weight
        
        logger.info(f"Updated ensemble weights: traditional={self.ensemble_weights['traditional']:.3f}, ml={self.ensemble_weights['ml']:.3f}")
    
    def start_adaptive_learning(self):
        """Start adaptive learning thread."""
        if self.adaptive_thread and self.adaptive_thread.is_alive():
            logger.warning("Adaptive learning already running")
            return
        
        self.stop_adaptive.clear()
        self.adaptive_thread = threading.Thread(target=self._adaptive_learning_loop, daemon=True)
        self.adaptive_thread.start()
        logger.info("Adaptive learning started")
    
    def stop_adaptive_learning(self):
        """Stop adaptive learning thread."""
        if self.adaptive_thread:
            self.stop_adaptive.set()
            self.adaptive_thread.join(timeout=5.0)
            logger.info("Adaptive learning stopped")
    
    def _adaptive_learning_loop(self):
        """Adaptive learning main loop."""
        last_retrain = time.time()
        
        while not self.stop_adaptive.is_set():
            try:
                current_time = time.time()
                
                # Check if retraining is needed
                if (current_time - last_retrain) > self.config['retrain_interval']:
                    if len(self.residual_history) >= self.config['min_training_samples']:
                        logger.info("Performing adaptive retraining...")
                        self.train_ml_models()
                        last_retrain = current_time
                
                # Update adaptive thresholds
                if self.adaptive_thresholds:
                    self._update_adaptive_thresholds()
                
                # Sleep for update interval
                time.sleep(1.0 / self.update_rate)
                
            except Exception as e:
                logger.error(f"Error in adaptive learning loop: {str(e)}")
                time.sleep(1.0)
    
    def _update_adaptive_thresholds(self):
        """Update adaptive thresholds based on recent performance."""
        if len(self.anomaly_history) < 10:
            return
        
        # Analyze recent performance
        recent_anomalies = self.anomaly_history[-50:]  # Last 50 anomalies
        
        # Calculate false positive rate
        false_positives = sum(1 for anomaly in recent_anomalies 
                            if anomaly['final_decision']['confidence'] < self.config['confidence_threshold'])
        
        false_positive_rate = false_positives / len(recent_anomalies) if recent_anomalies else 0
        
        # Adjust thresholds based on performance
        if false_positive_rate > 0.2:  # High false positive rate
            self.config['ensemble_threshold'] = min(0.8, self.config['ensemble_threshold'] + 0.05)
        elif false_positive_rate < 0.05:  # Low false positive rate
            self.config['ensemble_threshold'] = max(0.3, self.config['ensemble_threshold'] - 0.05)
        
        logger.debug(f"Updated ensemble threshold to {self.config['ensemble_threshold']:.3f}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = {
            'traditional_performance': self.traditional_detector.get_anomaly_summary(),
            'ml_performance': self.ml_detector.get_model_performance() if self.ml_enabled else {},
            'ensemble_weights': self.ensemble_weights.copy(),
            'config': self.config.copy(),
            'total_anomalies': len(self.anomaly_history),
            'total_residuals': len(self.residual_history)
        }
        
        # Calculate accuracy metrics
        if self.anomaly_history:
            recent_anomalies = self.anomaly_history[-100:]  # Last 100 anomalies
            high_confidence_anomalies = [a for a in recent_anomalies 
                                       if a['final_decision']['confidence'] > self.config['confidence_threshold']]
            
            metrics['recent_anomalies'] = len(recent_anomalies)
            metrics['high_confidence_anomalies'] = len(high_confidence_anomalies)
            metrics['confidence_rate'] = len(high_confidence_anomalies) / len(recent_anomalies) if recent_anomalies else 0
        
        return metrics
    
    def export_detection_data(self, filename: str = None) -> str:
        """Export detection data to CSV."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'enhanced_detection_data_{timestamp}.csv'
        
        # Prepare data for export
        export_data = []
        for anomaly in self.anomaly_history:
            row = {
                'timestamp': anomaly['timestamp'],
                'is_anomaly': anomaly['final_decision']['is_anomaly'],
                'confidence': anomaly['final_decision']['confidence'],
                'severity': anomaly['final_decision']['severity'].value,
                'method_used': anomaly['final_decision']['method_used']
            }
            
            # Add residual data
            for sensor, value in anomaly['residuals'].items():
                row[f'residual_{sensor}'] = value
            
            # Add traditional results
            traditional = anomaly.get('traditional_results', {})
            row['traditional_anomaly'] = traditional.get('overall_anomaly', False)
            row['traditional_severity'] = traditional.get('overall_severity', AnomalyType.NORMAL).value
            
            # Add ML results
            ml = anomaly.get('ml_results', {})
            if ml and 'ensemble_prediction' in ml:
                row['ml_anomaly'] = ml['ensemble_prediction']
                row['ml_score'] = ml['ensemble_score']
            
            export_data.append(row)
        
        # Create DataFrame and export
        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False)
        
        logger.info(f"Detection data exported to {filename}")
        return filename
    
    def reset(self):
        """Reset the enhanced detector."""
        self.traditional_detector.reset()
        
        if self.ml_detector:
            self.ml_detector.reset()
        
        self.residual_history = []
        self.anomaly_history = []
        self.performance_history = []
        self.ensemble_weights = {}
        self.method_performance = {}
        
        self.stop_adaptive_learning()
        
        logger.info("Enhanced detector reset")


def create_enhanced_detector(traditional_methods: List[DetectionMethod] = None,
                           ml_enabled: bool = True,
                           adaptive_thresholds: bool = True,
                           ensemble_voting: bool = True) -> EnhancedAnomalyDetector:
    """
    Create an enhanced anomaly detector with default configuration.
    
    Parameters:
    -----------
    traditional_methods : list, optional
        Traditional detection methods
    ml_enabled : bool
        Enable ML methods
    adaptive_thresholds : bool
        Enable adaptive thresholds
    ensemble_voting : bool
        Enable ensemble voting
        
    Returns:
    --------
    EnhancedAnomalyDetector
        Configured enhanced detector
    """
    return EnhancedAnomalyDetector(
        traditional_methods=traditional_methods,
        ml_enabled=ml_enabled,
        adaptive_thresholds=adaptive_thresholds,
        ensemble_voting=ensemble_voting
    )
