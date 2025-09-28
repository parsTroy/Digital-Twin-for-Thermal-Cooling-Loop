"""
Anomaly Detection Module for Thermal Cooling Loop Digital Twin

This module implements various anomaly detection algorithms including
residual-based detection, statistical methods, and machine learning approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from enum import Enum
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    FAULT = "fault"


class DetectionMethod(Enum):
    """Available anomaly detection methods."""
    RESIDUAL_THRESHOLD = "residual_threshold"
    Z_SCORE = "z_score"
    ROLLING_Z_SCORE = "rolling_z_score"
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    STATISTICAL_PROCESS_CONTROL = "spc"
    CUSUM = "cusum"


class AnomalyDetector:
    """
    Comprehensive anomaly detection system for thermal cooling loop.
    
    Implements multiple detection algorithms and provides unified interface
    for real-time anomaly detection and classification.
    """
    
    def __init__(self, 
                 detection_methods: List[DetectionMethod] = None,
                 window_size: int = 100,
                 threshold_multiplier: float = 2.0):
        """
        Initialize the anomaly detector.
        
        Parameters:
        -----------
        detection_methods : list
            List of detection methods to use
        window_size : int
            Rolling window size for statistical methods
        threshold_multiplier : float
            Multiplier for threshold-based detection
        """
        self.detection_methods = detection_methods or [
            DetectionMethod.RESIDUAL_THRESHOLD,
            DetectionMethod.ROLLING_Z_SCORE,
            DetectionMethod.ISOLATION_FOREST
        ]
        self.window_size = window_size
        self.threshold_multiplier = threshold_multiplier
        
        # Detection state
        self.is_trained = False
        self.baseline_stats = {}
        self.models = {}
        self.scalers = {}
        self.residual_history = []
        self.anomaly_history = []
        
        # Configuration
        self.config = {
            'z_score_threshold': 2.0,
            'rolling_z_score_threshold': 2.5,
            'residual_threshold': 5.0,
            'spc_control_limits': 3.0,
            'cusum_threshold': 5.0,
            'isolation_forest_contamination': 0.1,
            'one_class_svm_nu': 0.1
        }
        
        logger.info("Anomaly detector initialized")
    
    def add_residuals(self, residuals: Dict[str, float], timestamp: float = None):
        """
        Add new residual data for analysis.
        
        Parameters:
        -----------
        residuals : dict
            Dictionary of residuals for each sensor
        timestamp : float, optional
            Timestamp for the residuals
        """
        if timestamp is None:
            timestamp = len(self.residual_history)
        
        residual_data = {
            'timestamp': timestamp,
            'residuals': residuals.copy()
        }
        
        self.residual_history.append(residual_data)
        
        # Keep only recent history
        if len(self.residual_history) > self.window_size * 2:
            self.residual_history = self.residual_history[-self.window_size:]
    
    def detect_anomalies(self, residuals: Dict[str, float], timestamp: float = None) -> Dict[str, Any]:
        """
        Detect anomalies in the given residuals.
        
        Parameters:
        -----------
        residuals : dict
            Dictionary of residuals for each sensor
        timestamp : float, optional
            Timestamp for the residuals
            
        Returns:
        --------
        dict
            Detection results with anomaly scores and classifications
        """
        if timestamp is None:
            timestamp = len(self.residual_history)
        
        # Add residuals to history
        self.add_residuals(residuals, timestamp)
        
        # Initialize results
        results = {
            'timestamp': timestamp,
            'residuals': residuals,
            'anomaly_scores': {},
            'anomaly_types': {},
            'overall_anomaly': False,
            'overall_severity': AnomalyType.NORMAL,
            'detection_methods': {}
        }
        
        # Run each detection method
        for method in self.detection_methods:
            try:
                method_results = self._run_detection_method(method, residuals, timestamp)
                results['detection_methods'][method.value] = method_results
                
                # Update overall results
                if method_results['is_anomaly']:
                    results['overall_anomaly'] = True
                    results['anomaly_scores'][method.value] = method_results['score']
                    results['anomaly_types'][method.value] = method_results['anomaly_type']
                    
                    # Update overall severity
                    if method_results['anomaly_type'] == AnomalyType.CRITICAL:
                        results['overall_severity'] = AnomalyType.CRITICAL
                    elif method_results['anomaly_type'] == AnomalyType.WARNING and results['overall_severity'] != AnomalyType.CRITICAL:
                        results['overall_severity'] = AnomalyType.WARNING
                    elif method_results['anomaly_type'] == AnomalyType.FAULT and results['overall_severity'] not in [AnomalyType.CRITICAL, AnomalyType.WARNING]:
                        results['overall_severity'] = AnomalyType.FAULT
                        
            except Exception as e:
                logger.warning(f"Detection method {method.value} failed: {str(e)}")
                results['detection_methods'][method.value] = {
                    'is_anomaly': False,
                    'score': 0.0,
                    'anomaly_type': AnomalyType.NORMAL,
                    'error': str(e)
                }
        
        # Store anomaly history
        self.anomaly_history.append(results)
        if len(self.anomaly_history) > self.window_size:
            self.anomaly_history = self.anomaly_history[-self.window_size:]
        
        return results
    
    def _run_detection_method(self, method: DetectionMethod, residuals: Dict[str, float], timestamp: float) -> Dict[str, Any]:
        """Run a specific detection method."""
        if method == DetectionMethod.RESIDUAL_THRESHOLD:
            return self._residual_threshold_detection(residuals)
        elif method == DetectionMethod.Z_SCORE:
            return self._z_score_detection(residuals)
        elif method == DetectionMethod.ROLLING_Z_SCORE:
            return self._rolling_z_score_detection(residuals, timestamp)
        elif method == DetectionMethod.ISOLATION_FOREST:
            return self._isolation_forest_detection(residuals)
        elif method == DetectionMethod.ONE_CLASS_SVM:
            return self._one_class_svm_detection(residuals)
        elif method == DetectionMethod.STATISTICAL_PROCESS_CONTROL:
            return self._spc_detection(residuals, timestamp)
        elif method == DetectionMethod.CUSUM:
            return self._cusum_detection(residuals, timestamp)
        else:
            raise ValueError(f"Unknown detection method: {method}")
    
    def _residual_threshold_detection(self, residuals: Dict[str, float]) -> Dict[str, Any]:
        """Residual threshold-based detection."""
        max_residual = max(abs(r) for r in residuals.values())
        threshold = self.config['residual_threshold']
        
        is_anomaly = max_residual > threshold
        score = max_residual / threshold if threshold > 0 else 0
        
        if is_anomaly:
            if score > 2.0:
                anomaly_type = AnomalyType.CRITICAL
            elif score > 1.5:
                anomaly_type = AnomalyType.WARNING
            else:
                anomaly_type = AnomalyType.FAULT
        else:
            anomaly_type = AnomalyType.NORMAL
        
        return {
            'is_anomaly': is_anomaly,
            'score': score,
            'anomaly_type': anomaly_type,
            'details': {
                'max_residual': max_residual,
                'threshold': threshold
            }
        }
    
    def _z_score_detection(self, residuals: Dict[str, float]) -> Dict[str, Any]:
        """Z-score based detection."""
        if not self.is_trained:
            return {'is_anomaly': False, 'score': 0.0, 'anomaly_type': AnomalyType.NORMAL}
        
        max_z_score = 0
        for sensor, residual in residuals.items():
            if sensor in self.baseline_stats:
                mean = self.baseline_stats[sensor]['mean']
                std = self.baseline_stats[sensor]['std']
                if std > 0:
                    z_score = abs((residual - mean) / std)
                    max_z_score = max(max_z_score, z_score)
        
        threshold = self.config['z_score_threshold']
        is_anomaly = max_z_score > threshold
        score = max_z_score / threshold if threshold > 0 else 0
        
        if is_anomaly:
            if score > 2.0:
                anomaly_type = AnomalyType.CRITICAL
            elif score > 1.5:
                anomaly_type = AnomalyType.WARNING
            else:
                anomaly_type = AnomalyType.FAULT
        else:
            anomaly_type = AnomalyType.NORMAL
        
        return {
            'is_anomaly': is_anomaly,
            'score': score,
            'anomaly_type': anomaly_type,
            'details': {'max_z_score': max_z_score, 'threshold': threshold}
        }
    
    def _rolling_z_score_detection(self, residuals: Dict[str, float], timestamp: float) -> Dict[str, Any]:
        """Rolling z-score based detection."""
        if len(self.residual_history) < self.window_size:
            return {'is_anomaly': False, 'score': 0.0, 'anomaly_type': AnomalyType.NORMAL}
        
        # Get recent residuals
        recent_residuals = self.residual_history[-self.window_size:]
        
        max_z_score = 0
        for sensor, residual in residuals.items():
            sensor_residuals = [r['residuals'].get(sensor, 0) for r in recent_residuals]
            if len(sensor_residuals) > 1:
                mean = np.mean(sensor_residuals)
                std = np.std(sensor_residuals)
                if std > 0:
                    z_score = abs((residual - mean) / std)
                    max_z_score = max(max_z_score, z_score)
        
        threshold = self.config['rolling_z_score_threshold']
        is_anomaly = max_z_score > threshold
        score = max_z_score / threshold if threshold > 0 else 0
        
        if is_anomaly:
            if score > 2.0:
                anomaly_type = AnomalyType.CRITICAL
            elif score > 1.5:
                anomaly_type = AnomalyType.WARNING
            else:
                anomaly_type = AnomalyType.FAULT
        else:
            anomaly_type = AnomalyType.NORMAL
        
        return {
            'is_anomaly': is_anomaly,
            'score': score,
            'anomaly_type': anomaly_type,
            'details': {'max_z_score': max_z_score, 'threshold': threshold}
        }
    
    def _isolation_forest_detection(self, residuals: Dict[str, float]) -> Dict[str, Any]:
        """Isolation Forest based detection."""
        if not self.is_trained or 'isolation_forest' not in self.models:
            return {'is_anomaly': False, 'score': 0.0, 'anomaly_type': AnomalyType.NORMAL}
        
        # Prepare data
        residual_array = np.array([list(residuals.values())])
        
        # Predict anomaly
        prediction = self.models['isolation_forest'].predict(residual_array)
        score = self.models['isolation_forest'].score_samples(residual_array)
        
        is_anomaly = prediction[0] == -1
        anomaly_score = -score[0]  # Negative score indicates anomaly
        
        if is_anomaly:
            if anomaly_score > 0.5:
                anomaly_type = AnomalyType.CRITICAL
            elif anomaly_score > 0.3:
                anomaly_type = AnomalyType.WARNING
            else:
                anomaly_type = AnomalyType.FAULT
        else:
            anomaly_type = AnomalyType.NORMAL
        
        return {
            'is_anomaly': is_anomaly,
            'score': anomaly_score,
            'anomaly_type': anomaly_type,
            'details': {'prediction': prediction[0], 'score': score[0]}
        }
    
    def _one_class_svm_detection(self, residuals: Dict[str, float]) -> Dict[str, Any]:
        """One-Class SVM based detection."""
        if not self.is_trained or 'one_class_svm' not in self.models:
            return {'is_anomaly': False, 'score': 0.0, 'anomaly_type': AnomalyType.NORMAL}
        
        # Prepare data
        residual_array = np.array([list(residuals.values())])
        
        # Predict anomaly
        prediction = self.models['one_class_svm'].predict(residual_array)
        score = self.models['one_class_svm'].score_samples(residual_array)
        
        is_anomaly = prediction[0] == -1
        anomaly_score = -score[0]  # Negative score indicates anomaly
        
        if is_anomaly:
            if anomaly_score > 0.5:
                anomaly_type = AnomalyType.CRITICAL
            elif anomaly_score > 0.3:
                anomaly_type = AnomalyType.WARNING
            else:
                anomaly_type = AnomalyType.FAULT
        else:
            anomaly_type = AnomalyType.NORMAL
        
        return {
            'is_anomaly': is_anomaly,
            'score': anomaly_score,
            'anomaly_type': anomaly_type,
            'details': {'prediction': prediction[0], 'score': score[0]}
        }
    
    def _spc_detection(self, residuals: Dict[str, float], timestamp: float) -> Dict[str, Any]:
        """Statistical Process Control detection."""
        if len(self.residual_history) < self.window_size:
            return {'is_anomaly': False, 'score': 0.0, 'anomaly_type': AnomalyType.NORMAL}
        
        # Get recent residuals
        recent_residuals = self.residual_history[-self.window_size:]
        
        max_violation = 0
        for sensor, residual in residuals.items():
            sensor_residuals = [r['residuals'].get(sensor, 0) for r in recent_residuals]
            if len(sensor_residuals) > 1:
                mean = np.mean(sensor_residuals)
                std = np.std(sensor_residuals)
                control_limit = self.config['spc_control_limits'] * std
                
                violation = abs(residual - mean) / control_limit if control_limit > 0 else 0
                max_violation = max(max_violation, violation)
        
        is_anomaly = max_violation > 1.0
        score = max_violation
        
        if is_anomaly:
            if score > 2.0:
                anomaly_type = AnomalyType.CRITICAL
            elif score > 1.5:
                anomaly_type = AnomalyType.WARNING
            else:
                anomaly_type = AnomalyType.FAULT
        else:
            anomaly_type = AnomalyType.NORMAL
        
        return {
            'is_anomaly': is_anomaly,
            'score': score,
            'anomaly_type': anomaly_type,
            'details': {'max_violation': max_violation}
        }
    
    def _cusum_detection(self, residuals: Dict[str, float], timestamp: float) -> Dict[str, Any]:
        """CUSUM (Cumulative Sum) detection."""
        if len(self.residual_history) < 10:
            return {'is_anomaly': False, 'score': 0.0, 'anomaly_type': AnomalyType.NORMAL}
        
        # Calculate CUSUM for each sensor
        max_cusum = 0
        for sensor, residual in residuals.items():
            sensor_residuals = [r['residuals'].get(sensor, 0) for r in self.residual_history]
            
            if len(sensor_residuals) > 1:
                mean = np.mean(sensor_residuals)
                std = np.std(sensor_residuals)
                
                # Calculate CUSUM
                cusum = 0
                for r in sensor_residuals[-10:]:  # Last 10 points
                    cusum += (r - mean) / std if std > 0 else 0
                
                max_cusum = max(max_cusum, abs(cusum))
        
        threshold = self.config['cusum_threshold']
        is_anomaly = max_cusum > threshold
        score = max_cusum / threshold if threshold > 0 else 0
        
        if is_anomaly:
            if score > 2.0:
                anomaly_type = AnomalyType.CRITICAL
            elif score > 1.5:
                anomaly_type = AnomalyType.WARNING
            else:
                anomaly_type = AnomalyType.FAULT
        else:
            anomaly_type = AnomalyType.NORMAL
        
        return {
            'is_anomaly': is_anomaly,
            'score': score,
            'anomaly_type': anomaly_type,
            'details': {'max_cusum': max_cusum, 'threshold': threshold}
        }
    
    def train(self, training_data: List[Dict[str, float]]):
        """
        Train the anomaly detection models.
        
        Parameters:
        -----------
        training_data : list
            List of residual dictionaries for training
        """
        if not training_data:
            logger.warning("No training data provided")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(training_data)
        
        # Calculate baseline statistics
        for sensor in df.columns:
            self.baseline_stats[sensor] = {
                'mean': df[sensor].mean(),
                'std': df[sensor].std(),
                'min': df[sensor].min(),
                'max': df[sensor].max()
            }
        
        # Train ML models
        if DetectionMethod.ISOLATION_FOREST in self.detection_methods:
            self.models['isolation_forest'] = IsolationForest(
                contamination=self.config['isolation_forest_contamination'],
                random_state=42
            )
            self.models['isolation_forest'].fit(df.values)
        
        if DetectionMethod.ONE_CLASS_SVM in self.detection_methods:
            self.scalers['one_class_svm'] = StandardScaler()
            scaled_data = self.scalers['one_class_svm'].fit_transform(df.values)
            
            self.models['one_class_svm'] = OneClassSVM(
                nu=self.config['one_class_svm_nu'],
                kernel='rbf'
            )
            self.models['one_class_svm'].fit(scaled_data)
        
        self.is_trained = True
        logger.info("Anomaly detection models trained successfully")
    
    def get_anomaly_summary(self, time_window: int = None) -> Dict[str, Any]:
        """
        Get summary of recent anomalies.
        
        Parameters:
        -----------
        time_window : int, optional
            Number of recent anomalies to include
            
        Returns:
        --------
        dict
            Summary of anomalies
        """
        if not self.anomaly_history:
            return {'total_anomalies': 0, 'anomaly_types': {}, 'recent_anomalies': []}
        
        recent_anomalies = self.anomaly_history
        if time_window:
            recent_anomalies = recent_anomalies[-time_window:]
        
        # Count anomalies by type
        anomaly_types = {}
        for anomaly in recent_anomalies:
            severity = anomaly['overall_severity'].value
            anomaly_types[severity] = anomaly_types.get(severity, 0) + 1
        
        return {
            'total_anomalies': len([a for a in recent_anomalies if a['overall_anomaly']]),
            'anomaly_types': anomaly_types,
            'recent_anomalies': recent_anomalies[-10:],  # Last 10 anomalies
            'detection_methods_used': [method.value for method in self.detection_methods]
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update detection configuration."""
        self.config.update(new_config)
        logger.info("Detection configuration updated")
    
    def reset(self):
        """Reset the detector state."""
        self.is_trained = False
        self.baseline_stats = {}
        self.models = {}
        self.scalers = {}
        self.residual_history = []
        self.anomaly_history = []
        logger.info("Anomaly detector reset")


def create_default_detector() -> AnomalyDetector:
    """Create a default anomaly detector with standard configuration."""
    return AnomalyDetector(
        detection_methods=[
            DetectionMethod.RESIDUAL_THRESHOLD,
            DetectionMethod.ROLLING_Z_SCORE,
            DetectionMethod.ISOLATION_FOREST
        ],
        window_size=100,
        threshold_multiplier=2.0
    )


def create_advanced_detector() -> AnomalyDetector:
    """Create an advanced anomaly detector with all methods."""
    return AnomalyDetector(
        detection_methods=[
            DetectionMethod.RESIDUAL_THRESHOLD,
            DetectionMethod.Z_SCORE,
            DetectionMethod.ROLLING_Z_SCORE,
            DetectionMethod.ISOLATION_FOREST,
            DetectionMethod.ONE_CLASS_SVM,
            DetectionMethod.STATISTICAL_PROCESS_CONTROL,
            DetectionMethod.CUSUM
        ],
        window_size=200,
        threshold_multiplier=1.5
    )
