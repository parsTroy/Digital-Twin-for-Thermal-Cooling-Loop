"""
Unit tests for the digital twin manager module.
"""

import unittest
import numpy as np
import pandas as pd
import time
import sys
import os

# Add the parent directory to the path to import the twin module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from twin.digital_twin import DigitalTwinManager, create_digital_twin_manager
from twin.model import create_default_parameters
from twin.plant_simulator import PlantSimulator, FaultType
from twin.detector import DetectionMethod, AnomalyType


class TestDigitalTwinManager(unittest.TestCase):
    """Test cases for the DigitalTwinManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.plant_simulator = PlantSimulator(create_default_parameters())
        self.twin_params = create_default_parameters()
        
        self.manager = DigitalTwinManager(
            plant_simulator=self.plant_simulator,
            twin_params=self.twin_params,
            detection_methods=[DetectionMethod.RESIDUAL_THRESHOLD],
            update_rate=10.0  # High update rate for testing
        )
    
    def test_initialization(self):
        """Test digital twin manager initialization."""
        self.assertIsInstance(self.manager, DigitalTwinManager)
        self.assertEqual(self.manager.plant_simulator, self.plant_simulator)
        self.assertEqual(self.manager.twin_params, self.twin_params)
        self.assertFalse(self.manager.is_running)
        self.assertEqual(self.manager.current_time, 0.0)
    
    def test_start_stop(self):
        """Test starting and stopping the digital twin."""
        # Test starting
        initial_conditions = np.array([350.0, 300.0])
        self.manager.start(initial_conditions)
        
        self.assertTrue(self.manager.is_running)
        self.assertIsNotNone(self.manager.thread)
        
        # Let it run for a short time
        time.sleep(0.2)
        
        # Test stopping
        self.manager.stop()
        
        self.assertFalse(self.manager.is_running)
    
    def test_get_current_data(self):
        """Test getting current data."""
        # Start the manager
        initial_conditions = np.array([350.0, 300.0])
        self.manager.start(initial_conditions)
        
        # Let it run briefly
        time.sleep(0.1)
        
        # Get current data
        data = self.manager.get_current_data()
        
        self.assertIn('timestamp', data)
        self.assertIn('plant_state', data)
        self.assertIn('twin_state', data)
        self.assertIn('residuals', data)
        self.assertIn('anomaly_results', data)
        self.assertIn('is_running', data)
        
        self.assertTrue(data['is_running'])
        self.assertIsNotNone(data['plant_state'])
        self.assertIsNotNone(data['twin_state'])
        
        # Stop the manager
        self.manager.stop()
    
    def test_get_history(self):
        """Test getting historical data."""
        # Start the manager
        initial_conditions = np.array([350.0, 300.0])
        self.manager.start(initial_conditions)
        
        # Let it run for a bit
        time.sleep(0.2)
        
        # Get history
        history = self.manager.get_history()
        
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)
        
        # Check history structure
        for data_point in history:
            self.assertIn('timestamp', data_point)
            self.assertIn('plant_state', data_point)
            self.assertIn('twin_state', data_point)
            self.assertIn('residuals', data_point)
            self.assertIn('anomaly_results', data_point)
        
        # Test time window
        recent_history = self.manager.get_history(time_window=0.1)
        self.assertLessEqual(len(recent_history), len(history))
        
        # Stop the manager
        self.manager.stop()
    
    def test_residual_computation(self):
        """Test residual computation between plant and twin."""
        # Start the manager
        initial_conditions = np.array([350.0, 300.0])
        self.manager.start(initial_conditions)
        
        # Let it run briefly
        time.sleep(0.1)
        
        # Check that residuals are computed
        data = self.manager.get_current_data()
        residuals = data['residuals']
        
        self.assertIn('T_hot', residuals)
        self.assertIn('T_cold', residuals)
        self.assertIn('m_dot', residuals)
        
        # Residuals should be finite
        for sensor, residual in residuals.items():
            self.assertTrue(np.isfinite(residual))
        
        # Stop the manager
        self.manager.stop()
    
    def test_anomaly_detection(self):
        """Test anomaly detection functionality."""
        # Start the manager
        initial_conditions = np.array([350.0, 300.0])
        self.manager.start(initial_conditions)
        
        # Let it run briefly
        time.sleep(0.1)
        
        # Check anomaly detection
        data = self.manager.get_current_data()
        anomaly_results = data['anomaly_results']
        
        self.assertIn('timestamp', anomaly_results)
        self.assertIn('residuals', anomaly_results)
        self.assertIn('anomaly_scores', anomaly_results)
        self.assertIn('anomaly_types', anomaly_results)
        self.assertIn('overall_anomaly', anomaly_results)
        self.assertIn('overall_severity', anomaly_results)
        self.assertIn('detection_methods', anomaly_results)
        
        # Stop the manager
        self.manager.stop()
    
    def test_residual_statistics(self):
        """Test residual statistics calculation."""
        # Start the manager
        initial_conditions = np.array([350.0, 300.0])
        self.manager.start(initial_conditions)
        
        # Let it run for a bit
        time.sleep(0.2)
        
        # Get residual statistics
        stats = self.manager.get_residual_statistics()
        
        self.assertIsInstance(stats, dict)
        
        for sensor in ['T_hot', 'T_cold', 'm_dot']:
            if sensor in stats:
                sensor_stats = stats[sensor]
                self.assertIn('mean', sensor_stats)
                self.assertIn('std', sensor_stats)
                self.assertIn('min', sensor_stats)
                self.assertIn('max', sensor_stats)
                self.assertIn('rms', sensor_stats)
                
                # Check that statistics are finite
                for stat_name, stat_value in sensor_stats.items():
                    self.assertTrue(np.isfinite(stat_value), f"{sensor}.{stat_name} is not finite")
        
        # Stop the manager
        self.manager.stop()
    
    def test_anomaly_summary(self):
        """Test anomaly summary generation."""
        # Start the manager
        initial_conditions = np.array([350.0, 300.0])
        self.manager.start(initial_conditions)
        
        # Let it run for a bit
        time.sleep(0.2)
        
        # Get anomaly summary
        summary = self.manager.get_anomaly_summary()
        
        self.assertIn('total_anomalies', summary)
        self.assertIn('anomaly_types', summary)
        self.assertIn('anomaly_rate', summary)
        self.assertIn('recent_anomalies', summary)
        
        self.assertIsInstance(summary['total_anomalies'], int)
        self.assertIsInstance(summary['anomaly_types'], dict)
        self.assertIsInstance(summary['anomaly_rate'], float)
        self.assertIsInstance(summary['recent_anomalies'], list)
        
        # Stop the manager
        self.manager.stop()
    
    def test_callbacks(self):
        """Test callback functionality."""
        # Set up callbacks
        anomaly_callback_called = []
        data_callback_called = []
        
        def anomaly_callback(anomaly_results):
            anomaly_callback_called.append(anomaly_results)
        
        def data_callback(data):
            data_callback_called.append(data)
        
        self.manager.add_anomaly_callback(anomaly_callback)
        self.manager.add_data_callback(data_callback)
        
        # Start the manager
        initial_conditions = np.array([350.0, 300.0])
        self.manager.start(initial_conditions)
        
        # Let it run briefly
        time.sleep(0.1)
        
        # Check that callbacks were called
        self.assertGreater(len(data_callback_called), 0)
        
        # Stop the manager
        self.manager.stop()
    
    def test_training(self):
        """Test detector training."""
        # Generate training data
        training_data = []
        for _ in range(50):
            training_data.append({
                'T_hot': np.random.normal(0, 1.0),
                'T_cold': np.random.normal(0, 0.8),
                'm_dot': np.random.normal(0, 0.1)
            })
        
        # Train detector
        self.manager.train_detector(training_data)
        
        # Check that detector is trained
        self.assertTrue(self.manager.detector.is_trained)
    
    def test_parameter_update(self):
        """Test twin parameter update."""
        # Start the manager
        initial_conditions = np.array([350.0, 300.0])
        self.manager.start(initial_conditions)
        
        # Update parameters
        new_params = {'C': 1500.0, 'm_dot': 0.15}
        self.manager.update_twin_parameters(new_params)
        
        # Check that parameters were updated
        self.assertEqual(self.manager.twin_params['C'], 1500.0)
        self.assertEqual(self.manager.twin_params['m_dot'], 0.15)
        
        # Stop the manager
        self.manager.stop()
    
    def test_fault_injection(self):
        """Test fault injection into plant simulator."""
        # Start the manager
        initial_conditions = np.array([350.0, 300.0])
        self.manager.start(initial_conditions)
        
        # Inject fault
        self.manager.inject_plant_fault(
            FaultType.PUMP_DEGRADATION,
            start_time=1.0,
            parameters={'degradation_rate': 0.1}
        )
        
        # Check that fault was injected
        self.assertEqual(len(self.plant_simulator.active_faults), 1)
        
        # Clear faults
        self.manager.clear_plant_faults()
        
        # Check that faults were cleared
        self.assertEqual(len(self.plant_simulator.active_faults), 0)
        
        # Stop the manager
        self.manager.stop()
    
    def test_export_data(self):
        """Test data export functionality."""
        # Start the manager
        initial_conditions = np.array([350.0, 300.0])
        self.manager.start(initial_conditions)
        
        # Let it run briefly
        time.sleep(0.1)
        
        # Export data
        filename = self.manager.export_data()
        
        # Check that file was created
        self.assertTrue(os.path.exists(filename))
        
        # Check file content
        df = pd.read_csv(filename)
        self.assertGreater(len(df), 0)
        
        # Check required columns
        required_columns = [
            'timestamp', 'plant_T_hot', 'plant_T_cold',
            'twin_T_hot', 'twin_T_cold',
            'residual_T_hot', 'residual_T_cold', 'residual_m_dot',
            'anomaly_detected', 'anomaly_severity'
        ]
        
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Clean up
        os.remove(filename)
        
        # Stop the manager
        self.manager.stop()
    
    def test_multiple_starts_stops(self):
        """Test multiple start/stop cycles."""
        initial_conditions = np.array([350.0, 300.0])
        
        # First cycle
        self.manager.start(initial_conditions)
        self.assertTrue(self.manager.is_running)
        time.sleep(0.1)
        self.manager.stop()
        self.assertFalse(self.manager.is_running)
        
        # Second cycle
        self.manager.start(initial_conditions)
        self.assertTrue(self.manager.is_running)
        time.sleep(0.1)
        self.manager.stop()
        self.assertFalse(self.manager.is_running)
    
    def test_error_handling(self):
        """Test error handling in the main loop."""
        # Create manager with invalid parameters to cause errors
        invalid_params = {'C': -1000, 'm_dot': 0.1, 'cp': 4180, 'UA': 50, 'Q_in': 1000, 'Q_out': 1000}
        
        manager = DigitalTwinManager(
            plant_simulator=self.plant_simulator,
            twin_params=invalid_params,
            detection_methods=[DetectionMethod.RESIDUAL_THRESHOLD]
        )
        
        # This should not raise an exception
        initial_conditions = np.array([350.0, 300.0])
        manager.start(initial_conditions)
        time.sleep(0.1)
        manager.stop()


class TestDigitalTwinFactory(unittest.TestCase):
    """Test cases for the digital twin factory function."""
    
    def test_create_digital_twin_manager(self):
        """Test creating digital twin manager with factory function."""
        manager = create_digital_twin_manager()
        
        self.assertIsInstance(manager, DigitalTwinManager)
        self.assertIsNotNone(manager.plant_simulator)
        self.assertIsNotNone(manager.twin)
        self.assertIsNotNone(manager.detector)
    
    def test_create_with_custom_parameters(self):
        """Test creating with custom parameters."""
        plant_simulator = PlantSimulator(create_default_parameters())
        twin_params = {'C': 1500, 'm_dot': 0.15, 'cp': 4180, 'UA': 60, 'Q_in': 1200, 'Q_out': 1200}
        detection_methods = [DetectionMethod.RESIDUAL_THRESHOLD, DetectionMethod.ROLLING_Z_SCORE]
        
        manager = create_digital_twin_manager(
            plant_simulator=plant_simulator,
            twin_params=twin_params,
            detection_methods=detection_methods
        )
        
        self.assertEqual(manager.plant_simulator, plant_simulator)
        self.assertEqual(manager.twin_params, twin_params)
        self.assertEqual(manager.detector.detection_methods, detection_methods)


if __name__ == '__main__':
    unittest.main()
