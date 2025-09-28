"""
Unit tests for the plant simulator.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add the parent directory to the path to import the twin module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from twin.model import ThermalCoolingTwin, create_default_parameters
from twin.plant_simulator import PlantSimulator, FaultType


class TestPlantSimulator(unittest.TestCase):
    """Test cases for the PlantSimulator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_params = create_default_parameters()
        self.simulator = PlantSimulator(self.base_params)
        self.twin = ThermalCoolingTwin(self.base_params)
        self.t_span = (0, 100)
        self.y0 = np.array([350.0, 300.0])
        
    def test_initialization(self):
        """Test plant simulator initialization."""
        self.assertIsInstance(self.simulator, PlantSimulator)
        self.assertEqual(self.simulator.base_params, self.base_params)
        self.assertEqual(self.simulator.current_fault, FaultType.NONE)
        
    def test_sensor_characteristics(self):
        """Test sensor characteristics setup."""
        char = self.simulator.sensor_characteristics
        
        # Check that all required sensors are present
        self.assertIn('T_hot', char)
        self.assertIn('T_cold', char)
        self.assertIn('m_dot', char)
        
        # Check sensor characteristics structure
        for sensor in ['T_hot', 'T_cold', 'm_dot']:
            sensor_char = char[sensor]
            self.assertIn('noise_std', sensor_char)
            self.assertIn('bias', sensor_char)
            self.assertIn('drift_rate', sensor_char)
            self.assertIn('resolution', sensor_char)
            self.assertIn('range', sensor_char)
    
    def test_fault_injection(self):
        """Test fault injection functionality."""
        # Test pump degradation fault
        self.simulator.inject_fault(
            FaultType.PUMP_DEGRADATION,
            start_time=50.0,
            parameters={'degradation_rate': 0.02}
        )
        
        self.assertEqual(self.simulator.current_fault, FaultType.PUMP_DEGRADATION)
        self.assertEqual(self.simulator.fault_start_time, 50.0)
        self.assertEqual(len(self.simulator.active_faults), 1)
        
    def test_clear_faults(self):
        """Test fault clearing functionality."""
        # Inject a fault
        self.simulator.inject_fault(FaultType.PUMP_DEGRADATION, 50.0)
        
        # Clear faults
        self.simulator.clear_faults()
        
        self.assertEqual(self.simulator.current_fault, FaultType.NONE)
        self.assertIsNone(self.simulator.fault_start_time)
        self.assertEqual(len(self.simulator.active_faults), 0)
    
    def test_get_modified_parameters(self):
        """Test parameter modification by faults."""
        # Test without fault
        params_no_fault = self.simulator.get_modified_parameters(25.0)
        self.assertEqual(params_no_fault, self.base_params)
        
        # Test with pump degradation fault
        self.simulator.inject_fault(
            FaultType.PUMP_DEGRADATION,
            start_time=50.0,
            parameters={'degradation_rate': 0.1}
        )
        
        # Before fault
        params_before = self.simulator.get_modified_parameters(25.0)
        self.assertEqual(params_before['m_dot'], self.base_params['m_dot'])
        
        # After fault
        params_after = self.simulator.get_modified_parameters(75.0)
        self.assertLess(params_after['m_dot'], self.base_params['m_dot'])
    
    def test_sensor_noise(self):
        """Test sensor noise addition."""
        T_hot = 350.0
        T_cold = 300.0
        m_dot = 0.1
        t = 50.0
        
        # Test multiple times to check randomness
        results = []
        for _ in range(10):
            T_hot_noisy, T_cold_noisy, m_dot_noisy = self.simulator.add_sensor_noise(
                T_hot, T_cold, m_dot, t
            )
            results.append((T_hot_noisy, T_cold_noisy, m_dot_noisy))
        
        # Check that noise is applied (values should be different)
        T_hot_values = [r[0] for r in results]
        T_cold_values = [r[1] for r in results]
        m_dot_values = [r[2] for r in results]
        
        # Should have some variation due to noise
        self.assertGreater(np.std(T_hot_values), 0)
        self.assertGreater(np.std(T_cold_values), 0)
        self.assertGreater(np.std(m_dot_values), 0)
        
        # Values should be within reasonable ranges
        for T_hot_noisy, T_cold_noisy, m_dot_noisy in results:
            self.assertGreater(T_hot_noisy, 0)
            self.assertGreater(T_cold_noisy, 0)
            self.assertGreater(m_dot_noisy, 0)
    
    def test_sensor_characteristics_application(self):
        """Test sensor characteristics application."""
        # Test with custom sensor characteristics
        custom_char = {
            'T_hot': {
                'noise_std': 1.0,
                'bias': 2.0,
                'drift_rate': 0.01,
                'resolution': 0.5,
                'range': (200, 500)
            }
        }
        
        simulator = PlantSimulator(self.base_params, sensor_characteristics=custom_char)
        
        T_hot = 350.0
        t = 100.0
        
        noisy_value = simulator._apply_sensor_characteristics('T_hot', T_hot, t)
        
        # Should have bias and drift applied
        expected_bias = 2.0
        expected_drift = 0.01 * 100.0
        expected_total = T_hot + expected_bias + expected_drift
        
        # Allow for noise variation
        self.assertAlmostEqual(noisy_value, expected_total, places=0)
    
    def test_simulate_sensor_data(self):
        """Test sensor data simulation."""
        # Run simulation
        df = self.simulator.simulate_sensor_data(
            self.twin, self.t_span, self.y0
        )
        
        # Check DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        
        # Check required columns
        required_columns = [
            't', 'T_hot_true', 'T_cold_true', 'm_dot_true',
            'T_hot', 'T_cold', 'm_dot', 'fault_type', 'fault_active'
        ]
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Check data types
        self.assertTrue(pd.api.types.is_numeric_dtype(df['t']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['T_hot']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['T_cold']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['m_dot']))
        
        # Check that noisy values are different from true values
        self.assertFalse(np.allclose(df['T_hot'], df['T_hot_true']))
        self.assertFalse(np.allclose(df['T_cold'], df['T_cold_true']))
    
    def test_simulate_with_fault(self):
        """Test sensor data simulation with fault."""
        # Inject a fault
        self.simulator.inject_fault(
            FaultType.PUMP_DEGRADATION,
            start_time=50.0,
            parameters={'degradation_rate': 0.1}
        )
        
        # Run simulation
        df = self.simulator.simulate_sensor_data(
            self.twin, self.t_span, self.y0
        )
        
        # Check fault information
        self.assertTrue((df['fault_type'] == 'pump_degradation').any())
        self.assertTrue((df['fault_active'] == True).any())
        self.assertTrue((df['fault_active'] == False).any())
        
        # Check that mass flow rate is affected by fault
        before_fault = df[df['t'] < 50.0]['m_dot_true']
        after_fault = df[df['t'] > 50.0]['m_dot_true']
        
        if len(before_fault) > 0 and len(after_fault) > 0:
            self.assertLess(np.mean(after_fault), np.mean(before_fault))
    
    def test_demo_scenarios(self):
        """Test demo scenario creation."""
        scenarios = self.simulator.create_demo_scenarios()
        
        self.assertIsInstance(scenarios, list)
        self.assertGreater(len(scenarios), 0)
        
        # Check scenario structure
        for scenario in scenarios:
            self.assertIn('name', scenario)
            self.assertIn('description', scenario)
            self.assertIn('duration', scenario)
            self.assertIn('faults', scenario)
            
            self.assertIsInstance(scenario['name'], str)
            self.assertIsInstance(scenario['description'], str)
            self.assertIsInstance(scenario['duration'], (int, float))
            self.assertIsInstance(scenario['faults'], list)
    
    def test_multiple_faults(self):
        """Test multiple concurrent faults."""
        # Inject multiple faults
        self.simulator.inject_fault(
            FaultType.PUMP_DEGRADATION,
            start_time=30.0,
            parameters={'degradation_rate': 0.05}
        )
        
        self.simulator.inject_fault(
            FaultType.SENSOR_BIAS,
            start_time=60.0,
            parameters={'bias_magnitude': 5.0}
        )
        
        # Check that both faults are active
        self.assertEqual(len(self.simulator.active_faults), 2)
        
        # Test parameter modification with multiple faults
        params = self.simulator.get_modified_parameters(80.0)
        
        # Should be affected by pump degradation
        self.assertLess(params['m_dot'], self.base_params['m_dot'])
    
    def test_sensor_range_limits(self):
        """Test sensor range limits."""
        # Create simulator with restrictive ranges
        custom_char = {
            'T_hot': {
                'noise_std': 0.1,
                'bias': 0.0,
                'drift_rate': 0.0,
                'resolution': 0.1,
                'range': (300, 400)  # Restrictive range
            }
        }
        
        simulator = PlantSimulator(self.base_params, sensor_characteristics=custom_char)
        
        # Test with value outside range
        T_hot = 500.0  # Above range
        noisy_value = simulator._apply_sensor_characteristics('T_hot', T_hot, 0.0)
        
        # Should be clipped to range
        self.assertLessEqual(noisy_value, 400.0)
        self.assertGreaterEqual(noisy_value, 300.0)
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        # Create two simulators with same parameters
        sim1 = PlantSimulator(self.base_params)
        sim2 = PlantSimulator(self.base_params)
        
        # Run same simulation
        df1 = sim1.simulate_sensor_data(self.twin, self.t_span, self.y0)
        df2 = sim2.simulate_sensor_data(self.twin, self.t_span, self.y0)
        
        # Results should be identical (due to same random seed)
        pd.testing.assert_frame_equal(df1, df2)


if __name__ == '__main__':
    unittest.main()
