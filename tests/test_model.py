"""
Unit tests for the thermal cooling loop digital twin model.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path to import the twin module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from twin.model import (
    ThermalCoolingTwin, 
    create_default_parameters,
    create_time_varying_heat_input,
    create_step_heat_input,
    create_ramp_heat_input,
    create_pulse_heat_input,
    create_complex_heat_input
)


class TestThermalCoolingTwin(unittest.TestCase):
    """Test cases for the ThermalCoolingTwin class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.params = create_default_parameters()
        self.twin = ThermalCoolingTwin(self.params)
        self.t_span = (0, 100)
        self.y0 = np.array([350.0, 300.0])  # Initial temperatures
        
    def test_initialization(self):
        """Test twin model initialization."""
        self.assertIsInstance(self.twin, ThermalCoolingTwin)
        self.assertEqual(self.twin.params, self.params)
        
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test missing parameter
        invalid_params = {'C': 1000, 'm_dot': 0.1}  # Missing required params
        with self.assertRaises(ValueError):
            ThermalCoolingTwin(invalid_params)
            
        # Test negative parameter
        invalid_params = create_default_parameters()
        invalid_params['C'] = -1000
        with self.assertRaises(ValueError):
            ThermalCoolingTwin(invalid_params)
    
    def test_twin_rhs(self):
        """Test the right-hand side function."""
        t = 0.0
        y = np.array([350.0, 300.0])
        
        dydt = self.twin.twin_rhs(t, y)
        
        self.assertIsInstance(dydt, np.ndarray)
        self.assertEqual(len(dydt), 2)
        self.assertTrue(np.isfinite(dydt).all())
    
    def test_simulation_basic(self):
        """Test basic simulation functionality."""
        results = self.twin.simulate(self.t_span, self.y0)
        
        # Check results structure
        self.assertIn('t', results)
        self.assertIn('T_hot', results)
        self.assertIn('T_cold', results)
        self.assertIn('success', results)
        self.assertIn('message', results)
        
        # Check data types and shapes
        self.assertTrue(results['success'])
        self.assertIsInstance(results['t'], np.ndarray)
        self.assertIsInstance(results['T_hot'], np.ndarray)
        self.assertIsInstance(results['T_cold'], np.ndarray)
        
        # Check array lengths
        n_points = len(results['t'])
        self.assertEqual(len(results['T_hot']), n_points)
        self.assertEqual(len(results['T_cold']), n_points)
        
        # Check that temperatures are finite
        self.assertTrue(np.isfinite(results['T_hot']).all())
        self.assertTrue(np.isfinite(results['T_cold']).all())
    
    def test_simulation_with_t_eval(self):
        """Test simulation with specific time evaluation points."""
        t_eval = np.linspace(0, 100, 50)
        results = self.twin.simulate(self.t_span, self.y0, t_eval=t_eval)
        
        self.assertTrue(results['success'])
        np.testing.assert_array_almost_equal(results['t'], t_eval)
    
    def test_simulation_different_methods(self):
        """Test simulation with different ODE solver methods."""
        methods = ['RK45', 'RK23', 'DOP853']
        
        for method in methods:
            with self.subTest(method=method):
                results = self.twin.simulate(
                    self.t_span, self.y0, method=method
                )
                self.assertTrue(results['success'])
    
    def test_steady_state_calculation(self):
        """Test steady-state temperature calculation."""
        T_hot_ss, T_cold_ss = self.twin.get_steady_state()
        
        self.assertIsInstance(T_hot_ss, float)
        self.assertIsInstance(T_cold_ss, float)
        self.assertTrue(np.isfinite(T_hot_ss))
        self.assertTrue(np.isfinite(T_cold_ss))
        self.assertGreater(T_hot_ss, T_cold_ss)  # Hot should be hotter than cold
    
    def test_energy_balance(self):
        """Test energy balance in steady state."""
        # For steady state, Q_in should equal Q_out
        T_hot_ss, T_cold_ss = self.twin.get_steady_state()
        
        # Simulate steady state
        y0_ss = np.array([T_hot_ss, T_cold_ss])
        results = self.twin.simulate((0, 10), y0_ss)
        
        # Check that temperatures remain approximately constant
        T_hot_final = results['T_hot'][-1]
        T_cold_final = results['T_cold'][-1]
        
        self.assertAlmostEqual(T_hot_final, T_hot_ss, places=1)
        self.assertAlmostEqual(T_cold_final, T_cold_ss, places=1)
    
    def test_time_varying_heat_input(self):
        """Test simulation with time-varying heat input."""
        # Create time-varying heat input
        Q_in_func = create_time_varying_heat_input(
            base_power=1000.0,
            amplitude=200.0,
            frequency=0.1
        )
        
        # Update parameters
        params = self.params.copy()
        params['Q_in'] = Q_in_func
        twin = ThermalCoolingTwin(params)
        
        # Run simulation
        results = twin.simulate(self.t_span, self.y0)
        
        self.assertTrue(results['success'])
        self.assertTrue(np.isfinite(results['T_hot']).all())
        self.assertTrue(np.isfinite(results['T_cold']).all())
    
    def test_step_heat_input(self):
        """Test simulation with step heat input."""
        Q_in_func = create_step_heat_input(
            base_power=1000.0,
            step_power=1500.0,
            step_time=50.0
        )
        
        params = self.params.copy()
        params['Q_in'] = Q_in_func
        twin = ThermalCoolingTwin(params)
        
        results = twin.simulate(self.t_span, self.y0)
        
        self.assertTrue(results['success'])
        
        # Check that temperatures respond to step change
        # Temperature should increase after step time
        step_idx = np.argmin(np.abs(results['t'] - 50.0))
        T_hot_before = np.mean(results['T_hot'][:step_idx])
        T_hot_after = np.mean(results['T_hot'][step_idx:])
        
        self.assertGreater(T_hot_after, T_hot_before)
    
    def test_ramp_heat_input(self):
        """Test simulation with ramp heat input."""
        Q_in_func = create_ramp_heat_input(
            base_power=1000.0,
            final_power=1500.0,
            ramp_start=20.0,
            ramp_duration=30.0
        )
        
        params = self.params.copy()
        params['Q_in'] = Q_in_func
        twin = ThermalCoolingTwin(params)
        
        results = twin.simulate(self.t_span, self.y0)
        
        self.assertTrue(results['success'])
        self.assertTrue(np.isfinite(results['T_hot']).all())
    
    def test_pulse_heat_input(self):
        """Test simulation with pulse heat input."""
        Q_in_func = create_pulse_heat_input(
            base_power=1000.0,
            pulse_power=2000.0,
            pulse_start=30.0,
            pulse_duration=10.0
        )
        
        params = self.params.copy()
        params['Q_in'] = Q_in_func
        twin = ThermalCoolingTwin(params)
        
        results = twin.simulate(self.t_span, self.y0)
        
        self.assertTrue(results['success'])
        self.assertTrue(np.isfinite(results['T_hot']).all())
    
    def test_complex_heat_input(self):
        """Test simulation with complex heat input."""
        components = [
            {'type': 'sin', 'amplitude': 100, 'frequency': 0.05},
            {'type': 'step', 'amplitude': 200, 'start_time': 50},
            {'type': 'pulse', 'amplitude': 500, 'start_time': 100, 'duration': 20}
        ]
        
        Q_in_func = create_complex_heat_input(
            base_power=1000.0,
            components=components
        )
        
        params = self.params.copy()
        params['Q_in'] = Q_in_func
        twin = ThermalCoolingTwin(params)
        
        # Extend simulation time to see all components
        t_span = (0, 150)
        results = twin.simulate(t_span, self.y0)
        
        self.assertTrue(results['success'])
        self.assertTrue(np.isfinite(results['T_hot']).all())
    
    def test_parameter_sensitivity(self):
        """Test sensitivity to parameter changes."""
        # Test mass flow rate sensitivity
        params_low_flow = self.params.copy()
        params_low_flow['m_dot'] = 0.05
        
        twin_low_flow = ThermalCoolingTwin(params_low_flow)
        results_low_flow = twin_low_flow.simulate(self.t_span, self.y0)
        
        # Lower mass flow should result in higher temperatures
        T_hot_low_flow = results_low_flow['T_hot'][-1]
        T_hot_normal = self.twin.simulate(self.t_span, self.y0)['T_hot'][-1]
        
        self.assertGreater(T_hot_low_flow, T_hot_normal)
    
    def test_convergence(self):
        """Test that simulation converges to steady state."""
        # Start from arbitrary initial conditions
        y0_arbitrary = np.array([400.0, 250.0])
        
        # Run long simulation
        t_span_long = (0, 500)
        results = self.twin.simulate(t_span_long, y0_arbitrary)
        
        self.assertTrue(results['success'])
        
        # Check that final temperatures are close to steady state
        T_hot_ss, T_cold_ss = self.twin.get_steady_state()
        T_hot_final = results['T_hot'][-1]
        T_cold_final = results['T_cold'][-1]
        
        self.assertAlmostEqual(T_hot_final, T_hot_ss, places=1)
        self.assertAlmostEqual(T_cold_final, T_cold_ss, places=1)


class TestHeatInputFunctions(unittest.TestCase):
    """Test cases for heat input functions."""
    
    def test_time_varying_heat_input(self):
        """Test time-varying heat input function."""
        Q_in = create_time_varying_heat_input(1000.0, 200.0, 0.1)
        
        # Test at different times
        self.assertAlmostEqual(Q_in(0), 1000.0)
        self.assertAlmostEqual(Q_in(2.5), 1000.0)  # Should be at peak
        self.assertAlmostEqual(Q_in(5.0), 1000.0)  # Should be at zero
    
    def test_step_heat_input(self):
        """Test step heat input function."""
        Q_in = create_step_heat_input(1000.0, 1500.0, 50.0)
        
        self.assertEqual(Q_in(25.0), 1000.0)  # Before step
        self.assertEqual(Q_in(50.0), 1500.0)  # At step
        self.assertEqual(Q_in(75.0), 1500.0)  # After step
    
    def test_ramp_heat_input(self):
        """Test ramp heat input function."""
        Q_in = create_ramp_heat_input(1000.0, 1500.0, 20.0, 30.0)
        
        self.assertEqual(Q_in(10.0), 1000.0)  # Before ramp
        self.assertEqual(Q_in(35.0), 1500.0)  # After ramp
        self.assertAlmostEqual(Q_in(27.5), 1250.0)  # Mid-ramp
    
    def test_pulse_heat_input(self):
        """Test pulse heat input function."""
        Q_in = create_pulse_heat_input(1000.0, 2000.0, 30.0, 10.0)
        
        self.assertEqual(Q_in(20.0), 1000.0)  # Before pulse
        self.assertEqual(Q_in(35.0), 2000.0)  # During pulse
        self.assertEqual(Q_in(45.0), 1000.0)  # After pulse
    
    def test_complex_heat_input(self):
        """Test complex heat input function."""
        components = [
            {'type': 'sin', 'amplitude': 100, 'frequency': 0.1},
            {'type': 'step', 'amplitude': 200, 'start_time': 50}
        ]
        
        Q_in = create_complex_heat_input(1000.0, components)
        
        # Test before step
        self.assertAlmostEqual(Q_in(25.0), 1000.0, places=1)
        
        # Test after step
        self.assertAlmostEqual(Q_in(75.0), 1200.0, places=1)


if __name__ == '__main__':
    unittest.main()
