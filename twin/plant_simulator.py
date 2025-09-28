"""
Plant Simulator Implementation

This module implements a synthetic plant simulator that generates realistic
sensor data with noise, bias drift, and injected faults for testing the
digital twin and anomaly detection algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaultType(Enum):
    """Types of faults that can be injected into the plant simulator."""
    NONE = "none"
    PUMP_DEGRADATION = "pump_degradation"
    HEAT_EXCHANGER_FOULING = "heat_exchanger_fouling"
    SENSOR_BIAS = "sensor_bias"
    SENSOR_NOISE_INCREASE = "sensor_noise_increase"
    MASS_FLOW_REDUCTION = "mass_flow_reduction"


class PlantSimulator:
    """
    Synthetic plant simulator for thermal cooling loop system.
    
    Generates realistic sensor data with various types of noise and faults
    for testing digital twin and anomaly detection capabilities.
    """
    
    def __init__(self, 
                 base_params: Dict[str, float],
                 noise_level: float = 0.01,
                 sample_rate: float = 1.0):
        """
        Initialize the plant simulator.
        
        Parameters:
        -----------
        base_params : dict
            Base system parameters (same as twin model)
        noise_level : float
            Standard deviation of Gaussian noise (relative to signal)
        sample_rate : float
            Sampling rate [Hz]
        """
        self.base_params = base_params.copy()
        self.noise_level = noise_level
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        
        # Fault state
        self.current_fault = FaultType.NONE
        self.fault_start_time = None
        self.fault_parameters = {}
        
        # Sensor bias and drift
        self.sensor_bias = {'T_hot': 0.0, 'T_cold': 0.0, 'm_dot': 0.0}
        self.sensor_drift = {'T_hot': 0.0, 'T_cold': 0.0, 'm_dot': 0.0}
        
        # Data storage
        self.sensor_data = []
        self.time_history = []
        
        logger.info("Plant simulator initialized successfully")
    
    def inject_fault(self, 
                    fault_type: FaultType, 
                    start_time: float,
                    parameters: Optional[Dict] = None) -> None:
        """
        Inject a fault into the plant simulator.
        
        Parameters:
        -----------
        fault_type : FaultType
            Type of fault to inject
        start_time : float
            Time when fault starts [s]
        parameters : dict, optional
            Fault-specific parameters
        """
        self.current_fault = fault_type
        self.fault_start_time = start_time
        self.fault_parameters = parameters or {}
        
        logger.info(f"Fault injected: {fault_type.value} at t={start_time}s")
    
    def clear_fault(self) -> None:
        """Clear any active faults."""
        self.current_fault = FaultType.NONE
        self.fault_start_time = None
        self.fault_parameters = {}
        logger.info("All faults cleared")
    
    def get_modified_parameters(self, t: float) -> Dict[str, float]:
        """
        Get system parameters modified by active faults.
        
        Parameters:
        -----------
        t : float
            Current time [s]
            
        Returns:
        --------
        dict
            Modified system parameters
        """
        params = self.base_params.copy()
        
        if self.current_fault == FaultType.NONE or t < self.fault_start_time:
            return params
        
        fault_duration = t - self.fault_start_time
        
        if self.current_fault == FaultType.PUMP_DEGRADATION:
            # Gradual reduction in mass flow rate
            degradation_rate = self.fault_parameters.get('degradation_rate', 0.1)
            flow_reduction = min(degradation_rate * fault_duration, 0.8)
            params['m_dot'] *= (1.0 - flow_reduction)
            
        elif self.current_fault == FaultType.HEAT_EXCHANGER_FOULING:
            # Gradual reduction in heat exchanger effectiveness
            fouling_rate = self.fault_parameters.get('fouling_rate', 0.05)
            effectiveness_reduction = min(fouling_rate * fault_duration, 0.6)
            params['UA'] *= (1.0 - effectiveness_reduction)
            
        elif self.current_fault == FaultType.MASS_FLOW_REDUCTION:
            # Sudden reduction in mass flow rate
            reduction_factor = self.fault_parameters.get('reduction_factor', 0.5)
            params['m_dot'] *= reduction_factor
        
        return params
    
    def add_sensor_noise(self, 
                        T_hot: float, 
                        T_cold: float, 
                        m_dot: float,
                        t: float) -> Tuple[float, float, float]:
        """
        Add realistic sensor noise to measurements.
        
        Parameters:
        -----------
        T_hot : float
            True hot temperature [K]
        T_cold : float
            True cold temperature [K]
        m_dot : float
            True mass flow rate [kg/s]
        t : float
            Current time [s]
            
        Returns:
        --------
        tuple
            Noisy measurements (T_hot_noisy, T_cold_noisy, m_dot_noisy)
        """
        # Base noise level
        noise_level = self.noise_level
        
        # Increase noise for sensor noise fault
        if (self.current_fault == FaultType.SENSOR_NOISE_INCREASE and 
            t >= self.fault_start_time):
            noise_multiplier = self.fault_parameters.get('noise_multiplier', 3.0)
            noise_level *= noise_multiplier
        
        # Add Gaussian noise
        T_hot_noisy = T_hot + np.random.normal(0, noise_level * T_hot)
        T_cold_noisy = T_cold + np.random.normal(0, noise_level * T_cold)
        m_dot_noisy = m_dot + np.random.normal(0, noise_level * m_dot)
        
        # Add sensor bias
        if (self.current_fault == FaultType.SENSOR_BIAS and 
            t >= self.fault_start_time):
            bias_magnitude = self.fault_parameters.get('bias_magnitude', 5.0)
            T_hot_noisy += bias_magnitude
            T_cold_noisy += bias_magnitude
        
        # Add sensor drift
        drift_rate = 0.001  # K/s
        T_hot_noisy += self.sensor_drift['T_hot'] * t
        T_cold_noisy += self.sensor_drift['T_cold'] * t
        m_dot_noisy += self.sensor_drift['m_dot'] * t
        
        return T_hot_noisy, T_cold_noisy, m_dot_noisy
    
    def simulate_sensor_data(self, 
                           twin_model,
                           t_span: Tuple[float, float],
                           y0: np.ndarray,
                           t_eval: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Simulate sensor data from the plant.
        
        Parameters:
        -----------
        twin_model : ThermalCoolingTwin
            Reference twin model for baseline simulation
        t_span : tuple
            Time span (t_start, t_end) [s]
        y0 : np.ndarray
            Initial conditions [T_hot_0, T_cold_0] [K]
        t_eval : np.ndarray, optional
            Time points for evaluation [s]
            
        Returns:
        --------
        pd.DataFrame
            Sensor data with columns: t, T_hot, T_cold, m_dot, fault_type
        """
        # Generate time points if not provided
        if t_eval is None:
            t_eval = np.arange(t_span[0], t_span[1] + self.dt, self.dt)
        
        # Simulate using modified parameters
        sensor_data = []
        
        for i, t in enumerate(t_eval):
            # Get modified parameters for this time step
            modified_params = self.get_modified_parameters(t)
            
            # Create temporary twin with modified parameters
            temp_twin = twin_model.__class__(modified_params)
            
            # Simulate single time step
            if i == 0:
                y_current = y0
            else:
                # Use previous state as initial condition
                y_current = np.array([sensor_data[-1]['T_hot_true'], 
                                    sensor_data[-1]['T_cold_true']])
            
            # Single step simulation
            sol = temp_twin.simulate((t, t + self.dt), y_current, 
                                   t_eval=np.array([t + self.dt]))
            
            if sol['success']:
                T_hot_true = sol['T_hot'][0]
                T_cold_true = sol['T_cold'][0]
                m_dot_true = modified_params['m_dot']
                
                # Add sensor noise
                T_hot_noisy, T_cold_noisy, m_dot_noisy = self.add_sensor_noise(
                    T_hot_true, T_cold_true, m_dot_true, t)
                
                # Store data
                data_point = {
                    't': t,
                    'T_hot_true': T_hot_true,
                    'T_cold_true': T_cold_true,
                    'm_dot_true': m_dot_true,
                    'T_hot': T_hot_noisy,
                    'T_cold': T_cold_noisy,
                    'm_dot': m_dot_noisy,
                    'fault_type': self.current_fault.value,
                    'fault_active': t >= self.fault_start_time if self.fault_start_time else False
                }
                
                sensor_data.append(data_point)
            else:
                logger.warning(f"Simulation failed at t={t}")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(sensor_data)
        
        logger.info(f"Generated {len(df)} sensor data points")
        return df
    
    def create_demo_scenarios(self) -> List[Dict]:
        """
        Create predefined demo scenarios for testing.
        
        Returns:
        --------
        list
            List of scenario dictionaries
        """
        scenarios = [
            {
                'name': 'Nominal Operation',
                'description': 'Normal operation with no faults',
                'duration': 100.0,
                'faults': []
            },
            {
                'name': 'Pump Degradation',
                'description': 'Gradual pump degradation over time',
                'duration': 200.0,
                'faults': [
                    {
                        'type': FaultType.PUMP_DEGRADATION,
                        'start_time': 50.0,
                        'parameters': {'degradation_rate': 0.02}
                    }
                ]
            },
            {
                'name': 'Heat Exchanger Fouling',
                'description': 'Gradual heat exchanger fouling',
                'duration': 300.0,
                'faults': [
                    {
                        'type': FaultType.HEAT_EXCHANGER_FOULING,
                        'start_time': 100.0,
                        'parameters': {'fouling_rate': 0.01}
                    }
                ]
            },
            {
                'name': 'Sensor Bias',
                'description': 'Temperature sensor bias fault',
                'duration': 150.0,
                'faults': [
                    {
                        'type': FaultType.SENSOR_BIAS,
                        'start_time': 75.0,
                        'parameters': {'bias_magnitude': 10.0}
                    }
                ]
            },
            {
                'name': 'Multiple Faults',
                'description': 'Combination of pump degradation and sensor noise',
                'duration': 250.0,
                'faults': [
                    {
                        'type': FaultType.PUMP_DEGRADATION,
                        'start_time': 50.0,
                        'parameters': {'degradation_rate': 0.015}
                    },
                    {
                        'type': FaultType.SENSOR_NOISE_INCREASE,
                        'start_time': 150.0,
                        'parameters': {'noise_multiplier': 2.5}
                    }
                ]
            }
        ]
        
        return scenarios


def create_default_plant_simulator() -> PlantSimulator:
    """
    Create a plant simulator with default parameters.
    
    Returns:
    --------
    PlantSimulator
        Default plant simulator instance
    """
    from .model import create_default_parameters
    
    base_params = create_default_parameters()
    return PlantSimulator(base_params, noise_level=0.005, sample_rate=2.0)
