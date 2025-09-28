"""
Digital Twin Manager for Thermal Cooling Loop

This module implements the digital twin that runs in lockstep with the plant
simulator, computes residuals, and performs real-time anomaly detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging
import time
from threading import Thread, Event
import queue

from .model import ThermalCoolingTwin, create_default_parameters
from .plant_simulator import PlantSimulator, FaultType
from .detector import AnomalyDetector, DetectionMethod, AnomalyType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DigitalTwinManager:
    """
    Digital Twin Manager for real-time monitoring and anomaly detection.
    
    Runs the digital twin in lockstep with the plant simulator, computes
    residuals, and performs real-time anomaly detection.
    """
    
    def __init__(self, 
                 plant_simulator: PlantSimulator,
                 twin_params: Dict[str, float] = None,
                 detection_methods: List[DetectionMethod] = None,
                 update_rate: float = 1.0):
        """
        Initialize the digital twin manager.
        
        Parameters:
        -----------
        plant_simulator : PlantSimulator
            Plant simulator instance
        twin_params : dict, optional
            Parameters for the digital twin model
        detection_methods : list, optional
            Anomaly detection methods to use
        update_rate : float
            Update rate in Hz
        """
        self.plant_simulator = plant_simulator
        self.twin_params = twin_params or create_default_parameters()
        self.update_rate = update_rate
        self.dt = 1.0 / update_rate
        
        # Create digital twin
        self.twin = ThermalCoolingTwin(self.twin_params)
        
        # Create anomaly detector
        self.detector = AnomalyDetector(
            detection_methods=detection_methods,
            window_size=100
        )
        
        # State variables
        self.is_running = False
        self.current_time = 0.0
        self.twin_state = None
        self.plant_state = None
        self.residuals = {}
        self.anomaly_results = {}
        
        # Data storage
        self.history = []
        self.max_history_size = 10000
        
        # Threading
        self.thread = None
        self.stop_event = Event()
        self.data_queue = queue.Queue()
        
        # Callbacks
        self.anomaly_callbacks = []
        self.data_callbacks = []
        
        logger.info("Digital twin manager initialized")
    
    def start(self, initial_conditions: np.ndarray = None):
        """
        Start the digital twin in real-time mode.
        
        Parameters:
        -----------
        initial_conditions : np.ndarray, optional
            Initial conditions [T_hot, T_cold]
        """
        if self.is_running:
            logger.warning("Digital twin is already running")
            return
        
        if initial_conditions is None:
            initial_conditions = np.array([350.0, 300.0])
        
        self.twin_state = initial_conditions.copy()
        self.plant_state = initial_conditions.copy()
        self.current_time = 0.0
        
        # Reset detector
        self.detector.reset()
        
        # Start thread
        self.is_running = True
        self.stop_event.clear()
        self.thread = Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        
        logger.info("Digital twin started")
    
    def stop(self):
        """Stop the digital twin."""
        if not self.is_running:
            logger.warning("Digital twin is not running")
            return
        
        self.is_running = False
        self.stop_event.set()
        
        if self.thread:
            self.thread.join(timeout=5.0)
        
        logger.info("Digital twin stopped")
    
    def _run_loop(self):
        """Main execution loop for the digital twin."""
        logger.info("Digital twin loop started")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # Update plant simulator
                self._update_plant()
                
                # Update digital twin
                self._update_twin()
                
                # Compute residuals
                self._compute_residuals()
                
                # Detect anomalies
                self._detect_anomalies()
                
                # Store data
                self._store_data()
                
                # Notify callbacks
                self._notify_callbacks()
                
                # Update time
                self.current_time += self.dt
                
                # Sleep to maintain update rate
                time.sleep(self.dt)
                
            except Exception as e:
                logger.error(f"Error in digital twin loop: {str(e)}")
                break
        
        logger.info("Digital twin loop stopped")
    
    def _update_plant(self):
        """Update the plant simulator state."""
        # Get current plant parameters (may be modified by faults)
        plant_params = self.plant_simulator.get_modified_parameters(self.current_time)
        
        # Create temporary twin with plant parameters
        temp_twin = ThermalCoolingTwin(plant_params)
        
        # Simulate single step
        t_span = (self.current_time, self.current_time + self.dt)
        result = temp_twin.simulate(t_span, self.plant_state, method='RK45')
        
        if result['success']:
            self.plant_state = np.array([result['T_hot'][-1], result['T_cold'][-1]])
        else:
            logger.warning("Plant simulation failed")
    
    def _update_twin(self):
        """Update the digital twin state."""
        # Simulate single step with twin parameters
        t_span = (self.current_time, self.current_time + self.dt)
        result = self.twin.simulate(t_span, self.twin_state, method='RK45')
        
        if result['success']:
            self.twin_state = np.array([result['T_hot'][-1], result['T_cold'][-1]])
        else:
            logger.warning("Twin simulation failed")
    
    def _compute_residuals(self):
        """Compute residuals between plant and twin."""
        if self.plant_state is None or self.twin_state is None:
            return
        
        # Compute temperature residuals
        T_hot_residual = self.plant_state[0] - self.twin_state[0]
        T_cold_residual = self.plant_state[1] - self.twin_state[1]
        
        # Get mass flow rate residual (if available)
        plant_params = self.plant_simulator.get_modified_parameters(self.current_time)
        m_dot_residual = plant_params['m_dot'] - self.twin_params['m_dot']
        
        self.residuals = {
            'T_hot': T_hot_residual,
            'T_cold': T_cold_residual,
            'm_dot': m_dot_residual
        }
    
    def _detect_anomalies(self):
        """Detect anomalies in the residuals."""
        if not self.residuals:
            return
        
        # Detect anomalies
        self.anomaly_results = self.detector.detect_anomalies(
            self.residuals, 
            self.current_time
        )
        
        # Check for anomalies and notify callbacks
        if self.anomaly_results.get('overall_anomaly', False):
            self._notify_anomaly_callbacks(self.anomaly_results)
    
    def _store_data(self):
        """Store current data in history."""
        data_point = {
            'timestamp': self.current_time,
            'plant_state': self.plant_state.copy() if self.plant_state is not None else None,
            'twin_state': self.twin_state.copy() if self.twin_state is not None else None,
            'residuals': self.residuals.copy(),
            'anomaly_results': self.anomaly_results.copy()
        }
        
        self.history.append(data_point)
        
        # Limit history size
        if len(self.history) > self.max_history_size:
            self.history = self.history[-self.max_history_size:]
    
    def _notify_callbacks(self):
        """Notify data callbacks."""
        for callback in self.data_callbacks:
            try:
                callback(self.get_current_data())
            except Exception as e:
                logger.warning(f"Data callback failed: {str(e)}")
    
    def _notify_anomaly_callbacks(self, anomaly_results: Dict[str, Any]):
        """Notify anomaly callbacks."""
        for callback in self.anomaly_callbacks:
            try:
                callback(anomaly_results)
            except Exception as e:
                logger.warning(f"Anomaly callback failed: {str(e)}")
    
    def get_current_data(self) -> Dict[str, Any]:
        """Get current system data."""
        return {
            'timestamp': self.current_time,
            'plant_state': self.plant_state.copy() if self.plant_state is not None else None,
            'twin_state': self.twin_state.copy() if self.twin_state is not None else None,
            'residuals': self.residuals.copy(),
            'anomaly_results': self.anomaly_results.copy(),
            'is_running': self.is_running
        }
    
    def get_history(self, time_window: float = None) -> List[Dict[str, Any]]:
        """
        Get historical data.
        
        Parameters:
        -----------
        time_window : float, optional
            Time window in seconds
            
        Returns:
        --------
        list
            Historical data points
        """
        if time_window is None:
            return self.history.copy()
        
        cutoff_time = self.current_time - time_window
        return [data for data in self.history if data['timestamp'] >= cutoff_time]
    
    def get_residual_statistics(self, time_window: float = None) -> Dict[str, Dict[str, float]]:
        """
        Get residual statistics.
        
        Parameters:
        -----------
        time_window : float, optional
            Time window in seconds
            
        Returns:
        --------
        dict
            Residual statistics for each sensor
        """
        history = self.get_history(time_window)
        
        if not history:
            return {}
        
        residuals_data = {}
        for sensor in ['T_hot', 'T_cold', 'm_dot']:
            sensor_residuals = [data['residuals'].get(sensor, 0) for data in history if data['residuals']]
            
            if sensor_residuals:
                residuals_data[sensor] = {
                    'mean': np.mean(sensor_residuals),
                    'std': np.std(sensor_residuals),
                    'min': np.min(sensor_residuals),
                    'max': np.max(sensor_residuals),
                    'rms': np.sqrt(np.mean(np.square(sensor_residuals)))
                }
        
        return residuals_data
    
    def get_anomaly_summary(self, time_window: float = None) -> Dict[str, Any]:
        """
        Get anomaly summary.
        
        Parameters:
        -----------
        time_window : float, optional
            Time window in seconds
            
        Returns:
        --------
        dict
            Anomaly summary
        """
        history = self.get_history(time_window)
        
        if not history:
            return {'total_anomalies': 0, 'anomaly_types': {}}
        
        anomalies = [data for data in history if data.get('anomaly_results', {}).get('overall_anomaly', False)]
        
        # Count by severity
        anomaly_types = {}
        for anomaly in anomalies:
            severity = anomaly['anomaly_results'].get('overall_severity', AnomalyType.NORMAL).value
            anomaly_types[severity] = anomaly_types.get(severity, 0) + 1
        
        return {
            'total_anomalies': len(anomalies),
            'anomaly_types': anomaly_types,
            'anomaly_rate': len(anomalies) / len(history) if history else 0,
            'recent_anomalies': anomalies[-10:] if anomalies else []
        }
    
    def add_anomaly_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add anomaly detection callback."""
        self.anomaly_callbacks.append(callback)
    
    def add_data_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add data update callback."""
        self.data_callbacks.append(callback)
    
    def train_detector(self, training_data: List[Dict[str, float]]):
        """Train the anomaly detector."""
        self.detector.train(training_data)
        logger.info("Anomaly detector trained")
    
    def update_twin_parameters(self, new_params: Dict[str, float]):
        """Update digital twin parameters."""
        self.twin_params.update(new_params)
        self.twin = ThermalCoolingTwin(self.twin_params)
        logger.info("Twin parameters updated")
    
    def inject_plant_fault(self, fault_type: FaultType, start_time: float, parameters: Dict = None):
        """Inject fault into plant simulator."""
        self.plant_simulator.inject_fault(fault_type, start_time, parameters)
        logger.info(f"Plant fault injected: {fault_type.value}")
    
    def clear_plant_faults(self):
        """Clear all plant faults."""
        self.plant_simulator.clear_faults()
        logger.info("Plant faults cleared")
    
    def export_data(self, filename: str = None) -> str:
        """
        Export historical data to CSV.
        
        Parameters:
        -----------
        filename : str, optional
            Output filename
            
        Returns:
        --------
        str
            Exported filename
        """
        if filename is None:
            filename = f"digital_twin_data_{int(time.time())}.csv"
        
        # Prepare data for export
        export_data = []
        for data in self.history:
            if data['plant_state'] is not None and data['twin_state'] is not None:
                row = {
                    'timestamp': data['timestamp'],
                    'plant_T_hot': data['plant_state'][0],
                    'plant_T_cold': data['plant_state'][1],
                    'twin_T_hot': data['twin_state'][0],
                    'twin_T_cold': data['twin_state'][1],
                    'residual_T_hot': data['residuals'].get('T_hot', 0),
                    'residual_T_cold': data['residuals'].get('T_cold', 0),
                    'residual_m_dot': data['residuals'].get('m_dot', 0),
                    'anomaly_detected': data['anomaly_results'].get('overall_anomaly', False),
                    'anomaly_severity': data['anomaly_results'].get('overall_severity', AnomalyType.NORMAL).value
                }
                export_data.append(row)
        
        # Create DataFrame and export
        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False)
        
        logger.info(f"Data exported to {filename}")
        return filename


def create_digital_twin_manager(plant_simulator: PlantSimulator = None,
                               twin_params: Dict[str, float] = None,
                               detection_methods: List[DetectionMethod] = None) -> DigitalTwinManager:
    """
    Create a digital twin manager with default configuration.
    
    Parameters:
    -----------
    plant_simulator : PlantSimulator, optional
        Plant simulator instance
    twin_params : dict, optional
        Twin model parameters
    detection_methods : list, optional
        Detection methods to use
        
    Returns:
    --------
    DigitalTwinManager
        Configured digital twin manager
    """
    if plant_simulator is None:
        from .plant_simulator import PlantSimulator
        plant_simulator = PlantSimulator(create_default_parameters())
    
    if twin_params is None:
        twin_params = create_default_parameters()
    
    if detection_methods is None:
        detection_methods = [
            DetectionMethod.RESIDUAL_THRESHOLD,
            DetectionMethod.ROLLING_Z_SCORE,
            DetectionMethod.ISOLATION_FOREST
        ]
    
    return DigitalTwinManager(
        plant_simulator=plant_simulator,
        twin_params=twin_params,
        detection_methods=detection_methods,
        update_rate=1.0
    )
