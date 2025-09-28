"""
Digital Twin Model Implementation

This module implements the core ODE-based thermal cooling loop model
using lumped capacitance approach for real-time simulation.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Callable, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThermalCoolingTwin:
    """
    Digital twin model for thermal cooling loop system.
    
    Implements the lumped capacitance model with two state variables:
    - T_hot: Fluid temperature after heat source
    - T_cold: Fluid temperature after heat exchanger
    """
    
    def __init__(self, params: Dict[str, float]):
        """
        Initialize the thermal cooling twin model.
        
        Parameters:
        -----------
        params : dict
            System parameters including:
            - C: Thermal capacitance [J/K]
            - m_dot: Mass flow rate [kg/s]
            - cp: Specific heat capacity [J/(kg·K)]
            - UA: Heat exchanger effectiveness [W/K]
            - Q_in: Heat input function [W]
            - Q_out: Heat output [W] (optional)
        """
        self.params = params
        self.validate_parameters()
        
    def validate_parameters(self) -> None:
        """Validate that all required parameters are present and positive."""
        required_params = ['C', 'm_dot', 'cp', 'UA']
        
        for param in required_params:
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")
            if self.params[param] <= 0:
                raise ValueError(f"Parameter {param} must be positive")
                
        # Set default Q_out if not provided
        if 'Q_out' not in self.params:
            self.params['Q_out'] = 0.0
            
        logger.info("Thermal cooling twin parameters validated successfully")
    
    def twin_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Right-hand side of the ODE system for the thermal cooling loop.
        
        Parameters:
        -----------
        t : float
            Current time [s]
        y : np.ndarray
            State vector [T_hot, T_cold] [K]
            
        Returns:
        --------
        np.ndarray
            Derivative vector [dT_hot/dt, dT_cold/dt] [K/s]
        """
        T_hot, T_cold = y
        
        # Extract parameters
        C = self.params['C']
        m_dot = self.params['m_dot']
        cp = self.params['cp']
        UA = self.params['UA']
        Q_out = self.params['Q_out']
        
        # Get heat input (can be time-dependent)
        if callable(self.params['Q_in']):
            Q_in = self.params['Q_in'](t)
        else:
            Q_in = self.params['Q_in']
        
        # Governing equations
        dT_hot_dt = (Q_in - UA * (T_hot - T_cold) - m_dot * cp * (T_hot - T_cold)) / C
        dT_cold_dt = (m_dot * cp * (T_hot - T_cold) - Q_out) / C
        
        return np.array([dT_hot_dt, dT_cold_dt])
    
    def simulate(self, 
                 t_span: Tuple[float, float], 
                 y0: np.ndarray,
                 t_eval: Optional[np.ndarray] = None,
                 rtol: float = 1e-6,
                 atol: float = 1e-9,
                 method: str = 'RK45') -> Dict[str, np.ndarray]:
        """
        Simulate the thermal cooling loop system.
        
        Parameters:
        -----------
        t_span : tuple
            Time span (t_start, t_end) [s]
        y0 : np.ndarray
            Initial conditions [T_hot_0, T_cold_0] [K]
        t_eval : np.ndarray, optional
            Time points for evaluation [s]
        rtol : float
            Relative tolerance for ODE solver
        atol : float
            Absolute tolerance for ODE solver
        method : str
            ODE solver method ('RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA')
            
        Returns:
        --------
        dict
            Simulation results with keys:
            - 't': time array [s]
            - 'T_hot': hot temperature [K]
            - 'T_cold': cold temperature [K]
            - 'success': solver success flag
            - 'message': solver message
            - 'nfev': number of function evaluations
            - 'njev': number of jacobian evaluations
        """
        try:
            # Solve ODE system
            sol = solve_ivp(
                fun=self.twin_rhs,
                t_span=t_span,
                y0=y0,
                t_eval=t_eval,
                rtol=rtol,
                atol=atol,
                method=method
            )
            
            if not sol.success:
                logger.warning(f"ODE solver failed: {sol.message}")
            
            # Extract results
            results = {
                't': sol.t,
                'T_hot': sol.y[0],
                'T_cold': sol.y[1],
                'success': sol.success,
                'message': sol.message,
                'nfev': sol.nfev,
                'njev': sol.njev
            }
            
            logger.info(f"Simulation completed: {len(sol.t)} time points")
            return results
            
        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            raise
    
    def get_steady_state(self, 
                        T_hot_guess: float = 350.0,
                        T_cold_guess: float = 300.0) -> Tuple[float, float]:
        """
        Calculate steady-state temperatures.
        
        Parameters:
        -----------
        T_hot_guess : float
            Initial guess for hot temperature [K]
        T_cold_guess : float
            Initial guess for cold temperature [K]
            
        Returns:
        --------
        tuple
            Steady-state temperatures (T_hot_ss, T_cold_ss) [K]
        """
        # For steady state, dT/dt = 0
        # This gives us a system of algebraic equations
        
        # Extract parameters
        m_dot = self.params['m_dot']
        cp = self.params['cp']
        UA = self.params['UA']
        Q_out = self.params['Q_out']
        
        # Get steady-state heat input
        if callable(self.params['Q_in']):
            # Use a large time value to approximate steady state
            Q_in = self.params['Q_in'](1000.0)
        else:
            Q_in = self.params['Q_in']
        
        # Solve steady-state equations
        # From dT_hot/dt = 0: Q_in - UA*(T_hot - T_cold) - m_dot*cp*(T_hot - T_cold) = 0
        # From dT_cold/dt = 0: m_dot*cp*(T_hot - T_cold) - Q_out = 0
        
        # From second equation: T_hot - T_cold = Q_out / (m_dot * cp)
        delta_T = Q_out / (m_dot * cp)
        
        # From first equation: Q_in - UA * delta_T - m_dot * cp * delta_T = 0
        # This should be satisfied if Q_in = Q_out (energy balance)
        
        # If Q_in != Q_out, we need to solve the full system
        if abs(Q_in - Q_out) > 1e-6:
            # Use numerical solution
            from scipy.optimize import fsolve
            
            def equations(vars):
                T_hot, T_cold = vars
                eq1 = Q_in - UA * (T_hot - T_cold) - m_dot * cp * (T_hot - T_cold)
                eq2 = m_dot * cp * (T_hot - T_cold) - Q_out
                return [eq1, eq2]
            
            solution = fsolve(equations, [T_hot_guess, T_cold_guess])
            T_hot_ss, T_cold_ss = solution
        else:
            # Simple case: Q_in = Q_out
            T_cold_ss = T_cold_guess
            T_hot_ss = T_cold_ss + delta_T
        
        logger.info(f"Steady-state temperatures: T_hot={T_hot_ss:.2f}K, T_cold={T_cold_ss:.2f}K")
        return T_hot_ss, T_cold_ss


def create_default_parameters() -> Dict[str, float]:
    """
    Create default system parameters for the thermal cooling loop.
    
    Returns:
    --------
    dict
        Default parameter values
    """
    return {
        'C': 1000.0,        # Thermal capacitance [J/K]
        'm_dot': 0.1,       # Mass flow rate [kg/s]
        'cp': 4180.0,       # Specific heat capacity [J/(kg·K)]
        'UA': 50.0,         # Heat exchanger effectiveness [W/K]
        'Q_in': 1000.0,     # Heat input [W]
        'Q_out': 1000.0     # Heat output [W]
    }


def create_time_varying_heat_input(base_power: float = 1000.0,
                                  amplitude: float = 200.0,
                                  frequency: float = 0.1) -> Callable[[float], float]:
    """
    Create a time-varying heat input function.
    
    Parameters:
    -----------
    base_power : float
        Base heat input power [W]
    amplitude : float
        Amplitude of variation [W]
    frequency : float
        Frequency of variation [Hz]
        
    Returns:
    --------
    callable
        Heat input function Q_in(t)
    """
    def Q_in(t):
        return base_power + amplitude * np.sin(2 * np.pi * frequency * t)
    
    return Q_in


def create_step_heat_input(base_power: float = 1000.0,
                          step_power: float = 1500.0,
                          step_time: float = 50.0) -> Callable[[float], float]:
    """
    Create a step heat input function.
    
    Parameters:
    -----------
    base_power : float
        Initial heat input power [W]
    step_power : float
        Final heat input power [W]
    step_time : float
        Time when step occurs [s]
        
    Returns:
    --------
    callable
        Heat input function Q_in(t)
    """
    def Q_in(t):
        return step_power if t >= step_time else base_power
    
    return Q_in


def create_ramp_heat_input(base_power: float = 1000.0,
                          final_power: float = 1500.0,
                          ramp_start: float = 20.0,
                          ramp_duration: float = 30.0) -> Callable[[float], float]:
    """
    Create a ramp heat input function.
    
    Parameters:
    -----------
    base_power : float
        Initial heat input power [W]
    final_power : float
        Final heat input power [W]
    ramp_start : float
        Time when ramp starts [s]
    ramp_duration : float
        Duration of ramp [s]
        
    Returns:
    --------
    callable
        Heat input function Q_in(t)
    """
    def Q_in(t):
        if t < ramp_start:
            return base_power
        elif t < ramp_start + ramp_duration:
            # Linear ramp
            progress = (t - ramp_start) / ramp_duration
            return base_power + progress * (final_power - base_power)
        else:
            return final_power
    
    return Q_in


def create_pulse_heat_input(base_power: float = 1000.0,
                           pulse_power: float = 2000.0,
                           pulse_start: float = 30.0,
                           pulse_duration: float = 10.0) -> Callable[[float], float]:
    """
    Create a pulse heat input function.
    
    Parameters:
    -----------
    base_power : float
        Base heat input power [W]
    pulse_power : float
        Pulse heat input power [W]
    pulse_start : float
        Time when pulse starts [s]
    pulse_duration : float
        Duration of pulse [s]
        
    Returns:
    --------
    callable
        Heat input function Q_in(t)
    """
    def Q_in(t):
        if pulse_start <= t <= pulse_start + pulse_duration:
            return pulse_power
        else:
            return base_power
    
    return Q_in


def create_complex_heat_input(base_power: float = 1000.0,
                             components: list = None) -> Callable[[float], float]:
    """
    Create a complex heat input function with multiple components.
    
    Parameters:
    -----------
    base_power : float
        Base heat input power [W]
    components : list
        List of component dictionaries with keys:
        - 'type': 'sin', 'cos', 'step', 'ramp', 'pulse'
        - 'amplitude': amplitude [W]
        - 'frequency': frequency [Hz] (for sin/cos)
        - 'start_time': start time [s]
        - 'duration': duration [s] (for ramp/pulse)
        - 'end_power': end power [W] (for ramp)
        
    Returns:
    --------
    callable
        Heat input function Q_in(t)
    """
    if components is None:
        components = [
            {'type': 'sin', 'amplitude': 100, 'frequency': 0.05},
            {'type': 'step', 'amplitude': 200, 'start_time': 50},
            {'type': 'pulse', 'amplitude': 500, 'start_time': 100, 'duration': 20}
        ]
    
    def Q_in(t):
        total = base_power
        
        for comp in components:
            comp_type = comp['type']
            amplitude = comp['amplitude']
            
            if comp_type == 'sin':
                frequency = comp.get('frequency', 0.1)
                total += amplitude * np.sin(2 * np.pi * frequency * t)
                
            elif comp_type == 'cos':
                frequency = comp.get('frequency', 0.1)
                total += amplitude * np.cos(2 * np.pi * frequency * t)
                
            elif comp_type == 'step':
                start_time = comp.get('start_time', 0)
                if t >= start_time:
                    total += amplitude
                    
            elif comp_type == 'ramp':
                start_time = comp.get('start_time', 0)
                duration = comp.get('duration', 10)
                end_power = comp.get('end_power', base_power + amplitude)
                
                if t >= start_time and t <= start_time + duration:
                    progress = (t - start_time) / duration
                    total += progress * amplitude
                elif t > start_time + duration:
                    total += amplitude
                    
            elif comp_type == 'pulse':
                start_time = comp.get('start_time', 0)
                duration = comp.get('duration', 5)
                
                if start_time <= t <= start_time + duration:
                    total += amplitude
        
        return total
    
    return Q_in
