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
                 atol: float = 1e-9) -> Dict[str, np.ndarray]:
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
            
        Returns:
        --------
        dict
            Simulation results with keys:
            - 't': time array [s]
            - 'T_hot': hot temperature [K]
            - 'T_cold': cold temperature [K]
            - 'success': solver success flag
            - 'message': solver message
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
                method='RK45'
            )
            
            if not sol.success:
                logger.warning(f"ODE solver failed: {sol.message}")
            
            # Extract results
            results = {
                't': sol.t,
                'T_hot': sol.y[0],
                'T_cold': sol.y[1],
                'success': sol.success,
                'message': sol.message
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
