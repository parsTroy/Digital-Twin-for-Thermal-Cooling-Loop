"""
Streamlit Dashboard for Thermal Cooling Loop Digital Twin

This module provides an interactive web dashboard for the thermal cooling loop
digital twin system, allowing users to explore different scenarios and
visualize results in real-time.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
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
from twin.plant_simulator import PlantSimulator, FaultType


def setup_page():
    """Set up the Streamlit page configuration."""
    st.set_page_config(
        page_title="Thermal Cooling Loop Digital Twin",
        page_icon="🌡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌡️ Thermal Cooling Loop Digital Twin")
    st.markdown("Real-time monitoring and analysis of thermal cooling systems")


def create_sidebar():
    """Create the sidebar with control parameters."""
    st.sidebar.header("System Parameters")
    
    # System parameters
    C = st.sidebar.slider(
        "Thermal Capacitance (J/K)",
        min_value=500.0,
        max_value=2000.0,
        value=1000.0,
        step=50.0,
        help="Thermal capacitance of the system"
    )
    
    m_dot = st.sidebar.slider(
        "Mass Flow Rate (kg/s)",
        min_value=0.05,
        max_value=0.2,
        value=0.1,
        step=0.01,
        help="Mass flow rate of the cooling fluid"
    )
    
    cp = st.sidebar.slider(
        "Specific Heat Capacity (J/kg·K)",
        min_value=3000.0,
        max_value=5000.0,
        value=4180.0,
        step=20.0,
        help="Specific heat capacity of the fluid"
    )
    
    UA = st.sidebar.slider(
        "Heat Exchanger Effectiveness (W/K)",
        min_value=20.0,
        max_value=100.0,
        value=50.0,
        step=5.0,
        help="Heat exchanger effectiveness"
    )
    
    Q_in = st.sidebar.slider(
        "Heat Input (W)",
        min_value=500.0,
        max_value=2000.0,
        value=1000.0,
        step=50.0,
        help="Heat input to the system"
    )
    
    Q_out = st.sidebar.slider(
        "Heat Output (W)",
        min_value=500.0,
        max_value=2000.0,
        value=1000.0,
        step=50.0,
        help="Heat output from the system"
    )
    
    # Simulation parameters
    st.sidebar.header("Simulation Parameters")
    
    simulation_time = st.sidebar.slider(
        "Simulation Time (s)",
        min_value=50,
        max_value=500,
        value=200,
        step=25,
        help="Total simulation time"
    )
    
    initial_temp_hot = st.sidebar.slider(
        "Initial Hot Temperature (K)",
        min_value=300.0,
        max_value=400.0,
        value=350.0,
        step=5.0,
        help="Initial temperature of hot fluid"
    )
    
    initial_temp_cold = st.sidebar.slider(
        "Initial Cold Temperature (K)",
        min_value=250.0,
        max_value=350.0,
        value=300.0,
        step=5.0,
        help="Initial temperature of cold fluid"
    )
    
    # Heat input scenario
    st.sidebar.header("Heat Input Scenario")
    
    heat_scenario = st.sidebar.selectbox(
        "Select Heat Input Type",
        ["Constant", "Time Varying", "Step", "Ramp", "Pulse", "Complex"],
        help="Choose the type of heat input to simulate"
    )
    
    # Heat input parameters based on scenario
    if heat_scenario == "Time Varying":
        amplitude = st.sidebar.slider("Amplitude (W)", 50.0, 500.0, 200.0, 25.0)
        frequency = st.sidebar.slider("Frequency (Hz)", 0.01, 0.5, 0.1, 0.01)
        Q_in_func = create_time_varying_heat_input(Q_in, amplitude, frequency)
    elif heat_scenario == "Step":
        step_power = st.sidebar.slider("Step Power (W)", 1000.0, 3000.0, 1500.0, 100.0)
        step_time = st.sidebar.slider("Step Time (s)", 10.0, simulation_time/2, 50.0, 10.0)
        Q_in_func = create_step_heat_input(Q_in, step_power, step_time)
    elif heat_scenario == "Ramp":
        final_power = st.sidebar.slider("Final Power (W)", 1000.0, 3000.0, 1500.0, 100.0)
        ramp_start = st.sidebar.slider("Ramp Start (s)", 10.0, simulation_time/2, 30.0, 10.0)
        ramp_duration = st.sidebar.slider("Ramp Duration (s)", 10.0, simulation_time/2, 40.0, 10.0)
        Q_in_func = create_ramp_heat_input(Q_in, final_power, ramp_start, ramp_duration)
    elif heat_scenario == "Pulse":
        pulse_power = st.sidebar.slider("Pulse Power (W)", 1000.0, 4000.0, 2000.0, 100.0)
        pulse_start = st.sidebar.slider("Pulse Start (s)", 10.0, simulation_time/2, 30.0, 10.0)
        pulse_duration = st.sidebar.slider("Pulse Duration (s)", 5.0, simulation_time/4, 20.0, 5.0)
        Q_in_func = create_pulse_heat_input(Q_in, pulse_power, pulse_start, pulse_duration)
    elif heat_scenario == "Complex":
        # Complex heat input with multiple components
        components = [
            {'type': 'sin', 'amplitude': 100, 'frequency': 0.05},
            {'type': 'step', 'amplitude': 200, 'start_time': 50},
            {'type': 'pulse', 'amplitude': 500, 'start_time': 100, 'duration': 20}
        ]
        Q_in_func = create_complex_heat_input(Q_in, components)
    else:  # Constant
        Q_in_func = Q_in
    
    return {
        'C': C,
        'm_dot': m_dot,
        'cp': cp,
        'UA': UA,
        'Q_in': Q_in_func,
        'Q_out': Q_out,
        'simulation_time': simulation_time,
        'initial_temp_hot': initial_temp_hot,
        'initial_temp_cold': initial_temp_cold,
        'heat_scenario': heat_scenario
    }


def run_simulation(params):
    """Run the thermal cooling loop simulation."""
    # Create twin model
    twin_params = {
        'C': params['C'],
        'm_dot': params['m_dot'],
        'cp': params['cp'],
        'UA': params['UA'],
        'Q_in': params['Q_in'],
        'Q_out': params['Q_out']
    }
    
    twin = ThermalCoolingTwin(twin_params)
    
    # Set up simulation
    t_span = (0, params['simulation_time'])
    y0 = np.array([params['initial_temp_hot'], params['initial_temp_cold']])
    t_eval = np.linspace(0, params['simulation_time'], 1000)
    
    # Run simulation
    results = twin.simulate(t_span, y0, t_eval=t_eval)
    
    return results, twin


def run_plant_simulation(params):
    """Run the plant simulator with sensor noise."""
    # Create plant simulator
    twin_params = {
        'C': params['C'],
        'm_dot': params['m_dot'],
        'cp': params['cp'],
        'UA': params['UA'],
        'Q_in': params['Q_in'],
        'Q_out': params['Q_out']
    }
    
    simulator = PlantSimulator(twin_params, noise_level=0.005, sample_rate=2.0)
    twin = ThermalCoolingTwin(twin_params)
    
    # Set up simulation
    t_span = (0, params['simulation_time'])
    y0 = np.array([params['initial_temp_hot'], params['initial_temp_cold']])
    
    # Run simulation
    df = simulator.simulate_sensor_data(twin, t_span, y0)
    
    return df, simulator


def create_temperature_plot(results):
    """Create temperature vs time plot."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results['t'],
        y=results['T_hot'],
        mode='lines',
        name='Hot Temperature',
        line=dict(color='red', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=results['t'],
        y=results['T_cold'],
        mode='lines',
        name='Cold Temperature',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title="Temperature vs Time",
        xaxis_title="Time (s)",
        yaxis_title="Temperature (K)",
        hovermode='x unified',
        height=400
    )
    
    return fig


def create_plant_simulator_plot(df):
    """Create plant simulator plot with true and noisy values."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Temperature Measurements", "Mass Flow Rate"),
        vertical_spacing=0.1
    )
    
    # Temperature plot
    fig.add_trace(go.Scatter(
        x=df['t'],
        y=df['T_hot_true'],
        mode='lines',
        name='True T_hot',
        line=dict(color='red', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['t'],
        y=df['T_hot'],
        mode='lines',
        name='Measured T_hot',
        line=dict(color='red', width=1, dash='dash')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['t'],
        y=df['T_cold_true'],
        mode='lines',
        name='True T_cold',
        line=dict(color='blue', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['t'],
        y=df['T_cold'],
        mode='lines',
        name='Measured T_cold',
        line=dict(color='blue', width=1, dash='dash')
    ), row=1, col=1)
    
    # Mass flow rate plot
    fig.add_trace(go.Scatter(
        x=df['t'],
        y=df['m_dot_true'],
        mode='lines',
        name='True m_dot',
        line=dict(color='green', width=2)
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['t'],
        y=df['m_dot'],
        mode='lines',
        name='Measured m_dot',
        line=dict(color='green', width=1, dash='dash')
    ), row=2, col=1)
    
    fig.update_layout(
        title="Plant Simulator - True vs Measured Values",
        height=600,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Temperature (K)", row=1, col=1)
    fig.update_yaxes(title_text="Mass Flow Rate (kg/s)", row=2, col=1)
    
    return fig


def create_heat_input_plot(params, results):
    """Create heat input vs time plot."""
    if callable(params['Q_in']):
        # Calculate heat input values
        t = results['t']
        Q_in_values = [params['Q_in'](ti) for ti in t]
    else:
        Q_in_values = [params['Q_in']] * len(results['t'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results['t'],
        y=Q_in_values,
        mode='lines',
        name='Heat Input',
        line=dict(color='orange', width=2)
    ))
    
    fig.update_layout(
        title=f"Heat Input - {params['heat_scenario']}",
        xaxis_title="Time (s)",
        yaxis_title="Heat Input (W)",
        height=300
    )
    
    return fig


def display_system_info(params, results, twin):
    """Display system information and metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Final Hot Temperature",
            f"{results['T_hot'][-1]:.1f} K",
            delta=f"{results['T_hot'][-1] - results['T_hot'][0]:.1f} K"
        )
    
    with col2:
        st.metric(
            "Final Cold Temperature",
            f"{results['T_cold'][-1]:.1f} K",
            delta=f"{results['T_cold'][-1] - results['T_cold'][0]:.1f} K"
        )
    
    with col3:
        # Calculate steady state
        T_hot_ss, T_cold_ss = twin.get_steady_state()
        st.metric(
            "Steady State Hot Temp",
            f"{T_hot_ss:.1f} K"
        )
    
    with col4:
        st.metric(
            "Steady State Cold Temp",
            f"{T_cold_ss:.1f} K"
        )


def display_plant_simulator_info(df):
    """Display plant simulator information."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Calculate noise statistics
        T_hot_noise = df['T_hot'] - df['T_hot_true']
        st.metric(
            "T_hot Noise Std",
            f"{T_hot_noise.std():.3f} K"
        )
    
    with col2:
        T_cold_noise = df['T_cold'] - df['T_cold_true']
        st.metric(
            "T_cold Noise Std",
            f"{T_cold_noise.std():.3f} K"
        )
    
    with col3:
        m_dot_noise = df['m_dot'] - df['m_dot_true']
        st.metric(
            "m_dot Noise Std",
            f"{m_dot_noise.std():.6f} kg/s"
        )


def main():
    """Main dashboard function."""
    setup_page()
    
    # Create sidebar
    params = create_sidebar()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Digital Twin", "Plant Simulator", "Analysis"])
    
    with tab1:
        st.header("Digital Twin Simulation")
        
        if st.button("Run Simulation", type="primary"):
            with st.spinner("Running simulation..."):
                results, twin = run_simulation(params)
                
                if results['success']:
                    st.success("Simulation completed successfully!")
                    
                    # Display system info
                    display_system_info(params, results, twin)
                    
                    # Create plots
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.plotly_chart(create_temperature_plot(results), use_container_width=True)
                    
                    with col2:
                        st.plotly_chart(create_heat_input_plot(params, results), use_container_width=True)
                    
                    # Display simulation details
                    st.subheader("Simulation Details")
                    st.write(f"**Method:** {results.get('method', 'RK45')}")
                    st.write(f"**Function Evaluations:** {results.get('nfev', 'N/A')}")
                    st.write(f"**Jacobian Evaluations:** {results.get('njev', 'N/A')}")
                    st.write(f"**Time Points:** {len(results['t'])}")
                    
                else:
                    st.error(f"Simulation failed: {results['message']}")
    
    with tab2:
        st.header("Plant Simulator with Sensor Noise")
        
        if st.button("Run Plant Simulation", type="primary"):
            with st.spinner("Running plant simulation..."):
                df, simulator = run_plant_simulation(params)
                
                st.success("Plant simulation completed successfully!")
                
                # Display plant simulator info
                display_plant_simulator_info(df)
                
                # Create plant simulator plot
                st.plotly_chart(create_plant_simulator_plot(df), use_container_width=True)
                
                # Display data summary
                st.subheader("Data Summary")
                st.write(f"**Data Points:** {len(df)}")
                st.write(f"**Sampling Rate:** {simulator.sample_rate} Hz")
                st.write(f"**Noise Level:** {simulator.noise_level}")
                
                # Show data table
                if st.checkbox("Show Data Table"):
                    st.dataframe(df.head(20))
    
    with tab3:
        st.header("System Analysis")
        
        st.subheader("Parameter Sensitivity")
        st.write("Adjust the parameters in the sidebar to see how they affect the system behavior.")
        
        st.subheader("Heat Input Scenarios")
        st.write("Try different heat input scenarios to understand system response:")
        st.write("- **Constant**: Steady heat input")
        st.write("- **Time Varying**: Sinusoidal variation")
        st.write("- **Step**: Sudden change in heat input")
        st.write("- **Ramp**: Gradual change in heat input")
        st.write("- **Pulse**: Temporary spike in heat input")
        st.write("- **Complex**: Combination of multiple effects")
        
        st.subheader("System Characteristics")
        st.write("The thermal cooling loop system is characterized by:")
        st.write("- **Thermal Capacitance (C)**: Determines thermal inertia")
        st.write("- **Mass Flow Rate (m_dot)**: Controls heat transfer rate")
        st.write("- **Heat Exchanger Effectiveness (UA)**: Determines cooling efficiency")
        st.write("- **Heat Input/Output**: Energy balance components")
        
        # Add some educational content
        st.subheader("Educational Notes")
        st.info("""
        **Phase 1 Implementation Features:**
        - ODE-based thermal cooling loop model
        - Multiple heat input scenarios
        - Realistic plant simulator with sensor noise
        - Interactive parameter adjustment
        - Real-time visualization
        - Comprehensive testing framework
        """)


if __name__ == "__main__":
    main()
