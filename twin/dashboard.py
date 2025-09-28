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
from twin.digital_twin import DigitalTwinManager, create_digital_twin_manager
from twin.detector import DetectionMethod, AnomalyType
from twin.enhanced_detector import EnhancedAnomalyDetector, create_enhanced_detector
from twin.ml_detector import MLAnomalyDetector, create_ml_detector


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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Digital Twin", "Plant Simulator", "Real-time Monitoring", "ML Detection", "Analysis"])
    
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
        st.header("Real-time Digital Twin Monitoring")
        
        # Detection method selection
        col1, col2 = st.columns(2)
        
        with col1:
            detection_methods = st.multiselect(
                "Detection Methods",
                ["Residual Threshold", "Rolling Z-Score", "Isolation Forest", "One-Class SVM", "SPC", "CUSUM"],
                default=["Residual Threshold", "Rolling Z-Score", "Isolation Forest"]
            )
        
        with col2:
            update_rate = st.slider("Update Rate (Hz)", 0.1, 5.0, 1.0, 0.1)
        
        # Convert detection methods
        method_map = {
            "Residual Threshold": DetectionMethod.RESIDUAL_THRESHOLD,
            "Rolling Z-Score": DetectionMethod.ROLLING_Z_SCORE,
            "Isolation Forest": DetectionMethod.ISOLATION_FOREST,
            "One-Class SVM": DetectionMethod.ONE_CLASS_SVM,
            "SPC": DetectionMethod.STATISTICAL_PROCESS_CONTROL,
            "CUSUM": DetectionMethod.CUSUM
        }
        
        selected_methods = [method_map[m] for m in detection_methods]
        
        # Initialize session state for digital twin
        if 'digital_twin_manager' not in st.session_state:
            st.session_state.digital_twin_manager = None
        if 'monitoring_data' not in st.session_state:
            st.session_state.monitoring_data = []
        if 'anomaly_count' not in st.session_state:
            st.session_state.anomaly_count = 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Start Real-time Monitoring", type="primary"):
                # Create plant simulator
                plant_params = {
                    'C': params['C'],
                    'm_dot': params['m_dot'],
                    'cp': params['cp'],
                    'UA': params['UA'],
                    'Q_in': params['Q_in'],
                    'Q_out': params['Q_out']
                }
                
                plant_simulator = PlantSimulator(plant_params, noise_level=0.005, sample_rate=update_rate)
                
                # Create digital twin manager
                twin_manager = DigitalTwinManager(
                    plant_simulator=plant_simulator,
                    twin_params=plant_params,
                    detection_methods=selected_methods,
                    update_rate=update_rate
                )
                
                # Set up callbacks
                def anomaly_callback(anomaly_results):
                    st.session_state.anomaly_count += 1
                    st.session_state.monitoring_data.append({
                        'time': anomaly_results['timestamp'],
                        'type': 'anomaly',
                        'severity': anomaly_results['overall_severity'].value,
                        'residuals': anomaly_results['residuals']
                    })
                
                def data_callback(data):
                    st.session_state.monitoring_data.append({
                        'time': data['timestamp'],
                        'type': 'data',
                        'plant_state': data['plant_state'],
                        'twin_state': data['twin_state'],
                        'residuals': data['residuals'],
                        'anomaly': data['anomaly_results']['overall_anomaly']
                    })
                
                twin_manager.add_anomaly_callback(anomaly_callback)
                twin_manager.add_data_callback(data_callback)
                
                # Start monitoring
                initial_conditions = np.array([params['initial_temp_hot'], params['initial_temp_cold']])
                twin_manager.start(initial_conditions)
                
                st.session_state.digital_twin_manager = twin_manager
                st.success("Real-time monitoring started!")
        
        with col2:
            if st.button("Stop Monitoring"):
                if st.session_state.digital_twin_manager:
                    st.session_state.digital_twin_manager.stop()
                    st.session_state.digital_twin_manager = None
                    st.success("Monitoring stopped!")
        
        with col3:
            if st.button("Clear Data"):
                st.session_state.monitoring_data = []
                st.session_state.anomaly_count = 0
                st.success("Data cleared!")
        
        # Display current status
        if st.session_state.digital_twin_manager:
            current_data = st.session_state.digital_twin_manager.get_current_data()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Plant T_hot", f"{current_data['plant_state'][0]:.1f} K")
            
            with col2:
                st.metric("Twin T_hot", f"{current_data['twin_state'][0]:.1f} K")
            
            with col3:
                residual = current_data['residuals'].get('T_hot', 0)
                st.metric("Residual", f"{residual:.2f} K")
            
            with col4:
                anomaly_status = "Yes" if current_data['anomaly_results']['overall_anomaly'] else "No"
                st.metric("Anomaly", anomaly_status)
            
            # Real-time plots
            if st.session_state.monitoring_data:
                df = pd.DataFrame(st.session_state.monitoring_data)
                data_df = df[df['type'] == 'data']
                
                if len(data_df) > 0:
                    # Temperature comparison plot
                    fig1 = go.Figure()
                    
                    fig1.add_trace(go.Scatter(
                        x=data_df['time'],
                        y=[state[0] for state in data_df['plant_state']],
                        mode='lines',
                        name='Plant T_hot',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig1.add_trace(go.Scatter(
                        x=data_df['time'],
                        y=[state[0] for state in data_df['twin_state']],
                        mode='lines',
                        name='Twin T_hot',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    fig1.update_layout(
                        title="Real-time Temperature Comparison",
                        xaxis_title="Time (s)",
                        yaxis_title="Temperature (K)",
                        height=300
                    )
                    
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Residuals plot
                    fig2 = go.Figure()
                    
                    residuals = [r.get('T_hot', 0) for r in data_df['residuals']]
                    fig2.add_trace(go.Scatter(
                        x=data_df['time'],
                        y=residuals,
                        mode='lines',
                        name='T_hot Residual',
                        line=dict(color='green', width=2)
                    ))
                    
                    fig2.add_hline(y=0, line_dash="dash", line_color="gray", alpha=0.5)
                    
                    fig2.update_layout(
                        title="Real-time Residuals",
                        xaxis_title="Time (s)",
                        yaxis_title="Residual (K)",
                        height=300
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Anomaly summary
            st.subheader("Anomaly Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Anomalies", st.session_state.anomaly_count)
            
            with col2:
                if st.session_state.monitoring_data:
                    anomaly_df = df[df['type'] == 'anomaly']
                    if len(anomaly_df) > 0:
                        severity_counts = anomaly_df['severity'].value_counts()
                        st.metric("Critical Anomalies", severity_counts.get('critical', 0))
                    else:
                        st.metric("Critical Anomalies", 0)
                else:
                    st.metric("Critical Anomalies", 0)
            
            with col3:
                if st.session_state.monitoring_data:
                    recent_anomalies = df[df['type'] == 'anomaly'].tail(5)
                    st.metric("Recent Anomalies", len(recent_anomalies))
                else:
                    st.metric("Recent Anomalies", 0)
            
            # Recent anomalies table
            if st.session_state.monitoring_data:
                anomaly_df = df[df['type'] == 'anomaly']
                if len(anomaly_df) > 0:
                    st.subheader("Recent Anomalies")
                    recent_anomalies = anomaly_df.tail(10)[['time', 'severity', 'residuals']]
                    st.dataframe(recent_anomalies, use_container_width=True)
        
        else:
            st.info("Click 'Start Real-time Monitoring' to begin monitoring the digital twin.")
    
    with tab4:
        st.header("Advanced ML Anomaly Detection")
        
        # ML Detection Configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Detection Configuration")
            
            # Traditional methods
            traditional_methods = st.multiselect(
                "Traditional Methods",
                ["Residual Threshold", "Rolling Z-Score", "Isolation Forest", "One-Class SVM", "SPC", "CUSUM"],
                default=["Residual Threshold", "Rolling Z-Score", "Isolation Forest"]
            )
            
            # ML features
            ml_enabled = st.checkbox("Enable ML Detection", value=True)
            feature_engineering = st.checkbox("Enable Feature Engineering", value=True)
            ensemble_voting = st.checkbox("Enable Ensemble Voting", value=True)
            adaptive_thresholds = st.checkbox("Enable Adaptive Thresholds", value=True)
            
        with col2:
            st.subheader("ML Model Configuration")
            
            # Model parameters
            auto_tuning = st.checkbox("Enable Auto-tuning", value=True)
            ensemble_methods = st.checkbox("Enable Ensemble Methods", value=True)
            model_persistence = st.checkbox("Enable Model Persistence", value=True)
            
            # Training parameters
            min_training_samples = st.slider("Min Training Samples", 50, 500, 100)
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
            cv_folds = st.slider("CV Folds", 3, 10, 5)
        
        # Convert detection methods
        method_map = {
            "Residual Threshold": DetectionMethod.RESIDUAL_THRESHOLD,
            "Rolling Z-Score": DetectionMethod.ROLLING_Z_SCORE,
            "Isolation Forest": DetectionMethod.ISOLATION_FOREST,
            "One-Class SVM": DetectionMethod.ONE_CLASS_SVM,
            "SPC": DetectionMethod.STATISTICAL_PROCESS_CONTROL,
            "CUSUM": DetectionMethod.CUSUM
        }
        
        selected_traditional_methods = [method_map[m] for m in traditional_methods]
        
        # Initialize session state for ML detector
        if 'ml_detector' not in st.session_state:
            st.session_state.ml_detector = None
        if 'enhanced_detector' not in st.session_state:
            st.session_state.enhanced_detector = None
        if 'ml_training_data' not in st.session_state:
            st.session_state.ml_training_data = []
        if 'ml_anomaly_count' not in st.session_state:
            st.session_state.ml_anomaly_count = 0
        
        # ML Detector Controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Create ML Detector", type="primary"):
                # Create enhanced detector
                enhanced_detector = create_enhanced_detector(
                    traditional_methods=selected_traditional_methods,
                    ml_enabled=ml_enabled,
                    adaptive_thresholds=adaptive_thresholds,
                    ensemble_voting=ensemble_voting
                )
                
                # Create ML detector
                ml_detector = create_ml_detector(
                    feature_engineering=feature_engineering,
                    auto_tuning=auto_tuning,
                    ensemble_methods=ensemble_methods
                )
                
                st.session_state.enhanced_detector = enhanced_detector
                st.session_state.ml_detector = ml_detector
                st.success("ML detectors created successfully!")
        
        with col2:
            if st.button("Generate Training Data"):
                if st.session_state.ml_detector:
                    # Generate synthetic training data
                    np.random.seed(42)
                    training_data = []
                    
                    # Normal data
                    for _ in range(100):
                        training_data.append({
                            'T_hot': np.random.normal(0, 1.0),
                            'T_cold': np.random.normal(0, 0.8),
                            'm_dot': np.random.normal(0, 0.1)
                        })
                    
                    # Anomalous data
                    for _ in range(25):
                        training_data.append({
                            'T_hot': np.random.normal(0, 4.0),
                            'T_cold': np.random.normal(0, 3.0),
                            'm_dot': np.random.normal(0, 0.3)
                        })
                    
                    st.session_state.ml_training_data = training_data
                    st.success(f"Generated {len(training_data)} training samples!")
                else:
                    st.error("Please create ML detector first!")
        
        with col3:
            if st.button("Train Models"):
                if st.session_state.ml_detector and st.session_state.ml_training_data:
                    with st.spinner("Training ML models..."):
                        # Train enhanced detector
                        enhanced_results = st.session_state.enhanced_detector.train_ml_models(st.session_state.ml_training_data)
                        
                        # Train ML detector
                        ml_results = st.session_state.ml_detector.train_models()
                        
                        st.session_state.ml_training_results = {
                            'enhanced': enhanced_results,
                            'ml': ml_results
                        }
                        st.success("Models trained successfully!")
                else:
                    st.error("Please generate training data first!")
        
        with col4:
            if st.button("Reset Detectors"):
                st.session_state.ml_detector = None
                st.session_state.enhanced_detector = None
                st.session_state.ml_training_data = []
                st.session_state.ml_anomaly_count = 0
                st.success("Detectors reset!")
        
        # Display detector status
        if st.session_state.enhanced_detector:
            st.subheader("Detector Status")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Enhanced Detector", "Ready" if st.session_state.enhanced_detector.is_trained else "Not Trained")
            
            with col2:
                if st.session_state.ml_detector:
                    st.metric("ML Detector", "Ready" if st.session_state.ml_detector.is_trained else "Not Trained")
                else:
                    st.metric("ML Detector", "Not Created")
            
            with col3:
                st.metric("Training Data", f"{len(st.session_state.ml_training_data)} samples")
        
        # ML Detection Testing
        if st.session_state.enhanced_detector and st.session_state.enhanced_detector.is_trained:
            st.subheader("ML Detection Testing")
            
            # Test scenarios
            test_scenarios = {
                "Normal Operation": {'T_hot': 0.5, 'T_cold': 0.3, 'm_dot': 0.05},
                "Temperature Anomaly": {'T_hot': 8.0, 'T_cold': 6.0, 'm_dot': 0.1},
                "Flow Rate Anomaly": {'T_hot': 1.0, 'T_cold': 0.8, 'm_dot': 0.8},
                "Multiple Anomalies": {'T_hot': 10.0, 'T_cold': 8.0, 'm_dot': 0.9}
            }
            
            selected_scenario = st.selectbox("Select Test Scenario", list(test_scenarios.keys()))
            
            if st.button("Test Detection"):
                test_residuals = test_scenarios[selected_scenario]
                
                # Test with enhanced detector
                result = st.session_state.enhanced_detector.detect_anomalies(test_residuals, use_ensemble=True)
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Anomaly Detected", "Yes" if result['final_decision']['is_anomaly'] else "No")
                
                with col2:
                    st.metric("Confidence", f"{result['final_decision']['confidence']:.3f}")
                
                with col3:
                    st.metric("Severity", result['final_decision']['severity'].value)
                
                with col4:
                    st.metric("Method Used", result['final_decision']['method_used'])
                
                # Show detailed results
                with st.expander("Detailed Results"):
                    st.json({
                        "residuals": test_residuals,
                        "traditional_results": result.get('traditional_results', {}),
                        "ml_results": result.get('ml_results', {}),
                        "ensemble_result": result.get('ensemble_result', {}),
                        "final_decision": result['final_decision']
                    })
        
        # Model Performance Analysis
        if 'ml_training_results' in st.session_state:
            st.subheader("Model Performance Analysis")
            
            results = st.session_state.ml_training_results
            
            # Enhanced detector results
            if 'enhanced' in results:
                st.write("**Enhanced Detector Results:**")
                enhanced_results = results['enhanced']
                
                if isinstance(enhanced_results, dict) and 'error' not in enhanced_results:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Traditional Methods:**")
                        for method, perf in enhanced_results.items():
                            if isinstance(perf, dict) and 'test_accuracy' in perf:
                                st.write(f"- {method}: {perf['test_accuracy']:.3f}")
                    
                    with col2:
                        st.write("**ML Methods:**")
                        if 'ml_performance' in enhanced_results:
                            ml_perf = enhanced_results['ml_performance']
                            for method, perf in ml_perf.items():
                                if isinstance(perf, dict) and 'test_accuracy' in perf:
                                    st.write(f"- {method}: {perf['test_accuracy']:.3f}")
            
            # ML detector results
            if 'ml' in results:
                st.write("**ML Detector Results:**")
                ml_results = results['ml']
                
                if isinstance(ml_results, dict):
                    for method, perf in ml_results.items():
                        if isinstance(perf, dict) and 'test_accuracy' in perf:
                            st.write(f"- {method}: {perf['test_accuracy']:.3f}")
        
        # Feature Importance Analysis
        if st.session_state.ml_detector and st.session_state.ml_detector.is_trained:
            st.subheader("Feature Importance Analysis")
            
            if st.button("Show Feature Importance"):
                importance = st.session_state.ml_detector.get_feature_importance('random_forest')
                
                if importance:
                    # Create feature importance plot
                    features = list(importance.keys())[:10]  # Top 10 features
                    importances = list(importance.values())[:10]
                    
                    fig = go.Figure(data=[
                        go.Bar(x=features, y=importances, name='Feature Importance')
                    ])
                    
                    fig.update_layout(
                        title="Top 10 Most Important Features",
                        xaxis_title="Features",
                        yaxis_title="Importance",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show feature importance table
                    importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
                    st.dataframe(importance_df.head(20), use_container_width=True)
                else:
                    st.warning("Feature importance not available for the selected model.")
        
        # Real-time ML Monitoring
        if st.session_state.enhanced_detector and st.session_state.enhanced_detector.is_trained:
            st.subheader("Real-time ML Monitoring")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Start ML Monitoring"):
                    st.session_state.ml_monitoring_active = True
                    st.success("ML monitoring started!")
            
            with col2:
                if st.button("Stop ML Monitoring"):
                    st.session_state.ml_monitoring_active = False
                    st.success("ML monitoring stopped!")
            
            if st.session_state.get('ml_monitoring_active', False):
                # Simulate real-time monitoring
                if st.button("Simulate Data Point"):
                    # Generate random residuals
                    np.random.seed(int(time.time()))
                    residuals = {
                        'T_hot': np.random.normal(0, 2.0),
                        'T_cold': np.random.normal(0, 1.5),
                        'm_dot': np.random.normal(0, 0.2)
                    }
                    
                    # Detect anomalies
                    result = st.session_state.enhanced_detector.detect_anomalies(residuals, use_ensemble=True)
                    
                    # Update counter
                    if result['final_decision']['is_anomaly']:
                        st.session_state.ml_anomaly_count += 1
                    
                    # Display current status
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Residuals", f"T_hot: {residuals['T_hot']:.2f}")
                    
                    with col2:
                        st.metric("Anomaly Detected", "Yes" if result['final_decision']['is_anomaly'] else "No")
                    
                    with col3:
                        st.metric("Total Anomalies", st.session_state.ml_anomaly_count)
                    
                    # Show detection details
                    with st.expander("Detection Details"):
                        st.json(result['final_decision'])
    
    with tab5:
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
