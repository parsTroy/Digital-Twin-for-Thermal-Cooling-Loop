"""
Phase 2 Demo: Digital Twin & Residual Generation

This script demonstrates the digital twin running in lockstep with the plant
simulator, residual computation, and real-time anomaly detection.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys
import os

# Add the parent directory to the path to import the twin module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from twin.model import create_default_parameters, create_step_heat_input
from twin.plant_simulator import PlantSimulator, FaultType
from twin.digital_twin import DigitalTwinManager, create_digital_twin_manager
from twin.detector import DetectionMethod, AnomalyType


def demo_basic_digital_twin():
    """Demonstrate basic digital twin functionality."""
    print("=== Basic Digital Twin Demo ===")
    
    # Create plant simulator and digital twin manager
    plant_params = create_default_parameters()
    plant_simulator = PlantSimulator(plant_params, noise_level=0.01, sample_rate=2.0)
    
    twin_manager = create_digital_twin_manager(
        plant_simulator=plant_simulator,
        detection_methods=[
            DetectionMethod.RESIDUAL_THRESHOLD,
            DetectionMethod.ROLLING_Z_SCORE,
            DetectionMethod.ISOLATION_FOREST
        ]
    )
    
    # Set up callbacks for monitoring
    anomaly_count = 0
    
    def anomaly_callback(anomaly_results):
        nonlocal anomaly_count
        anomaly_count += 1
        print(f"Anomaly detected! Severity: {anomaly_results['overall_severity'].value}")
        print(f"  Residuals: {anomaly_results['residuals']}")
        print(f"  Anomaly scores: {anomaly_results['anomaly_scores']}")
    
    def data_callback(data):
        if len(data['residuals']) > 0:
            print(f"Time: {data['timestamp']:.1f}s, "
                  f"Plant: T_hot={data['plant_state'][0]:.1f}K, "
                  f"Twin: T_hot={data['twin_state'][0]:.1f}K, "
                  f"Residual: {data['residuals']['T_hot']:.2f}K")
    
    twin_manager.add_anomaly_callback(anomaly_callback)
    twin_manager.add_data_callback(data_callback)
    
    # Start digital twin
    initial_conditions = np.array([350.0, 300.0])
    print(f"Starting digital twin with initial conditions: {initial_conditions}")
    
    twin_manager.start(initial_conditions)
    
    # Let it run for a while
    print("Running digital twin for 10 seconds...")
    time.sleep(10)
    
    # Get current data
    current_data = twin_manager.get_current_data()
    print(f"\nCurrent state:")
    print(f"  Plant: T_hot={current_data['plant_state'][0]:.1f}K, T_cold={current_data['plant_state'][1]:.1f}K")
    print(f"  Twin:  T_hot={current_data['twin_state'][0]:.1f}K, T_cold={current_data['twin_state'][1]:.1f}K")
    print(f"  Residuals: {current_data['residuals']}")
    
    # Get residual statistics
    residual_stats = twin_manager.get_residual_statistics()
    print(f"\nResidual statistics:")
    for sensor, stats in residual_stats.items():
        print(f"  {sensor}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, rms={stats['rms']:.3f}")
    
    # Get anomaly summary
    anomaly_summary = twin_manager.get_anomaly_summary()
    print(f"\nAnomaly summary:")
    print(f"  Total anomalies: {anomaly_summary['total_anomalies']}")
    print(f"  Anomaly types: {anomaly_summary['anomaly_types']}")
    print(f"  Anomaly rate: {anomaly_summary['anomaly_rate']:.3f}")
    
    # Stop digital twin
    twin_manager.stop()
    print(f"\nDigital twin stopped. Total anomalies detected: {anomaly_count}")
    
    return twin_manager


def demo_fault_detection():
    """Demonstrate fault detection capabilities."""
    print("\n=== Fault Detection Demo ===")
    
    # Create plant simulator with fault injection
    plant_params = create_default_parameters()
    plant_simulator = PlantSimulator(plant_params, noise_level=0.005, sample_rate=2.0)
    
    twin_manager = create_digital_twin_manager(
        plant_simulator=plant_simulator,
        detection_methods=[
            DetectionMethod.RESIDUAL_THRESHOLD,
            DetectionMethod.ROLLING_Z_SCORE,
            DetectionMethod.ISOLATION_FOREST
        ]
    )
    
    # Set up monitoring
    anomalies_detected = []
    
    def anomaly_callback(anomaly_results):
        anomalies_detected.append({
            'time': anomaly_results['timestamp'],
            'severity': anomaly_results['overall_severity'].value,
            'residuals': anomaly_results['residuals'].copy(),
            'scores': anomaly_results['anomaly_scores'].copy()
        })
        print(f"FAULT DETECTED at t={anomaly_results['timestamp']:.1f}s: "
              f"{anomaly_results['overall_severity'].value}")
    
    twin_manager.add_anomaly_callback(anomaly_callback)
    
    # Start digital twin
    initial_conditions = np.array([350.0, 300.0])
    twin_manager.start(initial_conditions)
    
    print("Running normal operation for 5 seconds...")
    time.sleep(5)
    
    # Inject pump degradation fault
    print("Injecting pump degradation fault...")
    twin_manager.inject_plant_fault(
        FaultType.PUMP_DEGRADATION,
        start_time=twin_manager.current_time + 2.0,
        parameters={'degradation_rate': 0.05}
    )
    
    print("Running with fault for 10 seconds...")
    time.sleep(10)
    
    # Inject sensor bias fault
    print("Injecting sensor bias fault...")
    twin_manager.inject_plant_fault(
        FaultType.SENSOR_BIAS,
        start_time=twin_manager.current_time + 2.0,
        parameters={'bias_magnitude': 8.0}
    )
    
    print("Running with multiple faults for 8 seconds...")
    time.sleep(8)
    
    # Stop and analyze
    twin_manager.stop()
    
    print(f"\nFault detection results:")
    print(f"  Total anomalies detected: {len(anomalies_detected)}")
    
    if anomalies_detected:
        print(f"  First anomaly at: {anomalies_detected[0]['time']:.1f}s")
        print(f"  Last anomaly at: {anomalies_detected[-1]['time']:.1f}s")
        
        # Analyze by severity
        severity_counts = {}
        for anomaly in anomalies_detected:
            severity = anomaly['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        print(f"  Anomalies by severity: {severity_counts}")
    
    return twin_manager, anomalies_detected


def demo_detection_methods():
    """Demonstrate different anomaly detection methods."""
    print("\n=== Detection Methods Demo ===")
    
    # Create plant simulator
    plant_params = create_default_parameters()
    plant_simulator = PlantSimulator(plant_params, noise_level=0.01, sample_rate=2.0)
    
    # Test different detection method combinations
    detection_configs = [
        {
            'name': 'Threshold Only',
            'methods': [DetectionMethod.RESIDUAL_THRESHOLD]
        },
        {
            'name': 'Statistical Methods',
            'methods': [DetectionMethod.RESIDUAL_THRESHOLD, DetectionMethod.ROLLING_Z_SCORE]
        },
        {
            'name': 'ML Methods',
            'methods': [DetectionMethod.ISOLATION_FOREST, DetectionMethod.ONE_CLASS_SVM]
        },
        {
            'name': 'All Methods',
            'methods': [
                DetectionMethod.RESIDUAL_THRESHOLD,
                DetectionMethod.ROLLING_Z_SCORE,
                DetectionMethod.ISOLATION_FOREST,
                DetectionMethod.ONE_CLASS_SVM,
                DetectionMethod.STATISTICAL_PROCESS_CONTROL,
                DetectionMethod.CUSUM
            ]
        }
    ]
    
    results = {}
    
    for config in detection_configs:
        print(f"\nTesting {config['name']}...")
        
        # Create twin manager with specific methods
        twin_manager = DigitalTwinManager(
            plant_simulator=plant_simulator,
            detection_methods=config['methods']
        )
        
        # Train detector with some data first
        training_data = []
        for _ in range(50):
            training_data.append({
                'T_hot': np.random.normal(0, 1.0),
                'T_cold': np.random.normal(0, 0.8),
                'm_dot': np.random.normal(0, 0.1)
            })
        twin_manager.train_detector(training_data)
        
        # Set up monitoring
        anomalies_detected = []
        
        def anomaly_callback(anomaly_results):
            anomalies_detected.append(anomaly_results)
        
        twin_manager.add_anomaly_callback(anomaly_callback)
        
        # Run simulation
        initial_conditions = np.array([350.0, 300.0])
        twin_manager.start(initial_conditions)
        
        # Inject fault after 2 seconds
        time.sleep(2)
        twin_manager.inject_plant_fault(
            FaultType.PUMP_DEGRADATION,
            start_time=twin_manager.current_time + 1.0,
            parameters={'degradation_rate': 0.1}
        )
        
        # Run for 8 more seconds
        time.sleep(8)
        twin_manager.stop()
        
        # Analyze results
        anomaly_summary = twin_manager.get_anomaly_summary()
        results[config['name']] = {
            'total_anomalies': anomaly_summary['total_anomalies'],
            'anomaly_rate': anomaly_summary['anomaly_rate'],
            'methods_used': len(config['methods'])
        }
        
        print(f"  Anomalies detected: {anomaly_summary['total_anomalies']}")
        print(f"  Anomaly rate: {anomaly_summary['anomaly_rate']:.3f}")
    
    # Compare results
    print(f"\nDetection method comparison:")
    for name, result in results.items():
        print(f"  {name}: {result['total_anomalies']} anomalies, "
              f"rate={result['anomaly_rate']:.3f}, methods={result['methods_used']}")
    
    return results


def demo_real_time_monitoring():
    """Demonstrate real-time monitoring capabilities."""
    print("\n=== Real-time Monitoring Demo ===")
    
    # Create plant simulator with time-varying heat input
    plant_params = create_default_parameters()
    plant_params['Q_in'] = create_step_heat_input(1000, 1500, 10)  # Step at 10s
    plant_simulator = PlantSimulator(plant_params, noise_level=0.005, sample_rate=1.0)
    
    twin_manager = create_digital_twin_manager(
        plant_simulator=plant_simulator,
        detection_methods=[
            DetectionMethod.RESIDUAL_THRESHOLD,
            DetectionMethod.ROLLING_Z_SCORE
        ]
    )
    
    # Set up real-time monitoring
    monitoring_data = []
    
    def data_callback(data):
        monitoring_data.append({
            'time': data['timestamp'],
            'plant_T_hot': data['plant_state'][0],
            'twin_T_hot': data['twin_state'][0],
            'residual_T_hot': data['residuals']['T_hot'],
            'anomaly': data['anomaly_results']['overall_anomaly']
        })
        
        # Print every 2 seconds
        if int(data['timestamp']) % 2 == 0 and data['timestamp'] > 0:
            print(f"t={data['timestamp']:.1f}s: "
                  f"Plant={data['plant_state'][0]:.1f}K, "
                  f"Twin={data['twin_state'][0]:.1f}K, "
                  f"Residual={data['residuals']['T_hot']:.2f}K, "
                  f"Anomaly={data['anomaly_results']['overall_anomaly']}")
    
    twin_manager.add_data_callback(data_callback)
    
    # Start monitoring
    initial_conditions = np.array([350.0, 300.0])
    twin_manager.start(initial_conditions)
    
    print("Real-time monitoring for 20 seconds...")
    print("Time(s)  Plant(K)  Twin(K)   Residual(K)  Anomaly")
    print("-" * 50)
    
    time.sleep(20)
    twin_manager.stop()
    
    # Analyze monitoring data
    df = pd.DataFrame(monitoring_data)
    
    print(f"\nMonitoring summary:")
    print(f"  Data points collected: {len(df)}")
    print(f"  Time range: {df['time'].min():.1f}s - {df['time'].max():.1f}s")
    print(f"  Temperature range: {df['plant_T_hot'].min():.1f}K - {df['plant_T_hot'].max():.1f}K")
    print(f"  Residual range: {df['residual_T_hot'].min():.3f}K - {df['residual_T_hot'].max():.3f}K")
    print(f"  Anomalies detected: {df['anomaly'].sum()}")
    
    return df


def create_plots(twin_manager, anomalies_detected=None, monitoring_data=None):
    """Create visualization plots for the demo results."""
    print("\n=== Creating Plots ===")
    
    # Get historical data
    history = twin_manager.get_history()
    
    if not history:
        print("No historical data available for plotting")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(history)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Digital Twin Phase 2 Demo Results', fontsize=16)
    
    # Plot 1: Plant vs Twin temperatures
    ax1 = axes[0, 0]
    if 'plant_state' in df.columns and 'twin_state' in df.columns:
        plant_t_hot = [state[0] for state in df['plant_state'] if state is not None]
        twin_t_hot = [state[0] for state in df['twin_state'] if state is not None]
        times = df['timestamp'].values[:len(plant_t_hot)]
        
        ax1.plot(times, plant_t_hot, 'b-', label='Plant T_hot', linewidth=2)
        ax1.plot(times, twin_t_hot, 'r--', label='Twin T_hot', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Temperature (K)')
        ax1.set_title('Plant vs Twin Temperatures')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    ax2 = axes[0, 1]
    if 'residuals' in df.columns:
        residuals = df['residuals'].apply(lambda x: x.get('T_hot', 0) if x else 0)
        times = df['timestamp'].values
        
        ax2.plot(times, residuals, 'g-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Residual (K)')
        ax2.set_title('Temperature Residuals')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Anomaly detection
    ax3 = axes[1, 0]
    if 'anomaly_results' in df.columns:
        anomalies = df['anomaly_results'].apply(lambda x: x.get('overall_anomaly', False) if x else False)
        times = df['timestamp'].values
        
        # Plot anomaly indicators
        anomaly_times = times[anomalies]
        if len(anomaly_times) > 0:
            ax3.scatter(anomaly_times, [1] * len(anomaly_times), c='red', s=50, alpha=0.7, label='Anomalies')
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Anomaly')
        ax3.set_title('Anomaly Detection')
        ax3.set_ylim(-0.1, 1.5)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Residual statistics
    ax4 = axes[1, 1]
    if 'residuals' in df.columns:
        residuals = df['residuals'].apply(lambda x: x.get('T_hot', 0) if x else 0)
        
        # Calculate rolling statistics
        window_size = min(50, len(residuals) // 4)
        if window_size > 1:
            rolling_mean = residuals.rolling(window=window_size).mean()
            rolling_std = residuals.rolling(window=window_size).std()
            times = df['timestamp'].values
            
            ax4.plot(times, rolling_mean, 'b-', label=f'Rolling Mean (w={window_size})', linewidth=2)
            ax4.fill_between(times, 
                           rolling_mean - 2*rolling_std, 
                           rolling_mean + 2*rolling_std, 
                           alpha=0.3, label='±2σ')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Residual (K)')
            ax4.set_title('Residual Statistics')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phase2_demo_results.png', dpi=300, bbox_inches='tight')
    print("Plots saved as 'phase2_demo_results.png'")
    plt.show()


def main():
    """Main demo function."""
    print("Thermal Cooling Loop Digital Twin - Phase 2 Demo")
    print("=" * 60)
    
    # Run all demos
    print("Running Phase 2 demonstrations...")
    
    # Basic digital twin demo
    twin_manager = demo_basic_digital_twin()
    
    # Fault detection demo
    fault_manager, anomalies = demo_fault_detection()
    
    # Detection methods demo
    detection_results = demo_detection_methods()
    
    # Real-time monitoring demo
    monitoring_data = demo_real_time_monitoring()
    
    # Create plots
    create_plots(twin_manager, anomalies, monitoring_data)
    
    print("\n=== Phase 2 Demo Complete ===")
    print("Key Features Demonstrated:")
    print("- Digital twin running in lockstep with plant simulator")
    print("- Real-time residual computation and analysis")
    print("- Multiple anomaly detection algorithms")
    print("- Fault injection and detection capabilities")
    print("- Real-time monitoring and callbacks")
    print("- Comprehensive statistical analysis")
    print("- Export and visualization capabilities")


if __name__ == '__main__':
    main()
