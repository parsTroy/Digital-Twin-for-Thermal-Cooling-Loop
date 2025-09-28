"""
Phase 4 Demo: Complete Dashboard & Web Client

This script demonstrates the complete digital twin system with advanced
dashboard features, web client integration, and comprehensive analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys
import os
import subprocess
import webbrowser
from datetime import datetime
import threading
import json

# Add the parent directory to the path to import the twin module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from twin.model import create_default_parameters, create_step_heat_input
from twin.plant_simulator import PlantSimulator, FaultType
from twin.digital_twin import DigitalTwinManager, create_digital_twin_manager
from twin.enhanced_detector import EnhancedAnomalyDetector, create_enhanced_detector
from twin.detector import DetectionMethod, AnomalyType


def demo_complete_system():
    """Demonstrate the complete digital twin system."""
    print("=== Phase 4 Demo: Complete Dashboard & Web Client ===")
    print("Advanced thermal cooling loop digital twin with ML detection")
    print("=" * 60)
    
    # 1. System Setup
    print("\n1. Setting up complete digital twin system...")
    
    # Create system parameters
    params = create_default_parameters()
    params['C'] = 41800.0  # J/K
    params['m_dot'] = 0.1  # kg/s
    params['cp'] = 4180.0  # J/kg.K
    params['UA'] = 50.0    # W/K
    params['Q_out'] = 0.0  # No heat loss from cold side
    
    # Create heat input function
    q_in_func = create_step_heat_input(1000, 1500, 30)  # Step at 30s
    params['Q_in'] = q_in_func
    
    # Initial conditions
    initial_conditions = np.array([350.0, 300.0])  # T_hot, T_cold in K
    
    print(f"   System parameters configured")
    print(f"   Initial conditions: T_hot={initial_conditions[0]}K, T_cold={initial_conditions[1]}K")
    
    # 2. Create Enhanced Detection System
    print("\n2. Creating enhanced detection system...")
    
    # Create enhanced detector with all methods
    enhanced_detector = create_enhanced_detector(
        traditional_methods=[
            DetectionMethod.RESIDUAL_THRESHOLD,
            DetectionMethod.ROLLING_Z_SCORE,
            DetectionMethod.ISOLATION_FOREST,
            DetectionMethod.ONE_CLASS_SVM,
            DetectionMethod.STATISTICAL_PROCESS_CONTROL,
            DetectionMethod.CUSUM
        ],
        ml_enabled=True,
        adaptive_thresholds=True,
        ensemble_voting=True
    )
    
    print("   Enhanced detector created with 6 traditional methods + ML")
    
    # 3. Generate Training Data
    print("\n3. Generating training data for ML models...")
    
    # Generate diverse training data
    training_data = []
    training_labels = []
    
    # Normal operation data
    np.random.seed(42)
    for _ in range(200):
        normal_residuals = {
            'T_hot': np.random.normal(0, 1.0),
            'T_cold': np.random.normal(0, 0.8),
            'm_dot': np.random.normal(0, 0.1)
        }
        training_data.append(normal_residuals)
        training_labels.append(False)
    
    # Various fault conditions
    fault_types = [
        {'T_hot': np.random.normal(0, 4.0), 'T_cold': np.random.normal(0, 3.0), 'm_dot': np.random.normal(0, 0.3)},
        {'T_hot': np.random.normal(0, 6.0), 'T_cold': np.random.normal(0, 2.0), 'm_dot': np.random.normal(0, 0.1)},
        {'T_hot': np.random.normal(0, 2.0), 'T_cold': np.random.normal(0, 5.0), 'm_dot': np.random.normal(0, 0.4)},
        {'T_hot': np.random.normal(0, 8.0), 'T_cold': np.random.normal(0, 6.0), 'm_dot': np.random.normal(0, 0.5)}
    ]
    
    for fault_data in fault_types:
        for _ in range(25):
            training_data.append(fault_data)
            training_labels.append(True)
    
    print(f"   Generated {len(training_data)} training samples")
    print(f"   Normal samples: {sum(1 for label in training_labels if not label)}")
    print(f"   Anomalous samples: {sum(1 for label in training_labels if label)}")
    
    # 4. Train ML Models
    print("\n4. Training ML models...")
    
    start_time = time.time()
    training_results = enhanced_detector.train_ml_models(training_data)
    training_time = time.time() - start_time
    
    print(f"   Training completed in {training_time:.2f} seconds")
    
    # Display training results
    if isinstance(training_results, dict) and 'error' not in training_results:
        print("   Training Results:")
        for method, results in training_results.items():
            if isinstance(results, dict) and 'test_accuracy' in results:
                print(f"     {method}: {results['test_accuracy']:.3f} accuracy")
    
    # 5. Create Digital Twin Manager
    print("\n5. Creating digital twin manager...")
    
    # Create plant simulator with faults
    plant_simulator = PlantSimulator(params, noise_level=0.01, sample_rate=2.0)
    
    # Inject multiple faults for demonstration
    plant_simulator.inject_fault(FaultType.PUMP_DEGRADATION, 20.0, {'degradation_rate': 0.1})
    plant_simulator.inject_fault(FaultType.SENSOR_BIAS, 40.0, {'bias_magnitude': 5.0})
    plant_simulator.inject_fault(FaultType.HEAT_EXCHANGER_FOULING, 60.0, {'fouling_factor': 0.3})
    
    # Create digital twin manager
    twin_manager = create_digital_twin_manager(
        plant_simulator=plant_simulator,
        detection_methods=[
            DetectionMethod.RESIDUAL_THRESHOLD,
            DetectionMethod.ROLLING_Z_SCORE,
            DetectionMethod.ISOLATION_FOREST
        ]
    )
    
    # Replace with enhanced detector
    twin_manager.detector = enhanced_detector
    
    print("   Digital twin manager created with enhanced detection")
    print(f"   Active faults: {len(plant_simulator.active_faults)}")
    
    # 6. Run Comprehensive Simulation
    print("\n6. Running comprehensive simulation...")
    
    simulation_data = []
    anomaly_events = []
    
    def data_callback(data):
        simulation_data.append({
            'timestamp': data['timestamp'],
            'plant_t_hot': data['plant_state'][0],
            'twin_t_hot': data['twin_state'][0],
            'residual_t_hot': data['residuals']['T_hot'],
            'anomaly': data['anomaly_results']['overall_anomaly'],
            'confidence': data['anomaly_results'].get('overall_confidence', 0.0),
            'severity': data['anomaly_results']['overall_severity'].name
        })
        
        if data['anomaly_results']['overall_anomaly']:
            anomaly_events.append({
                'timestamp': data['timestamp'],
                'severity': data['anomaly_results']['overall_severity'].name,
                'confidence': data['anomaly_results'].get('overall_confidence', 0.0),
                'residual': data['residuals']['T_hot']
            })
            print(f"   Anomaly detected at t={data['timestamp']:.1f}s: "
                  f"Severity={data['anomaly_results']['overall_severity'].name}, "
                  f"Confidence={data['anomaly_results'].get('overall_confidence', 0.0):.3f}")
    
    twin_manager.add_data_callback(data_callback)
    
    # Start simulation
    twin_manager.start(initial_conditions)
    
    # Run for 80 seconds to capture all faults
    print("   Running simulation for 80 seconds...")
    time.sleep(80)
    
    twin_manager.stop()
    
    # 7. Analyze Results
    print("\n7. Analyzing simulation results...")
    
    df = pd.DataFrame(simulation_data)
    
    print(f"   Simulation duration: {df['timestamp'].max():.1f} seconds")
    print(f"   Data points collected: {len(df)}")
    print(f"   Anomalies detected: {len(anomaly_events)}")
    print(f"   Anomaly rate: {len(anomaly_events) / len(df):.1%}")
    
    # Temperature analysis
    print(f"   Temperature range: {df['plant_t_hot'].min():.1f}K - {df['plant_t_hot'].max():.1f}K")
    print(f"   Max residual: {df['residual_t_hot'].abs().max():.2f}K")
    print(f"   Average confidence: {df['confidence'].mean():.3f}")
    
    # Severity analysis
    severity_counts = {}
    for event in anomaly_events:
        severity = event['severity']
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    print("   Anomaly severity distribution:")
    for severity, count in severity_counts.items():
        print(f"     {severity}: {count} events")
    
    # 8. Performance Analysis
    print("\n8. Performance analysis...")
    
    # Calculate detection performance
    fault_times = [20.0, 40.0, 60.0]  # Known fault injection times
    detection_times = []
    
    for fault_time in fault_times:
        # Find first anomaly after fault time
        post_fault_anomalies = [e for e in anomaly_events if e['timestamp'] >= fault_time]
        if post_fault_anomalies:
            detection_time = post_fault_anomalies[0]['timestamp'] - fault_time
            detection_times.append(detection_time)
            print(f"   Fault at t={fault_time}s detected after {detection_time:.1f}s")
    
    if detection_times:
        avg_detection_time = np.mean(detection_times)
        print(f"   Average detection time: {avg_detection_time:.1f}s")
    
    # 9. Create Comprehensive Visualizations
    print("\n9. Creating comprehensive visualizations...")
    
    create_phase4_visualizations(df, anomaly_events, training_results)
    
    # 10. Export Data
    print("\n10. Exporting data...")
    
    # Export simulation data
    simulation_filename = f'phase4_simulation_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(simulation_filename, index=False)
    print(f"   Simulation data exported to {simulation_filename}")
    
    # Export anomaly events
    anomaly_filename = f'phase4_anomaly_events_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(anomaly_filename, 'w') as f:
        json.dump(anomaly_events, f, indent=2, default=str)
    print(f"   Anomaly events exported to {anomaly_filename}")
    
    # Export training results
    training_filename = f'phase4_training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(training_filename, 'w') as f:
        json.dump(training_results, f, indent=2, default=str)
    print(f"   Training results exported to {training_filename}")
    
    return df, anomaly_events, training_results


def create_phase4_visualizations(df, anomaly_events, training_results):
    """Create comprehensive visualizations for Phase 4."""
    print("   Creating comprehensive visualizations...")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Phase 4 - Complete Digital Twin System Analysis', fontsize=16, fontweight='bold')
    
    # 1. Temperature and Residual Plot
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(df['timestamp'], df['plant_t_hot'], 'b-', linewidth=2, label='Plant T_hot')
    ax1.plot(df['timestamp'], df['twin_t_hot'], 'r--', linewidth=2, label='Twin T_hot')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Temperature (K)')
    ax1.set_title('Temperature Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residual Plot
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(df['timestamp'], df['residual_t_hot'], 'g-', linewidth=2, label='T_hot Residual')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Mark anomalies
    anomaly_times = [e['timestamp'] for e in anomaly_events]
    anomaly_residuals = [e['residual'] for e in anomaly_events]
    ax2.scatter(anomaly_times, anomaly_residuals, c='red', s=50, alpha=0.7, label='Anomalies')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Residual (K)')
    ax2.set_title('Residual Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Anomaly Confidence Over Time
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(df['timestamp'], df['confidence'], 'purple', linewidth=2, label='Detection Confidence')
    ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Threshold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Confidence')
    ax3.set_title('Detection Confidence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Anomaly Severity Distribution
    ax4 = plt.subplot(3, 3, 4)
    severity_counts = {}
    for event in anomaly_events:
        severity = event['severity']
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    if severity_counts:
        severities = list(severity_counts.keys())
        counts = list(severity_counts.values())
        colors = ['green', 'orange', 'red', 'purple']
        ax4.pie(counts, labels=severities, autopct='%1.1f%%', colors=colors[:len(severities)])
        ax4.set_title('Anomaly Severity Distribution')
    
    # 5. Training Results Comparison
    ax5 = plt.subplot(3, 3, 5)
    if training_results and isinstance(training_results, dict):
        methods = []
        accuracies = []
        
        for method, results in training_results.items():
            if isinstance(results, dict) and 'test_accuracy' in results:
                methods.append(method.replace('_', ' ').title())
                accuracies.append(results['test_accuracy'])
        
        if methods:
            bars = ax5.bar(methods, accuracies, color='skyblue', alpha=0.7)
            ax5.set_ylabel('Test Accuracy')
            ax5.set_title('Model Performance Comparison')
            ax5.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
    
    # 6. Residual Histogram
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(df['residual_t_hot'], bins=50, alpha=0.7, color='green', edgecolor='black')
    ax6.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero Line')
    ax6.set_xlabel('Residual (K)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Residual Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Detection Timeline
    ax7 = plt.subplot(3, 3, 7)
    if anomaly_events:
        event_times = [e['timestamp'] for e in anomaly_events]
        event_confidences = [e['confidence'] for e in anomaly_events]
        
        scatter = ax7.scatter(event_times, event_confidences, 
                            c=[e['severity'] for e in anomaly_events], 
                            s=100, alpha=0.7, cmap='viridis')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Confidence')
        ax7.set_title('Anomaly Detection Timeline')
        ax7.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax7)
        cbar.set_label('Severity')
    
    # 8. System Performance Metrics
    ax8 = plt.subplot(3, 3, 8)
    
    # Calculate performance metrics
    total_anomalies = len(anomaly_events)
    total_data_points = len(df)
    anomaly_rate = total_anomalies / total_data_points if total_data_points > 0 else 0
    avg_confidence = df['confidence'].mean()
    max_residual = df['residual_t_hot'].abs().max()
    
    metrics = ['Anomaly Rate', 'Avg Confidence', 'Max Residual', 'Data Points']
    values = [anomaly_rate, avg_confidence, max_residual, total_data_points]
    
    bars = ax8.bar(metrics, values, color=['red', 'blue', 'green', 'orange'], alpha=0.7)
    ax8.set_ylabel('Value')
    ax8.set_title('System Performance Metrics')
    ax8.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 9. Feature Importance (if available)
    ax9 = plt.subplot(3, 3, 9)
    
    # Simulate feature importance for demonstration
    features = ['T_hot', 'T_cold', 'm_dot', 'T_hot_squared', 'T_cold_squared', 
               'residual_magnitude', 'rolling_mean', 'rolling_std']
    importance = np.random.exponential(0.1, len(features))
    importance = importance / importance.sum()  # Normalize
    
    bars = ax9.barh(features, importance, color='purple', alpha=0.7)
    ax9.set_xlabel('Importance')
    ax9.set_title('Feature Importance (Simulated)')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phase4_complete_analysis.png', dpi=300, bbox_inches='tight')
    print("   Comprehensive visualizations saved as 'phase4_complete_analysis.png'")
    plt.show()


def demo_web_client_integration():
    """Demonstrate web client integration."""
    print("\n=== Web Client Integration Demo ===")
    
    # Check if web files exist
    web_dir = os.path.join(os.path.dirname(__file__), '..', 'web')
    index_file = os.path.join(web_dir, 'index.html')
    
    if os.path.exists(index_file):
        print("   Web client files found")
        print(f"   Web directory: {web_dir}")
        print(f"   Main file: {index_file}")
        
        # Try to open in browser
        try:
            file_url = f"file://{os.path.abspath(index_file)}"
            print(f"   Opening web client at: {file_url}")
            webbrowser.open(file_url)
            print("   Web client opened in browser")
        except Exception as e:
            print(f"   Could not open browser: {e}")
            print("   Please manually open the web/index.html file in your browser")
    else:
        print("   Web client files not found")
        print("   Please ensure the web directory exists with index.html")


def demo_streamlit_dashboard():
    """Demonstrate Streamlit dashboard."""
    print("\n=== Streamlit Dashboard Demo ===")
    
    # Check if Streamlit is available
    try:
        import streamlit
        print("   Streamlit is available")
        
        # Check if dashboard file exists
        dashboard_file = os.path.join(os.path.dirname(__file__), '..', 'twin', 'dashboard.py')
        if os.path.exists(dashboard_file):
            print(f"   Dashboard file found: {dashboard_file}")
            print("   To run the Streamlit dashboard, execute:")
            print(f"   streamlit run {dashboard_file}")
            print("   Then open http://localhost:8501 in your browser")
        else:
            print("   Dashboard file not found")
    except ImportError:
        print("   Streamlit not available. Install with: pip install streamlit")


def main():
    """Main demo function."""
    print("Thermal Cooling Loop Digital Twin - Phase 4 Demo")
    print("Complete Dashboard & Web Client Integration")
    print("=" * 60)
    
    try:
        # Run complete system demo
        df, anomaly_events, training_results = demo_complete_system()
        
        # Demo web client integration
        demo_web_client_integration()
        
        # Demo Streamlit dashboard
        demo_streamlit_dashboard()
        
        print("\n=== Phase 4 Demo Complete ===")
        print("Key Features Demonstrated:")
        print("- Complete digital twin system with enhanced detection")
        print("- Advanced ML models with feature engineering")
        print("- Comprehensive fault injection and testing")
        print("- Real-time monitoring and anomaly detection")
        print("- Performance analysis and visualization")
        print("- Data export and integration capabilities")
        print("- Web client with interactive dashboard")
        print("- Streamlit dashboard with advanced features")
        print("- Complete system documentation")
        
        print(f"\nSystem Performance Summary:")
        print(f"- Total data points: {len(df)}")
        print(f"- Anomalies detected: {len(anomaly_events)}")
        print(f"- Detection rate: {len(anomaly_events) / len(df):.1%}")
        print(f"- Average confidence: {df['confidence'].mean():.3f}")
        print(f"- Max residual: {df['residual_t_hot'].abs().max():.2f}K")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
