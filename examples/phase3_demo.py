"""
Phase 3 Demo: Advanced ML Anomaly Detection

This script demonstrates advanced machine learning-based anomaly detection
with feature engineering, ensemble methods, and adaptive learning.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys
import os
from datetime import datetime

# Add the parent directory to the path to import the twin module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from twin.model import create_default_parameters, create_step_heat_input
from twin.plant_simulator import PlantSimulator, FaultType
from twin.digital_twin import DigitalTwinManager, create_digital_twin_manager
from twin.enhanced_detector import EnhancedAnomalyDetector, create_enhanced_detector
from twin.detector import DetectionMethod, AnomalyType


def demo_ml_detector_training():
    """Demonstrate ML detector training and feature engineering."""
    print("=== ML Detector Training Demo ===")
    
    # Create enhanced detector
    detector = create_enhanced_detector(
        traditional_methods=[
            DetectionMethod.RESIDUAL_THRESHOLD,
            DetectionMethod.ROLLING_Z_SCORE,
            DetectionMethod.ISOLATION_FOREST
        ],
        ml_enabled=True,
        adaptive_thresholds=True,
        ensemble_voting=True
    )
    
    # Generate training data
    print("Generating training data...")
    np.random.seed(42)
    
    training_data = []
    training_labels = []
    
    # Normal data (80%)
    for _ in range(200):
        normal_residuals = {
            'T_hot': np.random.normal(0, 1.0),
            'T_cold': np.random.normal(0, 0.8),
            'm_dot': np.random.normal(0, 0.1)
        }
        training_data.append(normal_residuals)
        training_labels.append(False)
    
    # Anomalous data (20%)
    for _ in range(50):
        anomalous_residuals = {
            'T_hot': np.random.normal(0, 4.0),  # Higher variance
            'T_cold': np.random.normal(0, 3.0),
            'm_dot': np.random.normal(0, 0.3)
        }
        training_data.append(anomalous_residuals)
        training_labels.append(True)
    
    print(f"Generated {len(training_data)} training samples")
    print(f"Normal samples: {sum(1 for label in training_labels if not label)}")
    print(f"Anomalous samples: {sum(1 for label in training_labels if label)}")
    
    # Train ML models
    print("\nTraining ML models...")
    start_time = time.time()
    
    training_results = detector.train_ml_models(training_data)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Display training results
    print("\nTraining Results:")
    for model_name, results in training_results.items():
        if 'error' not in results:
            print(f"  {model_name}:")
            print(f"    Train Accuracy: {results['train_accuracy']:.3f}")
            print(f"    Test Accuracy: {results['test_accuracy']:.3f}")
            print(f"    CV Mean: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
    
    # Test feature engineering
    print("\nFeature Engineering Demo:")
    test_data = pd.DataFrame([{
        'T_hot': 1.0,
        'T_cold': 0.8,
        'm_dot': 0.1
    }])
    
    engineered_features = detector.ml_detector.engineer_features(test_data)
    print(f"Original features: {len(test_data.columns)}")
    print(f"Engineered features: {len(engineered_features.columns)}")
    print(f"Feature names: {list(engineered_features.columns)[:10]}...")  # Show first 10
    
    # Get feature importance
    if 'random_forest' in detector.ml_detector.models:
        importance = detector.ml_detector.get_feature_importance('random_forest')
        print(f"\nTop 5 Most Important Features:")
        for i, (feature, imp) in enumerate(list(importance.items())[:5]):
            print(f"  {i+1}. {feature}: {imp:.3f}")
    
    return detector


def demo_enhanced_detection():
    """Demonstrate enhanced detection with ensemble methods."""
    print("\n=== Enhanced Detection Demo ===")
    
    # Create enhanced detector
    detector = create_enhanced_detector()
    
    # Train with synthetic data
    print("Training enhanced detector...")
    training_data = []
    
    # Generate diverse training data
    np.random.seed(42)
    for _ in range(100):
        # Normal operation
        training_data.append({
            'T_hot': np.random.normal(0, 1.0),
            'T_cold': np.random.normal(0, 0.8),
            'm_dot': np.random.normal(0, 0.1)
        })
    
    for _ in range(25):
        # Various fault conditions
        training_data.append({
            'T_hot': np.random.normal(0, 3.0),
            'T_cold': np.random.normal(0, 2.5),
            'm_dot': np.random.normal(0, 0.2)
        })
    
    detector.train_ml_models(training_data)
    
    # Test detection scenarios
    test_scenarios = [
        {
            'name': 'Normal Operation',
            'residuals': {'T_hot': 0.5, 'T_cold': 0.3, 'm_dot': 0.05},
            'expected': False
        },
        {
            'name': 'Temperature Anomaly',
            'residuals': {'T_hot': 8.0, 'T_cold': 6.0, 'm_dot': 0.1},
            'expected': True
        },
        {
            'name': 'Flow Rate Anomaly',
            'residuals': {'T_hot': 1.0, 'T_cold': 0.8, 'm_dot': 0.8},
            'expected': True
        },
        {
            'name': 'Multiple Anomalies',
            'residuals': {'T_hot': 10.0, 'T_cold': 8.0, 'm_dot': 0.9},
            'expected': True
        }
    ]
    
    print("\nTesting Detection Scenarios:")
    print("-" * 60)
    
    for scenario in test_scenarios:
        result = detector.detect_anomalies(scenario['residuals'], use_ensemble=True)
        
        print(f"\n{scenario['name']}:")
        print(f"  Residuals: {scenario['residuals']}")
        print(f"  Detected: {result['final_decision']['is_anomaly']}")
        print(f"  Confidence: {result['final_decision']['confidence']:.3f}")
        print(f"  Severity: {result['final_decision']['severity'].value}")
        print(f"  Method: {result['final_decision']['method_used']}")
        
        # Show ensemble details
        if result['ensemble_result']:
            ensemble = result['ensemble_result']
            print(f"  Ensemble Details:")
            print(f"    Traditional Vote: {ensemble['details']['traditional_vote']:.3f}")
            print(f"    ML Vote: {ensemble['details']['ml_vote']:.3f}")
            print(f"    Weighted Score: {ensemble['details']['weighted_score']:.3f}")
            print(f"    Method Agreement: {ensemble['details']['method_agreement']}")
        
        # Check if detection matches expectation
        detected = result['final_decision']['is_anomaly']
        expected = scenario['expected']
        status = "✓" if detected == expected else "✗"
        print(f"  Status: {status}")
    
    return detector


def demo_adaptive_learning():
    """Demonstrate adaptive learning capabilities."""
    print("\n=== Adaptive Learning Demo ===")
    
    # Create detector with adaptive learning
    detector = create_enhanced_detector(adaptive_thresholds=True)
    
    # Start adaptive learning
    print("Starting adaptive learning...")
    detector.start_adaptive_learning()
    
    # Simulate real-time data with changing patterns
    print("Simulating real-time data with changing patterns...")
    
    np.random.seed(42)
    detection_results = []
    
    # Phase 1: Normal operation
    print("Phase 1: Normal operation (0-20s)")
    for i in range(20):
        residuals = {
            'T_hot': np.random.normal(0, 1.0),
            'T_cold': np.random.normal(0, 0.8),
            'm_dot': np.random.normal(0, 0.1)
        }
        
        result = detector.detect_anomalies(residuals, float(i))
        detection_results.append({
            'time': i,
            'phase': 'normal',
            'is_anomaly': result['final_decision']['is_anomaly'],
            'confidence': result['final_decision']['confidence']
        })
        
        time.sleep(0.1)  # Simulate real-time
    
    # Phase 2: Gradual degradation
    print("Phase 2: Gradual degradation (20-40s)")
    for i in range(20, 40):
        # Gradually increasing residuals
        factor = (i - 20) / 20.0
        residuals = {
            'T_hot': np.random.normal(0, 1.0 + factor * 2.0),
            'T_cold': np.random.normal(0, 0.8 + factor * 1.5),
            'm_dot': np.random.normal(0, 0.1 + factor * 0.1)
        }
        
        result = detector.detect_anomalies(residuals, float(i))
        detection_results.append({
            'time': i,
            'phase': 'degradation',
            'is_anomaly': result['final_decision']['is_anomaly'],
            'confidence': result['final_decision']['confidence']
        })
        
        time.sleep(0.1)
    
    # Phase 3: Fault condition
    print("Phase 3: Fault condition (40-60s)")
    for i in range(40, 60):
        residuals = {
            'T_hot': np.random.normal(0, 5.0),
            'T_cold': np.random.normal(0, 4.0),
            'm_dot': np.random.normal(0, 0.5)
        }
        
        result = detector.detect_anomalies(residuals, float(i))
        detection_results.append({
            'time': i,
            'phase': 'fault',
            'is_anomaly': result['final_decision']['is_anomaly'],
            'confidence': result['final_decision']['confidence']
        })
        
        time.sleep(0.1)
    
    # Stop adaptive learning
    detector.stop_adaptive_learning()
    
    # Analyze results
    df = pd.DataFrame(detection_results)
    
    print(f"\nAdaptive Learning Results:")
    print(f"  Total detections: {len(df)}")
    print(f"  Anomalies detected: {df['is_anomaly'].sum()}")
    print(f"  Average confidence: {df['confidence'].mean():.3f}")
    
    # Phase-wise analysis
    for phase in ['normal', 'degradation', 'fault']:
        phase_data = df[df['phase'] == phase]
        if len(phase_data) > 0:
            anomaly_rate = phase_data['is_anomaly'].mean()
            avg_confidence = phase_data['confidence'].mean()
            print(f"  {phase.capitalize()}: {anomaly_rate:.1%} anomaly rate, {avg_confidence:.3f} avg confidence")
    
    return detector, df


def demo_performance_comparison():
    """Demonstrate performance comparison between different methods."""
    print("\n=== Performance Comparison Demo ===")
    
    # Generate test data
    np.random.seed(42)
    test_data = []
    test_labels = []
    
    # Normal data
    for _ in range(100):
        test_data.append({
            'T_hot': np.random.normal(0, 1.0),
            'T_cold': np.random.normal(0, 0.8),
            'm_dot': np.random.normal(0, 0.1)
        })
        test_labels.append(False)
    
    # Anomalous data
    for _ in range(25):
        test_data.append({
            'T_hot': np.random.normal(0, 4.0),
            'T_cold': np.random.normal(0, 3.0),
            'm_dot': np.random.normal(0, 0.3)
        })
        test_labels.append(True)
    
    # Test different detector configurations
    configurations = [
        {
            'name': 'Traditional Only',
            'detector': create_enhanced_detector(ml_enabled=False, ensemble_voting=False)
        },
        {
            'name': 'ML Only',
            'detector': create_enhanced_detector(
                traditional_methods=[],
                ml_enabled=True,
                ensemble_voting=False
            )
        },
        {
            'name': 'Enhanced (Ensemble)',
            'detector': create_enhanced_detector()
        }
    ]
    
    results = {}
    
    for config in configurations:
        print(f"\nTesting {config['name']}...")
        
        detector = config['detector']
        
        # Train if ML is enabled
        if detector.ml_enabled:
            detector.train_ml_models(test_data)
        
        # Test detection
        predictions = []
        confidences = []
        times = []
        
        for i, (data, label) in enumerate(zip(test_data, test_labels)):
            start_time = time.time()
            result = detector.detect_anomalies(data, float(i))
            detection_time = time.time() - start_time
            
            predictions.append(result['final_decision']['is_anomaly'])
            confidences.append(result['final_decision']['confidence'])
            times.append(detection_time)
        
        # Calculate metrics
        accuracy = np.mean([p == l for p, l in zip(predictions, test_labels)])
        avg_confidence = np.mean(confidences)
        avg_time = np.mean(times) * 1000  # Convert to ms
        
        results[config['name']] = {
            'accuracy': accuracy,
            'confidence': avg_confidence,
            'time_ms': avg_time,
            'predictions': predictions
        }
        
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Avg Confidence: {avg_confidence:.3f}")
        print(f"  Avg Time: {avg_time:.2f} ms")
    
    # Compare results
    print(f"\nPerformance Comparison Summary:")
    print("-" * 50)
    print(f"{'Method':<20} {'Accuracy':<10} {'Confidence':<12} {'Time (ms)':<10}")
    print("-" * 50)
    
    for name, result in results.items():
        print(f"{name:<20} {result['accuracy']:<10.3f} {result['confidence']:<12.3f} {result['time_ms']:<10.2f}")
    
    return results


def demo_real_world_simulation():
    """Demonstrate real-world simulation with digital twin."""
    print("\n=== Real-World Simulation Demo ===")
    
    # Create plant simulator with faults
    plant_params = create_default_parameters()
    plant_params['Q_in'] = create_step_heat_input(1000, 1500, 30)  # Step at 30s
    
    plant_simulator = PlantSimulator(plant_params, noise_level=0.01, sample_rate=2.0)
    
    # Inject faults
    plant_simulator.inject_fault(FaultType.PUMP_DEGRADATION, 20.0, {'degradation_rate': 0.1})
    plant_simulator.inject_fault(FaultType.SENSOR_BIAS, 40.0, {'bias_magnitude': 5.0})
    
    # Create digital twin with enhanced detector
    twin_manager = create_digital_twin_manager(
        plant_simulator=plant_simulator,
        detection_methods=[
            DetectionMethod.RESIDUAL_THRESHOLD,
            DetectionMethod.ROLLING_Z_SCORE,
            DetectionMethod.ISOLATION_FOREST
        ]
    )
    
    # Replace detector with enhanced version
    enhanced_detector = create_enhanced_detector()
    twin_manager.detector = enhanced_detector
    
    # Set up monitoring
    simulation_data = []
    anomaly_count = 0
    
    def data_callback(data):
        nonlocal anomaly_count
        simulation_data.append({
            'time': data['timestamp'],
            'plant_t_hot': data['plant_state'][0],
            'twin_t_hot': data['twin_state'][0],
            'residual_t_hot': data['residuals']['T_hot'],
            'anomaly': data['anomaly_results']['overall_anomaly'],
            'confidence': data['anomaly_results'].get('overall_confidence', 0.0)
        })
        
        if data['anomaly_results']['overall_anomaly']:
            anomaly_count += 1
            print(f"Anomaly detected at t={data['timestamp']:.1f}s: "
                  f"Residual={data['residuals']['T_hot']:.2f}K, "
                  f"Confidence={data['anomaly_results'].get('overall_confidence', 0.0):.3f}")
    
    twin_manager.add_data_callback(data_callback)
    
    # Start simulation
    print("Starting real-world simulation...")
    initial_conditions = np.array([350.0, 300.0])
    twin_manager.start(initial_conditions)
    
    # Run for 60 seconds
    print("Running simulation for 60 seconds...")
    time.sleep(60)
    
    twin_manager.stop()
    
    # Analyze results
    df = pd.DataFrame(simulation_data)
    
    print(f"\nSimulation Results:")
    print(f"  Duration: {df['time'].max():.1f} seconds")
    print(f"  Data points: {len(df)}")
    print(f"  Anomalies detected: {anomaly_count}")
    print(f"  Anomaly rate: {anomaly_count / len(df):.1%}")
    
    # Temperature analysis
    print(f"  Temperature range: {df['plant_t_hot'].min():.1f}K - {df['plant_t_hot'].max():.1f}K")
    print(f"  Max residual: {df['residual_t_hot'].abs().max():.2f}K")
    print(f"  Avg confidence: {df['confidence'].mean():.3f}")
    
    return df


def create_visualization_plots(detection_data, simulation_data):
    """Create comprehensive visualization plots."""
    print("\n=== Creating Visualization Plots ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Phase 3 - Advanced ML Anomaly Detection Demo Results', fontsize=16)
    
    # Plot 1: Detection confidence over time
    ax1 = axes[0, 0]
    if detection_data is not None and len(detection_data) > 0:
        df = detection_data
        ax1.plot(df['time'], df['confidence'], 'b-', linewidth=2, label='Confidence')
        ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Threshold')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Detection Confidence')
        ax1.set_title('Adaptive Detection Confidence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Anomaly detection over time
    ax2 = axes[0, 1]
    if detection_data is not None and len(detection_data) > 0:
        anomaly_times = df[df['is_anomaly']]['time']
        if len(anomaly_times) > 0:
            ax2.scatter(anomaly_times, [1] * len(anomaly_times), c='red', s=50, alpha=0.7, label='Anomalies')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Anomaly Detection')
        ax2.set_title('Anomaly Detection Timeline')
        ax2.set_ylim(-0.1, 1.5)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Temperature and residuals
    ax3 = axes[1, 0]
    if simulation_data is not None and len(simulation_data) > 0:
        df_sim = simulation_data
        ax3_twin = ax3.twinx()
        
        line1 = ax3.plot(df_sim['time'], df_sim['plant_t_hot'], 'b-', linewidth=2, label='Plant T_hot')
        line2 = ax3.plot(df_sim['time'], df_sim['twin_t_hot'], 'r--', linewidth=2, label='Twin T_hot')
        line3 = ax3_twin.plot(df_sim['time'], df_sim['residual_t_hot'], 'g-', linewidth=2, label='Residual')
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Temperature (K)', color='b')
        ax3_twin.set_ylabel('Residual (K)', color='g')
        ax3.set_title('Temperature Comparison and Residuals')
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper left')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance comparison
    ax4 = axes[1, 1]
    # This would be populated with performance comparison data
    ax4.text(0.5, 0.5, 'Performance Comparison\n(Would show bar chart\nof different methods)', 
             ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    ax4.set_title('Method Performance Comparison')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('phase3_demo_results.png', dpi=300, bbox_inches='tight')
    print("Plots saved as 'phase3_demo_results.png'")
    plt.show()


def main():
    """Main demo function."""
    print("Thermal Cooling Loop Digital Twin - Phase 3 Demo")
    print("Advanced ML Anomaly Detection")
    print("=" * 60)
    
    # Run all demos
    print("Running Phase 3 demonstrations...")
    
    # ML detector training demo
    detector = demo_ml_detector_training()
    
    # Enhanced detection demo
    enhanced_detector = demo_enhanced_detection()
    
    # Adaptive learning demo
    adaptive_detector, detection_data = demo_adaptive_learning()
    
    # Performance comparison demo
    performance_results = demo_performance_comparison()
    
    # Real-world simulation demo
    simulation_data = demo_real_world_simulation()
    
    # Create visualization plots
    create_visualization_plots(detection_data, simulation_data)
    
    print("\n=== Phase 3 Demo Complete ===")
    print("Key Features Demonstrated:")
    print("- Advanced ML models with feature engineering")
    print("- Ensemble learning and voting mechanisms")
    print("- Adaptive learning and threshold adjustment")
    print("- Performance comparison between methods")
    print("- Real-world simulation with digital twin")
    print("- Comprehensive visualization and analysis")
    print("- Automated model training and optimization")
    print("- Multi-method anomaly detection integration")


if __name__ == '__main__':
    main()
