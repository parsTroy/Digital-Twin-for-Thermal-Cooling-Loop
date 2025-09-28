"""
Phase 1 Demo: Forward Model & Plant Simulator

This script demonstrates the core functionality of the thermal cooling loop
digital twin model and plant simulator with various heat input scenarios
and fault conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
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


def demo_basic_simulation():
    """Demonstrate basic simulation functionality."""
    print("=== Basic Simulation Demo ===")
    
    # Create twin model with default parameters
    params = create_default_parameters()
    twin = ThermalCoolingTwin(params)
    
    # Set up simulation
    t_span = (0, 200)
    y0 = np.array([350.0, 300.0])  # Initial temperatures
    
    print(f"Initial conditions: T_hot = {y0[0]} K, T_cold = {y0[1]} K")
    print(f"Simulation time: {t_span[0]} to {t_span[1]} seconds")
    
    # Run simulation
    results = twin.simulate(t_span, y0)
    
    if results['success']:
        print(f"Simulation completed successfully!")
        print(f"Number of time points: {len(results['t'])}")
        print(f"Final temperatures: T_hot = {results['T_hot'][-1]:.2f} K, T_cold = {results['T_cold'][-1]:.2f} K")
        
        # Calculate steady state
        T_hot_ss, T_cold_ss = twin.get_steady_state()
        print(f"Steady state: T_hot = {T_hot_ss:.2f} K, T_cold = {T_cold_ss:.2f} K")
    else:
        print(f"Simulation failed: {results['message']}")
    
    return results


def demo_heat_input_scenarios():
    """Demonstrate different heat input scenarios."""
    print("\n=== Heat Input Scenarios Demo ===")
    
    # Create twin model
    params = create_default_parameters()
    twin = ThermalCoolingTwin(params)
    
    # Set up simulation
    t_span = (0, 150)
    y0 = np.array([350.0, 300.0])
    t_eval = np.linspace(0, 150, 1000)
    
    scenarios = [
        ("Constant", params['Q_in']),
        ("Time Varying", create_time_varying_heat_input(1000, 200, 0.1)),
        ("Step", create_step_heat_input(1000, 1500, 50)),
        ("Ramp", create_ramp_heat_input(1000, 1500, 30, 40)),
        ("Pulse", create_pulse_heat_input(1000, 2000, 60, 20))
    ]
    
    results = {}
    
    for name, Q_in_func in scenarios:
        print(f"\nRunning {name} scenario...")
        
        # Update parameters
        scenario_params = params.copy()
        scenario_params['Q_in'] = Q_in_func
        scenario_twin = ThermalCoolingTwin(scenario_params)
        
        # Run simulation
        result = scenario_twin.simulate(t_span, y0, t_eval=t_eval)
        
        if result['success']:
            results[name] = result
            print(f"  Success: Final T_hot = {result['T_hot'][-1]:.2f} K")
        else:
            print(f"  Failed: {result['message']}")
    
    return results


def demo_plant_simulator():
    """Demonstrate plant simulator functionality."""
    print("\n=== Plant Simulator Demo ===")
    
    # Create plant simulator
    params = create_default_parameters()
    simulator = PlantSimulator(params, noise_level=0.005, sample_rate=2.0)
    twin = ThermalCoolingTwin(params)
    
    # Set up simulation
    t_span = (0, 100)
    y0 = np.array([350.0, 300.0])
    
    print("Running nominal simulation...")
    
    # Run nominal simulation
    df_nominal = simulator.simulate_sensor_data(twin, t_span, y0)
    
    print(f"Generated {len(df_nominal)} data points")
    print(f"Temperature range: T_hot = {df_nominal['T_hot'].min():.2f} - {df_nominal['T_hot'].max():.2f} K")
    print(f"Flow rate range: {df_nominal['m_dot'].min():.4f} - {df_nominal['m_dot'].max():.4f} kg/s")
    
    # Test fault injection
    print("\nTesting fault injection...")
    
    # Inject pump degradation fault
    simulator.inject_fault(
        FaultType.PUMP_DEGRADATION,
        start_time=50.0,
        parameters={'degradation_rate': 0.02}
    )
    
    # Run simulation with fault
    df_fault = simulator.simulate_sensor_data(twin, t_span, y0)
    
    # Compare before and after fault
    before_fault = df_fault[df_fault['t'] < 50.0]
    after_fault = df_fault[df_fault['t'] > 50.0]
    
    if len(before_fault) > 0 and len(after_fault) > 0:
        avg_flow_before = before_fault['m_dot_true'].mean()
        avg_flow_after = after_fault['m_dot_true'].mean()
        
        print(f"Average flow before fault: {avg_flow_before:.4f} kg/s")
        print(f"Average flow after fault: {avg_flow_after:.4f} kg/s")
        print(f"Flow reduction: {((avg_flow_before - avg_flow_after) / avg_flow_before * 100):.1f}%")
    
    return df_nominal, df_fault


def demo_sensor_characteristics():
    """Demonstrate sensor characteristics and noise."""
    print("\n=== Sensor Characteristics Demo ===")
    
    # Create simulator with custom sensor characteristics
    params = create_default_parameters()
    
    custom_characteristics = {
        'T_hot': {
            'noise_std': 1.0,  # Higher noise
            'bias': 2.0,       # Temperature bias
            'drift_rate': 0.005,  # Drift rate
            'resolution': 0.5,    # Resolution
            'range': (200, 500)   # Range limits
        },
        'T_cold': {
            'noise_std': 0.5,
            'bias': 1.0,
            'drift_rate': 0.002,
            'resolution': 0.2,
            'range': (200, 400)
        },
        'm_dot': {
            'noise_std': 0.002,
            'bias': 0.001,
            'drift_rate': 0.00001,
            'resolution': 0.0001,
            'range': (0.01, 0.5)
        }
    }
    
    simulator = PlantSimulator(params, sensor_characteristics=custom_characteristics)
    twin = ThermalCoolingTwin(params)
    
    # Run simulation
    t_span = (0, 50)
    y0 = np.array([350.0, 300.0])
    
    df = simulator.simulate_sensor_data(twin, t_span, y0)
    
    # Analyze sensor characteristics
    print("Sensor noise analysis:")
    for sensor in ['T_hot', 'T_cold', 'm_dot']:
        true_col = f'{sensor}_true'
        noisy_col = sensor
        
        if true_col in df.columns:
            noise = df[noisy_col] - df[true_col]
            print(f"  {sensor}:")
            print(f"    Noise std: {noise.std():.4f}")
            print(f"    Noise mean: {noise.mean():.4f}")
            print(f"    Max noise: {noise.max():.4f}")
            print(f"    Min noise: {noise.min():.4f}")
    
    return df


def demo_fault_scenarios():
    """Demonstrate different fault scenarios."""
    print("\n=== Fault Scenarios Demo ===")
    
    params = create_default_parameters()
    simulator = PlantSimulator(params)
    twin = ThermalCoolingTwin(params)
    
    t_span = (0, 120)
    y0 = np.array([350.0, 300.0])
    
    # Get demo scenarios
    scenarios = simulator.create_demo_scenarios()
    
    print(f"Available scenarios: {len(scenarios)}")
    for i, scenario in enumerate(scenarios):
        print(f"  {i+1}. {scenario['name']}: {scenario['description']}")
    
    # Run a few scenarios
    selected_scenarios = ['Nominal Operation', 'Pump Degradation', 'Sensor Bias']
    
    results = {}
    
    for scenario_name in selected_scenarios:
        print(f"\nRunning {scenario_name} scenario...")
        
        # Find scenario
        scenario = next((s for s in scenarios if s['name'] == scenario_name), None)
        if not scenario:
            print(f"  Scenario '{scenario_name}' not found")
            continue
        
        # Clear previous faults
        simulator.clear_faults()
        
        # Inject faults for this scenario
        for fault in scenario['faults']:
            fault_type = FaultType(fault['type'])
            simulator.inject_fault(
                fault_type,
                fault['start_time'],
                fault.get('parameters', {})
            )
        
        # Run simulation
        df = simulator.simulate_sensor_data(twin, t_span, y0)
        results[scenario_name] = df
        
        print(f"  Generated {len(df)} data points")
        print(f"  Active faults: {len(simulator.active_faults)}")
    
    return results


def create_plots(results_dict, save_plots=True):
    """Create visualization plots for the demo results."""
    print("\n=== Creating Plots ===")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Thermal Cooling Loop Digital Twin - Phase 1 Demo', fontsize=16)
    
    # Plot 1: Heat input scenarios
    ax1 = axes[0, 0]
    if 'heat_scenarios' in results_dict:
        scenarios = results_dict['heat_scenarios']
        for name, result in scenarios.items():
            ax1.plot(result['t'], result['T_hot'], label=f'{name} - T_hot', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Temperature (K)')
        ax1.set_title('Heat Input Scenarios - Hot Temperature')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Plant simulator with fault
    ax2 = axes[0, 1]
    if 'plant_simulator' in results_dict:
        df_nominal, df_fault = results_dict['plant_simulator']
        
        ax2.plot(df_nominal['t'], df_nominal['T_hot'], 'b-', label='Nominal T_hot', linewidth=2)
        ax2.plot(df_fault['t'], df_fault['T_hot'], 'r-', label='With Fault T_hot', linewidth=2)
        ax2.axvline(x=50, color='gray', linestyle='--', alpha=0.7, label='Fault Start')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Temperature (K)')
        ax2.set_title('Plant Simulator - Fault Injection')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sensor characteristics
    ax3 = axes[1, 0]
    if 'sensor_characteristics' in results_dict:
        df = results_dict['sensor_characteristics']
        
        ax3.plot(df['t'], df['T_hot_true'], 'b-', label='True T_hot', linewidth=2)
        ax3.plot(df['t'], df['T_hot'], 'r--', label='Noisy T_hot', linewidth=1)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Temperature (K)')
        ax3.set_title('Sensor Characteristics - Noise and Bias')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Fault scenarios comparison
    ax4 = axes[1, 1]
    if 'fault_scenarios' in results_dict:
        scenarios = results_dict['fault_scenarios']
        
        for name, df in scenarios.items():
            ax4.plot(df['t'], df['T_hot'], label=name, linewidth=2)
        
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Temperature (K)')
        ax4.set_title('Fault Scenarios Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('phase1_demo_results.png', dpi=300, bbox_inches='tight')
        print("Plots saved as 'phase1_demo_results.png'")
    
    plt.show()


def main():
    """Main demo function."""
    print("Thermal Cooling Loop Digital Twin - Phase 1 Demo")
    print("=" * 50)
    
    # Run all demos
    results = {}
    
    # Basic simulation
    results['basic'] = demo_basic_simulation()
    
    # Heat input scenarios
    results['heat_scenarios'] = demo_heat_input_scenarios()
    
    # Plant simulator
    results['plant_simulator'] = demo_plant_simulator()
    
    # Sensor characteristics
    results['sensor_characteristics'] = demo_sensor_characteristics()
    
    # Fault scenarios
    results['fault_scenarios'] = demo_fault_scenarios()
    
    # Create plots
    create_plots(results)
    
    print("\n=== Demo Complete ===")
    print("Phase 1 implementation successfully demonstrated!")
    print("\nKey Features Demonstrated:")
    print("- ODE-based thermal cooling loop model")
    print("- Multiple heat input scenarios")
    print("- Realistic plant simulator with sensor noise")
    print("- Fault injection and testing")
    print("- Comprehensive sensor modeling")
    print("- Automated test scenarios")


if __name__ == '__main__':
    main()
