# Digital Twin for Thermal Cooling Loop

A comprehensive digital twin implementation for closed-loop cooling systems with real-time anomaly detection and interactive web dashboard.

## Project Overview

This project implements a digital twin model of a thermal cooling loop system commonly used in aerospace thermal management and nuclear auxiliary systems. The system includes real-time monitoring, anomaly detection, and fault simulation capabilities.

## System Model

The thermal cooling loop is modeled using lumped capacitance approach with two state variables:
- **T_hot**: Fluid temperature after heat source
- **T_cold**: Fluid temperature after heat exchanger

### Governing Equations

```
C * dT_hot/dt = Q_in - UA*(T_hot - T_cold) - m_dot*cp*(T_hot - T_cold)
C * dT_cold/dt = m_dot*cp*(T_hot - T_cold) - Q_out
```

Where:
- C: Thermal capacitance
- Q_in: Heat input from payload
- UA: Heat exchanger effectiveness
- m_dot: Mass flow rate
- cp: Specific heat capacity
- Q_out: Heat rejection

## Features

### Core Functionality
- **Digital Twin Model**: Real-time ODE-based thermal simulation
- **Plant Simulator**: Synthetic sensor data with noise and faults
- **Anomaly Detection**: Residual-based and ML-based detection
- **Fault Simulation**: Pump failures, heat exchanger fouling
- **Real-time Monitoring**: Live system state visualization

### Web Dashboard
- **Interactive Interface**: Real-time parameter adjustment
- **Live Visualization**: Temperature, flow, and residual plots
- **Fault Injection**: Simulate various system failures
- **Performance Metrics**: System efficiency and stability analysis
- **Mobile Responsive**: Works on all devices

## Project Structure

```
thermal-digital-twin/
├─ README.md
├─ requirements.txt
├─ twin/
│  ├─ model.py              # ODE system implementation
│  ├─ plant_simulator.py    # Synthetic sensor data generator
│  ├─ detector.py           # Anomaly detection algorithms
│  └─ dashboard.py          # Streamlit dashboard
├─ web/                     # Web client dashboard
│  ├─ index.html
│  ├─ css/
│  ├─ js/
│  └─ docs/
├─ tests/                   # Unit and integration tests
├─ examples/                # Demo scenarios and playbooks
└─ .github/workflows/       # GitHub Actions for deployment
```

## Quick Start

### Python Environment
```bash
pip install -r requirements.txt
python -m twin.dashboard
```

### Web Dashboard
Open `web/index.html` in a modern web browser or visit the GitHub Pages deployment.

## Applications

- **Aerospace**: Thermal management for spacecraft and aircraft systems
- **Nuclear**: Auxiliary cooling systems for nuclear power plants
- **Industrial**: Process cooling and temperature control systems
- **Research**: Thermal system modeling and fault detection studies

## Technology Stack

- **Backend**: Python, SciPy, Pandas, Scikit-learn
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Visualization**: Streamlit, Chart.js
- **Deployment**: GitHub Pages, GitHub Actions

## License

MIT License - see LICENSE file for details.
