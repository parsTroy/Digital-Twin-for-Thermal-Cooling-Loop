# Running the Digital Twin Thermal Cooling Loop System

This guide explains how to run the complete system with both the backend server and web client.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Quick Start

### 1. Install Dependencies

```bash
# Install main project dependencies
pip install -r requirements.txt

# Install backend dependencies
pip install -r backend/requirements.txt
```

### 2. Start the Backend Server

```bash
# Option 1: Use the startup script (recommended)
python start_backend.py

# Option 2: Start manually
cd backend
python main.py
```

The backend server will start on `http://localhost:8000`

### 3. Access the Web Client

Open your web browser and go to:
- **Local development**: `http://localhost:8000`
- **GitHub Pages**: `https://parsTroy.github.io/Digital-Twin-for-Thermal-Cooling-Loop/`

## System Architecture

```
┌─────────────────┐    HTTP/WebSocket    ┌─────────────────┐
│   Web Client    │ ◄─────────────────► │  Backend Server │
│  (HTML/CSS/JS)  │                      │   (FastAPI)     │
└─────────────────┘                      └─────────────────┘
                                                  │
                                                  ▼
                                         ┌─────────────────┐
                                         │ Digital Twin    │
                                         │ Simulation      │
                                         └─────────────────┘
```

## Features

### Web Client Features
- **Real-time Monitoring**: Live temperature and flow data
- **Interactive Controls**: Start/stop simulation, adjust parameters
- **Anomaly Detection**: Real-time anomaly alerts and visualization
- **Data Visualization**: Interactive charts with Chart.js
- **Fault Injection**: Test system with various fault types

### Backend Features
- **FastAPI Server**: RESTful API and WebSocket support
- **Digital Twin Engine**: Real-time thermal simulation
- **Anomaly Detection**: Traditional and ML-based detection
- **Fault Simulation**: Comprehensive fault injection system
- **Data Management**: Historical data storage and retrieval

## API Endpoints

### HTTP Endpoints
- `GET /` - Serve web client
- `GET /api/status` - Get system status
- `POST /api/start` - Start simulation
- `POST /api/stop` - Stop simulation
- `POST /api/fault` - Inject fault
- `GET /api/history` - Get simulation history

### WebSocket Endpoints
- `WS /ws` - Real-time data streaming

## Configuration

### System Parameters
- **Thermal Capacitance (C)**: 1000 J/K (default)
- **Mass Flow Rate (m_dot)**: 0.1 kg/s (default)
- **Heat Capacity (cp)**: 4180 J/(kg·K) (default)
- **Heat Transfer Coefficient (UA)**: 50 W/K (default)
- **Heat Input (Q_in)**: 1000 W (default)
- **Heat Output (Q_out)**: 1000 W (default)

### Anomaly Detection
- **Traditional Methods**: Residual threshold, Z-score, CUSUM
- **ML Methods**: Isolation Forest, Random Forest
- **Thresholds**: Configurable warning and critical levels

## Troubleshooting

### Backend Won't Start
1. Check Python version: `python --version` (should be 3.8+)
2. Install dependencies: `pip install -r backend/requirements.txt`
3. Check port 8000 is available
4. Check for import errors in the console

### Web Client Can't Connect
1. Ensure backend is running on `http://localhost:8000`
2. Check browser console for errors
3. Verify WebSocket connection in Network tab
4. Check CORS settings if running from different domain

### Simulation Issues
1. Check parameter values are reasonable
2. Verify all required parameters are set
3. Check backend logs for error messages
4. Try resetting simulation

## Development

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_model.py

# Run with verbose output
python -m pytest tests/ -v
```

### Running Examples
```bash
# Phase 1: Basic model and simulator
python examples/phase1_demo.py

# Phase 2: Digital twin and detection
python examples/phase2_demo.py

# Phase 3: ML detection
python examples/phase3_demo.py

# Phase 4: Full system demo
python examples/phase4_demo.py
```

### Streamlit Dashboard
```bash
# Run the Streamlit dashboard
streamlit run twin/dashboard.py
```

## Deployment

### Local Deployment
1. Start backend server: `python start_backend.py`
2. Access web client: `http://localhost:8000`

### GitHub Pages Deployment
1. Push changes to main branch
2. GitHub Actions will automatically deploy
3. Access at: `https://parsTroy.github.io/Digital-Twin-for-Thermal-Cooling-Loop/`

### Production Deployment
1. Use a production WSGI server (e.g., Gunicorn)
2. Set up reverse proxy (e.g., Nginx)
3. Configure SSL/TLS
4. Set up monitoring and logging

## Support

For issues or questions:
1. Check this documentation
2. Review the main README.md
3. Check GitHub Issues
4. Contact: [your-email@domain.com]

## License

This project is licensed under the MIT License - see the LICENSE file for details.
