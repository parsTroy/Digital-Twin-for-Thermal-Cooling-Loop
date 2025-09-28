import asyncio
import json
import logging
import threading
import time
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import sys
import os

# Add the parent directory to the path to import the twin module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from twin.model import ThermalModel, default_params, create_step_heat_input
from twin.plant_simulator import PlantSimulator, FaultType
from twin.detector import DetectionMethod, AnomalyType
from twin.ml_detector import MLModelType, MLAnomalyType
from twin.enhanced_detector import EnhancedAnomalyDetector, create_enhanced_detector
from twin.digital_twin import DigitalTwinManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Digital Twin Thermal Cooling Loop API")

# CORS middleware for web client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the web client
app.mount("/static", StaticFiles(directory="../web"), name="static")

# Global manager for the digital twin simulation
dt_manager_instance: Optional[DigitalTwinManager] = None
manager_lock = threading.Lock()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected: {websocket.client}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def get_root():
    """Serve the main web client page"""
    with open("../web/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/status")
async def get_status():
    """Get current system status"""
    with manager_lock:
        if dt_manager_instance and dt_manager_instance._running:
            return {
                "status": "running",
                "current_time": dt_manager_instance.current_time,
                "data_points": len(dt_manager_instance.data_history)
            }
        else:
            return {"status": "stopped"}

@app.post("/api/start")
async def start_simulation(params: Dict[str, Any]):
    """Start the digital twin simulation"""
    global dt_manager_instance
    
    with manager_lock:
        if dt_manager_instance and dt_manager_instance._running:
            return {"status": "error", "message": "Simulation already running"}
        
        try:
            # Extract parameters
            initial_temp_hot = params.get('initial_temp_hot', 300.0)
            initial_temp_cold = params.get('initial_temp_cold', 280.0)
            heat_input = params.get('heat_input', 1000.0)
            mass_flow = params.get('mass_flow', 0.1)
            heat_capacity = params.get('heat_capacity', 41800.0)
            heat_transfer_coeff = params.get('heat_transfer_coeff', 50.0)
            
            # Create system parameters
            system_params = default_params.copy()
            system_params.update({
                'C': heat_capacity,
                'm_dot': mass_flow,
                'cp': 4180.0,
                'UA': heat_transfer_coeff,
                'Q_in': create_step_heat_input(heat_input, heat_input * 1.5, 50.0),
                'Q_out': 0.0
            })
            
            # Create enhanced detector
            traditional_methods = [
                DetectionMethod.RESIDUAL_THRESHOLD,
                DetectionMethod.ROLLING_Z_SCORE,
                DetectionMethod.CUSUM
            ]
            ml_model_types = [
                MLModelType.ISOLATION_FOREST,
                MLModelType.RANDOM_FOREST
            ]
            
            enhanced_detector = create_enhanced_detector(
                traditional_methods=traditional_methods,
                ml_enabled=True,
                ml_model_types=ml_model_types,
                feature_engineering=True,
                auto_tuning=True,
                ensemble_voting=True,
                adaptive_thresholds=False,
                anomaly_detector_params={
                    'thresholds': {
                        'T_hot': {'warning': 2.0, 'critical': 4.0},
                        'T_cold': {'warning': 1.5, 'critical': 3.0},
                        'm_dot': {'warning': 0.005, 'critical': 0.01}
                    },
                    'z_score_window': 30,
                    'cusum_threshold': 2.0,
                    'cusum_drift': 0.05
                }
            )
            
            # Create plant simulator
            sensor_chars = {
                'T_hot': {'noise_std': 0.5, 'bias': 0.0, 'drift_rate': 0.001, 'resolution': 0.1, 'range': (200, 500)},
                'T_cold': {'noise_std': 0.3, 'bias': 0.0, 'drift_rate': 0.0005, 'resolution': 0.1, 'range': (200, 400)},
                'm_dot': {'noise_std': 0.001, 'bias': 0.0, 'drift_rate': 0.00001, 'resolution': 0.0001, 'range': (0.01, 0.5)}
            }
            
            plant_simulator = PlantSimulator(
                base_params=system_params,
                sample_rate=1.0,
                sensor_characteristics=sensor_chars
            )
            
            # Create digital twin manager
            dt_manager_instance = DigitalTwinManager(
                plant_simulator=plant_simulator,
                twin_params=system_params,
                detection_methods=traditional_methods,
                update_rate=1.0,
                anomaly_detector_params={
                    'thresholds': {
                        'T_hot': {'warning': 2.0, 'critical': 4.0},
                        'T_cold': {'warning': 1.5, 'critical': 3.0},
                        'm_dot': {'warning': 0.005, 'critical': 0.01}
                    },
                    'z_score_window': 30,
                    'cusum_threshold': 2.0,
                    'cusum_drift': 0.05
                }
            )
            
            # Set up callbacks for broadcasting data
            def data_callback(data):
                asyncio.create_task(manager.broadcast(json.dumps({
                    'type': 'data_update',
                    'payload': {
                        'timestamp': data['timestamp'],
                        'plant_T_hot': data['plant_measured_state']['T_hot'],
                        'plant_T_cold': data['plant_measured_state']['T_cold'],
                        'plant_m_dot': data['plant_measured_state']['m_dot'],
                        'twin_T_hot': data['twin_state'][0],
                        'twin_T_cold': data['twin_state'][1],
                        'residual_T_hot': data['residuals']['T_hot'],
                        'residual_T_cold': data['residuals']['T_cold'],
                        'residual_m_dot': data['residuals']['m_dot'],
                        'overall_anomaly': data['anomaly_results']['overall_anomaly'],
                        'overall_severity': data['anomaly_results']['overall_severity'].name,
                        'anomaly_confidence': data['anomaly_results'].get('confidence', 0.0),
                        'anomaly_method': data['anomaly_results'].get('method_used', 'Unknown')
                    }
                })))
            
            def anomaly_callback(anomaly_event):
                asyncio.create_task(manager.broadcast(json.dumps({
                    'type': 'anomaly_event',
                    'payload': {
                        'timestamp': anomaly_event['timestamp'],
                        'severity': anomaly_event['severity'],
                        'details': anomaly_event['details']
                    }
                })))
            
            dt_manager_instance.add_data_callback(data_callback)
            dt_manager_instance.add_anomaly_callback(anomaly_callback)
            
            # Start simulation
            initial_conditions = np.array([initial_temp_hot, initial_temp_cold])
            dt_manager_instance.start(initial_conditions)
            
            return {"status": "success", "message": "Simulation started"}
            
        except Exception as e:
            logger.error(f"Error starting simulation: {e}")
            return {"status": "error", "message": str(e)}

@app.post("/api/stop")
async def stop_simulation():
    """Stop the digital twin simulation"""
    global dt_manager_instance
    
    with manager_lock:
        if dt_manager_instance and dt_manager_instance._running:
            dt_manager_instance.stop()
            return {"status": "success", "message": "Simulation stopped"}
        else:
            return {"status": "error", "message": "Simulation not running"}

@app.post("/api/fault")
async def inject_fault(fault_data: Dict[str, Any]):
    """Inject a fault into the simulation"""
    with manager_lock:
        if dt_manager_instance:
            try:
                fault_type = FaultType[fault_data['type'].upper().replace('-', '_')]
                dt_manager_instance.plant_simulator.inject_fault(
                    fault_type,
                    fault_data['start_time'],
                    fault_data['params']
                )
                return {"status": "success", "message": f"Fault {fault_data['type']} injected"}
            except Exception as e:
                logger.error(f"Error injecting fault: {e}")
                return {"status": "error", "message": str(e)}
        else:
            return {"status": "error", "message": "Simulation not running"}

@app.get("/api/history")
async def get_history():
    """Get simulation history data"""
    with manager_lock:
        if dt_manager_instance:
            history_df = dt_manager_instance.get_data_history()
            anomaly_df = dt_manager_instance.get_anomaly_history()
            return {
                "data_history": history_df.to_dict(orient='records') if not history_df.empty else [],
                "anomaly_history": anomaly_df.to_dict(orient='records') if not anomaly_df.empty else []
            }
        else:
            return {"data_history": [], "anomaly_history": []}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            logger.info(f"Received WebSocket message: {message['type']}")
            
            if message['type'] == 'ping':
                await manager.send_personal_message(json.dumps({'type': 'pong'}), websocket)
            elif message['type'] == 'get_status':
                status = await get_status()
                await manager.send_personal_message(json.dumps({'type': 'status', 'payload': status}), websocket)
            elif message['type'] == 'get_history':
                history = await get_history()
                await manager.send_personal_message(json.dumps({'type': 'history', 'payload': history}), websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
