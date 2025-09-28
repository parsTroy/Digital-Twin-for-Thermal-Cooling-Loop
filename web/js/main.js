/**
 * Main JavaScript for Thermal Cooling Loop Digital Twin
 * Handles navigation, initialization, and core functionality with backend integration
 */

// Global variables
let websocket = null;
let isConnected = false;
let simulationRunning = false;
let charts = {};

// Backend configuration
const BACKEND_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws';

class ThermalCoolingApp {
    constructor() {
        this.currentTab = 'simulation';
        this.simulation = null;
        this.monitoring = null;
        this.testing = null;
        this.isSimulationRunning = false;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.initializeCharts();
        this.updateParameterDisplays();
        this.connectToBackend();
        console.log('Thermal Cooling Loop Digital Twin initialized');
    }
    
    // Backend connection functions
    async connectToBackend() {
        try {
            // Test HTTP connection
            const response = await fetch(`${BACKEND_URL}/api/status`);
            if (response.ok) {
                console.log('Backend HTTP connection successful');
                this.updateConnectionStatus(true);
            } else {
                throw new Error('Backend not responding');
            }
            
            // Connect WebSocket
            this.connectWebSocket();
            
        } catch (error) {
            console.error('Failed to connect to backend:', error);
            this.updateConnectionStatus(false);
            this.showNotification('Backend connection failed. Please start the backend server.', 'error');
        }
    }

    connectWebSocket() {
        try {
            websocket = new WebSocket(WS_URL);
            
            websocket.onopen = (event) => {
                console.log('WebSocket connected');
                isConnected = true;
                this.updateConnectionStatus(true);
            };
            
            websocket.onmessage = (event) => {
                const message = JSON.parse(event.data);
                this.handleWebSocketMessage(message);
            };
            
            websocket.onclose = (event) => {
                console.log('WebSocket disconnected');
                isConnected = false;
                this.updateConnectionStatus(false);
                
                // Attempt to reconnect after 3 seconds
                setTimeout(() => this.connectWebSocket(), 3000);
            };
            
            websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                isConnected = false;
                this.updateConnectionStatus(false);
            };
            
        } catch (error) {
            console.error('WebSocket connection failed:', error);
            isConnected = false;
            this.updateConnectionStatus(false);
        }
    }

    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'data_update':
                this.updateRealTimeData(message.payload);
                break;
            case 'anomaly_event':
                this.handleAnomalyEvent(message.payload);
                break;
            case 'status':
                this.updateSystemStatus(message.payload);
                break;
            case 'history':
                this.updateHistoryData(message.payload);
                break;
            case 'pong':
                // Heartbeat response
                break;
            default:
                console.log('Unknown message type:', message.type);
        }
    }

    updateConnectionStatus(connected, mode = 'Connected') {
        const statusElement = document.getElementById('connectionStatus');
        if (statusElement) {
            statusElement.textContent = connected ? mode : 'Disconnected';
            statusElement.className = connected ? 'status-connected' : 'status-disconnected';
        }
    }

    updateRealTimeData(data) {
        // Update temperature displays
        this.updateElement('plant-temp-hot', data.plant_T_hot.toFixed(2));
        this.updateElement('plant-temp-cold', data.plant_T_cold.toFixed(2));
        this.updateElement('twin-temp-hot', data.twin_T_hot.toFixed(2));
        this.updateElement('twin-temp-cold', data.twin_T_cold.toFixed(2));
        this.updateElement('mass-flow', data.plant_m_dot.toFixed(4));
        
        // Update residuals
        this.updateElement('residual-temp-hot', data.residual_T_hot.toFixed(3));
        this.updateElement('residual-temp-cold', data.residual_T_cold.toFixed(3));
        this.updateElement('residual-mass-flow', data.residual_m_dot.toFixed(6));
        
        // Update anomaly status
        this.updateAnomalyStatus(data.overall_anomaly, data.overall_severity, data.anomaly_confidence);
        
        // Update charts
        this.updateCharts(data);
    }

    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }

    updateAnomalyStatus(isAnomaly, severity, confidence) {
        const statusElement = document.getElementById('anomaly-status');
        const severityElement = document.getElementById('anomaly-severity');
        const confidenceElement = document.getElementById('anomaly-confidence');
        
        if (statusElement) {
            statusElement.textContent = isAnomaly ? 'ANOMALY DETECTED' : 'NORMAL';
            statusElement.className = isAnomaly ? 'anomaly-detected' : 'anomaly-normal';
        }
        
        if (severityElement) {
            severityElement.textContent = severity;
        }
        
        if (confidenceElement) {
            confidenceElement.textContent = `${(confidence * 100).toFixed(1)}%`;
        }
    }

    updateCharts(data) {
        // Update temperature chart
        if (charts.temperature) {
            charts.temperature.data.labels.push(new Date().toLocaleTimeString());
            charts.temperature.data.datasets[0].data.push(data.plant_T_hot);
            charts.temperature.data.datasets[1].data.push(data.plant_T_cold);
            charts.temperature.data.datasets[2].data.push(data.twin_T_hot);
            charts.temperature.data.datasets[3].data.push(data.twin_T_cold);
            
            // Keep only last 50 points
            if (charts.temperature.data.labels.length > 50) {
                charts.temperature.data.labels.shift();
                charts.temperature.data.datasets.forEach(dataset => dataset.data.shift());
            }
            
            charts.temperature.update('none');
        }
        
        // Update residuals chart
        if (charts.residuals) {
            charts.residuals.data.labels.push(new Date().toLocaleTimeString());
            charts.residuals.data.datasets[0].data.push(data.residual_T_hot);
            charts.residuals.data.datasets[1].data.push(data.residual_T_cold);
            charts.residuals.data.datasets[2].data.push(data.residual_m_dot);
            
            // Keep only last 50 points
            if (charts.residuals.data.labels.length > 50) {
                charts.residuals.data.labels.shift();
                charts.residuals.data.datasets.forEach(dataset => dataset.data.shift());
            }
            
            charts.residuals.update('none');
        }
    }

    handleAnomalyEvent(anomaly) {
        console.log('Anomaly detected:', anomaly);
        
        // Add to anomaly list
        this.addAnomalyToList(anomaly);
        
        // Show notification
        this.showNotification(`Anomaly detected: ${anomaly.severity}`, 'warning');
    }

    addAnomalyToList(anomaly) {
        const anomalyList = document.getElementById('anomaly-list');
        if (anomalyList) {
            const anomalyItem = document.createElement('div');
            anomalyItem.className = 'anomaly-item';
            anomalyItem.innerHTML = `
                <div class="anomaly-time">${new Date(anomaly.timestamp * 1000).toLocaleTimeString()}</div>
                <div class="anomaly-severity">${anomaly.severity}</div>
                <div class="anomaly-details">${JSON.stringify(anomaly.details, null, 2)}</div>
            `;
            anomalyList.insertBefore(anomalyItem, anomalyList.firstChild);
            
            // Keep only last 10 anomalies
            while (anomalyList.children.length > 10) {
                anomalyList.removeChild(anomalyList.lastChild);
            }
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
    }
    
    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });
        
        // Parameter controls
        document.querySelectorAll('input[type="range"]').forEach(input => {
            input.addEventListener('input', (e) => {
                this.updateParameterDisplays();
            });
        });
        
        // Simulation buttons
        const startBtn = document.getElementById('start-simulation');
        const stopBtn = document.getElementById('stop-simulation');
        const resetBtn = document.getElementById('reset-simulation');
        
        if (startBtn) {
            startBtn.addEventListener('click', () => {
                this.startSimulation();
            });
        }
        
        if (stopBtn) {
            stopBtn.addEventListener('click', () => {
                this.stopSimulation();
            });
        }
        
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                this.resetSimulation();
            });
        }
        
        // Documentation navigation
        document.querySelectorAll('.doc-nav-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchDocSection(e.target.dataset.doc);
            });
        });
    }
    
    switchTab(tabName) {
        // Update navigation
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.remove('active');
        });
        const activeBtn = document.querySelector(`[onclick="showTab('${tabName}')"]`);
        if (activeBtn) {
            activeBtn.classList.add('active');
        }
        
        // Update content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        const activeContent = document.getElementById(tabName);
        if (activeContent) {
            activeContent.classList.add('active');
        }
        
        this.currentTab = tabName;
        
        // Initialize tab-specific functionality
        if (tabName === 'monitoring' && !this.monitoring) {
            this.initializeMonitoring();
        } else if (tabName === 'testing' && !this.testing) {
            this.initializeTesting();
        }
    }
    
    switchDocSection(sectionName) {
        // Update documentation navigation
        document.querySelectorAll('.doc-nav-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-doc="${sectionName}"]`).classList.add('active');
        
        // Update content
        document.querySelectorAll('.doc-section').forEach(section => {
            section.classList.remove('active');
        });
        document.getElementById(sectionName).classList.add('active');
    }
    
    updateParameterDisplays() {
        const parameters = [
            'thermal-capacitance',
            'mass-flow',
            'heat-capacity',
            'heat-exchanger',
            'heat-input',
            'heat-output'
        ];
        
        parameters.forEach(param => {
            const input = document.getElementById(param);
            const display = document.getElementById(`${param}-value`);
            if (input && display) {
                display.textContent = input.value;
            }
        });
    }
    
    getSimulationParameters() {
        return {
            initial_temp_hot: parseFloat(document.getElementById('thermal-capacitance').value),
            initial_temp_cold: parseFloat(document.getElementById('mass-flow').value),
            heat_input: parseFloat(document.getElementById('heat-input').value),
            mass_flow: parseFloat(document.getElementById('mass-flow').value),
            heat_capacity: parseFloat(document.getElementById('heat-capacity').value),
            heat_transfer_coeff: parseFloat(document.getElementById('heat-exchanger').value)
        };
    }
    
    async startSimulation() {
        if (this.isSimulationRunning) return;
        
        try {
            const params = this.getSimulationParameters();
            const result = await this.apiStartSimulation(params);
            
            if (result.status === 'success') {
                this.isSimulationRunning = true;
                document.getElementById('start-simulation').disabled = true;
                document.getElementById('stop-simulation').disabled = false;
                this.showNotification('Simulation started successfully', 'success');
            } else {
                this.showNotification(`Failed to start simulation: ${result.message}`, 'error');
            }
        } catch (error) {
            console.error('Error starting simulation:', error);
            this.showNotification('Failed to connect to backend', 'error');
        }
    }
    
    async stopSimulation() {
        if (!this.isSimulationRunning) return;
        
        try {
            const result = await this.apiStopSimulation();
            
            if (result.status === 'success') {
                this.isSimulationRunning = false;
                document.getElementById('start-simulation').disabled = false;
                document.getElementById('stop-simulation').disabled = true;
                this.showNotification('Simulation stopped', 'info');
            } else {
                this.showNotification(`Failed to stop simulation: ${result.message}`, 'error');
            }
        } catch (error) {
            console.error('Error stopping simulation:', error);
            this.showNotification('Failed to connect to backend', 'error');
        }
    }
    
    resetSimulation() {
        this.stopSimulation();
        
        // Reset parameters to defaults
        document.getElementById('thermal-capacitance').value = 1000;
        document.getElementById('mass-flow').value = 0.1;
        document.getElementById('heat-capacity').value = 4180;
        document.getElementById('heat-exchanger').value = 50;
        document.getElementById('heat-input').value = 1000;
        document.getElementById('heat-output').value = 1000;
        
        this.updateParameterDisplays();
        
        // Clear charts
        this.clearCharts();
        
        console.log('Simulation reset');
    }
    
    clearCharts() {
        if (charts.temperature) {
            charts.temperature.data.labels = [];
            charts.temperature.data.datasets.forEach(dataset => dataset.data = []);
            charts.temperature.update();
        }
        
        if (charts.residuals) {
            charts.residuals.data.labels = [];
            charts.residuals.data.datasets.forEach(dataset => dataset.data = []);
            charts.residuals.update();
        }
    }
    
    initializeCharts() {
        // Initialize temperature chart
        charts.temperature = this.createTemperatureChart();
        
        // Initialize residuals chart
        charts.residuals = this.createResidualsChart();
    }
    
    createTemperatureChart() {
        const ctx = document.getElementById('temperature-chart');
        if (!ctx) return null;
        
        return new Chart(ctx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Plant T_hot',
                        data: [],
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        tension: 0.1
                    },
                    {
                        label: 'Plant T_cold',
                        data: [],
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.1
                    },
                    {
                        label: 'Twin T_hot',
                        data: [],
                        borderColor: '#e74c3c',
                        borderDash: [5, 5],
                        backgroundColor: 'transparent',
                        tension: 0.1
                    },
                    {
                        label: 'Twin T_cold',
                        data: [],
                        borderColor: '#3498db',
                        borderDash: [5, 5],
                        backgroundColor: 'transparent',
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'category',
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Temperature (K)'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Temperature vs Time'
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });
    }
    
    createResidualsChart() {
        const ctx = document.getElementById('residuals-chart');
        if (!ctx) return null;
        
        return new Chart(ctx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Residual T_hot',
                        data: [],
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        tension: 0.1
                    },
                    {
                        label: 'Residual T_cold',
                        data: [],
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.1
                    },
                    {
                        label: 'Residual m_dot',
                        data: [],
                        borderColor: '#27ae60',
                        backgroundColor: 'rgba(39, 174, 96, 0.1)',
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'category',
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Residual Value'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Residuals vs Time'
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });
    }
    
    initializeMonitoring() {
        if (!this.monitoring) {
            this.monitoring = new MonitoringManager();
        }
        this.monitoring.initialize();
    }
    
    initializeTesting() {
        if (!this.testing) {
            this.testing = new TestingManager();
        }
        this.testing.initialize();
    }
    
    // API functions
    async apiStartSimulation(params) {
        try {
            const response = await fetch(`${BACKEND_URL}/api/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(params)
            });
            
            return await response.json();
        } catch (error) {
            console.error('Error starting simulation:', error);
            return { status: 'error', message: error.message };
        }
    }

    async apiStopSimulation() {
        try {
            const response = await fetch(`${BACKEND_URL}/api/stop`, {
                method: 'POST'
            });
            
            return await response.json();
        } catch (error) {
            console.error('Error stopping simulation:', error);
            return { status: 'error', message: error.message };
        }
    }

    async apiInjectFault(faultData) {
        try {
            const response = await fetch(`${BACKEND_URL}/api/fault`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(faultData)
            });
            
            return await response.json();
        } catch (error) {
            console.error('Error injecting fault:', error);
            return { status: 'error', message: error.message };
        }
    }
    
    // Utility methods
    formatTime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    
    formatNumber(value, decimals = 2) {
        return parseFloat(value).toFixed(decimals);
    }
}

// Global function for tab switching (called from HTML)
function showTab(tabName) {
    if (window.thermalApp) {
        window.thermalApp.switchTab(tabName);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.thermalApp = new ThermalCoolingApp();
});