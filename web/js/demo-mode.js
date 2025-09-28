/**
 * Demo mode for the web client when backend is not available
 * This provides a simulated experience for demonstration purposes
 */

class DemoMode {
    constructor() {
        this.isRunning = false;
        this.simulationData = [];
        this.timeStep = 0;
        this.intervalId = null;
        this.charts = {};
        
        this.init();
    }
    
    init() {
        console.log('Demo mode initialized - simulating backend responses');
        this.setupDemoData();
        this.updateConnectionStatus(false, 'Demo Mode');
    }
    
    setupDemoData() {
        // Generate realistic simulation data
        this.baseTempHot = 300;
        this.baseTempCold = 280;
        this.baseMassFlow = 0.1;
        this.timeStep = 0;
        
        // Simulate some realistic thermal behavior
        this.heatInput = 1000;
        this.thermalCapacitance = 1000;
        this.massFlow = 0.1;
        this.heatCapacity = 4180;
        this.heatTransferCoeff = 50;
    }
    
    updateConnectionStatus(connected, mode = 'Connected') {
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            statusElement.textContent = mode;
            statusElement.className = connected ? 'status-connected' : 'status-disconnected';
        }
    }
    
    startSimulation() {
        if (this.isRunning) return;
        
        console.log('Starting demo simulation...');
        this.isRunning = true;
        this.timeStep = 0;
        this.simulationData = [];
        
        // Update UI
        document.getElementById('start-simulation').disabled = true;
        document.getElementById('stop-simulation').disabled = false;
        
        // Start simulation loop
        this.intervalId = setInterval(() => {
            this.simulateStep();
        }, 1000); // Update every second
        
        this.showNotification('Demo simulation started', 'success');
    }
    
    stopSimulation() {
        if (!this.isRunning) return;
        
        console.log('Stopping demo simulation...');
        this.isRunning = false;
        
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
        
        // Update UI
        document.getElementById('start-simulation').disabled = false;
        document.getElementById('stop-simulation').disabled = true;
        
        this.showNotification('Demo simulation stopped', 'info');
    }
    
    simulateStep() {
        this.timeStep++;
        
        // Simulate realistic thermal dynamics
        const dt = 1.0; // 1 second time step
        
        // Add some noise and realistic behavior
        const noise = (Math.random() - 0.5) * 0.5;
        const heatInputVariation = Math.sin(this.timeStep * 0.1) * 50;
        
        // Simple thermal model simulation
        const currentHeatInput = this.heatInput + heatInputVariation;
        
        // Temperature evolution (simplified)
        const tempHot = this.baseTempHot + noise + (currentHeatInput / this.thermalCapacitance) * 0.1;
        const tempCold = this.baseTempCold + noise * 0.5 + (tempHot - this.baseTempCold) * 0.1;
        
        // Mass flow with some variation
        const massFlow = this.baseMassFlow + (Math.random() - 0.5) * 0.01;
        
        // Create "twin" predictions (slightly different from plant)
        const twinTempHot = tempHot + (Math.random() - 0.5) * 0.2;
        const twinTempCold = tempCold + (Math.random() - 0.5) * 0.2;
        
        // Calculate residuals
        const residualHot = tempHot - twinTempHot;
        const residualCold = tempCold - twinTempCold;
        const residualMassFlow = massFlow - this.baseMassFlow;
        
        // Simulate anomaly detection
        const anomalyThreshold = 2.0;
        const isAnomaly = Math.abs(residualHot) > anomalyThreshold || 
                         Math.abs(residualCold) > anomalyThreshold ||
                         Math.abs(residualMassFlow) > 0.005;
        
        const data = {
            timestamp: this.timeStep,
            plant_T_hot: tempHot,
            plant_T_cold: tempCold,
            plant_m_dot: massFlow,
            twin_T_hot: twinTempHot,
            twin_T_cold: twinTempCold,
            residual_T_hot: residualHot,
            residual_T_cold: residualCold,
            residual_m_dot: residualMassFlow,
            overall_anomaly: isAnomaly,
            overall_severity: isAnomaly ? 'WARNING' : 'NORMAL',
            anomaly_confidence: isAnomaly ? 0.8 : 0.1,
            anomaly_method: 'Demo Simulation'
        };
        
        this.simulationData.push(data);
        this.updateRealTimeData(data);
        
        // Simulate occasional anomalies
        if (Math.random() < 0.05 && this.timeStep > 10) { // 5% chance after 10 seconds
            this.simulateAnomaly();
        }
    }
    
    simulateAnomaly() {
        const anomalyTypes = [
            'PUMP_DEGRADATION',
            'HEAT_EXCHANGER_FOULING', 
            'SENSOR_BIAS',
            'MASS_FLOW_REDUCTION'
        ];
        
        const anomalyType = anomalyTypes[Math.floor(Math.random() * anomalyTypes.length)];
        
        const anomaly = {
            timestamp: this.timeStep,
            severity: 'WARNING',
            details: {
                type: anomalyType,
                description: `Simulated ${anomalyType.replace('_', ' ').toLowerCase()}`,
                confidence: 0.85
            }
        };
        
        this.handleAnomalyEvent(anomaly);
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
        if (window.thermalApp && window.thermalApp.charts && window.thermalApp.charts.temperature) {
            const chart = window.thermalApp.charts.temperature;
            chart.data.labels.push(new Date().toLocaleTimeString());
            chart.data.datasets[0].data.push(data.plant_T_hot);
            chart.data.datasets[1].data.push(data.plant_T_cold);
            chart.data.datasets[2].data.push(data.twin_T_hot);
            chart.data.datasets[3].data.push(data.twin_T_cold);
            
            // Keep only last 50 points
            if (chart.data.labels.length > 50) {
                chart.data.labels.shift();
                chart.data.datasets.forEach(dataset => dataset.data.shift());
            }
            
            chart.update('none');
        }
        
        // Update residuals chart
        if (window.thermalApp && window.thermalApp.charts && window.thermalApp.charts.residuals) {
            const chart = window.thermalApp.charts.residuals;
            chart.data.labels.push(new Date().toLocaleTimeString());
            chart.data.datasets[0].data.push(data.residual_T_hot);
            chart.data.datasets[1].data.push(data.residual_T_cold);
            chart.data.datasets[2].data.push(data.residual_m_dot);
            
            // Keep only last 50 points
            if (chart.data.labels.length > 50) {
                chart.data.labels.shift();
                chart.data.datasets.forEach(dataset => dataset.data.shift());
            }
            
            chart.update('none');
        }
    }
    
    handleAnomalyEvent(anomaly) {
        console.log('Demo anomaly detected:', anomaly);
        
        // Add to anomaly list
        this.addAnomalyToList(anomaly);
        
        // Show notification
        this.showNotification(`Demo anomaly detected: ${anomaly.severity}`, 'warning');
    }
    
    addAnomalyToList(anomaly) {
        const anomalyList = document.getElementById('anomaly-list');
        if (anomalyList) {
            const anomalyItem = document.createElement('div');
            anomalyItem.className = 'anomaly-item';
            anomalyItem.innerHTML = `
                <div class="anomaly-time">${new Date().toLocaleTimeString()}</div>
                <div class="anomaly-severity">${anomaly.severity}</div>
                <div class="anomaly-details">${anomaly.details.description}</div>
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
    
    // API simulation methods
    async startSimulation(params) {
        console.log('Demo: Starting simulation with params:', params);
        this.startSimulation();
        return { status: 'success', message: 'Demo simulation started' };
    }
    
    async stopSimulation() {
        console.log('Demo: Stopping simulation');
        this.stopSimulation();
        return { status: 'success', message: 'Demo simulation stopped' };
    }
    
    async injectFault(faultData) {
        console.log('Demo: Injecting fault:', faultData);
        this.simulateAnomaly();
        return { status: 'success', message: `Demo fault ${faultData.type} injected` };
    }
}

// Initialize demo mode when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Check if we're in demo mode (no backend connection)
    setTimeout(() => {
        const statusElement = document.getElementById('connection-status');
        if (statusElement && statusElement.textContent === 'Disconnected') {
            console.log('Backend not available, enabling demo mode');
            window.demoMode = new DemoMode();
            
            // Override the main app methods with demo versions
            if (window.thermalApp) {
                window.thermalApp.startSimulation = window.demoMode.startSimulation.bind(window.demoMode);
                window.thermalApp.stopSimulation = window.demoMode.stopSimulation.bind(window.demoMode);
                window.thermalApp.apiInjectFault = window.demoMode.injectFault.bind(window.demoMode);
            }
        }
    }, 2000); // Wait 2 seconds for backend connection attempt
});
