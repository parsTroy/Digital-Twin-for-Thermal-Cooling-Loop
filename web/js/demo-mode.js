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
        
        // Get current parameters from the UI
        this.updateParametersFromUI();
        
        // Simulate realistic thermal dynamics
        const dt = 1.0; // 1 second time step
        
        // Add some noise and realistic behavior
        const noise = (Math.random() - 0.5) * 0.5;
        const heatInputVariation = Math.sin(this.timeStep * 0.1) * 50;
        
        // Simple thermal model simulation with more realistic physics
        const currentHeatInput = this.heatInput + heatInputVariation;
        
        // Temperature evolution (more realistic thermal model)
        const heatTransfer = this.heatTransferCoeff * (this.baseTempHot - this.baseTempCold);
        const massFlowEffect = this.massFlow * this.heatCapacity * (this.baseTempHot - this.baseTempCold);
        
        // Update temperatures based on thermal model
        const dT_hot = (currentHeatInput - heatTransfer - massFlowEffect) / this.thermalCapacitance;
        const dT_cold = (massFlowEffect - this.heatOutput) / this.thermalCapacitance;
        
        this.baseTempHot += dT_hot * dt + noise;
        this.baseTempCold += dT_cold * dt + noise * 0.5;
        
        // Mass flow with some variation
        const massFlow = this.baseMassFlow + (Math.random() - 0.5) * 0.01;
        
        // Create "twin" predictions (slightly different from plant due to model uncertainty)
        const modelUncertainty = 0.1; // 10% uncertainty
        const twinTempHot = this.baseTempHot + (Math.random() - 0.5) * modelUncertainty * this.baseTempHot;
        const twinTempCold = this.baseTempCold + (Math.random() - 0.5) * modelUncertainty * this.baseTempCold;
        
        // Calculate residuals
        const residualHot = this.baseTempHot - twinTempHot;
        const residualCold = this.baseTempCold - twinTempCold;
        const residualMassFlow = massFlow - this.baseMassFlow;
        
        // Simulate anomaly detection with more sophisticated logic
        const hotThreshold = 2.0 + Math.abs(this.baseTempHot) * 0.01; // Adaptive threshold
        const coldThreshold = 1.5 + Math.abs(this.baseTempCold) * 0.01;
        const massFlowThreshold = 0.005;
        
        const isAnomaly = Math.abs(residualHot) > hotThreshold || 
                         Math.abs(residualCold) > coldThreshold ||
                         Math.abs(residualMassFlow) > massFlowThreshold;
        
        // Determine severity
        let severity = 'NORMAL';
        let confidence = 0.1;
        
        if (isAnomaly) {
            const maxResidual = Math.max(Math.abs(residualHot), Math.abs(residualCold), Math.abs(residualMassFlow) * 100);
            if (maxResidual > hotThreshold * 2) {
                severity = 'CRITICAL';
                confidence = 0.9;
            } else if (maxResidual > hotThreshold * 1.5) {
                severity = 'WARNING';
                confidence = 0.7;
            } else {
                severity = 'WARNING';
                confidence = 0.5;
            }
        }
        
        const data = {
            timestamp: this.timeStep,
            plant_T_hot: this.baseTempHot,
            plant_T_cold: this.baseTempCold,
            plant_m_dot: massFlow,
            twin_T_hot: twinTempHot,
            twin_T_cold: twinTempCold,
            residual_T_hot: residualHot,
            residual_T_cold: residualCold,
            residual_m_dot: residualMassFlow,
            overall_anomaly: isAnomaly,
            overall_severity: severity,
            anomaly_confidence: confidence,
            anomaly_method: 'Demo Simulation'
        };
        
        this.simulationData.push(data);
        this.updateRealTimeData(data);
        
        // Simulate occasional anomalies
        if (Math.random() < 0.03 && this.timeStep > 20) { // 3% chance after 20 seconds
            this.simulateAnomaly();
        }
    }
    
    updateParametersFromUI() {
        // Get current parameters from the UI sliders
        const thermalCapElement = document.getElementById('thermal-capacitance');
        const massFlowElement = document.getElementById('mass-flow');
        const heatCapacityElement = document.getElementById('heat-capacity');
        const heatExchangerElement = document.getElementById('heat-exchanger');
        const heatInputElement = document.getElementById('heat-input');
        const heatOutputElement = document.getElementById('heat-output');
        
        if (thermalCapElement) this.thermalCapacitance = parseFloat(thermalCapElement.value);
        if (massFlowElement) this.baseMassFlow = parseFloat(massFlowElement.value);
        if (heatCapacityElement) this.heatCapacity = parseFloat(heatCapacityElement.value);
        if (heatExchangerElement) this.heatTransferCoeff = parseFloat(heatExchangerElement.value);
        if (heatInputElement) this.heatInput = parseFloat(heatInputElement.value);
        if (heatOutputElement) this.heatOutput = parseFloat(heatOutputElement.value);
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
        if (typeof charts !== 'undefined' && charts.temperature) {
            const chart = charts.temperature;
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
        if (typeof charts !== 'undefined' && charts.residuals) {
            const chart = charts.residuals;
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
        this.startSimulationInternal();
        return { status: 'success', message: 'Demo simulation started' };
    }
    
    async stopSimulation() {
        console.log('Demo: Stopping simulation');
        this.stopSimulationInternal();
        return { status: 'success', message: 'Demo simulation stopped' };
    }
    
    startSimulationInternal() {
        if (this.isRunning) return;
        
        this.isRunning = true;
        this.timeStep = 0;
        this.simulationData = [];
        
        // Start simulation loop
        this.intervalId = setInterval(() => {
            this.simulateStep();
        }, 1000); // Update every second
        
        console.log('Demo simulation started internally');
    }
    
    stopSimulationInternal() {
        if (!this.isRunning) return;
        
        this.isRunning = false;
        
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
        
        console.log('Demo simulation stopped internally');
    }
    
    async injectFault(faultData) {
        console.log('Demo: Injecting fault:', faultData);
        this.simulateFault(faultData);
        return { status: 'success', message: `Demo fault ${faultData.type} injected` };
    }
    
    simulateFault(faultData) {
        const faultType = faultData.type || 'PUMP_DEGRADATION';
        const startTime = faultData.start_time || 0;
        const params = faultData.params || {};
        
        console.log(`Simulating fault: ${faultType} at time ${startTime}`);
        
        // Apply fault effects to the simulation
        switch (faultType.toUpperCase()) {
            case 'PUMP_DEGRADATION':
                const reductionFactor = params.reduction_factor || 0.3;
                this.baseMassFlow *= (1 - reductionFactor);
                this.showNotification(`Pump degradation: ${(reductionFactor * 100).toFixed(0)}% flow reduction`, 'warning');
                break;
                
            case 'HEAT_EXCHANGER_FOULING':
                const foulingFactor = params.fouling_factor || 0.5;
                this.heatTransferCoeff *= (1 - foulingFactor);
                this.showNotification(`Heat exchanger fouling: ${(foulingFactor * 100).toFixed(0)}% efficiency loss`, 'warning');
                break;
                
            case 'SENSOR_BIAS':
                const biasAmount = params.bias_amount || 5.0;
                this.baseTempHot += biasAmount;
                this.showNotification(`Sensor bias: +${biasAmount}K temperature offset`, 'warning');
                break;
                
            case 'MASS_FLOW_REDUCTION':
                const flowReduction = params.flow_reduction || 0.2;
                this.baseMassFlow *= (1 - flowReduction);
                this.showNotification(`Mass flow reduction: ${(flowReduction * 100).toFixed(0)}% flow decrease`, 'warning');
                break;
                
            default:
                this.showNotification(`Unknown fault type: ${faultType}`, 'error');
        }
        
        // Trigger an immediate anomaly
        setTimeout(() => {
            this.simulateAnomaly();
        }, 1000);
    }
}

// Demo mode will be initialized by the main app when needed
