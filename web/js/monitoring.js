/**
 * Monitoring Manager for Thermal Cooling Loop Digital Twin
 * Handles real-time monitoring, anomaly detection, and status updates
 */

class MonitoringManager {
    constructor() {
        this.isInitialized = false;
        this.monitoringChart = null;
        this.residualChart = null;
        this.anomalies = [];
        this.maxAnomalies = 100;
        this.residualThreshold = 2.0; // Z-score threshold
        this.residualWindow = 50; // Rolling window size
        this.residuals = {
            T_hot: [],
            T_cold: [],
            m_dot: []
        };
        this.baseline = {
            T_hot: 0,
            T_cold: 0,
            m_dot: 0
        };
    }
    
    initialize() {
        if (this.isInitialized) return;
        
        this.setupEventListeners();
        this.initializeCharts();
        this.startMonitoring();
        
        this.isInitialized = true;
        console.log('Monitoring manager initialized');
    }
    
    setupEventListeners() {
        // Add any monitoring-specific event listeners here
    }
    
    initializeCharts() {
        this.createMonitoringChart();
        this.createResidualChart();
    }
    
    createMonitoringChart() {
        const ctx = document.getElementById('monitoring-chart').getContext('2d');
        this.monitoringChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Hot Temperature (K)',
                        data: [],
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        tension: 0.1,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Cold Temperature (K)',
                        data: [],
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.1,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Mass Flow Rate (kg/s)',
                        data: [],
                        borderColor: '#27ae60',
                        backgroundColor: 'rgba(39, 174, 96, 0.1)',
                        tension: 0.1,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: 'Time (s)'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Temperature (K)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Mass Flow Rate (kg/s)'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Real-time System Monitoring'
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });
    }
    
    createResidualChart() {
        const ctx = document.getElementById('residual-chart').getContext('2d');
        this.residualChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'T_hot Residual',
                        data: [],
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        tension: 0.1
                    },
                    {
                        label: 'T_cold Residual',
                        data: [],
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.1
                    },
                    {
                        label: 'm_dot Residual',
                        data: [],
                        borderColor: '#27ae60',
                        backgroundColor: 'rgba(39, 174, 96, 0.1)',
                        tension: 0.1
                    },
                    {
                        label: 'Threshold',
                        data: [],
                        borderColor: '#f39c12',
                        backgroundColor: 'rgba(243, 156, 18, 0.1)',
                        borderDash: [5, 5],
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: 'Time (s)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Residual (Z-score)'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Anomaly Detection Residuals'
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });
    }
    
    startMonitoring() {
        // Start monitoring loop
        this.monitoringInterval = setInterval(() => {
            this.updateMonitoring();
        }, 1000); // Update every second
        
        console.log('Monitoring started');
    }
    
    stopMonitoring() {
        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
            this.monitoringInterval = null;
        }
        console.log('Monitoring stopped');
    }
    
    updateMonitoring() {
        if (!window.thermalApp || !window.thermalApp.simulation) return;
        
        const state = window.thermalApp.simulation.getCurrentState();
        this.updateStatusDisplay(state);
        this.updateCharts(state);
        this.performAnomalyDetection(state);
    }
    
    updateStatusDisplay(state) {
        // Update status values
        document.getElementById('hot-temp-value').textContent = 
            `${state.T_hot.toFixed(1)} K`;
        document.getElementById('cold-temp-value').textContent = 
            `${state.T_cold.toFixed(1)} K`;
        document.getElementById('flow-value').textContent = 
            `${state.m_dot.toFixed(3)} kg/s`;
        
        // Update trends (simplified)
        this.updateTrends(state);
        
        // Update system status
        this.updateSystemStatus(state);
    }
    
    updateTrends(state) {
        // Simple trend calculation (could be more sophisticated)
        const trends = this.calculateTrends(state);
        
        document.getElementById('hot-temp-trend').textContent = trends.T_hot;
        document.getElementById('cold-temp-trend').textContent = trends.T_cold;
        document.getElementById('flow-trend').textContent = trends.m_dot;
    }
    
    calculateTrends(state) {
        // This is a simplified trend calculation
        // In a real implementation, you'd analyze historical data
        const trends = {
            T_hot: 'Stable',
            T_cold: 'Stable',
            m_dot: 'Stable'
        };
        
        // Simple logic for demonstration
        if (state.T_hot > 350) trends.T_hot = 'Rising';
        if (state.T_cold < 280) trends.T_cold = 'Falling';
        if (state.m_dot < 0.08) trends.m_dot = 'Decreasing';
        
        return trends;
    }
    
    updateSystemStatus(state) {
        const status = this.determineSystemStatus(state);
        
        document.getElementById('system-status').textContent = status.text;
        const indicator = document.getElementById('status-indicator');
        indicator.className = `status-indicator ${status.type}`;
    }
    
    determineSystemStatus(state) {
        // Simple status determination logic
        if (state.T_hot > 400 || state.T_cold < 250) {
            return { text: 'Warning', type: 'warning' };
        } else if (state.T_hot > 450 || state.T_cold < 200) {
            return { text: 'Critical', type: 'error' };
        } else {
            return { text: 'Normal', type: 'normal' };
        }
    }
    
    updateCharts(state) {
        if (!this.monitoringChart || !this.residualChart) return;
        
        const time = state.time;
        
        // Update monitoring chart
        this.monitoringChart.data.labels.push(time.toFixed(1));
        this.monitoringChart.data.datasets[0].data.push({
            x: time,
            y: state.T_hot
        });
        this.monitoringChart.data.datasets[1].data.push({
            x: time,
            y: state.T_cold
        });
        this.monitoringChart.data.datasets[2].data.push({
            x: time,
            y: state.m_dot
        });
        
        // Limit data points
        if (this.monitoringChart.data.labels.length > 200) {
            this.monitoringChart.data.labels.shift();
            this.monitoringChart.data.datasets.forEach(dataset => {
                dataset.data.shift();
            });
        }
        
        this.monitoringChart.update('none');
        
        // Update residual chart
        this.updateResidualChart(time);
    }
    
    updateResidualChart(time) {
        if (!this.residualChart) return;
        
        // Calculate residuals (simplified)
        const residuals = this.calculateResiduals();
        
        this.residualChart.data.labels.push(time.toFixed(1));
        this.residualChart.data.datasets[0].data.push({
            x: time,
            y: residuals.T_hot
        });
        this.residualChart.data.datasets[1].data.push({
            x: time,
            y: residuals.T_cold
        });
        this.residualChart.data.datasets[2].data.push({
            x: time,
            y: residuals.m_dot
        });
        this.residualChart.data.datasets[3].data.push({
            x: time,
            y: this.residualThreshold
        });
        
        // Limit data points
        if (this.residualChart.data.labels.length > 200) {
            this.residualChart.data.labels.shift();
            this.residualChart.data.datasets.forEach(dataset => {
                dataset.data.shift();
            });
        }
        
        this.residualChart.update('none');
    }
    
    calculateResiduals() {
        // Simplified residual calculation
        // In a real implementation, you'd compare with digital twin predictions
        const residuals = {
            T_hot: (Math.random() - 0.5) * 4, // Simulated residual
            T_cold: (Math.random() - 0.5) * 4,
            m_dot: (Math.random() - 0.5) * 4
        };
        
        // Store residuals for analysis
        this.residuals.T_hot.push(residuals.T_hot);
        this.residuals.T_cold.push(residuals.T_cold);
        this.residuals.m_dot.push(residuals.m_dot);
        
        // Limit stored residuals
        if (this.residuals.T_hot.length > this.residualWindow) {
            this.residuals.T_hot.shift();
            this.residuals.T_cold.shift();
            this.residuals.m_dot.shift();
        }
        
        return residuals;
    }
    
    performAnomalyDetection(state) {
        const residuals = this.calculateResiduals();
        
        // Check for anomalies
        Object.keys(residuals).forEach(sensor => {
            const residual = Math.abs(residuals[sensor]);
            if (residual > this.residualThreshold) {
                this.addAnomaly(sensor, residual, state.time);
            }
        });
        
        this.updateAnomalyDisplay();
    }
    
    addAnomaly(sensor, residual, time) {
        const anomaly = {
            time: time,
            sensor: sensor,
            residual: residual,
            severity: this.determineSeverity(residual),
            timestamp: new Date()
        };
        
        this.anomalies.unshift(anomaly);
        
        // Limit stored anomalies
        if (this.anomalies.length > this.maxAnomalies) {
            this.anomalies.pop();
        }
        
        console.log('Anomaly detected:', anomaly);
    }
    
    determineSeverity(residual) {
        if (residual > this.residualThreshold * 2) {
            return 'error';
        } else if (residual > this.residualThreshold * 1.5) {
            return 'warning';
        } else {
            return 'normal';
        }
    }
    
    updateAnomalyDisplay() {
        const anomalyList = document.getElementById('anomaly-list');
        if (!anomalyList) return;
        
        // Clear existing anomalies
        anomalyList.innerHTML = '';
        
        if (this.anomalies.length === 0) {
            const noAnomalyItem = document.createElement('div');
            noAnomalyItem.className = 'anomaly-item';
            noAnomalyItem.innerHTML = `
                <span class="anomaly-time">--:--:--</span>
                <span class="anomaly-type">No anomalies detected</span>
                <span class="anomaly-severity normal">Normal</span>
            `;
            anomalyList.appendChild(noAnomalyItem);
            return;
        }
        
        // Display recent anomalies
        this.anomalies.slice(0, 10).forEach(anomaly => {
            const anomalyItem = document.createElement('div');
            anomalyItem.className = `anomaly-item ${anomaly.severity}`;
            anomalyItem.innerHTML = `
                <span class="anomaly-time">${this.formatTime(anomaly.time)}</span>
                <span class="anomaly-type">${anomaly.sensor} residual: ${anomaly.residual.toFixed(2)}</span>
                <span class="anomaly-severity ${anomaly.severity}">${anomaly.severity.toUpperCase()}</span>
            `;
            anomalyList.appendChild(anomalyItem);
        });
    }
    
    formatTime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    
    // Configuration methods
    setResidualThreshold(threshold) {
        this.residualThreshold = threshold;
        console.log('Residual threshold updated:', threshold);
    }
    
    setResidualWindow(windowSize) {
        this.residualWindow = windowSize;
        console.log('Residual window updated:', windowSize);
    }
    
    // Export methods
    exportAnomalies() {
        const csv = this.convertAnomaliesToCSV();
        this.downloadCSV(csv, 'anomalies.csv');
    }
    
    convertAnomaliesToCSV() {
        const headers = ['Timestamp', 'Time (s)', 'Sensor', 'Residual', 'Severity'];
        const rows = [headers.join(',')];
        
        this.anomalies.forEach(anomaly => {
            const row = [
                anomaly.timestamp.toISOString(),
                anomaly.time.toFixed(3),
                anomaly.sensor,
                anomaly.residual.toFixed(3),
                anomaly.severity
            ];
            rows.push(row.join(','));
        });
        
        return rows.join('\n');
    }
    
    downloadCSV(csv, filename) {
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    }
}
