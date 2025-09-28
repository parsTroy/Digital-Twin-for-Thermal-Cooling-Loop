/**
 * Main JavaScript for Thermal Cooling Loop Digital Twin
 * Handles navigation, initialization, and core functionality
 */

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
        console.log('Thermal Cooling Loop Digital Twin initialized');
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
                if (this.isSimulationRunning) {
                    this.updateSimulationParameters();
                }
            });
        });
        
        // Simulation buttons
        document.getElementById('start-simulation').addEventListener('click', () => {
            this.startSimulation();
        });
        
        document.getElementById('stop-simulation').addEventListener('click', () => {
            this.stopSimulation();
        });
        
        document.getElementById('reset-simulation').addEventListener('click', () => {
            this.resetSimulation();
        });
        
        // Documentation navigation
        document.querySelectorAll('.doc-nav-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchDocSection(e.target.dataset.doc);
            });
        });
    }
    
    switchTab(tabName) {
        // Update navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
        
        // Update content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(tabName).classList.add('active');
        
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
            C: parseFloat(document.getElementById('thermal-capacitance').value),
            m_dot: parseFloat(document.getElementById('mass-flow').value),
            cp: parseFloat(document.getElementById('heat-capacity').value),
            UA: parseFloat(document.getElementById('heat-exchanger').value),
            Q_in: parseFloat(document.getElementById('heat-input').value),
            Q_out: parseFloat(document.getElementById('heat-output').value)
        };
    }
    
    startSimulation() {
        if (this.isSimulationRunning) return;
        
        this.isSimulationRunning = true;
        document.getElementById('start-simulation').disabled = true;
        document.getElementById('stop-simulation').disabled = false;
        
        // Initialize simulation if not already done
        if (!this.simulation) {
            this.simulation = new SimulationManager();
        }
        
        this.simulation.start(this.getSimulationParameters());
        
        console.log('Simulation started');
    }
    
    stopSimulation() {
        if (!this.isSimulationRunning) return;
        
        this.isSimulationRunning = false;
        document.getElementById('start-simulation').disabled = false;
        document.getElementById('stop-simulation').disabled = true;
        
        if (this.simulation) {
            this.simulation.stop();
        }
        
        console.log('Simulation stopped');
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
        if (this.simulation) {
            this.simulation.reset();
        }
        
        console.log('Simulation reset');
    }
    
    updateSimulationParameters() {
        if (this.simulation && this.isSimulationRunning) {
            this.simulation.updateParameters(this.getSimulationParameters());
        }
    }
    
    initializeCharts() {
        // Initialize temperature chart
        this.temperatureChart = this.createTemperatureChart();
        
        // Initialize flow chart
        this.flowChart = this.createFlowChart();
    }
    
    createTemperatureChart() {
        const ctx = document.getElementById('temperature-chart').getContext('2d');
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Hot Temperature (K)',
                        data: [],
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        tension: 0.1
                    },
                    {
                        label: 'Cold Temperature (K)',
                        data: [],
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.1
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
    
    createFlowChart() {
        const ctx = document.getElementById('flow-chart').getContext('2d');
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Mass Flow Rate (kg/s)',
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
                            text: 'Mass Flow Rate (kg/s)'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Mass Flow Rate vs Time'
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

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.thermalApp = new ThermalCoolingApp();
});
