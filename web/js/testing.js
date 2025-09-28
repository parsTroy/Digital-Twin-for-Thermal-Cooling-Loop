/**
 * Testing Manager for Thermal Cooling Loop Digital Twin
 * Handles fault injection, test scenarios, and result analysis
 */

class TestingManager {
    constructor() {
        this.isInitialized = false;
        this.testResults = [];
        this.currentTest = null;
        this.testChart = null;
        this.scenarios = this.createTestScenarios();
        this.activeFaults = [];
    }
    
    initialize() {
        if (this.isInitialized) return;
        
        this.setupEventListeners();
        this.initializeCharts();
        this.updateTestResults();
        
        this.isInitialized = true;
        console.log('Testing manager initialized');
    }
    
    setupEventListeners() {
        // Scenario buttons
        document.querySelectorAll('.scenario-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.runScenario(e.target.dataset.scenario);
            });
        });
        
        // Fault injection
        document.getElementById('inject-fault').addEventListener('click', () => {
            this.injectFault();
        });
        
        document.getElementById('clear-faults').addEventListener('click', () => {
            this.clearAllFaults();
        });
    }
    
    initializeCharts() {
        this.createTestResultsChart();
    }
    
    createTestResultsChart() {
        const ctx = document.getElementById('test-results-chart').getContext('2d');
        this.testChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Passed', 'Failed', 'Warning'],
                datasets: [{
                    label: 'Test Results',
                    data: [0, 0, 0],
                    backgroundColor: [
                        'rgba(39, 174, 96, 0.8)',
                        'rgba(231, 76, 60, 0.8)',
                        'rgba(243, 156, 18, 0.8)'
                    ],
                    borderColor: [
                        'rgba(39, 174, 96, 1)',
                        'rgba(231, 76, 60, 1)',
                        'rgba(243, 156, 18, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Tests'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Test Results Summary'
                    },
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
    
    createTestScenarios() {
        return {
            'nominal': {
                name: 'Nominal Operation',
                description: 'Normal operation with no faults',
                duration: 100,
                faults: []
            },
            'pump-degradation': {
                name: 'Pump Degradation',
                description: 'Gradual pump degradation over time',
                duration: 200,
                faults: [{
                    type: 'pump-degradation',
                    startTime: 50,
                    parameters: { degradationRate: 0.02 }
                }]
            },
            'heat-exchanger-fouling': {
                name: 'Heat Exchanger Fouling',
                description: 'Gradual heat exchanger fouling',
                duration: 300,
                faults: [{
                    type: 'heat-exchanger-fouling',
                    startTime: 100,
                    parameters: { foulingRate: 0.01 }
                }]
            },
            'sensor-bias': {
                name: 'Sensor Bias',
                description: 'Temperature sensor bias fault',
                duration: 150,
                faults: [{
                    type: 'sensor-bias',
                    startTime: 75,
                    parameters: { biasMagnitude: 10.0 }
                }]
            },
            'sensor-noise': {
                name: 'Sensor Noise Increase',
                description: 'Increased sensor noise',
                duration: 180,
                faults: [{
                    type: 'sensor-noise',
                    startTime: 90,
                    parameters: { noiseMultiplier: 2.5 }
                }]
            },
            'mass-flow-reduction': {
                name: 'Mass Flow Reduction',
                description: 'Sudden reduction in mass flow rate',
                duration: 120,
                faults: [{
                    type: 'mass-flow-reduction',
                    startTime: 60,
                    parameters: { reductionFactor: 0.5 }
                }]
            },
            'multiple-faults': {
                name: 'Multiple Faults',
                description: 'Combination of pump degradation and sensor noise',
                duration: 250,
                faults: [
                    {
                        type: 'pump-degradation',
                        startTime: 50,
                        parameters: { degradationRate: 0.015 }
                    },
                    {
                        type: 'sensor-noise',
                        startTime: 150,
                        parameters: { noiseMultiplier: 2.0 }
                    }
                ]
            }
        };
    }
    
    runScenario(scenarioName) {
        const scenario = this.scenarios[scenarioName];
        if (!scenario) {
            console.error('Unknown scenario:', scenarioName);
            return;
        }
        
        console.log('Running scenario:', scenario.name);
        
        // Clear previous faults
        this.clearAllFaults();
        
        // Start simulation if not running
        if (!window.thermalApp || !window.thermalApp.isSimulationRunning) {
            window.thermalApp.startSimulation();
        }
        
        // Schedule faults
        scenario.faults.forEach(fault => {
            setTimeout(() => {
                this.injectFault(fault.type, fault.parameters);
            }, fault.startTime * 1000);
        });
        
        // Schedule test completion
        setTimeout(() => {
            this.completeTest(scenario);
        }, scenario.duration * 1000);
        
        // Update UI
        this.updateScenarioButtons(scenarioName);
    }
    
    updateScenarioButtons(activeScenario) {
        document.querySelectorAll('.scenario-btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.scenario === activeScenario) {
                btn.classList.add('active');
            }
        });
    }
    
    injectFault(faultType = null, parameters = {}) {
        if (!faultType) {
            faultType = document.getElementById('fault-type').value;
        }
        
        if (faultType === 'none') {
            console.log('No fault to inject');
            return;
        }
        
        const fault = {
            type: faultType,
            startTime: Date.now(),
            parameters: parameters,
            id: Math.random().toString(36).substr(2, 9)
        };
        
        this.activeFaults.push(fault);
        
        // Apply fault to simulation
        this.applyFault(fault);
        
        console.log('Fault injected:', fault);
        
        // Update UI
        this.updateFaultDisplay();
    }
    
    applyFault(fault) {
        // This would integrate with the actual simulation
        // For now, we'll just log the fault application
        console.log('Applying fault:', fault);
        
        // In a real implementation, this would modify the simulation parameters
        // or inject noise/errors into the system
    }
    
    clearAllFaults() {
        this.activeFaults = [];
        console.log('All faults cleared');
        this.updateFaultDisplay();
    }
    
    updateFaultDisplay() {
        // Update fault display in UI
        const faultCount = this.activeFaults.length;
        console.log(`Active faults: ${faultCount}`);
    }
    
    completeTest(scenario) {
        const testResult = this.analyzeTestResults(scenario);
        this.testResults.push(testResult);
        
        console.log('Test completed:', testResult);
        
        // Update test results display
        this.updateTestResults();
        
        // Clear active faults
        this.clearAllFaults();
    }
    
    analyzeTestResults(scenario) {
        // Simplified test analysis
        // In a real implementation, this would analyze actual simulation data
        
        const result = {
            scenario: scenario.name,
            timestamp: new Date(),
            duration: scenario.duration,
            status: 'passed', // This would be determined by actual analysis
            metrics: {
                maxTemperature: 350 + Math.random() * 50,
                minTemperature: 280 + Math.random() * 20,
                averageFlow: 0.1 + Math.random() * 0.02,
                stability: 0.95 + Math.random() * 0.05
            },
            anomalies: Math.floor(Math.random() * 5),
            faults: this.activeFaults.length
        };
        
        // Determine test status based on metrics
        if (result.metrics.maxTemperature > 400) {
            result.status = 'failed';
        } else if (result.metrics.maxTemperature > 380) {
            result.status = 'warning';
        }
        
        return result;
    }
    
    updateTestResults() {
        // Update summary statistics
        const totalTests = this.testResults.length;
        const passedTests = this.testResults.filter(r => r.status === 'passed').length;
        const failedTests = this.testResults.filter(r => r.status === 'failed').length;
        const warningTests = this.testResults.filter(r => r.status === 'warning').length;
        const successRate = totalTests > 0 ? (passedTests / totalTests * 100).toFixed(1) : 0;
        
        document.getElementById('tests-run').textContent = totalTests;
        document.getElementById('tests-passed').textContent = passedTests;
        document.getElementById('tests-failed').textContent = failedTests;
        document.getElementById('success-rate').textContent = `${successRate}%`;
        
        // Update chart
        if (this.testChart) {
            this.testChart.data.datasets[0].data = [passedTests, failedTests, warningTests];
            this.testChart.update();
        }
    }
    
    // Test analysis methods
    analyzeTemperatureResponse(data) {
        // Analyze temperature response characteristics
        const analysis = {
            riseTime: 0,
            settlingTime: 0,
            overshoot: 0,
            steadyStateError: 0
        };
        
        // Simplified analysis - in real implementation, would analyze actual data
        analysis.riseTime = 5 + Math.random() * 10;
        analysis.settlingTime = 20 + Math.random() * 30;
        analysis.overshoot = Math.random() * 5;
        analysis.steadyStateError = Math.random() * 2;
        
        return analysis;
    }
    
    analyzeStability(data) {
        // Analyze system stability
        const analysis = {
            isStable: true,
            margin: 0,
            frequency: 0
        };
        
        // Simplified analysis
        analysis.margin = 0.3 + Math.random() * 0.4;
        analysis.frequency = 0.1 + Math.random() * 0.2;
        analysis.isStable = analysis.margin > 0.2;
        
        return analysis;
    }
    
    analyzeAnomalyDetection(data) {
        // Analyze anomaly detection performance
        const analysis = {
            detectionRate: 0,
            falsePositiveRate: 0,
            responseTime: 0
        };
        
        // Simplified analysis
        analysis.detectionRate = 0.8 + Math.random() * 0.2;
        analysis.falsePositiveRate = Math.random() * 0.1;
        analysis.responseTime = 1 + Math.random() * 4;
        
        return analysis;
    }
    
    // Export methods
    exportTestResults() {
        const csv = this.convertTestResultsToCSV();
        this.downloadCSV(csv, 'test_results.csv');
    }
    
    convertTestResultsToCSV() {
        const headers = [
            'Timestamp',
            'Scenario',
            'Duration (s)',
            'Status',
            'Max Temperature (K)',
            'Min Temperature (K)',
            'Average Flow (kg/s)',
            'Stability',
            'Anomalies',
            'Faults'
        ];
        const rows = [headers.join(',')];
        
        this.testResults.forEach(result => {
            const row = [
                result.timestamp.toISOString(),
                result.scenario,
                result.duration,
                result.status,
                result.metrics.maxTemperature.toFixed(2),
                result.metrics.minTemperature.toFixed(2),
                result.metrics.averageFlow.toFixed(4),
                result.metrics.stability.toFixed(3),
                result.anomalies,
                result.faults
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
    
    // Configuration methods
    setTestParameters(parameters) {
        // Update test parameters
        console.log('Test parameters updated:', parameters);
    }
    
    setThresholds(thresholds) {
        // Update detection thresholds
        console.log('Detection thresholds updated:', thresholds);
    }
}
