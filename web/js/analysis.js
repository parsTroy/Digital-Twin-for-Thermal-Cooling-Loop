/**
 * Analysis JavaScript module
 * Handles system analysis and performance metrics
 */

// Analysis state
const AnalysisState = {
    analysisData: [],
    performanceMetrics: null,
    isAnalyzing: false,
    comparisonData: null
};

// Initialize Analysis tab
function initializeAnalysisTab() {
    console.log('Initializing Analysis tab...');
    
    setupAnalysisEventListeners();
    initializeAnalysisCharts();
    
    console.log('Analysis tab initialized');
}

// Set up analysis event listeners
function setupAnalysisEventListeners() {
    const runBtn = document.getElementById('runAnalysis');
    const exportBtn = document.getElementById('exportAnalysis');
    
    if (runBtn) {
        runBtn.addEventListener('click', runAnalysis);
    }
    
    if (exportBtn) {
        exportBtn.addEventListener('click', exportAnalysis);
    }
}

// Run analysis
async function runAnalysis() {
    console.log('Running system analysis...');
    
    try {
        AnalysisState.isAnalyzing = true;
        updateAnalysisButton(true);
        
        const analysisType = document.getElementById('analysisType').value;
        const timeWindow = parseInt(document.getElementById('analysisTimeWindow').value);
        
        // Simulate analysis delay
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        // Generate analysis data based on type
        switch (analysisType) {
            case 'performance':
                AnalysisState.performanceMetrics = generatePerformanceMetrics();
                break;
            case 'anomaly':
                AnalysisState.analysisData = generateAnomalyAnalysis();
                break;
            case 'residual':
                AnalysisState.analysisData = generateResidualAnalysis();
                break;
            case 'comparison':
                AnalysisState.comparisonData = generateComparisonData();
                break;
        }
        
        // Update displays
        updatePerformanceMetrics();
        updateAnalysisChart();
        updateComparisonTable();
        
        showNotification('Analysis completed successfully!', 'success');
        
    } catch (error) {
        console.error('Analysis failed:', error);
        showNotification('Analysis failed', 'error');
    } finally {
        AnalysisState.isAnalyzing = false;
        updateAnalysisButton(false);
    }
}

// Export analysis
function exportAnalysis() {
    console.log('Exporting analysis data...');
    
    try {
        const analysisType = document.getElementById('analysisType').value;
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        
        let exportData = {};
        
        switch (analysisType) {
            case 'performance':
                exportData = AnalysisState.performanceMetrics;
                break;
            case 'anomaly':
                exportData = { anomalies: AnalysisState.analysisData };
                break;
            case 'residual':
                exportData = { residuals: AnalysisState.analysisData };
                break;
            case 'comparison':
                exportData = { comparison: AnalysisState.comparisonData };
                break;
        }
        
        const dataStr = JSON.stringify(exportData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `thermal-twin-analysis-${analysisType}-${timestamp}.json`;
        link.click();
        
        showNotification('Analysis data exported successfully!', 'success');
        
    } catch (error) {
        console.error('Export failed:', error);
        showNotification('Export failed', 'error');
    }
}

// Generate performance metrics
function generatePerformanceMetrics() {
    return {
        detectionAccuracy: 0.92 + Math.random() * 0.06,
        falsePositiveRate: 0.02 + Math.random() * 0.03,
        responseTime: 50 + Math.random() * 100,
        systemStability: 0.95 + Math.random() * 0.04,
        throughput: 1000 + Math.random() * 500,
        latency: 10 + Math.random() * 20,
        uptime: 99.5 + Math.random() * 0.4,
        errorRate: 0.001 + Math.random() * 0.002
    };
}

// Generate anomaly analysis
function generateAnomalyAnalysis() {
    const anomalies = [];
    const count = 20 + Math.floor(Math.random() * 30);
    
    for (let i = 0; i < count; i++) {
        anomalies.push({
            timestamp: Date.now() - Math.random() * 3600000,
            type: ['temperature', 'flow', 'pressure', 'vibration'][Math.floor(Math.random() * 4)],
            severity: ['low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 4)],
            confidence: 0.7 + Math.random() * 0.3,
            description: `Anomaly detected in ${['temperature', 'flow', 'pressure', 'vibration'][Math.floor(Math.random() * 4)]} sensor`
        });
    }
    
    return anomalies.sort((a, b) => b.timestamp - a.timestamp);
}

// Generate residual analysis
function generateResidualAnalysis() {
    const residuals = [];
    const count = 100;
    
    for (let i = 0; i < count; i++) {
        residuals.push({
            timestamp: Date.now() - (count - i) * 1000,
            T_hot: (Math.random() - 0.5) * 4,
            T_cold: (Math.random() - 0.5) * 3,
            m_dot: (Math.random() - 0.5) * 0.2,
            magnitude: Math.sqrt(Math.pow((Math.random() - 0.5) * 4, 2) + 
                               Math.pow((Math.random() - 0.5) * 3, 2) + 
                               Math.pow((Math.random() - 0.5) * 0.2, 2))
        });
    }
    
    return residuals;
}

// Generate comparison data
function generateComparisonData() {
    return {
        methods: [
            {
                name: 'Residual Threshold',
                accuracy: 0.85,
                precision: 0.82,
                recall: 0.88,
                f1Score: 0.85,
                responseTime: 5
            },
            {
                name: 'Rolling Z-Score',
                accuracy: 0.88,
                precision: 0.85,
                recall: 0.91,
                f1Score: 0.88,
                responseTime: 8
            },
            {
                name: 'Isolation Forest',
                accuracy: 0.91,
                precision: 0.89,
                recall: 0.93,
                f1Score: 0.91,
                responseTime: 15
            },
            {
                name: 'One-Class SVM',
                accuracy: 0.89,
                precision: 0.87,
                recall: 0.91,
                f1Score: 0.89,
                responseTime: 20
            },
            {
                name: 'Ensemble',
                accuracy: 0.94,
                precision: 0.92,
                recall: 0.96,
                f1Score: 0.94,
                responseTime: 25
            }
        ]
    };
}

// Update performance metrics display
function updatePerformanceMetrics() {
    if (!AnalysisState.performanceMetrics) return;
    
    const metrics = AnalysisState.performanceMetrics;
    
    const detectionAccuracy = document.getElementById('detectionAccuracy');
    const falsePositiveRate = document.getElementById('falsePositiveRate');
    const responseTime = document.getElementById('responseTime');
    const systemStability = document.getElementById('systemStability');
    
    if (detectionAccuracy) {
        detectionAccuracy.textContent = `${(metrics.detectionAccuracy * 100).toFixed(1)}%`;
    }
    
    if (falsePositiveRate) {
        falsePositiveRate.textContent = `${(metrics.falsePositiveRate * 100).toFixed(2)}%`;
    }
    
    if (responseTime) {
        responseTime.textContent = `${metrics.responseTime.toFixed(1)} ms`;
    }
    
    if (systemStability) {
        systemStability.textContent = `${(metrics.systemStability * 100).toFixed(1)}%`;
    }
}

// Update analysis chart
function updateAnalysisChart() {
    const chartObj = AppState.charts['analysisChart'];
    if (!chartObj) return;
    
    const ctx = chartObj.ctx;
    const analysisType = document.getElementById('analysisType').value;
    
    // Destroy existing chart
    if (chartObj.chart) {
        chartObj.chart.destroy();
    }
    
    let chartConfig;
    
    switch (analysisType) {
        case 'performance':
            chartConfig = createPerformanceChart();
            break;
        case 'anomaly':
            chartConfig = createAnomalyChart();
            break;
        case 'residual':
            chartConfig = createResidualChart();
            break;
        case 'comparison':
            chartConfig = createComparisonChart();
            break;
        default:
            chartConfig = createDefaultChart();
    }
    
    chartObj.chart = new Chart(ctx, chartConfig);
}

// Create performance chart
function createPerformanceChart() {
    const metrics = AnalysisState.performanceMetrics;
    
    return {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Stability', 'Uptime'],
            datasets: [{
                label: 'System Performance',
                data: [
                    metrics.detectionAccuracy,
                    metrics.detectionAccuracy * 0.95,
                    metrics.detectionAccuracy * 1.05,
                    metrics.detectionAccuracy,
                    metrics.systemStability,
                    metrics.uptime / 100
                ],
                backgroundColor: 'rgba(52, 152, 219, 0.2)',
                borderColor: 'rgba(52, 152, 219, 1)',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'System Performance Overview'
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    };
}

// Create anomaly chart
function createAnomalyChart() {
    const anomalies = AnalysisState.analysisData;
    const severityCounts = {};
    
    anomalies.forEach(anomaly => {
        severityCounts[anomaly.severity] = (severityCounts[anomaly.severity] || 0) + 1;
    });
    
    return {
        type: 'doughnut',
        data: {
            labels: Object.keys(severityCounts),
            datasets: [{
                data: Object.values(severityCounts),
                backgroundColor: [
                    '#27ae60',
                    '#f39c12',
                    '#e74c3c',
                    '#8e44ad'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Anomaly Severity Distribution'
                }
            }
        }
    };
}

// Create residual chart
function createResidualChart() {
    const residuals = AnalysisState.analysisData;
    const timestamps = residuals.map(r => new Date(r.timestamp).toLocaleTimeString());
    const magnitudes = residuals.map(r => r.magnitude);
    
    return {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: [{
                label: 'Residual Magnitude',
                data: magnitudes,
                borderColor: 'rgba(231, 76, 60, 1)',
                backgroundColor: 'rgba(231, 76, 60, 0.1)',
                borderWidth: 2,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Residual Magnitude Over Time'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Residual Magnitude'
                    }
                }
            }
        }
    };
}

// Create comparison chart
function createComparisonChart() {
    const comparison = AnalysisState.comparisonData;
    const methods = comparison.methods;
    
    return {
        type: 'bar',
        data: {
            labels: methods.map(m => m.name),
            datasets: [
                {
                    label: 'Accuracy',
                    data: methods.map(m => m.accuracy),
                    backgroundColor: 'rgba(52, 152, 219, 0.8)',
                    borderColor: 'rgba(52, 152, 219, 1)',
                    borderWidth: 1
                },
                {
                    label: 'F1-Score',
                    data: methods.map(m => m.f1Score),
                    backgroundColor: 'rgba(39, 174, 96, 0.8)',
                    borderColor: 'rgba(39, 174, 96, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Method Performance Comparison'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    title: {
                        display: true,
                        text: 'Score'
                    }
                }
            }
        }
    };
}

// Create default chart
function createDefaultChart() {
    return {
        type: 'line',
        data: {
            labels: ['No Data'],
            datasets: [{
                label: 'No Data Available',
                data: [0],
                borderColor: 'rgba(108, 117, 125, 1)',
                backgroundColor: 'rgba(108, 117, 125, 0.1)',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'No Analysis Data Available'
                }
            }
        }
    };
}

// Update comparison table
function updateComparisonTable() {
    if (!AnalysisState.comparisonData) return;
    
    const tableBody = document.getElementById('comparisonTableBody');
    if (!tableBody) return;
    
    tableBody.innerHTML = '';
    
    AnalysisState.comparisonData.methods.forEach(method => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${method.name}</td>
            <td>${(method.accuracy * 100).toFixed(1)}%</td>
            <td>${(method.precision * 100).toFixed(1)}%</td>
            <td>${(method.recall * 100).toFixed(1)}%</td>
            <td>${(method.f1Score * 100).toFixed(1)}%</td>
        `;
        tableBody.appendChild(row);
    });
}

// Update analysis button
function updateAnalysisButton(isAnalyzing) {
    const runBtn = document.getElementById('runAnalysis');
    if (runBtn) {
        runBtn.textContent = isAnalyzing ? 'Analyzing...' : 'Run Analysis';
        runBtn.disabled = isAnalyzing;
    }
}

// Initialize analysis charts
function initializeAnalysisCharts() {
    // This will be called when the analysis tab is first loaded
    const chartObj = AppState.charts['analysisChart'];
    if (chartObj) {
        chartObj.chart = new Chart(chartObj.ctx, createDefaultChart());
    }
}

// Export functions
window.initializeAnalysisTab = initializeAnalysisTab;
window.runAnalysis = runAnalysis;
window.exportAnalysis = exportAnalysis;
