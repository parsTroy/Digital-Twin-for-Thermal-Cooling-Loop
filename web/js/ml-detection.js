/**
 * ML Detection JavaScript module
 * Handles advanced machine learning anomaly detection functionality
 */

// ML Detection state
const MLState = {
    enhancedDetector: null,
    mlDetector: null,
    trainingData: [],
    trainingResults: null,
    isTraining: false,
    testResults: null
};

// Initialize ML Detection tab
function initializeMLDetectionTab() {
    console.log('Initializing ML Detection tab...');
    
    setupMLEventListeners();
    updateMLStatus();
    
    console.log('ML Detection tab initialized');
}

// Set up ML event listeners
function setupMLEventListeners() {
    // ML Detector Controls
    const createBtn = document.getElementById('createMLDetector');
    const generateBtn = document.getElementById('generateTrainingData');
    const trainBtn = document.getElementById('trainModels');
    const resetBtn = document.getElementById('resetML');
    
    if (createBtn) {
        createBtn.addEventListener('click', createMLDetector);
    }
    
    if (generateBtn) {
        generateBtn.addEventListener('click', generateTrainingData);
    }
    
    if (trainBtn) {
        trainBtn.addEventListener('click', trainModels);
    }
    
    if (resetBtn) {
        resetBtn.addEventListener('click', resetML);
    }
    
    // ML Testing
    const testBtn = document.getElementById('testDetection');
    if (testBtn) {
        testBtn.addEventListener('click', testMLDetection);
    }
}

// Create ML detector
function createMLDetector() {
    console.log('Creating ML detector...');
    
    try {
        // Get configuration from UI
        const mlEnabled = document.getElementById('mlEnabled').checked;
        const featureEngineering = document.getElementById('featureEngineering').checked;
        const ensembleVoting = document.getElementById('ensembleVoting').checked;
        const adaptiveThresholds = document.getElementById('adaptiveThresholds').checked;
        
        // Create enhanced detector (simulated)
        MLState.enhancedDetector = {
            isTrained: false,
            config: {
                mlEnabled: mlEnabled,
                featureEngineering: featureEngineering,
                ensembleVoting: ensembleVoting,
                adaptiveThresholds: adaptiveThresholds
            }
        };
        
        // Create ML detector (simulated)
        MLState.mlDetector = {
            isTrained: false,
            config: {
                featureEngineering: featureEngineering,
                autoTuning: true,
                ensembleMethods: true
            }
        };
        
        updateMLStatus();
        showNotification('ML detectors created successfully!', 'success');
        
    } catch (error) {
        console.error('Failed to create ML detector:', error);
        showNotification('Failed to create ML detector', 'error');
    }
}

// Generate training data
function generateTrainingData() {
    console.log('Generating training data...');
    
    try {
        const samples = parseInt(document.getElementById('trainingSamples').value);
        MLState.trainingData = [];
        
        // Generate normal data (80%)
        const normalCount = Math.floor(samples * 0.8);
        for (let i = 0; i < normalCount; i++) {
            MLState.trainingData.push({
                T_hot: generateNormalResidual(0, 1.0),
                T_cold: generateNormalResidual(0, 0.8),
                m_dot: generateNormalResidual(0, 0.1),
                label: false
            });
        }
        
        // Generate anomalous data (20%)
        const anomalyCount = samples - normalCount;
        for (let i = 0; i < anomalyCount; i++) {
            MLState.trainingData.push({
                T_hot: generateNormalResidual(0, 4.0),
                T_cold: generateNormalResidual(0, 3.0),
                m_dot: generateNormalResidual(0, 0.3),
                label: true
            });
        }
        
        updateMLStatus();
        showNotification(`Generated ${samples} training samples!`, 'success');
        
    } catch (error) {
        console.error('Failed to generate training data:', error);
        showNotification('Failed to generate training data', 'error');
    }
}

// Train models
async function trainModels() {
    if (!MLState.enhancedDetector || !MLState.mlDetector) {
        showNotification('Please create ML detectors first', 'warning');
        return;
    }
    
    if (MLState.trainingData.length === 0) {
        showNotification('Please generate training data first', 'warning');
        return;
    }
    
    console.log('Training ML models...');
    
    try {
        MLState.isTraining = true;
        updateMLStatus();
        
        // Simulate training process
        const trainBtn = document.getElementById('trainModels');
        if (trainBtn) {
            trainBtn.textContent = 'Training...';
            trainBtn.disabled = true;
        }
        
        // Simulate training delay
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Simulate training results
        MLState.trainingResults = {
            enhanced: {
                random_forest: { test_accuracy: 0.92, train_accuracy: 0.95 },
                mlp: { test_accuracy: 0.89, train_accuracy: 0.93 },
                isolation_forest: { test_accuracy: 0.87, train_accuracy: 0.90 }
            },
            ml: {
                random_forest: { test_accuracy: 0.94, train_accuracy: 0.96 },
                mlp: { test_accuracy: 0.91, train_accuracy: 0.94 },
                isolation_forest: { test_accuracy: 0.88, train_accuracy: 0.91 },
                one_class_svm: { test_accuracy: 0.85, train_accuracy: 0.88 },
                ensemble: { test_accuracy: 0.95, train_accuracy: 0.97 }
            }
        };
        
        MLState.enhancedDetector.isTrained = true;
        MLState.mlDetector.isTrained = true;
        MLState.isTraining = false;
        
        updateMLStatus();
        updateMLTestingVisibility();
        showNotification('Models trained successfully!', 'success');
        
    } catch (error) {
        console.error('Failed to train models:', error);
        showNotification('Failed to train models', 'error');
        MLState.isTraining = false;
    } finally {
        const trainBtn = document.getElementById('trainModels');
        if (trainBtn) {
            trainBtn.textContent = 'Train Models';
            trainBtn.disabled = false;
        }
    }
}

// Test ML detection
function testMLDetection() {
    if (!MLState.enhancedDetector || !MLState.enhancedDetector.isTrained) {
        showNotification('Please train models first', 'warning');
        return;
    }
    
    console.log('Testing ML detection...');
    
    try {
        const scenario = document.getElementById('testScenario').value;
        const testData = getTestScenarioData(scenario);
        
        // Simulate ML detection
        const result = simulateMLDetection(testData, scenario);
        
        // Display results
        displayTestResults(result);
        
        // Update chart
        updateMLChart(testData, result);
        
        showNotification('ML detection test completed', 'success');
        
    } catch (error) {
        console.error('Failed to test ML detection:', error);
        showNotification('Failed to test ML detection', 'error');
    }
}

// Reset ML
function resetML() {
    console.log('Resetting ML detectors...');
    
    MLState.enhancedDetector = null;
    MLState.mlDetector = null;
    MLState.trainingData = [];
    MLState.trainingResults = null;
    MLState.isTraining = false;
    MLState.testResults = null;
    
    updateMLStatus();
    updateMLTestingVisibility();
    
    // Clear charts
    const mlChart = AppState.charts['mlChart'];
    if (mlChart && mlChart.chart) {
        mlChart.chart.destroy();
        mlChart.chart = null;
    }
    
    showNotification('ML detectors reset', 'info');
}

// Update ML status display
function updateMLStatus() {
    const enhancedStatus = document.getElementById('enhancedDetectorStatus');
    const mlStatus = document.getElementById('mlDetectorStatus');
    const trainingStatus = document.getElementById('trainingDataStatus');
    const performanceStatus = document.getElementById('modelPerformance');
    
    if (enhancedStatus) {
        enhancedStatus.textContent = MLState.enhancedDetector ? 
            (MLState.enhancedDetector.isTrained ? 'Ready' : 'Not Trained') : 'Not Created';
    }
    
    if (mlStatus) {
        mlStatus.textContent = MLState.mlDetector ? 
            (MLState.mlDetector.isTrained ? 'Ready' : 'Not Trained') : 'Not Created';
    }
    
    if (trainingStatus) {
        trainingStatus.textContent = `${MLState.trainingData.length} samples`;
    }
    
    if (performanceStatus && MLState.trainingResults) {
        const bestAccuracy = Math.max(
            ...Object.values(MLState.trainingResults.ml).map(r => r.test_accuracy)
        );
        performanceStatus.textContent = `${(bestAccuracy * 100).toFixed(1)}%`;
    }
}

// Update ML testing visibility
function updateMLTestingVisibility() {
    const mlTesting = document.getElementById('mlTesting');
    if (mlTesting) {
        mlTesting.style.display = MLState.enhancedDetector && MLState.enhancedDetector.isTrained ? 'block' : 'none';
    }
}

// Get test scenario data
function getTestScenarioData(scenario) {
    const scenarios = {
        normal: { T_hot: 0.5, T_cold: 0.3, m_dot: 0.05 },
        temp_anomaly: { T_hot: 8.0, T_cold: 6.0, m_dot: 0.1 },
        flow_anomaly: { T_hot: 1.0, T_cold: 0.8, m_dot: 0.8 },
        multiple_anomalies: { T_hot: 10.0, T_cold: 8.0, m_dot: 0.9 }
    };
    
    return scenarios[scenario] || scenarios.normal;
}

// Simulate ML detection
function simulateMLDetection(testData, scenario) {
    const isAnomaly = scenario !== 'normal';
    const confidence = isAnomaly ? 0.7 + Math.random() * 0.3 : Math.random() * 0.3;
    const method = isAnomaly ? 'ensemble' : 'traditional';
    const severity = isAnomaly ? (confidence > 0.9 ? 'critical' : 'warning') : 'normal';
    
    return {
        isAnomaly: isAnomaly,
        confidence: confidence,
        method: method,
        severity: severity,
        details: {
            traditional_vote: isAnomaly ? 0.8 : 0.2,
            ml_vote: isAnomaly ? 0.9 : 0.1,
            weighted_score: confidence,
            method_agreement: true
        }
    };
}

// Display test results
function displayTestResults(result) {
    const testResults = document.getElementById('testResults');
    if (testResults) {
        testResults.style.display = 'block';
    }
    
    const testAnomaly = document.getElementById('testAnomaly');
    const testConfidence = document.getElementById('testConfidence');
    const testMethod = document.getElementById('testMethod');
    const testSeverity = document.getElementById('testSeverity');
    
    if (testAnomaly) {
        testAnomaly.textContent = result.isAnomaly ? 'Yes' : 'No';
        testAnomaly.className = result.isAnomaly ? 'status-critical' : 'status-normal';
    }
    
    if (testConfidence) {
        testConfidence.textContent = `${(result.confidence * 100).toFixed(1)}%`;
    }
    
    if (testMethod) {
        testMethod.textContent = result.method;
    }
    
    if (testSeverity) {
        testSeverity.textContent = result.severity;
        testSeverity.className = `status-${result.severity}`;
    }
}

// Update ML chart
function updateMLChart(testData, result) {
    const chartObj = AppState.charts['mlChart'];
    if (!chartObj) return;
    
    const ctx = chartObj.ctx;
    
    // Destroy existing chart
    if (chartObj.chart) {
        chartObj.chart.destroy();
    }
    
    // Create new chart
    chartObj.chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['T_hot', 'T_cold', 'm_dot'],
            datasets: [{
                label: 'Residual Values',
                data: [testData.T_hot, testData.T_cold, testData.m_dot],
                backgroundColor: result.isAnomaly ? 
                    ['#e74c3c', '#e74c3c', '#e74c3c'] : 
                    ['#3498db', '#3498db', '#3498db'],
                borderColor: result.isAnomaly ? 
                    ['#c0392b', '#c0392b', '#c0392b'] : 
                    ['#2980b9', '#2980b9', '#2980b9'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `ML Detection Test - ${result.isAnomaly ? 'Anomaly Detected' : 'Normal Operation'}`
                },
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Residual Value'
                    }
                }
            }
        }
    });
}

// Generate normal residual
function generateNormalResidual(mean, std) {
    // Box-Muller transform for normal distribution
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mean + std * z0;
}

// Export functions
window.initializeMLDetectionTab = initializeMLDetectionTab;
window.createMLDetector = createMLDetector;
window.generateTrainingData = generateTrainingData;
window.trainModels = trainModels;
window.testMLDetection = testMLDetection;
window.resetML = resetML;
