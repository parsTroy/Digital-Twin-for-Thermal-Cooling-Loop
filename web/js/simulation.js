/**
 * Simulation Manager for Thermal Cooling Loop Digital Twin
 * Handles real-time simulation and chart updates
 */

class SimulationManager {
    constructor() {
        this.isRunning = false;
        this.animationId = null;
        this.time = 0;
        this.dt = 0.1; // Time step in seconds
        this.data = {
            time: [],
            T_hot: [],
            T_cold: [],
            m_dot: []
        };
        this.parameters = {};
        this.state = { T_hot: 300, T_cold: 300 }; // Initial state
        this.maxDataPoints = 1000; // Limit data points for performance
    }
    
    start(parameters) {
        this.parameters = parameters;
        this.isRunning = true;
        this.time = 0;
        this.data = {
            time: [],
            T_hot: [],
            T_cold: [],
            m_dot: []
        };
        
        // Set initial state based on parameters
        this.state = this.calculateSteadyState();
        
        // Start animation loop
        this.animate();
        
        console.log('Simulation started with parameters:', parameters);
    }
    
    stop() {
        this.isRunning = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        console.log('Simulation stopped');
    }
    
    reset() {
        this.stop();
        this.time = 0;
        this.data = {
            time: [],
            T_hot: [],
            T_cold: [],
            m_dot: []
        };
        this.state = { T_hot: 300, T_cold: 300 };
        
        // Clear charts
        if (window.thermalApp && window.thermalApp.temperatureChart) {
            window.thermalApp.temperatureChart.data.labels = [];
            window.thermalApp.temperatureChart.data.datasets[0].data = [];
            window.thermalApp.temperatureChart.data.datasets[1].data = [];
            window.thermalApp.temperatureChart.update();
        }
        
        if (window.thermalApp && window.thermalApp.flowChart) {
            window.thermalApp.flowChart.data.labels = [];
            window.thermalApp.flowChart.data.datasets[0].data = [];
            window.thermalApp.flowChart.update();
        }
        
        console.log('Simulation reset');
    }
    
    updateParameters(newParameters) {
        this.parameters = newParameters;
        console.log('Simulation parameters updated:', newParameters);
    }
    
    calculateSteadyState() {
        // Calculate steady-state temperatures
        const { C, m_dot, cp, UA, Q_in, Q_out } = this.parameters;
        
        // For steady state: dT/dt = 0
        // This gives us: Q_in - UA*(T_hot - T_cold) - m_dot*cp*(T_hot - T_cold) = 0
        // and: m_dot*cp*(T_hot - T_cold) - Q_out = 0
        
        // From second equation: T_hot - T_cold = Q_out / (m_dot * cp)
        const delta_T = Q_out / (m_dot * cp);
        
        // Assume T_cold = 300K (ambient)
        const T_cold = 300;
        const T_hot = T_cold + delta_T;
        
        return { T_hot, T_cold };
    }
    
    step() {
        if (!this.isRunning) return;
        
        // Simple Euler integration for the ODE system
        const { C, m_dot, cp, UA, Q_in, Q_out } = this.parameters;
        
        // Current state
        const T_hot = this.state.T_hot;
        const T_cold = this.state.T_cold;
        
        // Calculate derivatives
        const dT_hot_dt = (Q_in - UA * (T_hot - T_cold) - m_dot * cp * (T_hot - T_cold)) / C;
        const dT_cold_dt = (m_dot * cp * (T_hot - T_cold) - Q_out) / C;
        
        // Update state using Euler method
        this.state.T_hot += dT_hot_dt * this.dt;
        this.state.T_cold += dT_cold_dt * this.dt;
        
        // Add some realistic noise
        this.state.T_hot += (Math.random() - 0.5) * 0.1;
        this.state.T_cold += (Math.random() - 0.5) * 0.1;
        
        // Store data
        this.data.time.push(this.time);
        this.data.T_hot.push(this.state.T_hot);
        this.data.T_cold.push(this.state.T_cold);
        this.data.m_dot.push(m_dot);
        
        // Limit data points for performance
        if (this.data.time.length > this.maxDataPoints) {
            this.data.time.shift();
            this.data.T_hot.shift();
            this.data.T_cold.shift();
            this.data.m_dot.shift();
        }
        
        // Update time
        this.time += this.dt;
    }
    
    animate() {
        if (!this.isRunning) return;
        
        // Perform simulation step
        this.step();
        
        // Update charts
        this.updateCharts();
        
        // Continue animation
        this.animationId = requestAnimationFrame(() => this.animate());
    }
    
    updateCharts() {
        if (!window.thermalApp) return;
        
        // Update temperature chart
        if (window.thermalApp.temperatureChart) {
            const chart = window.thermalApp.temperatureChart;
            
            // Update data
            chart.data.labels = this.data.time.map(t => t.toFixed(1));
            chart.data.datasets[0].data = this.data.T_hot.map((temp, i) => ({
                x: this.data.time[i],
                y: temp
            }));
            chart.data.datasets[1].data = this.data.T_cold.map((temp, i) => ({
                x: this.data.time[i],
                y: temp
            }));
            
            // Update chart
            chart.update('none');
        }
        
        // Update flow chart
        if (window.thermalApp.flowChart) {
            const chart = window.thermalApp.flowChart;
            
            // Update data
            chart.data.labels = this.data.time.map(t => t.toFixed(1));
            chart.data.datasets[0].data = this.data.m_dot.map((flow, i) => ({
                x: this.data.time[i],
                y: flow
            }));
            
            // Update chart
            chart.update('none');
        }
    }
    
    getCurrentState() {
        return {
            time: this.time,
            T_hot: this.state.T_hot,
            T_cold: this.state.T_cold,
            m_dot: this.parameters.m_dot,
            parameters: this.parameters
        };
    }
    
    getData() {
        return {
            time: [...this.data.time],
            T_hot: [...this.data.T_hot],
            T_cold: [...this.data.T_cold],
            m_dot: [...this.data.m_dot]
        };
    }
    
    // Export data for analysis
    exportData() {
        const data = this.getData();
        const csv = this.convertToCSV(data);
        this.downloadCSV(csv, 'simulation_data.csv');
    }
    
    convertToCSV(data) {
        const headers = ['Time (s)', 'Hot Temperature (K)', 'Cold Temperature (K)', 'Mass Flow Rate (kg/s)'];
        const rows = [headers.join(',')];
        
        for (let i = 0; i < data.time.length; i++) {
            const row = [
                data.time[i].toFixed(3),
                data.T_hot[i].toFixed(3),
                data.T_cold[i].toFixed(3),
                data.m_dot[i].toFixed(6)
            ];
            rows.push(row.join(','));
        }
        
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
