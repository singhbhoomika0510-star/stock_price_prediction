// --- Stock Data Simulation ---
const stockData = {
    TCS: {
        current: 3260,
        predicted: 3348,
        change: 2.70,
        rmse: 31.4,
        mae: 192.4,
        modelRmse: 284.6,
        r2: 0.91,
        confidence: 87,
        history: [3150, 3180, 3175, 3210, 3240, 3220, 3260]
    },
    RELIANCE: {
        current: 1425,
        predicted: 1472,
        change: 3.29,
        rmse: 18.2,
        mae: 145.6,
        modelRmse: 210.3,
        r2: 0.88,
        confidence: 82,
        history: [1380, 1395, 1390, 1405, 1410, 1415, 1425]
    },
    INFY: {
        current: 1520,
        predicted: 1565,
        change: 2.96,
        rmse: 22.1,
        mae: 168.2,
        modelRmse: 235.1,
        r2: 0.89,
        confidence: 85,
        history: [1480, 1495, 1490, 1500, 1515, 1510, 1520]
    },
    HDFC: {
        current: 1680,
        predicted: 1645,
        change: -2.08,
        rmse: 25.4,
        mae: 175.8,
        modelRmse: 250.4,
        r2: 0.86,
        confidence: 79,
        history: [1710, 1720, 1705, 1690, 1695, 1685, 1680]
    }
};

let chartInstance = null;

// Formatter for Currency
const formatCurrency = (val) => `₹ ${val.toLocaleString('en-IN')}`;
const formatChange = (val) => `${val > 0 ? '+' : ''}${val.toFixed(2)}%`;

function initChart() {
    const ctx = document.getElementById('stockChart').getContext('2d');
    
    // Gradient for line chart
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(45, 212, 191, 0.4)');
    gradient.addColorStop(1, 'rgba(45, 212, 191, 0.0)');

    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Today'],
            datasets: [{
                label: 'Price History',
                data: [],
                borderColor: '#2dd4bf',
                backgroundColor: gradient,
                borderWidth: 3,
                pointBackgroundColor: '#0f172a',
                pointBorderColor: '#2dd4bf',
                pointBorderWidth: 2,
                pointRadius: 4,
                pointHoverRadius: 6,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(17, 26, 48, 0.9)',
                    titleFont: { family: 'Inter', size: 13 },
                    bodyFont: { family: 'Inter', size: 14, weight: 'bold' },
                    padding: 12,
                    cornerRadius: 8,
                    displayColors: false,
                    callbacks: {
                        label: (context) => `₹ ${context.parsed.y}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false, drawBorder: false },
                    ticks: { color: '#94a3b8', font: { family: 'Inter' } }
                },
                y: {
                    grid: { color: 'rgba(42, 56, 88, 0.3)', drawBorder: false },
                    ticks: { color: '#94a3b8', font: { family: 'Inter' }, callback: (value) => `₹${value}` }
                }
            }
        }
    });
}

function predictStock() {
    const stockKey = document.getElementById('stock').value;
    const data = stockData[stockKey];
    if(!data) return;

    // Animate numbers (simple implementation)
    document.getElementById('currentPrice').innerText = formatCurrency(data.current);
    document.getElementById('predictedPrice').innerText = formatCurrency(data.predicted);
    
    const changeEl = document.getElementById('change');
    changeEl.innerText = formatChange(data.change);
    
    // Dynamic color for change
    if(data.change < 0) {
        changeEl.className = 'negative-change';
        document.getElementById('predictedPrice').className = ''; // remove glow for negative
        document.getElementById('summaryPrice').className = '';
        document.getElementById('summaryPrice').style.color = 'var(--accent-rose)';
    } else {
        changeEl.className = 'positive-change';
        document.getElementById('predictedPrice').className = 'text-glow';
        document.getElementById('summaryPrice').className = 'text-glow';
        document.getElementById('summaryPrice').style.color = '';
    }
    
    document.getElementById('rmse').innerText = `₹ ${data.rmse}`;
    document.getElementById('summaryPrice').innerText = formatCurrency(data.predicted);
    document.getElementById('confidence').innerText = `${data.confidence}%`;
    document.getElementById('mae').innerText = `₹ ${data.mae}`;
    document.getElementById('modelRmse').innerText = `₹ ${data.modelRmse}`;
    document.getElementById('r2').innerText = data.r2;

    // Update Chart
    if(chartInstance) {
        chartInstance.data.datasets[0].data = data.history;
        
        // Update chart colors based on trend
        const ctx = document.getElementById('stockChart').getContext('2d');
        const isPositive = data.change >= 0;
        const mainColor = isPositive ? '#2dd4bf' : '#f43f5e';
        const gradient = ctx.createLinearGradient(0, 0, 0, 400);
        
        if (isPositive) {
            gradient.addColorStop(0, 'rgba(45, 212, 191, 0.4)');
            gradient.addColorStop(1, 'rgba(45, 212, 191, 0.0)');
        } else {
            gradient.addColorStop(0, 'rgba(244, 63, 94, 0.4)');
            gradient.addColorStop(1, 'rgba(244, 63, 94, 0.0)');
        }
        
        chartInstance.data.datasets[0].borderColor = mainColor;
        chartInstance.data.datasets[0].pointBorderColor = mainColor;
        chartInstance.data.datasets[0].backgroundColor = gradient;
        
        chartInstance.update();
    }
}

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    predictStock(); // Load initial data
});

// SPA Tab Switching Logic
function switchTab(event, tabId) {
    // Hide all sections
    document.querySelectorAll('.page-section').forEach(sec => {
        sec.style.display = 'none';
    });
    
    // Remove active class from all nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    
    // Show selected section
    document.getElementById('sec-' + tabId).style.display = 'block';
    
    // Add active class to clicked link
    if(event && event.currentTarget) {
        event.currentTarget.classList.add('active');
    }
}
