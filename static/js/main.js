/**
 * main.js
 * =======
 * Modern AI Fintech Dashboard Controller for StockPredict AI.
 * Handles API communication, Chart.js dark-theme rendering,
 * and Indian Rupee (₹) dynamic updates.
 */

let stockChart = null;
let currentData = null;
let currentChartMode = 'full';

// Indian Rupee formatting (e.g. ₹ 1,226.00)
function formatRupees(val) {
    if (val === undefined || val === null || isNaN(val)) return "₹ 0.00";
    return "₹ " + Number(val).toLocaleString('en-IN', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
}

// Percent formatting with explicit sign (e.g. +0.71%)
function formatPercent(val) {
    if (val === undefined || val === null || isNaN(val)) return "0.00%";
    const sign = val > 0 ? "+" : "";
    return `${sign}${Number(val).toFixed(2)}%`;
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    // Default load Reliance Industries on startup
    selectQuickStock('RELIANCE.NS');
});

// Dropdown handler
function handleStockPresetChange(value) {
    const customInput = document.getElementById('customTicker');
    if (value === 'CUSTOM') {
        customInput.value = '';
        customInput.focus();
    } else {
        customInput.value = value;
    }
}

// Quick select buttons (Reliance, TCS, Infosys, etc.)
function selectQuickStock(symbol) {
    const stockSelect = document.getElementById('stockSelect');
    const customInput = document.getElementById('customTicker');

    let matched = false;
    for (let i = 0; i < stockSelect.options.length; i++) {
        if (stockSelect.options[i].value === symbol) {
            stockSelect.selectedIndex = i;
            matched = true;
            break;
        }
    }
    if (!matched) {
        stockSelect.value = 'CUSTOM';
    }

    customInput.value = symbol;
    runPrediction(symbol);
}

// Form submit handler
function handleFormSubmit(e) {
    e.preventDefault();
    const symbol = document.getElementById('customTicker').value.trim();
    if (!symbol) {
        showError("Please enter a valid stock ticker symbol.");
        return;
    }
    runPrediction(symbol);
}

// Error alert presentation
function showError(msg) {
    const alertBox = document.getElementById('alertContainer');
    alertBox.innerHTML = `
        <div class="alert alert-warning py-2 px-3 small d-flex justify-content-between align-items-center mb-0 text-white" style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px;" role="alert">
            <span><i class="bi bi-exclamation-triangle-fill text-amber me-2"></i>${msg}</span>
            <button type="button" class="btn-close btn-close-white btn-close-sm" onclick="clearError()"></button>
        </div>
    `;
    alertBox.classList.remove('d-none');
}

function clearError() {
    const alertBox = document.getElementById('alertContainer');
    alertBox.innerHTML = '';
    alertBox.classList.add('d-none');
}

// Main API invocation
async function runPrediction(symbol) {
    clearError();
    setButtonLoading(true);

    const period = document.getElementById('periodSelect').value;
    const daysAhead = parseInt(document.getElementById('horizonSelect').value, 10);

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol: symbol,
                period: period,
                days_ahead: daysAhead
            })
        });

        const data = await response.json();

        if (!response.ok || data.status !== 'success') {
            throw new Error(data.message || "Failed to process stock data.");
        }

        currentData = data;
        updateResultsUI(data);
        renderChart();

    } catch (err) {
        console.error("Prediction error:", err);
        showError(err.message || "Failed to load stock data. Please check ticker symbol.");
    } finally {
        setButtonLoading(false);
    }
}

// Button loading state
function setButtonLoading(loading) {
    const btn = document.getElementById('trainBtn');
    const spinner = document.getElementById('btnSpinner');
    const text = document.getElementById('btnText');

    if (loading) {
        btn.disabled = true;
        spinner.classList.remove('d-none');
        text.textContent = 'Processing...';
    } else {
        btn.disabled = false;
        spinner.classList.add('d-none');
        text.textContent = 'Predict';
    }
}

// Update all UI elements with dynamic data
function updateResultsUI(data) {
    const forecast = data.forecast;
    const metrics = data.metrics;
    const meta = data.metadata;

    // 1. Stock Overview Header
    document.getElementById('stockNameDisplay').textContent = meta.company_name;
    document.getElementById('stockTickerDisplay').textContent = meta.symbol;
    document.getElementById('stockMetaDisplay').textContent = 
        `Historical Lookback: ${meta.period.toUpperCase()} • ${metrics.train_samples} Training Bars • 0 Leakage`;

    // 2. CURRENT PRICE
    document.getElementById('currentPriceDisplay').textContent = formatRupees(forecast.current_price);
    const lastDate = data.historical_chart.dates[data.historical_chart.dates.length - 1];
    document.getElementById('lastDateDisplay').innerHTML = `<i class="bi bi-clock-history me-1"></i> Latest close: ${lastDate}`;

    // 3. PREDICTED PRICE
    document.getElementById('predictedPriceDisplay').textContent = formatRupees(forecast.predicted_price);
    document.getElementById('targetDateDisplay').innerHTML = `<i class="bi bi-bullseye me-1"></i> Target: ${forecast.target_date} (${forecast.horizon_days}d ahead)`;

    // 4. EXPECTED CHANGE
    const diffEl = document.getElementById('priceDiffDisplay');
    const badgeEl = document.getElementById('changePercentBadge');
    const iconWrapper = document.getElementById('changeIconWrapper');
    const sentimentText = document.getElementById('sentimentText');
    const diffSign = forecast.price_diff >= 0 ? "+" : "";

    diffEl.textContent = `${diffSign}${formatRupees(forecast.price_diff).replace('₹ -', '-₹ ')}`;
    badgeEl.textContent = formatPercent(forecast.pct_change);

    if (forecast.price_diff >= 0) {
        diffEl.className = 'stat-value font-mono text-emerald';
        badgeEl.className = 'badge badge-trend trend-bullish';
        iconWrapper.className = 'stat-icon-wrapper icon-emerald';
        iconWrapper.innerHTML = '<i class="bi bi-graph-up-arrow"></i>';
        sentimentText.textContent = 'Bullish Trend Indicator';
    } else {
        diffEl.className = 'stat-value font-mono text-rose';
        badgeEl.className = 'badge badge-trend trend-bearish';
        iconWrapper.className = 'stat-icon-wrapper icon-rose';
        iconWrapper.innerHTML = '<i class="bi bi-graph-down-arrow"></i>';
        sentimentText.textContent = 'Bearish Trend Indicator';
    }

    // 5. Forecast Badge in Chart Section
    document.getElementById('forecastBadge').textContent = `Forecast: ${forecast.horizon_days} Days`;

    // 6. Model Performance Cards
    document.getElementById('r2ScoreDisplay').textContent = metrics.r2.toFixed(3);
    const r2Pct = (metrics.r2 * 100).toFixed(1);
    document.getElementById('r2Explain').textContent = `Explains roughly ${r2Pct}% of past price variance on test data.`;

    document.getElementById('maeDisplay').textContent = formatRupees(metrics.mae);
    document.getElementById('rmseDisplay').textContent = formatRupees(metrics.rmse);
    document.getElementById('trainSamplesDisplay').textContent = `${metrics.train_samples} Days`;

    // 7. AI Market Insight Card
    const movementDesc = forecast.pct_change >= 0 ? "upward movement" : "downward correction";
    const changeAbs = formatRupees(Math.abs(forecast.price_diff));
    document.getElementById('aiInsightParagraph').textContent = 
        `Based on historical price momentum and moving averages, the Linear Regression model projects a ${movementDesc} for ${meta.company_name} (${meta.symbol}) over the next ${forecast.horizon_days} trading days, targeting ${formatRupees(forecast.predicted_price)} (${formatPercent(forecast.pct_change)}).`;
}

// Chart mode switcher
function setChartMode(mode) {
    currentChartMode = mode;
    ['btnFullView', 'btnForecastView', 'btnTestView'].forEach(id => {
        document.getElementById(id)?.classList.remove('active');
    });

    if (mode === 'full') document.getElementById('btnFullView')?.classList.add('active');
    if (mode === 'forecast') document.getElementById('btnForecastView')?.classList.add('active');
    if (mode === 'test') document.getElementById('btnTestView')?.classList.add('active');

    renderChart();
}

// Initialize Chart.js with dark fintech theme
function initChart() {
    const ctx = document.getElementById('stockChart').getContext('2d');

    stockChart = new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#CBD5E1',
                        font: { family: "'Plus Jakarta Sans', sans-serif", size: 12, weight: 600 },
                        usePointStyle: true,
                        pointStyle: 'circle',
                        padding: 16
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    titleColor: '#F8FAFC',
                    bodyColor: '#CBD5E1',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) label += ': ';
                            if (context.parsed.y !== null) {
                                label += formatRupees(context.parsed.y);
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: {
                        color: '#94A3B8',
                        font: { family: "'JetBrains Mono', monospace", size: 11 },
                        maxRotation: 30
                    }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.06)' },
                    ticks: {
                        color: '#94A3B8',
                        font: { family: "'JetBrains Mono', monospace", size: 11 },
                        callback: function(val) {
                            return '₹ ' + Number(val).toLocaleString('en-IN');
                        }
                    }
                }
            }
        }
    });
}

// Render chart data based on active mode
function renderChart() {
    if (!currentData || !stockChart) return;

    const data = currentData;
    const histDates = data.historical_chart.dates;
    const histPrices = data.historical_chart.prices;

    const forecastTraj = data.forecast.trajectory;
    const forecastDates = forecastTraj.map(t => t.date);
    const forecastPrices = forecastTraj.map(t => t.price);

    if (currentChartMode === 'full') {
        // Full View: Historical + Future forecast
        const allLabels = [...histDates, ...forecastDates];
        const histPadded = [...histPrices, ...new Array(forecastDates.length).fill(null)];

        const forecastPadded = new Array(histDates.length - 1).fill(null);
        forecastPadded.push(histPrices[histPrices.length - 1]); // anchor to current price
        forecastPadded.push(...forecastPrices);

        stockChart.data.labels = allLabels;
        stockChart.data.datasets = [
            {
                label: 'Historical Price',
                data: histPadded,
                borderColor: '#94A3B8',
                backgroundColor: 'rgba(148, 163, 184, 0.05)',
                borderWidth: 2,
                pointRadius: 1,
                pointHoverRadius: 5,
                fill: true,
                tension: 0.15
            },
            {
                label: `Predicted Trajectory (${data.forecast.horizon_days}d Forecast)`,
                data: forecastPadded,
                borderColor: '#10B981',
                borderDash: [5, 4],
                backgroundColor: 'rgba(16, 185, 129, 0.08)',
                borderWidth: 2.5,
                pointRadius: 3.5,
                pointBackgroundColor: '#10B981',
                fill: true,
                tension: 0.15
            }
        ];

    } else if (currentChartMode === 'forecast') {
        // Forecast View: Recent 14 days + Future forecast
        const recentDates = histDates.slice(-14);
        const recentPrices = histPrices.slice(-14);

        const labels = [...recentDates, ...forecastDates];
        const histPadded = [...recentPrices, ...new Array(forecastDates.length).fill(null)];

        const forecastPadded = new Array(recentDates.length - 1).fill(null);
        forecastPadded.push(recentPrices[recentPrices.length - 1]);
        forecastPadded.push(...forecastPrices);

        stockChart.data.labels = labels;
        stockChart.data.datasets = [
            {
                label: 'Recent Close (Last 14 Days)',
                data: histPadded,
                borderColor: '#94A3B8',
                borderWidth: 2,
                pointRadius: 3,
                tension: 0.15
            },
            {
                label: `Projected Forecast (${data.forecast.horizon_days} Days)`,
                data: forecastPadded,
                borderColor: '#10B981',
                borderDash: [5, 4],
                borderWidth: 2.5,
                pointRadius: 4,
                pointBackgroundColor: '#10B981',
                backgroundColor: 'rgba(16, 185, 129, 0.12)',
                fill: true,
                tension: 0.15
            }
        ];

    } else if (currentChartMode === 'test') {
        // Model Fit Test: 20% unseen test data actual vs predicted
        const test = data.test_results;
        stockChart.data.labels = test.dates;
        stockChart.data.datasets = [
            {
                label: 'Actual Test Price',
                data: test.actual,
                borderColor: '#38BDF8',
                borderWidth: 2,
                pointRadius: 2.5,
                pointBackgroundColor: '#38BDF8',
                tension: 0.15
            },
            {
                label: 'Model Predicted Price (OLS)',
                data: test.predicted,
                borderColor: '#10B981',
                borderDash: [4, 4],
                borderWidth: 2,
                pointRadius: 2.5,
                pointBackgroundColor: '#10B981',
                tension: 0.15
            }
        ];
    }

    stockChart.update();
}
