Chart.defaults.color = '#9CA3AF';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';

function renderTrendChart(id, data) {
    const ctx = document.getElementById(id).getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(59, 130, 246, 0.4)');
    gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(d => d.month),
            datasets: [{
                label: 'Churn Rate %',
                data: data.map(d => d.rate),
                borderColor: '#3B82F6',
                borderWidth: 3,
                tension: 0.4,
                fill: true,
                backgroundColor: gradient,
                pointBackgroundColor: '#3B82F6',
                pointBorderColor: '#fff',
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: false, grid: { borderDash: [5, 5] } },
                x: { grid: { display: false } }
            },
            animation: { duration: 1500, easing: 'easeOutQuart' }
        }
    });
}

function renderBarChart(id, labels, values, label) {
    const ctx = document.getElementById(id).getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: values,
                backgroundColor: ['#3B82F6', '#60A5FA', '#93C5FD'],
                borderRadius: 8,
                barThickness: 40
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { grid: { borderDash: [5, 5] } },
                x: { grid: { display: false } }
            },
            animation: { duration: 1500, easing: 'easeOutQuart' }
        }
    });
}

function renderDonutChart(id, labels, values) {
    const ctx = document.getElementById(id).getContext('2d');
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: ['#EF4444', '#F59E0B', '#3B82F6', '#10B981'],
                borderWidth: 0,
                cutout: '75%'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { padding: 20, usePointStyle: true } }
            },
            animation: { duration: 1500, easing: 'easeOutQuart' }
        }
    });
}

function renderScatterChart(id, data) {
    const ctx = document.getElementById(id).getContext('2d');

    // Map segments to colors
    const segmentColors = {
        'High-value loyal': '#3B82F6',
        'New at-risk': '#EF4444',
        'Long-term stable': '#F59E0B',
        'Occasional user': '#10B981'
    };

    const datasets = Object.keys(segmentColors).map(seg => ({
        label: seg,
        data: data.filter(d => d.segment === seg).map(d => ({ x: d.x, y: d.y })),
        backgroundColor: segmentColors[seg],
        pointRadius: 4,
        pointHoverRadius: 6
    }));

    new Chart(ctx, {
        type: 'scatter',
        data: { datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'Tenure (months)' } },
                y: { title: { display: true, text: 'Monthly Charges ($)' } }
            },
            plugins: {
                legend: { position: 'bottom' },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `Tenure: ${ctx.raw.x}m, Charge: $${ctx.raw.y}`
                    }
                }
            }
        }
    });
}
