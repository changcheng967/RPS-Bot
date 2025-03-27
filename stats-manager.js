class StatsManager {
    constructor() {
        this.performanceChart = null;
        this.initChart();
    }

    initChart() {
        const ctx = document.getElementById('performanceChart').getContext('2d');
        this.performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Game 1', 'Game 2', 'Game 3', 'Game 4', 'Game 5'],
                datasets: [
                    {
                        label: 'AI Win Rate',
                        data: [0, 0, 0, 0, 0],
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1,
                        fill: false
                    },
                    {
                        label: 'Your Win Rate',
                        data: [0, 0, 0, 0, 0],
                        borderColor: 'rgb(54, 162, 235)',
                        tension: 0.1,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Performance Over Time'
                    },
                },
                scales: {
                    y: {
                        min: 0,
                        max: 100
                    }
                }
            }
        });
    }

    updateChart(aiWinRate, userWinRate) {
        if (!this.performanceChart) return;
        
        // Shift existing data
        this.performanceChart.data.labels.push(`Game ${this.performanceChart.data.labels.length + 1}`);
        if (this.performanceChart.data.labels.length > 5) {
            this.performanceChart.data.labels.shift();
        }
        
        // Update datasets
        this.performanceChart.data.datasets[0].data.push(aiWinRate);
        this.performanceChart.data.datasets[1].data.push(userWinRate);
        
        if (this.performanceChart.data.datasets[0].data.length > 5) {
            this.performanceChart.data.datasets[0].data.shift();
            this.performanceChart.data.datasets[1].data.shift();
        }
        
        this.performanceChart.update();
    }
}

const statsManager = new StatsManager();