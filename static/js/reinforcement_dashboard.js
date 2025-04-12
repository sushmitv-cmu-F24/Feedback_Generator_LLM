// Reinforcement Dashboard JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize distribution chart if data exists
    initializeDistributionChart();
});

/**
 * Initialize the distribution chart for rating visualization
 */
function initializeDistributionChart() {
    const distributionCanvas = document.getElementById('distributionChart');
    
    if (distributionCanvas) {
        // Check if we have distribution data
        const labels = distributionCanvas.dataset.labels;
        const counts = distributionCanvas.dataset.counts;
        
        if (labels && counts) {
            try {
                // Parse the data from the data attributes
                const labelsArray = JSON.parse(labels);
                const countsArray = JSON.parse(counts);
                
                // Create the chart
                const distributionChart = new Chart(distributionCanvas.getContext('2d'), {
                    type: 'bar',
                    data: {
                        labels: labelsArray,
                        datasets: [{
                            label: 'Number of Ratings',
                            data: countsArray,
                            backgroundColor: [
                                'rgba(255, 99, 132, 0.5)',
                                'rgba(255, 159, 64, 0.5)',
                                'rgba(54, 162, 235, 0.5)',
                                'rgba(75, 192, 192, 0.5)'
                            ],
                            borderColor: [
                                'rgb(255, 99, 132)',
                                'rgb(255, 159, 64)',
                                'rgb(54, 162, 235)',
                                'rgb(75, 192, 192)'
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
                                ticks: {
                                    stepSize: 1
                                }
                            }
                        }
                    }
                });
            } catch (e) {
                console.error('Error initializing distribution chart:', e);
            }
        }
    }
}

/**
 * Update the progress bar animation
 */
function animateProgressBar() {
    const progressBars = document.querySelectorAll('.progress-bar-striped');
    
    progressBars.forEach(bar => {
        // Add animation class if it doesn't exist
        if (!bar.classList.contains('progress-bar-animated')) {
            bar.classList.add('progress-bar-animated');
        }
    });
}

/**
 * Toggle collapsible sections
 */
function setupCollapsibleSections() {
    const collapsibles = document.querySelectorAll('.collapsible-header');
    
    collapsibles.forEach(header => {
        header.addEventListener('click', function() {
            // Toggle the 'collapsed' class on the header
            this.classList.toggle('collapsed');
            
            // Get the content element
            const content = this.nextElementSibling;
            
            // Toggle the content visibility
            if (content.style.maxHeight) {
                content.style.maxHeight = null;
            } else {
                content.style.maxHeight = content.scrollHeight + "px";
            }
            
            // Toggle the indicator
            const indicator = this.querySelector('.collapse-indicator');
            if (indicator) {
                indicator.textContent = indicator.textContent === '▼' ? '▲' : '▼';
            }
        });
    });
}

// Call animation setup when page loads
animateProgressBar();
setupCollapsibleSections();