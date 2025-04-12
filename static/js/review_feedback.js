// Review Feedback Page JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize markdown renderer
    initializeMarkdown();
    
    // Setup rating slider
    setupRatingSlider();
    
    // Initialize metrics chart
    initializeMetricsChart();
});

/**
 * Initialize markdown rendering
 */
function initializeMarkdown() {
    const markdownElements = document.querySelectorAll('.markdown-content');
    
    marked.setOptions({
        breaks: true,
        gfm: true,
        headerIds: false,
        mangle: false
    });
    
    markdownElements.forEach(function(element) {
        element.innerHTML = marked.parse(element.textContent);
    });
}

/**
 * Setup the rating slider with dynamic badge
 */
function setupRatingSlider() {
    const ratingSlider = document.getElementById('feedbackRating');
    const ratingValue = document.getElementById('ratingValue');
    
    if (ratingSlider && ratingValue) {
        ratingSlider.addEventListener('input', function() {
            ratingValue.textContent = this.value;
            
            // Change badge color based on value
            ratingValue.className = 'badge ';
            if (this.value >= 80) {
                ratingValue.className += 'bg-success';
            } else if (this.value >= 60) {
                ratingValue.className += 'bg-primary';
            } else if (this.value >= 40) {
                ratingValue.className += 'bg-warning';
            } else {
                ratingValue.className += 'bg-danger';
            }
        });
    }
}

/**
 * Initialize the radar chart for metrics visualization
 */
function initializeMetricsChart() {
    const ctx = document.getElementById('metricsChart');
    
    if (ctx) {
        // Get metrics data from the page
        const accuracyValue = parseFloat(ctx.dataset.accuracy || 0.7) * 100;
        const specificityValue = parseFloat(ctx.dataset.specificity || 0.8) * 100;
        const actionabilityValue = parseFloat(ctx.dataset.actionability || 0.6) * 100;
        const completenessValue = parseFloat(ctx.dataset.completeness || 0.7) * 100;
        const readabilityValue = parseFloat(ctx.dataset.readability || 0.8) * 100;
        
        const myChart = new Chart(ctx.getContext('2d'), {
            type: 'radar',
            data: {
                labels: ['Accuracy', 'Specificity', 'Actionability', 'Completeness', 'Readability'],
                datasets: [{
                    label: 'Model Feedback Quality',
                    data: [
                        accuracyValue,
                        specificityValue,
                        actionabilityValue,
                        completenessValue,
                        readabilityValue
                    ],
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    r: {
                        angleLines: {
                            display: true
                        },
                        suggestedMin: 0,
                        suggestedMax: 100
                    }
                },
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }
}