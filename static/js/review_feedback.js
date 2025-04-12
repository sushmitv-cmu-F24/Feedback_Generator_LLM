// Review Feedback Page JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize markdown renderer
    initializeMarkdown();
    
    // Setup rating slider
    setupRatingSlider();
    
    // Initialize metrics chart
    initializeMetricsChart();
    
    // Set up RLHF fields
    setupRlhfFields();
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

/**
 * Setup RLHF-specific fields
 */
function setupRlhfFields() {
    // Set up listeners for the RLHF checkboxes
    const rlhfCheckboxes = document.querySelectorAll('input[name="rlhf_improvements"]');
    const rlhfInstructions = document.getElementById('rlhfInstructions');
    
    if (rlhfCheckboxes && rlhfInstructions) {
        // When checkboxes are clicked, suggest instructions based on selection
        rlhfCheckboxes.forEach(function(checkbox) {
            checkbox.addEventListener('change', function() {
                updateRlhfInstructions();
            });
        });
    }
    
    // Link issue checkboxes to RLHF checkboxes
    const issueCheckboxes = document.querySelectorAll('.issue-item input[type="checkbox"]');
    if (issueCheckboxes) {
        issueCheckboxes.forEach(function(checkbox) {
            checkbox.addEventListener('change', function() {
                // Map issues to RLHF improvements
                const issueToRlhf = {
                    'missed_violations': 'improve_specificity',
                    'false_positives': 'improve_specificity',
                    'unclear_feedback': 'improve_clarity',
                    'no_actionable': 'improve_actionability',
                    'not_specific': 'improve_specificity'
                };
                
                const rlhfId = issueToRlhf[this.value];
                if (rlhfId) {
                    document.getElementById(rlhfId).checked = this.checked;
                    updateRlhfInstructions();
                }
            });
        });
    }
    
    // Add a form submit handler to collect additional metrics
    const form = document.getElementById('instructorFeedbackForm');
    if (form) {
        form.addEventListener('submit', function(event) {
            // Add timestamp hidden field
            const timestamp = document.createElement('input');
            timestamp.type = 'hidden';
            timestamp.name = 'feedback_timestamp';
            timestamp.value = new Date().toISOString();
            this.appendChild(timestamp);
            
            // Calculate and add edit distance between original and corrected feedback
            const originalFeedback = document.querySelector('.markdown-content').textContent;
            const correctedFeedback = document.getElementById('correctedFeedback').value;
            
            const editDistanceField = document.createElement('input');
            editDistanceField.type = 'hidden';
            editDistanceField.name = 'edit_distance';
            editDistanceField.value = calculateEditDistance(originalFeedback, correctedFeedback);
            this.appendChild(editDistanceField);
        });
    }
}

/**
 * Update RLHF instructions based on selected improvements
 */
function updateRlhfInstructions() {
    const rlhfInstructions = document.getElementById('rlhfInstructions');
    const checkedImprovements = document.querySelectorAll('input[name="rlhf_improvements"]:checked');
    
    if (rlhfInstructions && checkedImprovements.length > 0) {
        // Only update if the field is empty or auto-generated before
        if (!rlhfInstructions.dataset.userEdited || rlhfInstructions.value === '') {
            let instructions = 'Please improve the feedback by making it ';
            
            const improvements = Array.from(checkedImprovements).map(cb => {
                switch(cb.value) {
                    case 'specificity':
                        return 'more specific about code elements and locations';
                    case 'actionability':
                        return 'more actionable with concrete suggestions';
                    case 'solid_principles':
                        return 'more clear about SOLID principles and how they apply';
                    case 'clarity':
                        return 'more organized and readable';
                    default:
                        return cb.value;
                }
            });
            
            instructions += improvements.join(', ') + '.';
            rlhfInstructions.value = instructions;
        }
    }
    
    // Mark as user-edited when user types in the field
    rlhfInstructions.addEventListener('input', function() {
        this.dataset.userEdited = 'true';
    });
}

/**
 * Calculate a simple edit distance between two strings
 * This helps measure how much the instructor modified the feedback
 */
function calculateEditDistance(str1, str2) {
    // Simple implementation of Levenshtein distance
    const m = str1.length;
    const n = str2.length;
    
    // Create a matrix of size (m+1) x (n+1)
    const dp = Array(m + 1).fill().map(() => Array(n + 1).fill(0));
    
    // Fill the first row and column
    for (let i = 0; i <= m; i++) dp[i][0] = i;
    for (let j = 0; j <= n; j++) dp[0][j] = j;
    
    // Fill the dp table
    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            if (str1[i - 1] === str2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + Math.min(
                    dp[i - 1][j],    // deletion
                    dp[i][j - 1],    // insertion
                    dp[i - 1][j - 1] // substitution
                );
            }
        }
    }
    
    return dp[m][n];
}