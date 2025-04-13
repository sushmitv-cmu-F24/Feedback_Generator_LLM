// student_view.js

document.addEventListener('DOMContentLoaded', function() {
    // Initialize markdown renderer
    initializeMarkdown();
    
    // Setup rating buttons
    setupRatingButtons();
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
 * Setup rating buttons
 */
function setupRatingButtons() {
    const thumbsUp = document.getElementById('thumbsUp');
    const thumbsDown = document.getElementById('thumbsDown');
    const ratingInput = document.getElementById('ratingInput');
    const ratingForm = document.getElementById('ratingForm');
    
    if (thumbsUp && thumbsDown && ratingInput && ratingForm) {
        thumbsUp.addEventListener('click', function() {
            // Toggle active state
            if (!this.classList.contains('active')) {
                thumbsDown.classList.remove('active');
                this.classList.add('active');
                
                // Set rating value and submit
                ratingInput.value = '1';
                ratingForm.submit();
            }
        });
        
        thumbsDown.addEventListener('click', function() {
            // Toggle active state
            if (!this.classList.contains('active')) {
                thumbsUp.classList.remove('active');
                this.classList.add('active');
                
                // Set rating value and submit
                ratingInput.value = '0';
                ratingForm.submit();
            }
        });
    }
}