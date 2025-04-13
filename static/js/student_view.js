// student_view.js

document.addEventListener('DOMContentLoaded', function() {
    // Initialize markdown renderer
    initializeMarkdown();
    
    // Setup rating buttons
    setupRatingButtons();
    
    // Setup code tabs
    setupCodeTabs();
});

/**
 * Initialize markdown rendering
 */
function initializeMarkdown() {
    const markdownElements = document.querySelectorAll('.markdown-content');
    
    marked.setOptions({
        breaks: true,         // Add line breaks on single newlines
        gfm: true,            // GitHub Flavored Markdown
        headerIds: false,     // Don't add ids to headers
        mangle: false,        // Don't mangle email addresses
        smartLists: true,     // Use smarter list behavior
        smartypants: true     // Use smart punctuation
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

/**
 * Setup code tabs
 */
function setupCodeTabs() {
    const codeTabs = document.querySelectorAll('#codeTabs button');
    
    codeTabs.forEach(function(tab) {
        tab.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Remove active class from all tabs
            codeTabs.forEach(function(t) {
                t.classList.remove('active');
                
                // Hide tab content
                const target = document.querySelector(t.dataset.bsTarget);
                if (target) {
                    target.classList.remove('show');
                    target.classList.remove('active');
                }
            });
            
            // Add active class to clicked tab
            this.classList.add('active');
            
            // Show tab content
            const target = document.querySelector(this.dataset.bsTarget);
            if (target) {
                target.classList.add('show');
                target.classList.add('active');
            }
        });
    });
    
    // Initialize collapse for code section
    const codeToggle = document.querySelector('[data-bs-toggle="collapse"]');
    if (codeToggle) {
        codeToggle.addEventListener('click', function() {
            const target = document.querySelector(this.dataset.bsTarget);
            if (target) {
                target.classList.toggle('show');
            }
        });
    }
}