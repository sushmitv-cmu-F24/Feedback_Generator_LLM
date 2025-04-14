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
                
                // Show feedback notification
                showNotification('Thank you for your feedback!', 'success');
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
                
                // Show feedback notification
                showNotification('Thank you for your feedback. We\'ll work to improve.', 'warning');
            }
        });
    }
}

/**
 * Show a notification to the user
 */
function showNotification(message, type = 'info') {
    // Check if notification container exists, create if it doesn't
    let notificationContainer = document.getElementById('notificationContainer');
    
    if (!notificationContainer) {
        notificationContainer = document.createElement('div');
        notificationContainer.id = 'notificationContainer';
        notificationContainer.style.position = 'fixed';
        notificationContainer.style.bottom = '20px';
        notificationContainer.style.right = '20px';
        notificationContainer.style.zIndex = '1050';
        document.body.appendChild(notificationContainer);
    }
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show`;
    notification.role = 'alert';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Add notification to container
    notificationContainer.appendChild(notification);
    
    // Auto-dismiss after 3 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            notificationContainer.removeChild(notification);
        }, 150);
    }, 3000);
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