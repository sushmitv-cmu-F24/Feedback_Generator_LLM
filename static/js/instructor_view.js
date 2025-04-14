document.addEventListener('DOMContentLoaded', function() {
    // Initialize Markdown renderer with proper configuration
    initializeMarkdown();
    
    // Scroll chat to bottom
    scrollChatToBottom();
    
    // Set up form submission
    setupChatForm();
    
    // Set up textarea auto-resize
    setupTextareaAutoResize();
});

/**
 * Initialize Markdown rendering with improved handling for header formatting
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
        // First fix common markdown formatting issues
        let content = element.textContent;
        
        // Trim whitespace around headers
        content = content.replace(/\s+## /g, '\n## ');
        
        // Fix header formatting (ensure space after ##)
        content = content.replace(/##(\s*)([A-Za-z])/g, '## $2');
        
        // Parse the fixed markdown
        element.innerHTML = marked.parse(content);
        
        // Add 'processed' class to avoid re-processing
        element.classList.add('processed');
    });
}

/**
 * Scroll chat window to the bottom
 */
function scrollChatToBottom() {
    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages) {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

/**
 * Set up form submission with AJAX and validation
 */
function setupChatForm() {
    const chatForm = document.getElementById('chatForm');
    const messageInput = document.getElementById('messageInput');
    const chatMessages = document.getElementById('chatMessages');
    const inputError = document.getElementById('inputError');
    const sendButton = document.getElementById('sendButton');
    
    if (chatForm) {
        chatForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get the message and validate
            const message = messageInput.value.trim();
            if (!message) {
                if (inputError) inputError.textContent = "Message cannot be empty";
                return false;
            }
            
            // Clear any previous error
            if (inputError) inputError.textContent = "";
            
            // Disable button during submission
            if (sendButton) {
                sendButton.disabled = true;
                sendButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Sending...';
            }
            
            // Add user message to chat
            const userMessageHTML = createMessageHTML('instructor', message);
            chatMessages.insertAdjacentHTML('beforeend', userMessageHTML);
            
            // Clear input and reset textarea height
            messageInput.value = '';
            messageInput.style.height = 'auto';
            
            // Scroll to bottom
            scrollChatToBottom();
            
            // Submit the form via fetch API
            const formData = new FormData();
            formData.append('message', message);
            
            fetch(chatForm.action, {
                method: 'POST',
                body: formData,
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                // Re-enable the button
                if (sendButton) {
                    sendButton.disabled = false;
                    sendButton.innerHTML = '<i class="bi bi-send"></i> Send';
                }
                
                if (data.success) {
                    // Add AI response to chat
                    const aiMessageHTML = createMessageHTML('ai', data.response, true);
                    chatMessages.insertAdjacentHTML('beforeend', aiMessageHTML);
                    
                    // Initialize markdown for the new message
                    const newMarkdownElements = document.querySelectorAll('.markdown-content:not(.processed)');
                    newMarkdownElements.forEach(function(element) {
                        // Fix header formatting first
                        let content = element.textContent;
                        content = content.replace(/##(\s*)([A-Za-z])/g, '## $2');
                        
                        // Parse fixed markdown
                        element.innerHTML = marked.parse(content);
                        element.classList.add('processed');
                    });
                    
                    // Scroll to bottom
                    scrollChatToBottom();
                } else {
                    // Handle error
                    if (inputError) inputError.textContent = data.error || "An error occurred while sending your message.";
                    
                    // Add system message for error
                    const errorMessageHTML = createMessageHTML('system', 'Error: ' + (data.error || "An error occurred"));
                    chatMessages.insertAdjacentHTML('beforeend', errorMessageHTML);
                    scrollChatToBottom();
                }
            })
            .catch(error => {
                console.error('Error:', error);
                
                // Re-enable the button
                if (sendButton) {
                    sendButton.disabled = false;
                    sendButton.innerHTML = '<i class="bi bi-send"></i> Send';
                }
                
                // Display error
                if (inputError) inputError.textContent = "Network error. Please try again.";
                
                // Add system message for error
                const errorMessageHTML = createMessageHTML('system', 'Network error: Please try again');
                chatMessages.insertAdjacentHTML('beforeend', errorMessageHTML);
                scrollChatToBottom();
            });
        });
        
        // Clear error when user starts typing again
        if (messageInput && inputError) {
            messageInput.addEventListener('input', function() {
                inputError.textContent = "";
            });
        }
    }
}

/**
 * Create HTML for a new message
 */
function createMessageHTML(sender, content, isMarkdown = false) {
    const now = new Date();
    const time = now.getHours().toString().padStart(2, '0') + ':' + 
                 now.getMinutes().toString().padStart(2, '0');
    
    const messageClass = sender === 'instructor' ? 'user-message' : 
                          sender === 'ai' ? 'ai-message' : 'system-message';
    
    const contentClass = isMarkdown ? 'message-content markdown-content' : 'message-content';
    
    let senderName = sender === 'instructor' ? 'You' : 
                    sender === 'ai' ? 'AI Assistant' : 'System';
    
    let html = `
        <div class="message ${messageClass}">
            <div class="message-header">
                <span class="message-sender">${senderName}</span>
                <span class="message-time">${time}</span>
            </div>
            <div class="${contentClass} ${isMarkdown ? '' : 'processed'}">${content}</div>
        </div>
    `;
    
    return html;
}

/**
 * Set up textarea auto-resize
 */
function setupTextareaAutoResize() {
    const textarea = document.getElementById('messageInput');
    
    if (textarea) {
        textarea.addEventListener('input', function() {
            // Reset height to auto to get the correct scrollHeight
            this.style.height = 'auto';
            
            // Set new height based on scrollHeight (clamped to min and max)
            const newHeight = Math.min(Math.max(this.scrollHeight, 38), 150);
            this.style.height = newHeight + 'px';
        });
        
        // Initialize on page load
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(Math.max(textarea.scrollHeight, 38), 150) + 'px';
    }
}