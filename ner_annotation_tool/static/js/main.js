// Main JavaScript for NER Annotation Tool

// Professional Notification System
class NotificationManager {
    constructor() {
        this.container = null;
        this.init();
    }

    init() {
        // Create toast container if it doesn't exist
        if (!document.querySelector('.toast-container')) {
            this.container = document.createElement('div');
            this.container.className = 'toast-container';
            document.body.appendChild(this.container);
        } else {
            this.container = document.querySelector('.toast-container');
        }
    }

    showToast(type, title, message, options = {}) {
        const toast = document.createElement('div');
        toast.className = `toast-notification ${type}`;

        const iconMap = {
            success: '✓',
            error: '✕',
            info: 'i',
            warning: '!'
        };

        toast.innerHTML = `
            <div class="toast-icon">${iconMap[type] || 'i'}</div>
            <div class="toast-content">
                <div class="toast-title">${title}</div>
                <div class="toast-message">${message}</div>
            </div>
            <button class="toast-close" onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;

        this.container.appendChild(toast);

        // Trigger animation
        requestAnimationFrame(() => {
            toast.classList.add('show');
        });

        // Auto-dismiss for success and info messages
        if ((type === 'success' || type === 'info') && !options.persistent) {
            setTimeout(() => {
                this.dismissToast(toast);
            }, options.duration || 5000);
        }

        return toast;
    }

    dismissToast(toast) {
        toast.classList.add('hide');
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, 400);
    }

    showBanner(type, message, duration = 4000) {
        // Remove existing banner
        const existingBanner = document.querySelector('.banner-notification');
        if (existingBanner) {
            existingBanner.remove();
        }

        const banner = document.createElement('div');
        banner.className = `banner-notification ${type}`;
        banner.innerHTML = `
            <div class="banner-content">
                <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'} me-2"></i>
                ${message}
            </div>
        `;

        document.body.appendChild(banner);

        // Adjust page content
        const pageContent = document.querySelector('.container');
        if (pageContent) {
            pageContent.classList.add('banner-active');
        }

        // Show banner
        requestAnimationFrame(() => {
            banner.classList.add('show');
        });

        // Auto-dismiss
        setTimeout(() => {
            banner.classList.remove('show');
            if (pageContent) {
                pageContent.classList.remove('banner-active');
            }
            setTimeout(() => banner.remove(), 400);
        }, duration);
    }

    success(title, message, options = {}) {
        return this.showToast('success', title, message, options);
    }

    error(title, message, options = {}) {
        return this.showToast('error', title, message, { ...options, persistent: true });
    }

    info(title, message, options = {}) {
        return this.showToast('info', title, message, options);
    }

    warning(title, message, options = {}) {
        return this.showToast('warning', title, message, options);
    }
}

// Global notification manager
window.notifications = new NotificationManager();

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Remove old alert auto-dismiss (replaced with professional notifications)
});

// Text selection and annotation functionality
class AnnotationManager {
    constructor(sentenceElement, entityTags, sentenceId) {
        this.sentenceElement = sentenceElement;
        this.entityTags = entityTags;
        this.sentenceId = sentenceId;
        this.selectedText = '';
        this.selectedRange = null;
        this.annotations = [];
        this.originalText = sentenceElement.textContent;

        this.init();
    }

    init() {
        // Store original text and ensure clean state
        this.sentenceElement.textContent = this.originalText;

        this.sentenceElement.addEventListener('mouseup', (e) => this.handleTextSelection(e));
        this.sentenceElement.addEventListener('touchend', (e) => this.handleTextSelection(e));

        // Create entity selection modal
        this.createEntityModal();

        // Load existing annotations
        this.loadExistingAnnotations();
    }
    
    handleTextSelection(e) {
        const selection = window.getSelection();

        if (selection.rangeCount > 0 && !selection.isCollapsed) {
            const range = selection.getRangeAt(0);

            // Check if selection is within the sentence element and is text content
            if (this.sentenceElement.contains(range.commonAncestorContainer) &&
                range.commonAncestorContainer.nodeType === Node.TEXT_NODE ||
                (range.commonAncestorContainer === this.sentenceElement)) {

                this.selectedText = selection.toString().trim();
                this.selectedRange = range;

                if (this.selectedText.length > 0) {
                    // Get the position of the selection for popup positioning
                    const rect = range.getBoundingClientRect();
                    this.showEntityModal(rect.right + window.scrollX, rect.top + window.scrollY);
                }
            }
        }
    }
    
    createEntityModal() {
        // Remove existing modal if any
        const existingModal = document.getElementById('entityModal');
        if (existingModal) {
            existingModal.remove();
        }

        const modal = document.createElement('div');
        modal.id = 'entityModal';
        modal.className = 'entity-popup';
        modal.style.display = 'none';

        const modalContent = document.createElement('div');
        modalContent.className = 'entity-popup-content';

        const selectedTextDiv = document.createElement('div');
        selectedTextDiv.id = 'selectedTextDisplay';
        selectedTextDiv.className = 'selected-text-display';

        const buttonsContainer = document.createElement('div');
        buttonsContainer.className = 'entity-buttons-compact';

        // Create buttons for each entity type
        this.entityTags.forEach(tag => {
            const button = document.createElement('button');
            button.textContent = tag.name || tag; // Support both old and new format
            button.className = 'entity-btn-compact';

            // Apply custom color if available
            if (tag.color) {
                button.style.backgroundColor = tag.color;
                button.style.color = 'white';
                button.style.borderColor = tag.color;
            }

            button.onclick = () => this.saveAnnotation(tag.name || tag);
            buttonsContainer.appendChild(button);
        });

        modalContent.appendChild(selectedTextDiv);
        modalContent.appendChild(buttonsContainer);
        modal.appendChild(modalContent);

        document.body.appendChild(modal);

        // Close modal when clicking outside or pressing escape
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.hideEntityModal();
            }
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modal.style.display === 'block') {
                this.hideEntityModal();
            }
        });
    }
    
    showEntityModal(x, y) {
        const modal = document.getElementById('entityModal');
        const selectedTextDisplay = document.getElementById('selectedTextDisplay');

        selectedTextDisplay.textContent = `"${this.selectedText}"`;

        // Use fixed positioning to prevent layout shifts
        modal.style.display = 'block';
        modal.style.position = 'fixed';
        modal.style.zIndex = '1000';

        // Calculate optimal position
        const modalWidth = 250; // Approximate modal width
        const modalHeight = 100; // Approximate modal height

        let left = Math.min(x + 10, window.innerWidth - modalWidth - 20);
        let top = Math.max(y - modalHeight - 10, 10);

        // If popup would go above viewport, position it below the selection
        if (top < 10) {
            top = y + 20;
        }

        modal.style.left = left + 'px';
        modal.style.top = top + 'px';

        // Add smooth fade-in effect
        modal.style.opacity = '0';
        modal.style.transform = 'scale(0.95)';
        modal.style.transition = 'opacity 0.15s ease, transform 0.15s ease';

        // Trigger animation
        requestAnimationFrame(() => {
            modal.style.opacity = '1';
            modal.style.transform = 'scale(1)';
        });
    }

    hideEntityModal() {
        const modal = document.getElementById('entityModal');

        // Smooth fade-out effect
        modal.style.opacity = '0';
        modal.style.transform = 'scale(0.95)';

        setTimeout(() => {
            modal.style.display = 'none';
            modal.style.transition = '';
        }, 150);

        // Clear selection
        window.getSelection().removeAllRanges();
        this.selectedText = '';
        this.selectedRange = null;
    }
    
    async saveAnnotation(entityLabel) {
        if (!this.selectedRange || !this.selectedText) {
            return;
        }

        // Calculate character positions
        const startPos = this.getCharacterPosition(this.selectedRange.startContainer, this.selectedRange.startOffset);
        const endPos = startPos + this.selectedText.length;

        // Check for exact duplicate
        const exactDuplicate = this.checkForDuplicateAnnotation(startPos, endPos, entityLabel, this.selectedText);
        if (exactDuplicate) {
            window.notifications.warning('Duplicate Annotation', 'This exact annotation already exists');
            this.hideEntityModal();
            return;
        }

        // Check for overlapping annotation with same entity type
        const overlapping = this.checkForOverlappingAnnotation(startPos, endPos, entityLabel);
        if (overlapping && !this.isLegitimateOverlap(startPos, endPos, entityLabel)) {
            window.notifications.warning('Overlapping Annotation', `This text overlaps with existing ${entityLabel} annotation: "${overlapping.annotated_text}"`);
            this.hideEntityModal();
            return;
        }

        const annotationData = {
            sentence_id: this.sentenceId,
            start_pos: startPos,
            end_pos: endPos,
            entity_label: entityLabel,
            annotated_text: this.selectedText
        };
        
        try {
            const response = await fetch('/api/save_annotation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(annotationData)
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.addAnnotationToDisplay(annotationData, result.annotation_id);
                this.hideEntityModal();
                // Success notification removed for seamless workflow
            } else {
                console.error('Failed to save annotation:', result.error);
            }
        } catch (error) {
            console.error('Error saving annotation:', error.message);
        }
    }
    
    getCharacterPosition(node, offset) {
        // Since we're preserving original text structure, calculate position directly
        if (node === this.sentenceElement) {
            return offset;
        }

        // For text nodes, find position within the sentence element
        let position = 0;
        const walker = document.createTreeWalker(
            this.sentenceElement,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );

        let currentNode;
        while (currentNode = walker.nextNode()) {
            if (currentNode === node) {
                return position + offset;
            }
            position += currentNode.textContent.length;
        }

        return position;
    }
    
    addAnnotationToDisplay(annotation, annotationId) {
        // Add to annotations list
        this.annotations.push({...annotation, id: annotationId});

        // Update the annotations display
        this.updateAnnotationsDisplay();

        // Force refresh the highlighting immediately
        this.clearHighlights();
        this.highlightAnnotations();
    }
    
    updateAnnotationsDisplay() {
        const annotationsList = document.getElementById('annotationsList');
        if (!annotationsList) return;
        
        annotationsList.innerHTML = '';
        
        this.annotations.forEach(annotation => {
            const item = document.createElement('div');
            item.className = 'annotation-item d-flex justify-content-between align-items-center';

            // Find the entity color
            let entityColor = '#007bff'; // Default blue
            const entityTag = this.entityTags.find(tag =>
                (tag.name || tag) === annotation.entity_label
            );
            if (entityTag && entityTag.color) {
                entityColor = entityTag.color;
            }

            item.innerHTML = `
                <div>
                    <span class="badge me-2" style="background-color: ${entityColor}; color: white;">${annotation.entity_label}</span>
                    <span>"${annotation.annotated_text}"</span>
                </div>
                <button class="btn btn-sm btn-outline-danger" onclick="deleteAnnotation(${annotation.id})">
                    <i class="fas fa-trash"></i>
                </button>
            `;

            annotationsList.appendChild(item);
        });
    }
    
    highlightAnnotations() {
        // Always update highlighting when called
        if (this.annotations.length > 0) {
            this.updateHighlightedContent();
        }
    }

    clearHighlights() {
        // Reset to original text to ensure clean state
        this.sentenceElement.textContent = this.originalText;
    }

    updateHighlightedContent() {
        // Create highlighted version while preserving text selection capability
        const highlightedHTML = this.createHighlightedHTML();
        this.sentenceElement.innerHTML = highlightedHTML;

        // Re-attach event listeners to maintain functionality
        this.attachHighlightEventListeners();
    }

    attachHighlightEventListeners() {
        // Add click handlers to highlighted spans for potential future features
        const highlights = this.sentenceElement.querySelectorAll('.annotation-highlight');
        highlights.forEach(highlight => {
            highlight.addEventListener('click', (e) => {
                e.stopPropagation();
                // Could add edit/delete functionality here in the future
            });
        });
    }

    createHighlightOverlay(annotation) {
        // Use a more robust highlighting approach with inline spans
        const text = this.originalText;
        const beforeText = text.substring(0, annotation.start_pos);
        const annotatedText = text.substring(annotation.start_pos, annotation.end_pos);
        const afterText = text.substring(annotation.end_pos);

        // Find the entity color
        let entityColor = '#007bff'; // Default blue
        const entityTag = this.entityTags.find(tag =>
            (tag.name || tag) === annotation.entity_label
        );
        if (entityTag && entityTag.color) {
            entityColor = entityTag.color;
        }

        // Create highlighted version without destroying text selection capability
        if (!this.sentenceElement.querySelector('.annotation-highlight')) {
            // Only modify if no highlights exist yet
            const highlightedHTML = this.createHighlightedHTML();
            if (highlightedHTML !== this.sentenceElement.innerHTML) {
                this.sentenceElement.innerHTML = highlightedHTML;
            }
        }
    }

    createHighlightedHTML() {
        const text = this.originalText;

        // Create an array of text segments with their types
        const segments = [];
        let lastPos = 0;

        // Sort annotations by start position (ascending) for proper processing
        const sortedAnnotations = [...this.annotations].sort((a, b) => a.start_pos - b.start_pos);

        // Process each annotation
        sortedAnnotations.forEach(annotation => {
            // Add text before annotation if any
            if (annotation.start_pos > lastPos) {
                segments.push({
                    type: 'text',
                    content: text.substring(lastPos, annotation.start_pos)
                });
            }

            // Find the entity color
            let entityColor = '#007bff'; // Default blue
            const entityTag = this.entityTags.find(tag =>
                (tag.name || tag) === annotation.entity_label
            );
            if (entityTag && entityTag.color) {
                entityColor = entityTag.color;
            }

            // Add highlighted annotation
            const annotatedText = text.substring(annotation.start_pos, annotation.end_pos);
            segments.push({
                type: 'annotation',
                content: annotatedText,
                color: entityColor,
                label: annotation.entity_label,
                id: annotation.id
            });

            lastPos = annotation.end_pos;
        });

        // Add remaining text after last annotation
        if (lastPos < text.length) {
            segments.push({
                type: 'text',
                content: text.substring(lastPos)
            });
        }

        // Build HTML from segments
        return segments.map(segment => {
            if (segment.type === 'text') {
                return this.escapeHtml(segment.content);
            } else {
                return `<span class="annotation-highlight" style="background-color: ${segment.color}; color: white; padding: 2px 4px; border-radius: 3px; font-weight: 500;" title="${segment.label}: ${this.escapeHtml(segment.content)}" data-annotation-id="${segment.id}">${this.escapeHtml(segment.content)}</span>`;
            }
        }).join('');
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Duplicate detection functions
    checkForDuplicateAnnotation(startPos, endPos, entityLabel, annotatedText) {
        return this.annotations.find(annotation =>
            annotation.start_pos === startPos &&
            annotation.end_pos === endPos &&
            annotation.entity_label === entityLabel &&
            annotation.annotated_text === annotatedText
        );
    }

    checkForOverlappingAnnotation(startPos, endPos, entityLabel) {
        return this.annotations.find(annotation => {
            // Check for exact overlap with same entity type
            if (annotation.entity_label === entityLabel) {
                return (startPos < annotation.end_pos && endPos > annotation.start_pos);
            }
            return false;
        });
    }

    isLegitimateOverlap(startPos, endPos, entityLabel) {
        // Allow overlapping annotations for different entity types
        // This enables scenarios like "New York City" (LOCATION) containing "York" (PERSON)
        const overlapping = this.annotations.filter(annotation =>
            startPos < annotation.end_pos && endPos > annotation.start_pos
        );

        // Only block if there's an exact match with same entity type
        return !overlapping.some(annotation =>
            annotation.entity_label === entityLabel &&
            annotation.start_pos === startPos &&
            annotation.end_pos === endPos
        );
    }

    loadExistingAnnotations() {
        // This would be called with existing annotations from the server
        const existingAnnotations = window.existingAnnotations || [];
        this.annotations = existingAnnotations;
        this.updateAnnotationsDisplay();
        this.highlightAnnotations();
    }
    
    // Message functions removed for seamless workflow
}

// Global function to delete annotation (seamless, no confirmation)
async function deleteAnnotation(annotationId) {
    try {
        const response = await fetch(`/api/delete_annotation/${annotationId}`, {
            method: 'DELETE'
        });

        const result = await response.json();

        if (result.success) {
            // Reload the page to refresh annotations
            location.reload();
        } else {
            window.notifications.error('Delete Failed', result.error);
        }
    } catch (error) {
        window.notifications.error('Delete Error', error.message);
    }
}

// Navigation functions
function navigateToSentence(projectId, sentenceIndex) {
    window.location.href = `/annotate/${projectId}/${sentenceIndex}`;
}

function nextSentence(projectId, currentIndex, totalSentences) {
    if (currentIndex < totalSentences - 1) {
        navigateToSentence(projectId, currentIndex + 1);
    }
}

function previousSentence(projectId, currentIndex) {
    if (currentIndex > 0) {
        navigateToSentence(projectId, currentIndex - 1);
    }
}

function skipSentence(projectId, currentIndex, totalSentences) {
    nextSentence(projectId, currentIndex, totalSentences);
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Only activate shortcuts on annotation page
    if (!document.querySelector('.sentence-container')) return;
    
    const projectId = window.projectId;
    const currentIndex = window.currentSentenceIndex;
    const totalSentences = window.totalSentences;
    
    if (e.key === 'ArrowRight' && e.ctrlKey) {
        e.preventDefault();
        nextSentence(projectId, currentIndex, totalSentences);
    } else if (e.key === 'ArrowLeft' && e.ctrlKey) {
        e.preventDefault();
        previousSentence(projectId, currentIndex);
    } else if (e.key === 'Escape') {
        e.preventDefault();
        const modal = document.getElementById('entityModal');
        if (modal && modal.style.display === 'flex') {
            modal.style.display = 'none';
        }
    }
});
