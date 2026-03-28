// ============================================
// WiFi Sensing Paper - Interactive Animations
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all animations
    initScrollAnimations();
    initBarChartAnimations();
    initImageModal();
    initTableAnimations();
});

// ============================================
// Scroll-triggered Animations
// ============================================

function initScrollAnimations() {
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                
                // Trigger bar animations when chart section is visible
                if (entry.target.classList.contains('charts-section')) {
                    triggerBarAnimations();
                }
                
                // Trigger table row animations
                if (entry.target.querySelector('.animate-table')) {
                    entry.target.querySelectorAll('.animate-table').forEach(table => {
                        table.classList.add('visible');
                    });
                }
            }
        });
    }, observerOptions);

    // Observe all scroll-animated elements
    document.querySelectorAll('.animate-on-scroll').forEach(el => {
        observer.observe(el);
    });
}

// ============================================
// Bar Chart Animations
// ============================================

function initBarChartAnimations() {
    // Set initial state
    document.querySelectorAll('.animate-bar').forEach(bar => {
        bar.style.width = '0';
    });
}

function triggerBarAnimations() {
    document.querySelectorAll('.animate-bar').forEach(bar => {
        const delay = parseFloat(getComputedStyle(bar).getPropertyValue('--delay')) || 0;
        setTimeout(() => {
            bar.classList.add('visible');
        }, delay * 1000);
    });
}

// ============================================
// Image Modal
// ============================================

function initImageModal() {
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImg');
    const captionText = document.getElementById('caption');
    const closeBtn = document.querySelector('.close');

    // Add click handlers to all zoomable images
    document.querySelectorAll('.zoomable').forEach(img => {
        img.addEventListener('click', function() {
            modal.style.display = 'block';
            modalImg.src = this.src;
            captionText.innerHTML = this.alt;
            document.body.style.overflow = 'hidden';
        });
    });

    // Close modal
    closeBtn.addEventListener('click', closeModal);
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            closeModal();
        }
    });

    // Close on escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeModal();
        }
    });

    function closeModal() {
        modal.style.display = 'none';
        document.body.style.overflow = 'auto';
    }
}

// ============================================
// Table Animations
// ============================================

function initTableAnimations() {
    // Add hover effects to table rows
    document.querySelectorAll('.data-table tbody tr').forEach(row => {
        row.addEventListener('mouseenter', function() {
            this.style.transition = 'background-color 0.3s ease';
        });
    });

    // Highlight best values on hover
    document.querySelectorAll('.number.best').forEach(cell => {
        cell.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.1)';
            this.style.transition = 'transform 0.2s ease';
        });
        cell.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
        });
    });
}

// ============================================
// Smooth Scroll for Internal Links
// ============================================

document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// ============================================
// Dynamic Counter Animation for Stats
// ============================================

function animateCounter(element, target, duration = 1000) {
    const start = 0;
    const increment = target / (duration / 16);
    let current = start;

    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            element.textContent = target;
            clearInterval(timer);
        } else {
            element.textContent = Math.floor(current);
        }
    }, 16);
}

// ============================================
// Typing Effect for Key Findings
// ============================================

function typeWriter(element, text, speed = 50) {
    let i = 0;
    element.textContent = '';
    
    function type() {
        if (i < text.length) {
            element.textContent += text.charAt(i);
            i++;
            setTimeout(type, speed);
        }
    }
    
    type();
}

// ============================================
// Parallax Effect for Header
// ============================================

window.addEventListener('scroll', function() {
    const header = document.querySelector('.paper-header');
    if (header) {
        const scrolled = window.pageYOffset;
        const rate = scrolled * 0.3;
        header.style.backgroundPosition = `center ${rate}px`;
    }
});

// ============================================
// Interactive System Diagram
// ============================================

document.querySelectorAll('.diagram-block').forEach(block => {
    block.addEventListener('mouseenter', function() {
        this.style.transform = 'scale(1.05)';
        this.style.transition = 'transform 0.3s ease';
        this.style.zIndex = '10';
    });
    
    block.addEventListener('mouseleave', function() {
        this.style.transform = 'scale(1)';
        this.style.zIndex = '1';
    });
});

// ============================================
// Pipeline Step Highlighting
// ============================================

document.querySelectorAll('.pipeline-step').forEach((step, index) => {
    step.addEventListener('mouseenter', function() {
        // Highlight current and all previous steps
        document.querySelectorAll('.pipeline-step').forEach((s, i) => {
            if (i <= index) {
                s.style.background = 'linear-gradient(135deg, #c6f6d5 0%, #9ae6b4 100%)';
                s.style.borderColor = '#48bb78';
            }
        });
    });
    
    step.addEventListener('mouseleave', function() {
        document.querySelectorAll('.pipeline-step').forEach(s => {
            s.style.background = 'white';
            s.style.borderColor = '#e2e8f0';
        });
    });
});

// ============================================
// Feature Branch Comparison
// ============================================

document.querySelectorAll('.branch').forEach(branch => {
    branch.addEventListener('click', function() {
        // Remove active from all
        document.querySelectorAll('.branch').forEach(b => {
            b.classList.remove('active-branch');
        });
        // Add active to clicked
        this.classList.add('active-branch');
        
        // Show comparison info
        const branchType = this.classList.contains('rolling-var') ? 'rolling' :
                          this.classList.contains('sharp') ? 'sharp' : 'raw';
        showBranchInfo(branchType);
    });
});

function showBranchInfo(type) {
    const info = {
        rolling: {
            title: 'Rolling Variance (Proposed)',
            accuracy: '46.3% → 52.9%',
            speed: '0.027s',
            note: 'Best overall performance'
        },
        sharp: {
            title: 'SHARP Sanitisation',
            accuracy: '28.5%',
            speed: '3.508s',
            note: '100× slower, no accuracy gain'
        },
        raw: {
            title: 'Raw Amplitude',
            accuracy: '29.2%',
            speed: 'N/A',
            note: 'Baseline approach'
        }
    };
    
    console.log('Selected:', info[type]);
}

// ============================================
// CNN Layer Visualization
// ============================================

document.querySelectorAll('.layer-box').forEach(layer => {
    layer.addEventListener('mouseenter', function() {
        this.style.transform = 'scale(1.02)';
        this.style.boxShadow = '0 4px 15px rgba(0,0,0,0.15)';
        this.style.transition = 'all 0.3s ease';
    });
    
    layer.addEventListener('mouseleave', function() {
        this.style.transform = 'scale(1)';
        this.style.boxShadow = 'none';
    });
});

// ============================================
// Responsive Navigation
// ============================================

function createTableOfContents() {
    const sections = document.querySelectorAll('section h2');
    const toc = document.createElement('nav');
    toc.className = 'table-of-contents';
    
    sections.forEach((heading, index) => {
        const link = document.createElement('a');
        link.href = `#section-${index}`;
        link.textContent = heading.textContent;
        heading.id = `section-${index}`;
        toc.appendChild(link);
    });
    
    return toc;
}

// ============================================
// Print Optimization
// ============================================

window.addEventListener('beforeprint', function() {
    // Ensure all animations are complete before printing
    document.querySelectorAll('.animate-on-scroll').forEach(el => {
        el.classList.add('visible');
    });
    document.querySelectorAll('.animate-bar').forEach(bar => {
        bar.classList.add('visible');
    });
});

// ============================================
// Performance Optimization
// ============================================

// Debounce scroll events
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Throttle resize events
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Apply to scroll handler
const optimizedScrollHandler = debounce(function() {
    // Handle scroll-based updates
}, 10);

window.addEventListener('scroll', optimizedScrollHandler);

console.log('WiFi Sensing Paper - Interactive Page Loaded');
