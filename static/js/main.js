function animateCounter(el, target, duration = 1500) {
    if (!el) return;
    
    let start = 0;
    const prefix = el.dataset.prefix || "";
    const suffix = el.dataset.suffix || "";
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Ease out quadratic
        const ease = 1 - (1 - progress) * (1 - progress);
        const current = ease * target;

        if (suffix === "%") {
            el.textContent = prefix + current.toFixed(1) + suffix;
        } else if (prefix === "$") {
            el.textContent = prefix + current.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        } else {
            el.textContent = prefix + Math.floor(current).toLocaleString() + suffix;
        }

        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            if (suffix === "%") {
                el.textContent = prefix + target.toFixed(1) + suffix;
            } else if (prefix === "$") {
                el.textContent = prefix + target.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
            } else {
                el.textContent = prefix + target.toLocaleString() + suffix;
            }
        }
    }

    requestAnimationFrame(update);
}

// Sidebar toggle for mobile could be added here
document.addEventListener('DOMContentLoaded', () => {
    // Add active state to nav links if not handled by Jinja
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
});
