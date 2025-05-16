/**
 * Improves the appearance of code copy buttons
 * Replaces text with clipboard icon for a cleaner look
 */
document.addEventListener('DOMContentLoaded', function() {
  // Enhance code copy buttons with a small delay to ensure they're loaded
  setTimeout(function() {
    const copyButtons = document.querySelectorAll('.copy');

    copyButtons.forEach(function(button) {
      // Get button text
      const buttonText = button.textContent || button.innerText;

      // Check if this is a copy button
      if (buttonText.includes('Copy')) {
        // Add class for styling
        button.classList.add('copy-button');

        // Create icon element
        const icon = document.createElement('i');
        icon.innerHTML = '📋';
        icon.style.fontSize = '16px';

        // Wrap text in span to allow hiding
        const textSpan = document.createElement('span');
        textSpan.className = 'copy-button-text';
        textSpan.textContent = buttonText;

        // Clear button content and add new elements
        button.innerHTML = '';
        button.appendChild(icon);
        button.appendChild(textSpan);

        // Add tooltip
        button.setAttribute('title', 'Copy to clipboard');
      }
    });
  }, 500); // Small delay to ensure all elements are loaded
});