document.addEventListener('DOMContentLoaded', function() {
  // Fix navigation links that point to file:// URLs
  const allLinks = document.querySelectorAll('a');

  allLinks.forEach(function(link) {
    const href = link.getAttribute('href');

    // Skip links without href or external links
    if (!href || href.startsWith('http')) return;

    // Fix file:// links
    if (href.startsWith('file://')) {
      // Extract the file name from the path
      const pathParts = href.split('/');
      const fileName = pathParts[pathParts.length - 1];

      // Update href to point to the HTML file
      if (fileName) {
        const newHref = fileName.includes('.html') ? fileName : fileName + '.html';
        link.setAttribute('href', newHref);
      }
    }

    // Ensure .html extension for relative links
    if (!href.includes('#') && !href.endsWith('/') && !href.endsWith('.html')) {
      link.setAttribute('href', href + '.html');
    }
  });
});