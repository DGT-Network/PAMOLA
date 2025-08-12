/**
 * Custom JavaScript for PAMOLA.CORE Project Documentation
 */

document.addEventListener('DOMContentLoaded', function() {
  console.log('PAMOLA.CORE Documentation Scripts loaded');

  // Инициализация функций
  setupCodeCopy();
  enhanceImages();
  addTableClasses();
  setupCollapsibleSections();

  // Если мы на главной странице проекта
  if (document.querySelector('.language-switcher')) {
    handleLanguageSwitch();
  }
});

/**
 * Setup code block
 */
function setupCodeCopy() {
  // Find all code blocks
  const codeBlocks = document.querySelectorAll('pre');

  codeBlocks.forEach(function(block, index) {
    // Create wrapper for copy functionality
    block.classList.add('copyable-code');

    // Create copy button
    const copyButton = document.createElement('button');
    copyButton.className = 'copy-button';
    copyButton.innerHTML = '<span class="copy-button-icon">📋</span><span class="copy-button-text">Copy</span>';

    copyButton.setAttribute('aria-label', 'Copy code to clipboard');

    // Add button to block
    block.appendChild(copyButton);

    // Add copy functionality
    copyButton.addEventListener('click', function() {
      const code = block.querySelector('code')
        ? block.querySelector('code').innerText
        : block.innerText.replace('Copy', '').trim(); // Remove "Copy" text from selection

      navigator.clipboard.writeText(code).then(function() {
        // Change button text for feedback
        const textSpan = copyButton.querySelector('.copy-button-text');
        const originalText = textSpan.textContent;
        textSpan.textContent = 'Copied!';

        // Reset after 2 seconds
        setTimeout(function() {
          textSpan.textContent = originalText;
        }, 2000);
      }).catch(function(err) {
        console.error('Could not copy text: ', err);
      });
    });
  });
}

/**
 * Улучшение отображения изображений
 */
function enhanceImages() {
  const images = document.querySelectorAll('.rst-content img');

  images.forEach(function(img) {
    // Пропускаем изображения, которые уже находятся внутри figure
    if (img.parentNode.tagName === 'FIGURE' || img.classList.contains('no-enhance')) {
      return;
    }

    // Создаем обертку figure
    const figure = document.createElement('figure');
    figure.className = 'figure';

    // Клонируем изображение
    const newImg = img.cloneNode(true);

    // Добавляем атрибут alt в качестве подписи, если он существует
    if (img.alt && img.alt.trim() !== '') {
      const caption = document.createElement('figcaption');
      caption.className = 'figure-caption';
      caption.textContent = img.alt;
      figure.appendChild(newImg);
      figure.appendChild(caption);
    } else {
      figure.appendChild(newImg);
    }

    // Заменяем оригинальное изображение на figure
    img.parentNode.replaceChild(figure, img);
  });
}

/**
 * Добавление классов к таблицам для улучшения стиля
 */
function addTableClasses() {
  const tables = document.querySelectorAll('.rst-content table');

  tables.forEach(function(table) {
    table.classList.add('responsive-table');

    // Оборачиваем таблицу в div для горизонтальной прокрутки
    const wrapper = document.createElement('div');
    wrapper.className = 'table-wrapper';
    table.parentNode.insertBefore(wrapper, table);
    wrapper.appendChild(table);
  });
}

/**
 * Настройка сворачиваемых разделов
 */
function setupCollapsibleSections() {
  const details = document.querySelectorAll('details');

  details.forEach(function(detail) {
    // Добавляем иконку для обозначения состояния
    const summary = detail.querySelector('summary');
    if (summary) {
      summary.innerHTML = '<span class="toggle-icon">▶</span> ' + summary.innerHTML;

      // Обновляем иконку при изменении состояния
      detail.addEventListener('toggle', function() {
        const icon = detail.querySelector('.toggle-icon');
        if (detail.open) {
          icon.textContent = '▼';
        } else {
          icon.textContent = '▶';
        }
      });
    }
  });
}

/**
 * Обработка переключения языка
 */
function handleLanguageSwitch() {
  const links = document.querySelectorAll('.language-link');

  links.forEach(function(link) {
    link.addEventListener('click', function(e) {
      // Получаем целевой URL
      const targetUrl = link.getAttribute('href');

      // Определяем текущий язык из URL
      const currentPath = window.location.pathname;
      let currentLanguage = 'en';

      if (currentPath.includes('/ru/')) {
        currentLanguage = 'ru';
      } else if (currentPath.includes('/en/')) {
        currentLanguage = 'en';
      }

      // Если мы уже на целевом языке, предотвращаем переход
      if ((targetUrl.includes('/en/') && currentLanguage === 'en') ||
          (targetUrl.includes('/ru/') && currentLanguage === 'ru')) {
        e.preventDefault();
      }
    });
  });
}

/**
 * Функция для расчета времени чтения страницы
 */
function calculateReadingTime() {
  const content = document.querySelector('.rst-content');
  if (!content) return;

  // Получаем весь текст
  const text = content.textContent;

  // Считаем количество слов (приблизительно)
  const words = text.split(/\s+/).length;

  // Средняя скорость чтения - 200 слов в минуту
  const readingTime = Math.ceil(words / 200);

  // Создаем элемент для отображения времени чтения
  const readingTimeEl = document.createElement('div');
  readingTimeEl.className = 'reading-time';

  if (document.documentElement.lang === 'ru') {
    readingTimeEl.innerHTML = `Время чтения: ${readingTime} мин.`;
  } else {
    readingTimeEl.innerHTML = `Reading time: ${readingTime} min`;
  }

  // Добавляем элемент в начало содержимого
  const heading = content.querySelector('h1');
  if (heading) {
    heading.insertAdjacentElement('afterend', readingTimeEl);
  }
}

// Вызываем функцию расчета времени чтения при загрузке страницы
document.addEventListener('DOMContentLoaded', calculateReadingTime);