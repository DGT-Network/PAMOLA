"""
Word cloud implementation for the HHR profiling system.

This module provides a focused implementation for word clouds using the wordcloud library
and PIL.Image. It's designed specifically for integration with the text profiler
to generate word frequency visualizations.
"""

import logging
from typing import Dict, List, Any, Optional, Union

import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS

from pamola_core.utils.vis_helpers.base import BaseFigure, FigureRegistry

# Configure logger
logger = logging.getLogger(__name__)


class WordCloudGenerator(BaseFigure):
    """Word cloud implementation using wordcloud library and PIL.Image."""

    def create(self,
               text_data: Union[str, List[str], Dict[str, float]],
               title: str,
               max_words: int = 200,
               background_color: str = "white",
               width: int = 800,
               height: int = 400,
               colormap: Optional[str] = "viridis",
               mask: Optional[np.ndarray] = None,
               contour_width: int = 1,
               contour_color: str = 'steelblue',
               exclude_words: Optional[List[str]] = None,
               **kwargs) -> Dict[str, Any]:
        """
        Create a word cloud visualization using wordcloud and PIL.Image.

        Parameters:
        -----------
        text_data : str, List[str], or Dict[str, float]
            Text data to visualize. If string, the raw text. If list, each item is a document.
            If dictionary, word-frequency pairs.
        title : str
            Title for the plot
        max_words : int
            Maximum number of words to include in the word cloud
        background_color : str
            Background color for the word cloud
        width : int
            Width of the word cloud image
        height : int
            Height of the word cloud image
        colormap : str, optional
            Matplotlib colormap to use for word colors
        mask : np.ndarray, optional
            Image mask for the word cloud
        contour_width : int
            Width of the contour line around the word cloud
        contour_color : str
            Color of the contour line
        exclude_words : List[str], optional
            List of words to exclude from the word cloud
        **kwargs:
            Additional arguments to pass to WordCloud

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing:
            - 'image': PIL.Image.Image object
            - 'title': str, the title of the visualization
            - 'wordcloud': WordCloud object for further manipulation
        """
        try:
            # Process text data based on input type
            word_frequencies = None

            if isinstance(text_data, dict):
                # Word-frequency dictionary
                word_frequencies = text_data
            else:
                # Get word frequencies from text data
                # Delegate to external text processing module
                from pamola_core.utils.text_processor import generate_word_frequencies

                # Convert to a single string if it's a list of strings
                if isinstance(text_data, list):
                    text = ' '.join(text_data)
                else:
                    text = text_data

                word_frequencies = generate_word_frequencies(text, exclude_words)

            # Check if we have any words to visualize
            if not word_frequencies:
                return self._create_empty_result(
                    title=title,
                    message="No valid words to create a word cloud",
                    width=width,
                    height=height
                )

            # Create stop words set
            stopwords = set(STOPWORDS)
            if exclude_words:
                stopwords.update(exclude_words)

            # Create word cloud
            wc = WordCloud(
                max_words=max_words,
                background_color=background_color,
                width=width,
                height=height,
                colormap=colormap,
                mask=mask,
                contour_width=contour_width,
                contour_color=contour_color,
                stopwords=stopwords,
                **kwargs
            )

            # Generate from frequencies or text
            if word_frequencies:
                wc.generate_from_frequencies(word_frequencies)

            # Convert wordcloud to image
            wordcloud_array = wc.to_array()
            image = Image.fromarray(wordcloud_array)

            # Return a dictionary with the result
            return {
                'image': image,
                'title': title,
                'wordcloud': wc
            }

        except ImportError as e:
            logger.error(f"Required package not installed: {e}")
            return self._create_empty_result(
                title=title,
                message=f"Required package not installed: {str(e)}",
                width=width,
                height=height
            )
        except Exception as e:
            logger.error(f"Error creating word cloud: {e}")
            return self._create_empty_result(
                title=title,
                message=f"Error creating word cloud: {str(e)}",
                width=width,
                height=height
            )

    @staticmethod
    def _create_empty_result(title: str, message: str, width: int, height: int) -> Dict[str, Any]:
        """
        Create an empty result when word cloud generation fails.

        Parameters:
        -----------
        title : str
            The title for the visualization
        message : str
            Error message to display
        width : int
            Width of the image
        height : int
            Height of the image

        Returns:
        --------
        Dict[str, Any]
            Result dictionary with a placeholder image
        """
        # Create a blank image with the error message
        image = Image.new('RGB', (width, height), color='white')

        # Return a dictionary with the placeholder
        return {
            'image': image,
            'title': title,
            'message': message,
            'is_error': True
        }

    @staticmethod
    def save_as_png(result: Dict[str, Any], file_path: str, dpi: int = 300) -> Optional[str]:
        """
        Saves a word cloud as a PNG file through the IO system.

        Parameters:
        -----------
        result : Dict[str, Any]
            Result dictionary from create()
        file_path : str
            Path where the PNG file should be saved
        dpi : int
            Resolution for the saved image

        Returns:
        --------
        Optional[str]
            Path to the saved file or None if an error occurred
        """
        if 'image' not in result:
            logger.error("No image found in the result")
            return None

        try:
            # Use the IO module's save_visualization function
            from pamola_core.utils.io import save_visualization
            saved_path = save_visualization(result, file_path, format="png", dpi=dpi)
            return str(saved_path)
        except Exception as e:
            logger.error(f"Error saving PNG file: {e}")
            return None


# Register implementation
FigureRegistry.register("wordcloud", "default", WordCloudGenerator)