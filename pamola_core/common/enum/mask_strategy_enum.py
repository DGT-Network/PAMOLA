from enum import Enum

class MaskStrategyEnum(str, Enum):
    """
    Enum representing supported masking strategies for data anonymization.

    Members:
    - FIXED: Apply mask using fixed start/end positions.
    - PATTERN: Apply mask using format patterns or regular expressions.
    - RANDOM: Randomly mask characters in the string.
    - WORDS: Mask word-by-word, preserving word boundaries if desired.
    """

    FIXED = "fixed"            # Mask using fixed start/end indexes
    PATTERN = "pattern"        # Mask using regex/format patterns
    RANDOM = "random"          # Random character masking
    WORDS = "words"            # Word-based masking strategy