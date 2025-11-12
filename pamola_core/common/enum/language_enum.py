"""
PAMOLA.CORE - Language Enumeration
----------------------------------
Module:        language.py
Package:       pamola_core.common.enum
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-12
License:       BSD 3-Clause

Description:
    Simple language enumeration for multi-language support.
    Supports English, Vietnamese, and Russian.

Usage:
    from pamola_core.common.enum.language import Language
    
    # Use in code
    lang = Language.ENGLISH
    print(lang.value)  # "en"
"""

from enum import Enum


class Language(str, Enum):
    """Supported languages."""

    ENGLISH = "en"  # English
    VIETNAMESE = "vi"  # Vietnamese (Tiếng Việt)
    RUSSIAN = "ru"  # Russian (Русский)