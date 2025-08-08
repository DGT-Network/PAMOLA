"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Prompt Management Utilities
Package:       pamola_core.utils.nlp.llm.prompt
Version:       1.1.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause
Description:
This module provides utilities for managing, formatting, and optimizing
prompts for Large Language Models. It includes template management,
variable substitution, prompt engineering helpers, and pre-configured
prompt strategies for common tasks.

Key Features:
- Template-based prompt management with variable substitution
- Multi-language prompt support
- Prompt validation and optimization
- Pre-configured prompts for common tasks (anonymization, translation, etc.)
- Token counting and prompt truncation
- Prompt versioning and A/B testing support
- Context window management
- Few-shot and chain-of-thought prompt builders
- Enhanced validation for empty required variables
- Automatic cleanup of trailing punctuation

Framework:
Part of PAMOLA.CORE LLM utilities, providing high-level prompt management
for consistent and effective LLM interactions.

Changelog:
1.1.0 - Added validation for empty required variables
    - Added sanitize_colon_suffix for trailing punctuation cleanup
    - Enhanced PromptFormatter with post-processing options
    - Improved error messages for validation failures
1.0.0 - Initial implementation

Dependencies:
- Standard library for core functionality
- Optional: tiktoken for accurate token counting

TODO:
- Add prompt caching based on template and variables
- Implement prompt compression techniques
- Add support for dynamic few-shot selection
- Create prompt testing and evaluation framework
- Add prompt security validation (injection prevention)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from string import Template
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pamola_core.utils.nlp.base import DependencyManager

# Configure logger
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_PROMPT_LENGTH = 4000  # Conservative default
TOKEN_CHAR_RATIO = 4  # Approximate chars per token
TEMPLATE_PATTERN = re.compile(r'{(\w+)}')  # Pattern for {variable} placeholders
TRAILING_PUNCT_PATTERN = re.compile(r'[:—–-]\s*$', re.MULTILINE)  # Trailing punctuation


class PromptStrategy(Enum):
    """Enumeration of available prompt strategies."""
    DIRECT = "direct"  # Simple direct prompt
    INSTRUCTION = "instruction"  # Instruction-following format
    CHAT = "chat"  # Chat/conversation format
    FEW_SHOT = "few_shot"  # Few-shot learning format
    CHAIN_OF_THOUGHT = "chain_of_thought"  # Step-by-step reasoning
    ROLE_PLAY = "role_play"  # Role-based prompting


class PromptValidationError(Exception):
    """Exception raised when prompt validation fails."""
    pass


@dataclass
class PromptTemplate:
    """
    Container for prompt templates with metadata.

    Attributes
    ----------
    name : str
        Template identifier
    template : str
        Template string with {variable} placeholders
    description : str
        Template description
    strategy : PromptStrategy
        Prompting strategy type
    language : str
        Template language code (e.g., 'en', 'ru')
    version : str
        Template version
    variables : List[str]
        List of required variables
    optional_variables : List[str]
        List of optional variables
    metadata : Dict[str, Any]
        Additional template metadata
    examples : List[Dict[str, str]]
        Example variable values
    require_non_empty : bool
        Whether to enforce non-empty values for required variables
    """
    name: str
    template: str
    description: str = ""
    strategy: PromptStrategy = PromptStrategy.DIRECT
    language: str = "en"
    version: str = "1.0.0"
    variables: List[str] = field(default_factory=list)
    optional_variables: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, str]] = field(default_factory=list)
    require_non_empty: bool = True

    def __post_init__(self):
        """Extract variables from template if not provided."""
        if not self.variables:
            self.variables = self._extract_variables()

    def _extract_variables(self) -> List[str]:
        """Extract variable names from template."""
        # Find all {variable} patterns
        matches = TEMPLATE_PATTERN.findall(self.template)
        # Filter out optional variables
        required = [v for v in matches if v not in self.optional_variables]
        return list(set(required))  # Remove duplicates

    def validate_variables(self, provided_vars: Dict[str, Any]) -> None:
        """
        Validate that all required variables are provided and non-empty.

        Parameters
        ----------
        provided_vars : dict
            Variables provided for substitution

        Raises
        ------
        PromptValidationError
            If required variables are missing or empty
        """
        # Check for missing variables
        missing = set(self.variables) - set(provided_vars.keys())
        if missing:
            raise PromptValidationError(
                f"Missing required variables for template '{self.name}': {missing}"
            )

        # Check for empty required variables if enforce_non_empty is True
        if self.require_non_empty:
            empty_vars = []
            for var in self.variables:
                if var in provided_vars:
                    value = provided_vars[var]
                    # Check if the value is empty string
                    if isinstance(value, str) and not value.strip():
                        empty_vars.append(var)
                    # Check if the value is None
                    elif value is None:
                        empty_vars.append(var)

            if empty_vars:
                raise PromptValidationError(
                    f"Required variables cannot be empty for template '{self.name}': {empty_vars}"
                )

    def format(self, **kwargs) -> str:
        """
        Format template with provided variables.

        Parameters
        ----------
        **kwargs
            Variable values for substitution

        Returns
        -------
        str
            Formatted prompt

        Raises
        ------
        PromptValidationError
            If required variables are missing or empty
        """
        self.validate_variables(kwargs)

        # Use safe substitution to handle missing optional variables
        template_obj = Template(self.template)
        return template_obj.safe_substitute(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "template": self.template,
            "description": self.description,
            "strategy": self.strategy.value,
            "language": self.language,
            "version": self.version,
            "variables": self.variables,
            "optional_variables": self.optional_variables,
            "metadata": self.metadata,
            "examples": self.examples,
            "require_non_empty": self.require_non_empty
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTemplate':
        """Create from dictionary representation."""
        # Convert strategy string to enum
        if isinstance(data.get('strategy'), str):
            data['strategy'] = PromptStrategy(data['strategy'])
        return cls(**data)


@dataclass
class PromptConfig:
    """Configuration for prompt formatting and optimization."""
    max_length: int = DEFAULT_MAX_PROMPT_LENGTH
    truncation_strategy: str = "end"  # "end", "middle", "smart"
    preserve_examples: bool = True  # Keep examples when truncating
    add_ellipsis: bool = True  # Add ... when truncating
    token_counter: Optional[Callable[[str], int]] = None  # Custom token counter
    enable_validation: bool = True  # Validate prompts before formatting
    strip_extra_whitespace: bool = True  # Clean up formatting
    sanitize_trailing_punctuation: bool = True  # Remove trailing colons etc.


class PromptLibrary:
    """Library of pre-configured prompt templates."""

    # Built-in templates for common tasks
    BUILTIN_TEMPLATES = {
        "anonymize_experience_ru": PromptTemplate(
            name="anonymize_experience_ru",
            template="""Перепиши следующий текст опыта работы, удалив все личные и идентифицирующие данные (названия компаний, конкретные проекты, имена, специфичные технологии). Сохрани общий смысл и профессиональные навыки.

Исходный текст: {text}

Анонимизированный текст:""",
            description="Anonymize work experience in Russian",
            strategy=PromptStrategy.INSTRUCTION,
            language="ru",
            variables=["text"],
            require_non_empty=True
        ),

        "anonymize_experience_en": PromptTemplate(
            name="anonymize_experience_en",
            template="""Rewrite the following work experience removing all personal and identifying information (company names, specific projects, names, specific technologies). Preserve the general meaning and professional skills.

Original text: {text}

Anonymized text:""",
            description="Anonymize work experience in English",
            strategy=PromptStrategy.INSTRUCTION,
            language="en",
            variables=["text"],
            require_non_empty=True
        ),

        "extract_skills": PromptTemplate(
            name="extract_skills",
            template="""Extract all professional skills and technologies mentioned in the following text. List them as comma-separated values.

Text: {text}

Skills:""",
            description="Extract skills from text",
            strategy=PromptStrategy.INSTRUCTION,
            language="en",
            variables=["text"],
            require_non_empty=True
        ),

        "chat_anonymize": PromptTemplate(
            name="chat_anonymize",
            template="""<|system|>
You are a privacy expert specializing in text anonymization. Your task is to remove personal and identifying information while preserving meaning.
<|user|>
Please anonymize this text: {text}
<|assistant|>""",
            description="Chat-based anonymization prompt",
            strategy=PromptStrategy.CHAT,
            language="en",
            variables=["text"],
            require_non_empty=True
        ),

        "few_shot_anonymize": PromptTemplate(
            name="few_shot_anonymize",
            template="""Task: Anonymize work experience by removing company names and personal details.

Example 1:
Input: Worked at Google as a Senior Software Engineer on the AdWords team.
Output: Worked at a major technology company as a Senior Software Engineer on the advertising platform team.

Example 2:
Input: Developed iOS apps for Apple including work on iTunes Connect.
Output: Developed iOS apps for a leading technology company including work on app distribution platform.

Now anonymize this:
Input: {text}
Output:""",
            description="Few-shot learning for anonymization",
            strategy=PromptStrategy.FEW_SHOT,
            language="en",
            variables=["text"],
            require_non_empty=True,
            examples=[
                {"text": "Worked at Microsoft on Azure cloud services"},
                {"text": "Senior Developer at Facebook working on React framework"}
            ]
        )
    }

    def __init__(self, custom_templates: Optional[Dict[str, PromptTemplate]] = None):
        """
        Initialize prompt library.

        Parameters
        ----------
        custom_templates : dict, optional
            Additional custom templates to add
        """
        self.templates = self.BUILTIN_TEMPLATES.copy()
        if custom_templates:
            self.templates.update(custom_templates)

    def get(self, name: str) -> PromptTemplate:
        """
        Get template by name.

        Parameters
        ----------
        name : str
            Template name

        Returns
        -------
        PromptTemplate
            Requested template

        Raises
        ------
        KeyError
            If template not found
        """
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found. Available: {list(self.templates.keys())}")
        return self.templates[name]

    def add(self, template: PromptTemplate, overwrite: bool = False) -> None:
        """
        Add template to library.

        Parameters
        ----------
        template : PromptTemplate
            Template to add
        overwrite : bool
            Whether to overwrite existing template

        Raises
        ------
        ValueError
            If template exists and overwrite is False
        """
        if template.name in self.templates and not overwrite:
            raise ValueError(f"Template '{template.name}' already exists")
        self.templates[template.name] = template

    def list_templates(self, language: Optional[str] = None,
                       strategy: Optional[PromptStrategy] = None) -> List[str]:
        """
        List available templates with optional filtering.

        Parameters
        ----------
        language : str, optional
            Filter by language
        strategy : PromptStrategy, optional
            Filter by strategy

        Returns
        -------
        List[str]
            List of template names
        """
        templates = []
        for name, template in self.templates.items():
            if language and template.language != language:
                continue
            if strategy and template.strategy != strategy:
                continue
            templates.append(name)
        return sorted(templates)

    def export_to_file(self, filepath: Union[str, Path]) -> None:
        """Export all templates to JSON file."""
        path = Path(filepath)
        data = {
            name: template.to_dict()
            for name, template in self.templates.items()
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def import_from_file(self, filepath: Union[str, Path],
                         overwrite: bool = False) -> None:
        """Import templates from JSON file."""
        path = Path(filepath)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for name, template_data in data.items():
            template = PromptTemplate.from_dict(template_data)
            self.add(template, overwrite=overwrite)


class PromptFormatter:
    """Utility class for formatting and optimizing prompts."""

    def __init__(self, config: Optional[PromptConfig] = None):
        """
        Initialize formatter.

        Parameters
        ----------
        config : PromptConfig, optional
            Formatting configuration
        """
        self.config = config or PromptConfig()

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Parameters
        ----------
        text : str
            Text to count tokens for

        Returns
        -------
        int
            Estimated token count
        """
        if self.config.token_counter:
            return self.config.token_counter(text)

        # Try tiktoken if available
        tiktoken = DependencyManager.get_module('tiktoken')
        if tiktoken:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except Exception:
                pass

        # Fallback to character-based estimation
        return len(text) // TOKEN_CHAR_RATIO

    def truncate_text(self, text: str, max_tokens: int) -> Tuple[str, int]:
        """
        Truncate text to fit within token limit.

        Parameters
        ----------
        text : str
            Text to truncate
        max_tokens : int
            Maximum token count

        Returns
        -------
        tuple
            (truncated_text, tokens_removed)
        """
        current_tokens = self.estimate_tokens(text)

        if current_tokens <= max_tokens:
            return text, 0

        # Calculate target character count
        char_ratio = len(text) / current_tokens
        target_chars = int(max_tokens * char_ratio * 0.95)  # 95% to be safe

        if self.config.truncation_strategy == "end":
            truncated = self._truncate_end(text, target_chars)
        elif self.config.truncation_strategy == "middle":
            truncated = self._truncate_middle(text, target_chars)
        else:  # "smart"
            truncated = self._truncate_smart(text, target_chars)

        if self.config.add_ellipsis and not truncated.endswith('...'):
            truncated += '...'

        tokens_removed = current_tokens - self.estimate_tokens(truncated)
        return truncated, tokens_removed

    def _truncate_end(self, text: str, max_chars: int) -> str:
        """Truncate at the end, trying to preserve word boundaries."""
        if len(text) <= max_chars:
            return text

        truncated = text[:max_chars]
        # Try to truncate at word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.8:
            truncated = truncated[:last_space]

        return truncated.strip()

    def _truncate_middle(self, text: str, max_chars: int) -> str:
        """Truncate in the middle, keeping beginning and end."""
        if len(text) <= max_chars:
            return text

        half_chars = (max_chars - 5) // 2  # Reserve space for " ... "
        return f"{text[:half_chars]} ... {text[-half_chars:]}"

    def _truncate_smart(self, text: str, max_chars: int) -> str:
        """Smart truncation trying to preserve complete sentences."""
        if len(text) <= max_chars:
            return text

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        truncated = ""
        for sent in sentences:
            if len(truncated) + len(sent) + 1 < max_chars:
                truncated += sent + " "
            else:
                break

        return truncated.strip()

    def clean_whitespace(self, text: str) -> str:
        """Clean up extra whitespace in text."""
        if not self.config.strip_extra_whitespace:
            return text

        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        return text.strip()

    def sanitize_colon_suffix(self, text: str) -> str:
        """
        Remove trailing colons and similar punctuation.

        This is useful when variable substitution results in empty values,
        leaving trailing punctuation like "Анонимизируй:" or "Text—".

        Parameters
        ----------
        text : str
            Text to sanitize

        Returns
        -------
        str
            Text with trailing punctuation removed
        """
        if not self.config.sanitize_trailing_punctuation:
            return text

        # Remove trailing colons, em-dashes, en-dashes, and hyphens
        return TRAILING_PUNCT_PATTERN.sub('', text).rstrip()

    def format_prompt(self, template: PromptTemplate,
                      variables: Dict[str, Any]) -> str:
        """
        Format prompt from template with optimization.

        Parameters
        ----------
        template : PromptTemplate
            Template to use
        variables : dict
            Variables for substitution

        Returns
        -------
        str
            Formatted and optimized prompt
        """
        # Validate if enabled
        if self.config.enable_validation:
            template.validate_variables(variables)

        # Format template
        prompt = template.format(**variables)

        # Clean whitespace
        prompt = self.clean_whitespace(prompt)

        # Sanitize trailing punctuation if enabled
        prompt = self.sanitize_colon_suffix(prompt)

        # Check token limit
        tokens = self.estimate_tokens(prompt)
        if tokens > self.config.max_length:
            logger.warning(
                f"Prompt exceeds token limit ({tokens} > {self.config.max_length}), "
                f"truncating..."
            )
            # Truncate the variable content, not the template
            # This is a simplified approach - could be improved
            if 'text' in variables and isinstance(variables['text'], str):
                text_tokens = self.estimate_tokens(variables['text'])
                template_tokens = tokens - text_tokens

                max_text_tokens = self.config.max_length - template_tokens - 50  # Buffer
                truncated_text, _ = self.truncate_text(
                    variables['text'],
                    max_text_tokens
                )

                # Re-format with truncated text
                new_vars = variables.copy()
                new_vars['text'] = truncated_text
                prompt = template.format(**new_vars)

                # Re-apply post-processing
                prompt = self.clean_whitespace(prompt)
                prompt = self.sanitize_colon_suffix(prompt)

        return prompt


class PromptChainBuilder:
    """Builder for creating chain-of-thought prompts."""

    def __init__(self):
        """Initialize chain builder."""
        self.steps: List[str] = []
        self.context: Optional[str] = None
        self.examples: List[Tuple[str, str]] = []

    def set_context(self, context: str) -> 'PromptChainBuilder':
        """Set initial context."""
        self.context = context
        return self

    def add_step(self, step: str) -> 'PromptChainBuilder':
        """Add reasoning step."""
        self.steps.append(step)
        return self

    def add_example(self, input_text: str, output_text: str) -> 'PromptChainBuilder':
        """Add example input/output pair."""
        self.examples.append((input_text, output_text))
        return self

    def build(self, final_instruction: str) -> str:
        """Build the complete chain-of-thought prompt."""
        parts = []

        # Add context if provided
        if self.context:
            parts.append(f"Context: {self.context}\n")

        # Add examples if provided
        if self.examples:
            parts.append("Examples:")
            for i, (inp, out) in enumerate(self.examples, 1):
                parts.append(f"\nExample {i}:")
                parts.append(f"Input: {inp}")
                parts.append(f"Output: {out}")
            parts.append("")  # Empty line

        # Add reasoning steps
        if self.steps:
            parts.append("Let's think step by step:")
            for i, step in enumerate(self.steps, 1):
                parts.append(f"{i}. {step}")
            parts.append("")  # Empty line

        # Add final instruction
        parts.append(final_instruction)

        return "\n".join(parts)


# Convenience functions
def create_prompt_formatter(config: Optional[PromptConfig] = None) -> PromptFormatter:
    """Create a prompt formatter with optional configuration."""
    return PromptFormatter(config)


def load_prompt_library(template_file: Optional[Union[str, Path]] = None) -> PromptLibrary:
    """
    Load prompt library with optional custom templates.

    Parameters
    ----------
    template_file : Path, optional
        Path to JSON file with custom templates

    Returns
    -------
    PromptLibrary
        Initialized prompt library
    """
    library = PromptLibrary()

    if template_file:
        library.import_from_file(template_file)

    return library