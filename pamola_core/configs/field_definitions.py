"""
PAMOLA.CORE (Privacy-Preserving AI Data Processors) - Anonymization) - Field Definitions
--------------------------------------------------
This module defines the metadata for all fields in the resume dataset,
including field types, privacy categories, anonymization strategies,
and profiling tasks.

Categories:
- Direct Identifier: Directly identifies an individual (e.g., name, email)
- Indirect Identifier: Can identify when combined with other data (e.g., IDs)
- Quasi-Identifier: May contribute to identification in combination (e.g., age, gender)
- Sensitive Attribute: Private information that should be protected (e.g., salary)
- Non-Sensitive: Information that poses minimal privacy risk

Anonymization Strategies:
- PSEUDONYMIZATION: Replace with synthetic identifier that preserves relationships
- GENERALIZATION: Replace specific values with more general ones (e.g., age -> age group)
- SUPPRESSION: Remove the data completely
- KEEP_ORIGINAL: Retain the original value without modification
- NOISE_ADDITION: Add random noise to numerical values
- NER/LLM_RECONSTRUCTION: Use NLP to identify and replace named entities

Field Types:
- short_text: Short text fields (typically < 100 chars)
- long_text: Longer text fields that may contain multiple values or paragraphs
- double: Floating-point numeric values
- long: Integer numeric values
- date: Date values (may be stored as text but with date semantics)

(C) 2025 BDA

Author: V.Khvatov
"""

from enum import Enum
from typing import Dict, List, Any


class FieldType(str, Enum):
    """Enumeration of field data types."""
    SHORT_TEXT = "short_text"
    LONG_TEXT = "long_text"
    DOUBLE = "double"
    LONG = "long"
    DATE = "date"


class PrivacyCategory(str, Enum):
    """Enumeration of privacy categories for fields."""
    DIRECT_IDENTIFIER = "Direct Identifier"
    INDIRECT_IDENTIFIER = "Indirect Identifier"
    QUASI_IDENTIFIER = "Quasi-Identifier"
    SENSITIVE_ATTRIBUTE = "Sensitive Attribute"
    NON_SENSITIVE = "Non-Sensitive"


class AnonymizationStrategy(str, Enum):
    """Enumeration of anonymization strategies."""
    PSEUDONYMIZATION = "PSEUDONYMIZATION"
    GENERALIZATION = "GENERALIZATION"
    SUPPRESSION = "SUPPRESSION"
    KEEP_ORIGINAL = "KEEP_ORIGINAL"
    NOISE_ADDITION = "NOISE_ADDITION"
    NER_LLM_RECONSTRUCTION = "NER/LLM RECONSTRUCTION"


class ProfilingTask(str, Enum):
    """Enumeration of profiling tasks."""
    COMPLETENESS = "completeness"  # Check for missing values
    UNIQUENESS = "uniqueness"  # Count unique values
    DISTRIBUTION = "distribution"  # Distribution of values
    FREQUENCY = "frequency"  # Frequency counts
    OUTLIERS = "outliers"  # Identify outliers
    RARE_VALUES = "rare_values"  # Identify rare/unique values
    FORMAT_VALIDATION = "format_validation"  # Validate format (e.g., dates)
    CORRELATION = "correlation"  # Correlation with other fields
    TEXT_ANALYSIS = "text_analysis"  # Analyze text content
    LENGTH_ANALYSIS = "length_analysis"  # Analyze text length
    PATTERN_DETECTION = "pattern_detection"  # Detect patterns


# Table definitions
TABLES = {
    "IDENTIFICATION": [
        "resume_id", "first_name", "last_name", "middle_name",
        "gender", "birth_day", "file_as", "UID"
    ],
    "RESUME_DETAILS": [
        "UID", "post", "education_level", "salary", "salary_currency",
        "area_name", "relocation", "metro_station_name", "road_time_type",
        "business_trip_readiness", "work_schedules", "employments",
        "driver_license_types", "has_vehicle"
    ],
    "CONTACTS": [
        "UID", "email", "home_phone", "work_phone", "cell_phone"
    ],
    "SPECIALIZATION": [
        "UID", "key_skill_names", "specialization_names"
    ],
    "ATTESTATION": [
        "UID", "attestation_education_names", "attestation_education_results",
        "attestation_education_organizations", "attestation_education_end_dates"
    ],
    "PRIMARY_EDU": [
        "UID", "primary_education_names", "primary_education_faculties",
        "primary_education_diplomas", "primary_education_end_dates"
    ],
    "ADDITIONAL_EDU": [
        "UID", "additional_education_names", "additional_education_organizations",
        "additional_education_diplomas", "additional_education_end_dates"
    ],
    "ELEMENTARY_EDU": [
        "UID", "elementary_education_names", "elementary_education_end_dates"
    ],
    "EXPERIENCE": [
        "UID", "experience_start_dates", "experience_end_dates",
        "experience_organizations", "experience_descriptions",
        "experience_posts", "experience_company_urls"
    ]
}

# Field definitions
FIELD_DEFINITIONS = {
    # ID fields
    "resume_id": {
        "id": 1,
        "type": FieldType.LONG,
        "category": PrivacyCategory.INDIRECT_IDENTIFIER,
        "strategy": AnonymizationStrategy.PSEUDONYMIZATION,
        "description": "Уникальный идентификатор резюме, требует обратимой анонимизации для связи записей",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.UNIQUENESS,
            ProfilingTask.DISTRIBUTION,
            ProfilingTask.FREQUENCY
        ],
        "table": "IDENTIFICATION"
    },

    # Personal identifiers
    "first_name": {
        "id": 3,
        "type": FieldType.SHORT_TEXT,
        "category": PrivacyCategory.DIRECT_IDENTIFIER,
        "strategy": AnonymizationStrategy.PSEUDONYMIZATION,
        "description": "Имя человека, полная замена на синтетический идентификатор",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.UNIQUENESS,
            ProfilingTask.FREQUENCY,
            ProfilingTask.RARE_VALUES,
            ProfilingTask.TEXT_ANALYSIS
        ],
        "table": "IDENTIFICATION"
    },
    "last_name": {
        "id": 4,
        "type": FieldType.SHORT_TEXT,
        "category": PrivacyCategory.DIRECT_IDENTIFIER,
        "strategy": AnonymizationStrategy.PSEUDONYMIZATION,
        "description": "Фамилия человека, полная замена на синтетический идентификатор",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.UNIQUENESS,
            ProfilingTask.FREQUENCY,
            ProfilingTask.RARE_VALUES,
            ProfilingTask.TEXT_ANALYSIS
        ],
        "table": "IDENTIFICATION"
    },
    "middle_name": {
        "id": 5,
        "type": FieldType.SHORT_TEXT,
        "category": PrivacyCategory.QUASI_IDENTIFIER,
        "strategy": AnonymizationStrategy.PSEUDONYMIZATION,
        "description": "Отчество, может отсутствовать",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.UNIQUENESS,
            ProfilingTask.FREQUENCY,
            ProfilingTask.RARE_VALUES
        ],
        "table": "IDENTIFICATION"
    },

    # Demographics
    "gender": {
        "id": 6,
        "type": FieldType.SHORT_TEXT,
        "category": PrivacyCategory.QUASI_IDENTIFIER,
        "strategy": AnonymizationStrategy.KEEP_ORIGINAL,
        "description": "Пол человека (Мужчина, Женщина, может отсутствовать)",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.DISTRIBUTION,
            ProfilingTask.FREQUENCY
        ],
        "table": "IDENTIFICATION"
    },
    "birth_day": {
        "id": 7,
        "type": FieldType.DATE,
        "category": PrivacyCategory.QUASI_IDENTIFIER,
        "strategy": AnonymizationStrategy.GENERALIZATION,
        "description": "Дата рождения в формате YYYY-MM-DD, обобщение до возрастных групп",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.DISTRIBUTION,
            ProfilingTask.FORMAT_VALIDATION,
            ProfilingTask.OUTLIERS
        ],
        "table": "IDENTIFICATION"
    },

    # Job details
    "post": {
        "id": 2,
        "type": FieldType.SHORT_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.GENERALIZATION,
        "description": "Должность, на которую претендует человек, обобщение до категорий должностей",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY,
            ProfilingTask.TEXT_ANALYSIS,
            ProfilingTask.PATTERN_DETECTION
        ],
        "table": "RESUME_DETAILS"
    },
    "education_level": {
        "id": 8,
        "type": FieldType.SHORT_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.KEEP_ORIGINAL,
        "description": "Уровень образования (категориальные значения)",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY,
            ProfilingTask.DISTRIBUTION
        ],
        "table": "RESUME_DETAILS"
    },
    "salary": {
        "id": 9,
        "type": FieldType.DOUBLE,
        "category": PrivacyCategory.SENSITIVE_ATTRIBUTE,
        "strategy": AnonymizationStrategy.NOISE_ADDITION,
        "description": "Зарплата (много нулей), добавление случайного шума",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.DISTRIBUTION,
            ProfilingTask.OUTLIERS,
            ProfilingTask.CORRELATION
        ],
        "table": "RESUME_DETAILS"
    },
    "salary_currency": {
        "id": 10,
        "type": FieldType.SHORT_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.KEEP_ORIGINAL,
        "description": "Валюта зарплаты",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY,
            ProfilingTask.CORRELATION
        ],
        "table": "RESUME_DETAILS"
    },

    # Location
    "area_name": {
        "id": 11,
        "type": FieldType.SHORT_TEXT,
        "category": PrivacyCategory.QUASI_IDENTIFIER,
        "strategy": AnonymizationStrategy.GENERALIZATION,
        "description": "Название региона или города, обобщение до региона",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY,
            ProfilingTask.CORRELATION
        ],
        "table": "RESUME_DETAILS"
    },
    "metro_station_name": {
        "id": 17,
        "type": FieldType.SHORT_TEXT,
        "category": PrivacyCategory.QUASI_IDENTIFIER,
        "strategy": AnonymizationStrategy.GENERALIZATION,
        "description": "Название станции метро, обобщение до района",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY,
            ProfilingTask.CORRELATION
        ],
        "table": "RESUME_DETAILS"
    },

    # Preferences
    "relocation": {
        "id": 12,
        "type": FieldType.SHORT_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.KEEP_ORIGINAL,
        "description": "Готовность к переезду",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY,
            ProfilingTask.CORRELATION
        ],
        "table": "RESUME_DETAILS"
    },
    "road_time_type": {
        "id": 18,
        "type": FieldType.SHORT_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.KEEP_ORIGINAL,
        "description": "Тип времени в пути",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY
        ],
        "table": "RESUME_DETAILS"
    },
    "business_trip_readiness": {
        "id": 19,
        "type": FieldType.SHORT_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.KEEP_ORIGINAL,
        "description": "Готовность к командировкам (справочник - мало значений)",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY
        ],
        "table": "RESUME_DETAILS"
    },
    "work_schedules": {
        "id": 20,
        "type": FieldType.LONG_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.GENERALIZATION,
        "description": "График работы, кодирование списка значений",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY,
            ProfilingTask.TEXT_ANALYSIS,
            ProfilingTask.PATTERN_DETECTION
        ],
        "table": "RESUME_DETAILS"
    },
    "employments": {
        "id": 21,
        "type": FieldType.LONG_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.GENERALIZATION,
        "description": "Типы занятости, кодирование списка значений",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY,
            ProfilingTask.TEXT_ANALYSIS,
            ProfilingTask.PATTERN_DETECTION
        ],
        "table": "RESUME_DETAILS"
    },

    # Transportation
    "driver_license_types": {
        "id": 22,
        "type": FieldType.SHORT_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.KEEP_ORIGINAL,
        "description": "Типы водительских прав",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY,
            ProfilingTask.PATTERN_DETECTION
        ],
        "table": "RESUME_DETAILS"
    },
    "has_vehicle": {
        "id": 23,
        "type": FieldType.SHORT_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.KEEP_ORIGINAL,
        "description": "Наличие транспортного средства",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY,
            ProfilingTask.CORRELATION
        ],
        "table": "RESUME_DETAILS"
    },

    # Contact information
    "email": {
        "id": 13,
        "type": FieldType.SHORT_TEXT,
        "category": PrivacyCategory.DIRECT_IDENTIFIER,
        "strategy": AnonymizationStrategy.SUPPRESSION,
        "description": "Электронная почта - прямой идентификатор, удаление",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.UNIQUENESS,
            ProfilingTask.PATTERN_DETECTION,
            ProfilingTask.TEXT_ANALYSIS
        ],
        "table": "CONTACTS"
    },
    "home_phone": {
        "id": 14,
        "type": FieldType.SHORT_TEXT,
        "category": PrivacyCategory.DIRECT_IDENTIFIER,
        "strategy": AnonymizationStrategy.SUPPRESSION,
        "description": "Домашний телефон, удаление",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.UNIQUENESS,
            ProfilingTask.PATTERN_DETECTION
        ],
        "table": "CONTACTS"
    },
    "work_phone": {
        "id": 15,
        "type": FieldType.SHORT_TEXT,
        "category": PrivacyCategory.DIRECT_IDENTIFIER,
        "strategy": AnonymizationStrategy.SUPPRESSION,
        "description": "Рабочий телефон, удаление",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.UNIQUENESS,
            ProfilingTask.PATTERN_DETECTION
        ],
        "table": "CONTACTS"
    },
    "cell_phone": {
        "id": 16,
        "type": FieldType.LONG_TEXT,
        "category": PrivacyCategory.DIRECT_IDENTIFIER,
        "strategy": AnonymizationStrategy.SUPPRESSION,
        "description": "Мобильный телефон, удаление",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.UNIQUENESS,
            ProfilingTask.PATTERN_DETECTION,
            ProfilingTask.TEXT_ANALYSIS
        ],
        "table": "CONTACTS"
    },

    # Skills and specializations
    "key_skill_names": {
        "id": 30,
        "type": FieldType.LONG_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.GENERALIZATION,
        "description": "Названия ключевых навыков, группировка по категориям",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY,
            ProfilingTask.TEXT_ANALYSIS,
            ProfilingTask.PATTERN_DETECTION
        ],
        "table": "SPECIALIZATION"
    },
    "specialization_names": {
        "id": 39,
        "type": FieldType.LONG_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.GENERALIZATION,
        "description": "Названия специализаций, группировка",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY,
            ProfilingTask.TEXT_ANALYSIS,
            ProfilingTask.PATTERN_DETECTION
        ],
        "table": "SPECIALIZATION"
    },

    # Additional education
    "additional_education_names": {
        "id": 24,
        "type": FieldType.LONG_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.NER_LLM_RECONSTRUCTION,
        "description": "Названия дополнительного образования, обработка именованных сущностей",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.TEXT_ANALYSIS,
            ProfilingTask.PATTERN_DETECTION
        ],
        "table": "ADDITIONAL_EDU"
    },
    "additional_education_organizations": {
        "id": 25,
        "type": FieldType.LONG_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.NER_LLM_RECONSTRUCTION,
        "description": "Организации дополнительного образования, обработка именованных сущностей",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY,
            ProfilingTask.TEXT_ANALYSIS
        ],
        "table": "ADDITIONAL_EDU"
    },
    "additional_education_diplomas": {
        "id": 26,
        "type": FieldType.SHORT_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.GENERALIZATION,
        "description": "Дипломы дополнительного образования",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY
        ],
        "table": "ADDITIONAL_EDU"
    },
    "additional_education_end_dates": {
        "id": 27,
        "type": FieldType.DATE,
        "category": PrivacyCategory.QUASI_IDENTIFIER,
        "strategy": AnonymizationStrategy.GENERALIZATION,
        "description": "Даты окончания дополнительного образования, обобщение до года",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.DISTRIBUTION,
            ProfilingTask.FORMAT_VALIDATION
        ],
        "table": "ADDITIONAL_EDU"
    },

    # Elementary education
    "elementary_education_names": {
        "id": 28,
        "type": FieldType.SHORT_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.NER_LLM_RECONSTRUCTION,
        "description": "Названия начального образования",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY,
            ProfilingTask.TEXT_ANALYSIS
        ],
        "table": "ELEMENTARY_EDU"
    },
    "elementary_education_end_dates": {
        "id": 29,
        "type": FieldType.DATE,
        "category": PrivacyCategory.QUASI_IDENTIFIER,
        "strategy": AnonymizationStrategy.GENERALIZATION,
        "description": "Даты окончания начального образования, обобщение до года",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.DISTRIBUTION,
            ProfilingTask.FORMAT_VALIDATION
        ],
        "table": "ELEMENTARY_EDU"
    },

    # Attestation education
    "attestation_education_names": {
        "id": 31,
        "type": FieldType.LONG_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.NER_LLM_RECONSTRUCTION,
        "description": "Названия аттестационного образования",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY,
            ProfilingTask.TEXT_ANALYSIS
        ],
        "table": "ATTESTATION"
    },
    "attestation_education_results": {
        "id": 32,
        "type": FieldType.SHORT_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.GENERALIZATION,
        "description": "Результаты аттестационного образования",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY
        ],
        "table": "ATTESTATION"
    },
    "attestation_education_organizations": {
        "id": 33,
        "type": FieldType.LONG_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.NER_LLM_RECONSTRUCTION,
        "description": "Организации аттестационного образования",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY,
            ProfilingTask.TEXT_ANALYSIS
        ],
        "table": "ATTESTATION"
    },
    "attestation_education_end_dates": {
        "id": 34,
        "type": FieldType.DATE,
        "category": PrivacyCategory.QUASI_IDENTIFIER,
        "strategy": AnonymizationStrategy.GENERALIZATION,
        "description": "Даты окончания аттестационного образования, обобщение до года",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.DISTRIBUTION,
            ProfilingTask.FORMAT_VALIDATION
        ],
        "table": "ATTESTATION"
    },

    # Primary education
    "primary_education_names": {
        "id": 35,
        "type": FieldType.LONG_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.NER_LLM_RECONSTRUCTION,
        "description": "Названия основного образования (университет)",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY,
            ProfilingTask.TEXT_ANALYSIS
        ],
        "table": "PRIMARY_EDU"
    },
    "primary_education_faculties": {
        "id": 36,
        "type": FieldType.SHORT_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.GENERALIZATION,
        "description": "Факультеты основного образования",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY,
            ProfilingTask.TEXT_ANALYSIS
        ],
        "table": "PRIMARY_EDU"
    },
    "primary_education_diplomas": {
        "id": 37,
        "type": FieldType.LONG_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.GENERALIZATION,
        "description": "Дипломы основного образования",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY
        ],
        "table": "PRIMARY_EDU"
    },
    "primary_education_end_dates": {
        "id": 38,
        "type": FieldType.DATE,
        "category": PrivacyCategory.QUASI_IDENTIFIER,
        "strategy": AnonymizationStrategy.GENERALIZATION,
        "description": "Даты окончания основного образования, обобщение до года",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.DISTRIBUTION,
            ProfilingTask.FORMAT_VALIDATION
        ],
        "table": "PRIMARY_EDU"
    },

    # Work experience
    "experience_start_dates": {
        "id": 40,
        "type": FieldType.DATE,
        "category": PrivacyCategory.QUASI_IDENTIFIER,
        "strategy": AnonymizationStrategy.GENERALIZATION,
        "description": "Даты начала опыта работы, обобщение до года",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.DISTRIBUTION,
            ProfilingTask.FORMAT_VALIDATION
        ],
        "table": "EXPERIENCE"
    },
    "experience_end_dates": {
        "id": 41,
        "type": FieldType.DATE,
        "category": PrivacyCategory.QUASI_IDENTIFIER,
        "strategy": AnonymizationStrategy.GENERALIZATION,
        "description": "Даты окончания опыта работы, обобщение до года",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.DISTRIBUTION,
            ProfilingTask.FORMAT_VALIDATION
        ],
        "table": "EXPERIENCE"
    },
    "experience_organizations": {
        "id": 42,
        "type": FieldType.LONG_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.NER_LLM_RECONSTRUCTION,
        "description": "Организации, в которых работал человек",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY,
            ProfilingTask.TEXT_ANALYSIS
        ],
        "table": "EXPERIENCE"
    },
    "experience_descriptions": {
        "id": 43,
        "type": FieldType.LONG_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.NER_LLM_RECONSTRUCTION,
        "description": "Описания опыта работы",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.TEXT_ANALYSIS,
            ProfilingTask.LENGTH_ANALYSIS
        ],
        "table": "EXPERIENCE"
    },
    "experience_posts": {
        "id": 44,
        "type": FieldType.LONG_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.GENERALIZATION,
        "description": "Должности в опыте работы",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY,
            ProfilingTask.TEXT_ANALYSIS
        ],
        "table": "EXPERIENCE"
    },
    "experience_company_urls": {
        "id": 45,
        "type": FieldType.SHORT_TEXT,
        "category": PrivacyCategory.NON_SENSITIVE,
        "strategy": AnonymizationStrategy.SUPPRESSION,
        "description": "URL компаний из опыта работы",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.FREQUENCY,
            ProfilingTask.PATTERN_DETECTION
        ],
        "table": "EXPERIENCE"
    },

    # Custom fields for data linkage
    "file_as": {
        "id": 100,
        "type": FieldType.SHORT_TEXT,
        "category": PrivacyCategory.INDIRECT_IDENTIFIER,
        "strategy": AnonymizationStrategy.PSEUDONYMIZATION,
        "description": "Составное поле для идентификации (имя + фамилия)",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.UNIQUENESS,
            ProfilingTask.FREQUENCY
        ],
        "table": "IDENTIFICATION"
    },
    "UID": {
        "id": 101,
        "type": FieldType.SHORT_TEXT,
        "category": PrivacyCategory.INDIRECT_IDENTIFIER,
        "strategy": AnonymizationStrategy.PSEUDONYMIZATION,
        "description": "Уникальный идентификатор человека (хеш от комбинации идентифицирующих полей)",
        "profiling_tasks": [
            ProfilingTask.COMPLETENESS,
            ProfilingTask.UNIQUENESS
        ],
        "table": "IDENTIFICATION"
    }
}


# Helper functions
def get_fields_by_table(table_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Get all field definitions for a specific table.

    Parameters:
    -----------
    table_name : str
        Name of the table

    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Dictionary of field definitions for the specified table
    """
    return {
        field_name: field_def
        for field_name, field_def in FIELD_DEFINITIONS.items()
        if field_def.get("table") == table_name
    }


def get_fields_by_category(category: PrivacyCategory) -> Dict[str, Dict[str, Any]]:
    """
    Get all field definitions for a specific privacy category.

    Parameters:
    -----------
    category : PrivacyCategory
        Privacy category to filter by

    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Dictionary of field definitions for the specified category
    """
    return {
        field_name: field_def
        for field_name, field_def in FIELD_DEFINITIONS.items()
        if field_def.get("category") == category
    }


def get_fields_by_strategy(strategy: AnonymizationStrategy) -> Dict[str, Dict[str, Any]]:
    """
    Get all field definitions for a specific anonymization strategy.

    Parameters:
    -----------
    strategy : AnonymizationStrategy
        Anonymization strategy to filter by

    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Dictionary of field definitions for the specified strategy
    """
    return {
        field_name: field_def
        for field_name, field_def in FIELD_DEFINITIONS.items()
        if field_def.get("strategy") == strategy.value  # Используем .value для получения строкового значения
    }


def get_fields_by_type(field_type: FieldType) -> Dict[str, Dict[str, Any]]:
    """
    Get all field definitions for a specific field type.

    Parameters:
    -----------
    field_type : FieldType
        Field type to filter by

    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Dictionary of field definitions for the specified type
    """
    return {
        field_name: field_def
        for field_name, field_def in FIELD_DEFINITIONS.items()
        if field_def.get("type") == field_type
    }


def get_profiling_tasks_for_field(field_name: str) -> List[ProfilingTask]:
    """
    Get all profiling tasks for a specific field.

    Parameters:
    -----------
    field_name : str
        Name of the field

    Returns:
    --------
    List[ProfilingTask]
        List of profiling tasks for the field
    """
    field_def = FIELD_DEFINITIONS.get(field_name)
    if not field_def:
        return []

    return field_def.get("profiling_tasks", [])


def get_field_definition(field_name: str) -> Dict[str, Any]:
    """
    Get the definition for a specific field.

    Parameters:
    -----------
    field_name : str
        Name of the field

    Returns:
    --------
    Dict[str, Any]
        Field definition dictionary or empty dict if not found
    """
    return FIELD_DEFINITIONS.get(field_name, {})


def is_identifier(field_name: str) -> bool:
    """
    Check if a field is any type of identifier (direct, indirect, or quasi).

    Parameters:
    -----------
    field_name : str
        Name of the field

    Returns:
    --------
    bool
        True if the field is any kind of identifier, False otherwise
    """
    field_def = get_field_definition(field_name)
    if not field_def:
        return False

    category = field_def.get("category")
    return category in [
        PrivacyCategory.DIRECT_IDENTIFIER,
        PrivacyCategory.INDIRECT_IDENTIFIER,
        PrivacyCategory.QUASI_IDENTIFIER
    ]


def get_table_structure(table_name: str) -> Dict[str, List[str]]:
    """
    Get the structure of a specific table including all fields.

    Parameters:
    -----------
    table_name : str
        Name of the table

    Returns:
    --------
    Dict[str, List[str]]
        Dictionary with field names and their properties
    """
    fields = get_fields_by_table(table_name)

    # Group fields by category
    result = {
        "direct_identifiers": [],
        "indirect_identifiers": [],
        "quasi_identifiers": [],
        "sensitive_attributes": [],
        "non_sensitive": [],
        "all_fields": list(fields.keys())
    }

    for field_name, field_def in fields.items():
        category = field_def.get("category")

        if category == PrivacyCategory.DIRECT_IDENTIFIER:
            result["direct_identifiers"].append(field_name)
        elif category == PrivacyCategory.INDIRECT_IDENTIFIER:
            result["indirect_identifiers"].append(field_name)
        elif category == PrivacyCategory.QUASI_IDENTIFIER:
            result["quasi_identifiers"].append(field_name)
        elif category == PrivacyCategory.SENSITIVE_ATTRIBUTE:
            result["sensitive_attributes"].append(field_name)
        elif category == PrivacyCategory.NON_SENSITIVE:
            result["non_sensitive"].append(field_name)

    return result


def get_all_tables() -> List[str]:
    """
    Get a list of all table names.

    Returns:
    --------
    List[str]
        List of all table names
    """
    return list(TABLES.keys())


# If this module is run directly, print some summary information
if __name__ == "__main__":
    import sys

    print("PAMOLA.CORE (Privacy-Preserving AI Data Processors) - Anonymization - Field Definitions")
    print("=" * 50)
    print(f"Total fields: {len(FIELD_DEFINITIONS)}")
    print(f"Total tables: {len(TABLES)}")

    print("\nFields by privacy category:")
    for category in PrivacyCategory:
        fields = get_fields_by_category(category)
        print(f"  {category}: {len(fields)} fields")

    print("\nFields by anonymization strategy:")
    for strategy in AnonymizationStrategy:
        fields = get_fields_by_strategy(strategy)
        print(f"  {strategy}: {len(fields)} fields")

    print("\nFields by data type:")
    for field_type in FieldType:
        fields = get_fields_by_type(field_type)
        print(f"  {field_type}: {len(fields)} fields")

    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        print("\nDetailed table structure:")
        for table_name in TABLES:
            print(f"\n{table_name} Table:")
            structure = get_table_structure(table_name)
            for category, fields in structure.items():
                if category != "all_fields":  # Skip the full list to avoid repetition
                    print(f"  {category.replace('_', ' ').title()}: {', '.join(fields)}")