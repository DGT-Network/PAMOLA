"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        Schema Utilities
Package:       pamola_core.utils
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
   This module provides utility functions for manipulating and transforming
   JSON schema definitions. It includes helpers for flattening schemas with
   allOf composition and generating human-friendly property titles.

Key Features:
   - Flatten any JSON schema with allOf into a single-level schema
   - Auto-generate Title Case for property names
   - Designed for dynamic schema validation and UI generation
   - Reusable across all PAMOLA.CORE modules

Design Principles:
   - Simplicity: Minimal, clear, and well-documented functions
   - Flexibility: Works with any schema structure
   - Reusability: Generic for broad applicability

Dependencies:
   - re: Regular expressions for string manipulation

Changelog:
   1.0.0 - Initial implementation with flatten_schema
"""

from pathlib import Path
from typing import Dict, List, Optional, Type
import copy
import json
from pamola_core.utils.io import write_json
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.schema_helpers.form_builder import convert_json_schema_to_formily


def flatten_schema(schema: dict, unused_fields=None) -> dict:
    """
    Flatten any JSON schema (with allOf) into a single-level schema.
    Each property will have type, title (auto-generated if missing), and default if present.
    Optionally filter out properties by unused_fields (exclude these fields from result).

    Args:
        schema (dict): The JSON schema to flatten.
        unused_fields (list or None): List of property names to exclude. If None, keep all fields.

    Returns:
        dict: Flattened schema with filtered properties (all except those in unused_fields).

    Example:
        flatten_schema(schema, unused_fields=["scope", "config"])  # will remove 'scope' and 'config' from result
    """
    result = {
        "type": "object",
        "title": "",
        "description": "",
        "properties": {},
        "required": [],
    }
    dependencies_merged = {}
    if "title" in schema:
        result["title"] = schema["title"]
    if "description" in schema:
        result["description"] = schema["description"]

    # Separate schema objects (with properties) and validation schemas (if/then/else...)
    all_of = schema.get("allOf", [schema]) if "allOf" in schema else [schema]
    merged_validations = []
    for sub_schema in all_of:
        if isinstance(sub_schema, dict) and "properties" in sub_schema:
            props = sub_schema.get("properties", {})
            reqs = sub_schema.get("required", [])
            for k, v in props.items():
                if unused_fields and k in unused_fields:
                    continue
                prop = v.copy()
                if "title" not in prop:
                    prop["title"] = k.replace("_", " ").title()
                result["properties"][k] = prop
            result["required"].extend(
                [r for r in reqs if not unused_fields or r not in unused_fields]
            )
            # Merge dependencies nếu có
            if "dependencies" in sub_schema:
                for dep_key, dep_val in sub_schema["dependencies"].items():
                    dependencies_merged[dep_key] = dep_val
        else:
            # Keep validation schemas (if/then/else...)
            merged_validations.append(sub_schema)

    result["required"] = list(dict.fromkeys(result["required"]))
    # If there are validation schemas, keep them in allOf
    if merged_validations:
        result["allOf"] = merged_validations
    # Merge dependencies vào kết quả nếu có
    if "dependencies" in schema:
        for dep_key, dep_val in schema["dependencies"].items():
            dependencies_merged[dep_key] = dep_val
    if dependencies_merged:
        result["dependencies"] = dependencies_merged
    return result


def get_filtered_schema(schema: dict, exclude_fields: Optional[list] = None) -> dict:
    """
    Return a deep-copied schema with all fields in exclude_fields removed recursively at any nesting level.
    Args:
        schema (dict): The original JSON schema to be filtered.
        exclude_fields (Optional[list]): List of field names to remove from the schema (recursively). If None, no fields are excluded.
    Returns:
        dict: New schema with excluded fields removed.
    """
    exclude_fields = exclude_fields or []
    schema = copy.deepcopy(schema)
    # Recursively remove excluded fields from the schema (including nested objects/arrays)
    if exclude_fields:
        remove_fields_recursive(schema, exclude_fields)
    return schema


def remove_fields_recursive(block: dict, exclude_fields: list = []) -> None:
    """
    Recursively remove all fields in exclude_fields from the given schema block (or any nested subschema).
    - Removes fields from 'properties' and 'required'.
    - Recurses into nested objects, arrays of objects, and schema composition keywords.
    Args:
        block (dict): The current schema or subschema.
        exclude_fields (list): List of field names to remove.
    """
    # Remove excluded fields from 'properties' if present
    props = block.get("properties", {})
    for field in exclude_fields:
        props.pop(field, None)  # Remove the field if it exists
    # Remove excluded fields from 'required' if present
    reqs = block.get("required", [])
    block["required"] = [r for r in reqs if r not in exclude_fields]
    # Recurse into nested object properties
    for v in props.values():
        # If property is an object, recurse into its properties
        if v.get("type") == "object" and "properties" in v:
            remove_fields_recursive(v, exclude_fields)
        # If property is an array of objects, recurse into the items
        if v.get("type") == "array" and isinstance(v.get("items"), dict):
            if v["items"].get("type") == "object":
                remove_fields_recursive(v["items"], exclude_fields)
    # Recurse into schema composition keywords (allOf, anyOf, oneOf)
    for key in ["allOf", "anyOf", "oneOf"]:
        if key in block:
            for sub in block[key]:
                if isinstance(sub, dict):
                    remove_fields_recursive(sub, exclude_fields)
    # Recurse into conditional schema keywords (then, else, if)
    for key in ["then", "else", "if"]:
        if key in block and isinstance(block[key], dict):
            remove_fields_recursive(block[key], exclude_fields)


def remove_none_from_enum(schema):
    """
    Recursively remove None values from any 'enum' lists in the schema.
    Args:
        schema (dict or list): The schema or subschema to process.
    """
    if isinstance(schema, dict):
        for k, v in schema.items():
            if k == "enum" and isinstance(v, list):
                schema[k] = [
                    item for item in v if item is not None
                ]  # Remove None from enum
            else:
                remove_none_from_enum(v)
    elif isinstance(schema, list):
        for item in schema:
            remove_none_from_enum(item)


def get_schema_json(
    config_class: Type[OperationConfig], excluded_fields: List[str] = []
) -> str:
    """
    Filter the schema by removing all fields listed in excluded_fields of the config class, and return a pretty JSON string for frontend use.
    Args:
        config_class (class): Config class with attributes:
            - schema: dict, the original JSON schema.
            - EXCLUDE_FIELDS: list, fields to exclude from the schema for the frontend.
        excluded_fields (list, optional): List of field names to exclude from the schema. Defaults to [].
    Returns:
        str: Filtered JSON schema as a pretty-formatted string (indent=2, ensure_ascii=False).
    Example:
        get_schema_json(NumericGeneralizationConfig)
    """
    # Get filtered schema with excluded fields removed
    filtered_schema = get_filtered_schema(config_class.schema, excluded_fields)
    # Remove None values from enum lists (for frontend compatibility)
    remove_none_from_enum(filtered_schema)
    # Return pretty JSON string, Unicode supported
    return json.dumps(filtered_schema, ensure_ascii=False, indent=2)


def generate_schema_json(
    config_class: Type[OperationConfig],
    task_dir: Path,
    excluded_fields: List[str] = [],
    generate_formily_schema: bool = False,
    tooltip: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Write the schema (after excluding specified fields) of the given config_class to a JSON file.
    Args:
        config_class (class): Configuration class with a 'schema' attribute (dict).
        excluded_fields (list, optional): List of field names to exclude from the schema. Defaults to [].
    Returns:
        Path: Path to the written JSON file.
    Example:
        generate_schema_json(NumericGeneralizationConfig)
    """

    # Get filtered schema with excluded fields removed
    filtered_schema = get_filtered_schema(config_class.schema, excluded_fields)

    # Flatten allOf recursively at all levels
    filtered_schema = flatten_schema(filtered_schema)

    # Remove None values from enum lists (for frontend compatibility)
    remove_none_from_enum(filtered_schema)

    if generate_formily_schema:
        # Convert the filtered JSON schema to Formily schema
        filtered_schema = convert_json_schema_to_formily(filtered_schema, tooltip)

    # Use the class name as the output filename
    filename = f"{config_class.__name__}.json"

    # Define the output path
    output_path = task_dir / filename

    # Write the filtered schema to a JSON file
    path_file = write_json(filtered_schema, output_path)

    # Return the path to the written JSON file
    return path_file
