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
            # Merge dependencies if any
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
    # Merge dependencies into the result if any
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
    core_config_cls: Type[OperationConfig],
    ui_config_class: Type[OperationConfig],
    task_dir: Path,
    excluded_fields: List[str] = [],
    generate_formily_schema: bool = False,
    tooltip: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Write the schema (after excluding specified fields) of the given config_class to a JSON file.
    Args:
        core_config_cls (class): Core Config class with attributes:
            - schema: dict, the original JSON schema.
        ui_config_class (class): UI Config class with attributes:
            - schema: dict, the original JSON schema.
        task_dir (Path): Directory where the JSON file will be saved.
        excluded_fields (list, optional): List of field names to exclude from the schema. Defaults to [].
        generate_formily_schema (bool): Whether to convert the schema to Formily format. Defaults to False.
        tooltip (dict, optional): Tooltip information for Formily schema conversion. Defaults to None.
    Returns:
        Path: Path to the written JSON file.
    Example:
        generate_schema_json(NumericGeneralizationConfig, NumericGeneralizationUIConfig)
    """
    # Get core filtered schema with excluded fields removed
    core_filtered_schema = get_filtered_schema(core_config_cls.schema, excluded_fields)
    # Get UI filtered schema with excluded fields removed
    ui_filtered_schema = get_filtered_schema(ui_config_class.schema, excluded_fields)

    # Merge core and UI schemas
    filtered_schema = merge_schemas(core_filtered_schema, ui_filtered_schema)

    # Flatten allOf recursively at all levels
    filtered_schema = flatten_schema(filtered_schema)

    # Remove None values from enum lists (for frontend compatibility)
    remove_none_from_enum(filtered_schema)

    if generate_formily_schema:
        # Convert the filtered JSON schema to Formily schema
        filtered_schema = convert_json_schema_to_formily(
            filtered_schema, core_config_cls.__name__, tooltip
        )

    # Use the UI class name as the output filename
    filename = f"{core_config_cls.__name__}.json"

    # Define the output path
    output_path = task_dir / filename

    # Write the filtered schema to a JSON file
    path_file = write_json(filtered_schema, output_path)

    # Return the path to the written JSON file
    return path_file


def merge_schemas(core_schema: dict, ui_schema: dict) -> dict:
    """
    Deep merge two JSON schemas, combining properties from both.
    UI schema properties take precedence and are merged into core schema properties.

    Args:
        core_schema (dict): The core validation schema
        ui_schema (dict): The UI metadata schema

    Returns:
        dict: Merged schema with both core validation and UI metadata
    """
    merged = copy.deepcopy(core_schema)

    # Merge top-level allOf arrays if both exist
    if "allOf" in core_schema and "allOf" in ui_schema:
        # Merge properties from corresponding allOf elements
        core_allof = merged["allOf"]
        ui_allof = ui_schema["allOf"]

        for i, ui_item in enumerate(ui_allof):
            if (
                i < len(core_allof)
                and isinstance(ui_item, dict)
                and isinstance(core_allof[i], dict)
            ):
                if "properties" in ui_item and "properties" in core_allof[i]:
                    # Merge properties from UI into core
                    for prop_name, prop_value in ui_item["properties"].items():
                        if prop_name in core_allof[i]["properties"]:
                            # Merge UI metadata into existing core property
                            _merge_property_recursive(
                                core_allof[i]["properties"][prop_name], prop_value
                            )
                        else:
                            # Add new property from UI schema
                            core_allof[i]["properties"][prop_name] = copy.deepcopy(
                                prop_value
                            )
    elif "allOf" not in core_schema and "allOf" in ui_schema:
        # If core doesn't have allOf but UI does, add it
        merged["allOf"] = copy.deepcopy(ui_schema["allOf"])

    # Merge top-level properties if they exist
    if "properties" in ui_schema:
        if "properties" not in merged:
            merged["properties"] = {}
        for prop_name, prop_value in ui_schema["properties"].items():
            if prop_name in merged["properties"]:
                _merge_property_recursive(merged["properties"][prop_name], prop_value)
            else:
                merged["properties"][prop_name] = copy.deepcopy(prop_value)

    return merged


def _merge_property_recursive(core_prop: dict, ui_prop: dict) -> None:
    """
    Recursively merge UI property metadata into core property.
    Handles special cases:
    - x-items.properties from UI should merge into items.properties from core (Case 1: object arrays)
    - x-items from UI should merge into items from core (Case 2: primitive arrays)
    - items.properties UI metadata should be extracted to x-items
    - Top-level UI metadata (x-component, x-group, etc.) should be added to core

    Args:
        core_prop (dict): Core property definition (modified in place)
        ui_prop (dict): UI property definition with metadata
    """
    # Track if we need to create x-items for array items UI metadata
    ui_items_metadata = {}

    for key, value in ui_prop.items():
        if key == "x-items":
            # Special case: x-items from UI needs to be merged based on structure
            if isinstance(value, dict):
                # Case 1: x-items has a properties object (for object arrays)
                if (
                    "properties" in value
                    and "items" in core_prop
                    and "properties" in core_prop["items"]
                ):
                    # Merge x-items.properties metadata into items.properties
                    for prop_name, prop_ui_metadata in value["properties"].items():
                        if prop_name in core_prop["items"]["properties"]:
                            # Add UI metadata to existing property
                            core_prop["items"]["properties"][prop_name].update(
                                copy.deepcopy(prop_ui_metadata)
                            )
                        else:
                            # Property doesn't exist in core, store for later
                            if "properties" not in ui_items_metadata:
                                ui_items_metadata["properties"] = {}
                            ui_items_metadata["properties"][prop_name] = copy.deepcopy(
                                prop_ui_metadata
                            )
                # Case 2: x-items contains direct UI metadata (for primitive arrays)
                elif "items" in core_prop and "properties" not in core_prop["items"]:
                    # Merge x-items directly into items (for primitive arrays like range_limits)
                    core_prop["items"].update(copy.deepcopy(value))
                # Case 3: x-items has properties but core doesn't have items.properties (old structure fallback)
                elif "items" in core_prop and "properties" in core_prop["items"]:
                    # Handle old structure where x-items directly contains property metadata
                    for prop_name, prop_ui_metadata in value.items():
                        if (
                            prop_name != "properties"
                            and prop_name in core_prop["items"]["properties"]
                        ):
                            # Add UI metadata to existing property
                            core_prop["items"]["properties"][prop_name].update(
                                copy.deepcopy(prop_ui_metadata)
                            )
                        elif prop_name != "properties":
                            # Property doesn't exist in core, store for later
                            if prop_name not in ui_items_metadata:
                                ui_items_metadata[prop_name] = {}
                            ui_items_metadata[prop_name].update(
                                copy.deepcopy(prop_ui_metadata)
                            )
                else:
                    # If core doesn't have items structure, keep x-items as is
                    if "x-items" not in core_prop:
                        core_prop["x-items"] = {}
                    core_prop["x-items"].update(copy.deepcopy(value))

        elif (
            key == "items"
            and isinstance(value, dict)
            and isinstance(core_prop.get("items"), dict)
        ):
            # Special handling for items: merge nested properties recursively
            if "properties" in value and "properties" in core_prop["items"]:
                # Merge each property in items.properties
                for prop_name, prop_value in value["properties"].items():
                    if prop_name in core_prop["items"]["properties"]:
                        # Extract UI metadata (x-* fields) from nested properties
                        ui_metadata = {
                            k: v for k, v in prop_value.items() if k.startswith("x-")
                        }
                        core_metadata = {
                            k: v
                            for k, v in prop_value.items()
                            if not k.startswith("x-")
                        }

                        # Recursively merge non-UI metadata
                        if core_metadata:
                            _merge_property_recursive(
                                core_prop["items"]["properties"][prop_name],
                                core_metadata,
                            )

                        # Add UI metadata directly to items.properties
                        if ui_metadata:
                            core_prop["items"]["properties"][prop_name].update(
                                ui_metadata
                            )
                    else:
                        # Add new property (separate UI and core metadata)
                        ui_metadata = {
                            k: v for k, v in prop_value.items() if k.startswith("x-")
                        }
                        core_metadata = {
                            k: v
                            for k, v in prop_value.items()
                            if not k.startswith("x-")
                        }

                        # Merge both core and UI metadata
                        merged_property = copy.deepcopy(core_metadata)
                        merged_property.update(ui_metadata)
                        core_prop["items"]["properties"][prop_name] = merged_property
            else:
                # If no properties to merge, just update items directly
                for item_key, item_value in value.items():
                    if item_key == "properties":
                        continue  # Already handled above
                    if item_key not in core_prop["items"]:
                        core_prop["items"][item_key] = copy.deepcopy(item_value)

        elif (
            key == "properties"
            and isinstance(value, dict)
            and isinstance(core_prop.get("properties"), dict)
        ):
            # Recursively merge nested properties (for object types)
            for prop_name, prop_value in value.items():
                if prop_name in core_prop["properties"]:
                    _merge_property_recursive(
                        core_prop["properties"][prop_name], prop_value
                    )
                else:
                    core_prop["properties"][prop_name] = copy.deepcopy(prop_value)
        else:
            # Standard merge: add or update the property
            # This handles x-component, x-group, x-depend-on, etc.
            if key not in core_prop:
                core_prop[key] = (
                    copy.deepcopy(value) if isinstance(value, (dict, list)) else value
                )
            elif isinstance(core_prop[key], dict) and isinstance(value, dict):
                # Deep merge for dict values
                core_prop[key].update(copy.deepcopy(value))
            else:
                core_prop[key] = (
                    copy.deepcopy(value) if isinstance(value, (dict, list)) else value
                )

    # If we collected UI metadata that couldn't be merged, keep it in x-items
    if ui_items_metadata:
        if "x-items" not in core_prop:
            core_prop["x-items"] = {}
        core_prop["x-items"].update(ui_items_metadata)
