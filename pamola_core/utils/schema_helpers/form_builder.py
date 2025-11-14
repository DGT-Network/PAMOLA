"""
PAMOLA.CORE - Formily Schema Builder
------------------------------------
Module:        formily_builder.py
Package:       pamola_core.utils.schema_helpers
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-11-03
License:       BSD 3-Clause

Description:
    Utility functions to convert and merge JSON schemas into Formily-compatible schemas for UI form generation.
    Used for building dynamic forms in PAMOLA's web interfaces.

Usage:
    Import and use convert_json_schema_to_formily, and related helpers.
"""

import copy
from typing import Any, Dict, List, Optional, Union

from pamola_core.common.enum.custom_components import CustomComponents

from tomlkit import item

from pamola_core.common.enum.form_groups import (
    get_groups_with_titles,
)


def _merge_allOf(allOf_schemas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple allOf schemas into a single schema.

    Args:
        allOf_schemas: List of schemas to merge

    Returns:
        Merged schema with combined properties
    """
    merged = {}
    merged_properties = {}

    for sub_schema in allOf_schemas:
        # Convert sub-schema without operation_config_type to avoid adding groups
        sub_converted = convert_json_schema_to_formily(sub_schema)

        # Merge top-level keys except properties and group
        for k, v in sub_converted.items():
            if k == "properties":
                if isinstance(v, dict):
                    merged_properties.update(v)
            elif k != "group":
                merged[k] = v

    if merged_properties:
        merged["properties"] = merged_properties

    return merged


def _normalize_field_type(type: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Normalize field type from JSON Schema to Formily format.
    Handles both single type and union types (list of types).
    Keeps null values and only converts integer to number.

    Args:
        type: Field type - can be string or list of strings (e.g., ["string", "null"])

    Returns:
        Normalized type - can be string or list of strings
        - Single type: "integer" -> "number", "string" -> "string"
        - List types: ["string", "integer", "null"] -> ["string", "number", "null"]

    Examples:
        "integer" -> "number"
        ["integer", "null"] -> ["number", "null"]
        ["string", "integer", "null"] -> ["string", "number", "null"]
        "string" -> "string"
    """
    if isinstance(type, list):
        # Keep null values, only convert integer to number
        normalized_types = []
        for t in type:
            if t == "integer":
                normalized_types.append("number")
            else:
                normalized_types.append(t)

        return normalized_types
    else:
        # Single type - convert integer to number
        if type == "integer":
            return "number"
        return type


def _handle_array_items_component(
    field: Dict[str, Any],
    t: Any,
    formily_schema: Dict[str, Any],
    tooltip: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Handle ArrayItems component configuration.

    Args:
        field: The field configuration
        t: The field type

    Returns:
        Dict[str, Any]: Updated field configuration for ArrayItems
    """
    if "items" in field:
        items_schema = field["items"]
        if field["type"] == "array" or (isinstance(t, list) and "array" in t):
            field["type"] = "array"
            item_type = field["items"].get("type")
            field["items"] = {
                "type": "object",
                "x-component": "ArrayItems.Item",
                "properties": {},
            }
            if "properties" in items_schema:
                nested_required = field.get("required", [])
                formily_schema_item = formily_schema.get("properties", {})
                for k, v in items_schema["properties"].items():
                    if "x-component" in v:
                        converted = convert_property(
                            k,
                            v,
                            nested_required,
                            formily_schema_item[field["name"]].get("items", {}),
                            True,
                            tooltip,
                        )
                        field["items"]["properties"][k] = converted

            elif "x-items-title" in items_schema:
                items_titles = items_schema.get("x-items-title", [])
                item_params = items_schema.get("x-item-params", [])
                if item_params and len(item_params) == len(items_titles):
                    items_zip = zip(items_titles, item_params)
                else:
                    items_zip = zip(items_titles, items_titles)

                for title, param in items_zip:
                    item_key = param.lower().replace(" ", "_")
                    field["items"]["properties"][item_key] = {
                        "type": _normalize_field_type(item_type),
                        "title": f"{title}",
                        "x-decorator": "FormItem",
                        "x-component": items_schema["x-component"],
                        "x-component-props": {"placeholder": f"{title} value"},
                    }
                    if item_key == "max":
                        field["items"]["properties"][item_key]["x-decorator-props"] = {
                            "style": {"marginLeft": "8px"}
                        }
                        field["items"]["properties"][item_key]["x-reactions"] = [
                            {
                                "dependencies": [".min"],
                                "when": "{{$deps[0] !== undefined && $self.value !== undefined}}",
                                "fulfill": {
                                    "run": "if ($self.value <= $deps[0]) { $self.setFeedback({ type: 'error', code: 'range', messages: ['Max must be greater than Min'] }) } else { $self.setFeedback({ type: 'error', code: 'range', messages: [] }) }"
                                },
                            }
                        ]

                # Convert default values to array of objects format if needed
                # This handles both input formats:
                # 1. Simple array: ["value1", "value2"] -> [{"prop1": "value1", "prop2": "value2"}]
                # 2. Array of objects: [{"prop1": "value1", "prop2": "value2"}] -> keep as is
                if (
                    "default" in field
                    and isinstance(field["default"], list)
                    and len(field["default"]) > 0
                ):
                    if isinstance(field["default"][0], dict):
                        # Already in array of objects format - no conversion needed
                        default_obj = field["default"][0]
                    else:
                        # Convert simple array to object by mapping values to property keys
                        default_obj = {}
                        property_keys = list(field["items"]["properties"].keys())
                        for i, value in enumerate(field["default"]):
                            if i < len(property_keys):
                                default_obj[property_keys[i]] = value

                        # Convert to array of objects format
                        field["default"] = [default_obj]
            else:
                minItems = field.get("minItems", 1)
                for i in range(minItems):
                    item_key = f"value_{i+1}"
                    field["items"]["properties"][item_key] = {
                        "type": _normalize_field_type(item_type),
                        "x-decorator": "FormItem",
                        "x-component": items_schema["x-component"],
                        "x-component-props": {"placeholder": "Value"},
                    }
    else:
        field["type"] = field.get("type", "array")
        field["x-decorator"] = "FormItem"
        field["x-component"] = "ArrayItems"
        field["items"] = {
            "type": "object",
            "x-component": "ArrayItems.Item",
            "properties": {
                "value": {
                    "type": _normalize_field_type(field["items"]["type"]),
                    "x-decorator": "FormItem",
                    "x-component": "Input",
                },
            },
        }

    if (
        "minItems" in field
        and "maxItems" in field
        and field["minItems"] == field["maxItems"]
    ):
        field.pop("minItems", None)
        field.pop("maxItems", None)

    else:
        # Ensure items structure exists
        if "items" not in field:
            field["items"] = {}
        if "properties" not in field["items"]:
            field["items"]["properties"] = {}

        field["items"]["properties"]["remove"] = {
            "type": "void",
            "x-component": "ArrayItems.Remove",
            "x-component-props": {"style": {"marginLeft": "8px"}},
        }
        field["properties"] = {
            "add": {
                "type": "void",
                "title": "Add",
                "x-component": "ArrayItems.Addition",
                "x-component-props": {"style": {"marginTop": "8px"}},
            }
        }

    return field


def convert_property(
    name: str,
    prop: Dict[str, Any],
    required_fields: List[str] = [],
    formily_schema: Dict[str, Any] = {},
    is_nested: bool = False,
    tooltip: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    field = copy.deepcopy(prop)
    field["name"] = name
    field["x-decorator"] = "FormItem"
    if "description" in prop:
        field.pop("description", None)  # Remove description if present

    if tooltip is not None and name in tooltip:
        field["tooltip"] = tooltip[name]

    if "type" in field:
        field["type"] = _normalize_field_type(field["type"])

    if "default" in field:
        field["default"] = field["default"]

    # Required
    if name in required_fields:
        field["required"] = True

    if "x-component" not in field:
        return field

    if field["x-component"] == "Select":
        t = prop.get("type")
        if (
            field["type"] == "string"
            or (isinstance(t, list) and "string" in t)
            or field["type"] == "array"
            or (isinstance(t, list) and "array" in t)
        ):
            field["x-component-props"] = {
                "getPopupContainer": "{{(node) => node?.parentElement || document.body}}",
                "showSearch": True,
                "allowClear": True,
                "optionFilterProp": "label",
            }
            if field["type"] == "string" or (isinstance(t, list) and "string" in t):
                field["x-component-props"]["placeholder"] = field["title"]
            if field["type"] == "array" or (isinstance(t, list) and "array" in t):
                field["x-component-props"]["mode"] = "multiple"
                field["x-component-props"]["placeholder"] = "Select options"
        else:
            field["x-component-props"] = {
                "getPopupContainer": "{{(node) => node?.parentElement || document.body}}",
                "showSearch": True,
            }

        if "oneOf" in field and any(
            isinstance(opt, dict) and "const" in opt for opt in field["oneOf"]
        ):
            field["enum"] = [
                {
                    "value": opt["const"],
                    "label": opt.get("description", str(opt["const"])),
                }
                for opt in field["oneOf"]
                if opt.get("const") is not None
            ]
            field.pop("oneOf", None)

        if field["type"] == "array" or (isinstance(t, list) and "array" in t):
            if (
                "items" in field
                and isinstance(field["items"], dict)
                and "enum" in field["items"]
            ):
                field["enum"] = [
                    {
                        "value": value,
                        "label": str(value),
                    }
                    for value in field["items"]["enum"]
                ]

    elif field["x-component"] == "Switch":
        field["x-content"] = f"Enable {field['title']}"

    elif field["x-component"] == "NumberPicker":
        if "minimum" in field:
            field["x-validate"] = field.get("x-validate", [])
            field["x-validate"].append(
                {
                    "type": "minimum",
                    "message": f"{field['title']} must be at least {field['minimum']}",
                    "minimum": field["minimum"],
                }
            )
        if "maximum" in field:
            field["x-validate"] = field.get("x-validate", [])
            field["x-validate"].append(
                {
                    "type": "maximum",
                    "message": f"{field['title']} must be at most {field['maximum']}",
                    "maximum": field["maximum"],
                }
            )

    elif field["x-component"] == "FloatPicker":
        field["x-component"] = "NumberPicker"
        field["x-component-props"] = {"step": 0.1, "precision": 1, "min": 0}

        if "minimum" in field:
            field["x-validate"] = field.get("x-validate", [])
            field["x-validate"].append(
                {
                    "type": "minimum",
                    "message": f"{field['title']} must be at least {field['minimum']}",
                    "minimum": field["minimum"],
                }
            )
        if "maximum" in field:
            field["x-validate"] = field.get("x-validate", [])
            field["x-validate"].append(
                {
                    "type": "maximum",
                    "message": f"{field['title']} must be at most {field['maximum']}",
                    "maximum": field["maximum"],
                }
            )

    elif field["x-component"] == "ArrayItems":
        t = prop.get("type")
        field = _handle_array_items_component(field, t, formily_schema, tooltip)

    elif field["x-component"] == "DatePicker":
        field["x-decorator"] = "FormItem"

    elif field["x-component"] == "DateFormatArray":
        field["x-decorator"] = "FormItem"

    elif field.get("x-component") in [
        CustomComponents.NUMERIC_RANGE_MODE,
        CustomComponents.DEPEND_SELECT,
    ]:
        field = _handle_custom_component(field)

    # Add this NEW section before the existing x-reactions handling
    if "x-toggle-groups" in field:
        toggle_groups = field.pop("x-toggle-groups")
        group_reactions = []

        for value, groups in toggle_groups.items():
            for group_name in groups:
                # Get the actual string value from GroupName enum
                group_value = (
                    group_name.value
                    if hasattr(group_name, "value")
                    else str(group_name)
                )
                group_reactions.append(
                    {
                        "target": f"*({group_value})",
                        "fulfill": {
                            "state": {
                                "display": f"{{{{$self.value === '{value}' ? 'visible' : 'none'}}}}"
                            }
                        },
                    }
                )

        # Merge with existing x-reactions
        existing_reactions = field.get("x-reactions", [])
        field["x-reactions"] = group_reactions + existing_reactions

    # Handle x-depend-on, x-required-on, and x-custom-function
    if (
        "x-depend-on" in field
        or "x-required-on" in field
        or "x-custom-function" in field
    ):
        field = _add_x_reactions(field, formily_schema, is_nested)

    # Add Upload component initialization (AFTER dependencies)
    if field.get("x-component") == CustomComponents.UPLOAD:
        field = _add_upload_reaction(field)

    # Nested object - mark as nested when calling recursively
    if field.get("type") == "object" and "properties" in field:
        nested_required = field.get("required", [])
        field["properties"] = {
            k: convert_property(
                k, v, nested_required, field["properties"], True, tooltip
            )  # is_nested=True
            for k, v in field["properties"].items()
        }
    return field


def _handle_custom_component(field: dict) -> dict:
    """
    Transform custom component fields into standard Formily components.

    Parameters:
    -----------
    field : dict
        Field configuration dictionary

    Returns:
    --------
    dict
        Transformed field configuration
    """
    component = field.get("x-component")

    if component == CustomComponents.NUMERIC_RANGE_MODE:
        # Handle NUMERIC_RANGE_MODE: allows symmetric (number) or asymmetric (array) noise
        field["x-decorator"] = "FormItem"
        field["x-component-props"] = {"step": 0.1, "precision": 1}
        field["enum"] = [
            {
                "label": "Symetric",
                "value": "Symetric",
                "dataType": "number",
            },
            {
                "label": "Asymmetric",
                "value": "Asymmetric",
                "dataType": "array",
            },
        ]

    elif component == CustomComponents.DEPEND_SELECT:
        # Handle DependentSelect: options change based on another field
        depend_map = field.get("x-depend-map", {})
        depend_on = depend_map.get("depend_on")
        options_map = depend_map.get("options_map", {})

        # Convert to Select with reactions
        field["x-decorator"] = "FormItem"
        field["x-component"] = "Select"
        field["x-component-props"] = {
            "placeholder": f"Select {field.get('title', '').lower() or 'option'}"
        }

        # Add reactions to update options based on dependency
        field["x-reactions"] = [
            {
                "dependencies": [f".{depend_on}"],
                "fulfill": {
                    "schema": {
                        "enum": "{{ (function() { \
                            const map = "
                        + str(options_map).replace("'", '"')
                        + "; \
                            return map[$deps[0]] || []; \
                        })() }}"
                    },
                },
            },
            {
                "dependencies": [f".{depend_on}"],
                "fulfill": {
                    "state": {"visible": "{{ !!$deps[0] }}"},
                    "run": "{{ $self.setValue(undefined); }}",
                },
            },
        ]

        # Clean up custom properties
        field.pop("x-depend-map", None)

    return field


def _add_upload_reaction(field: dict) -> dict:
    """
    Add initialization reaction for Upload component.
    Overwrites existing run script in the first reaction instead of appending.
    """
    reactions = field.get("x-reactions", [])

    if reactions:
        # Overwrite run in first existing reaction
        if "fulfill" not in reactions[0]:
            reactions[0]["fulfill"] = {}
        reactions[0]["fulfill"]["run"] = "{{ init_upload($self) }}"
    else:
        # Create new reaction if none exist
        reactions.append(
            {
                "fulfill": {
                    "run": "{{ init_upload($self) }}",
                },
            }
        )

    field["x-reactions"] = reactions
    return field


def _get_default_value_str(field: Dict[str, Any]) -> str:
    """Get the default value string for a field based on its type and current value."""
    default_value = field.get("default", None)
    if default_value is None:
        if field.get("type") == "number":
            return str(field.get("minimum", 0))
        return "null"
    if isinstance(default_value, bool):
        return str(default_value).lower()
    if isinstance(default_value, (int, float)):
        return str(default_value)
    return f"'{default_value}'"


def _build_condition_expression(condition_value: Any, field_index: int) -> str:
    """Build a condition expression for field dependencies using $deps array."""
    if isinstance(condition_value, bool):
        js_value = "true" if condition_value else "false"
        return f" $deps[{field_index}] === {js_value} "
    if condition_value == "not_null":
        return f" !!$deps[{field_index}] "
    if condition_value == "null" or condition_value is None:
        return f" !$deps[{field_index}] "
    if isinstance(condition_value, list):
        return " || ".join(
            f" $deps[{field_index}] === '{val}' " for val in condition_value
        )
    return f" $deps[{field_index}] === '{condition_value}' "


def _process_field_conditions(
    conditions: Dict[str, Any],
    formily_schema: Dict[str, Any],
    depend_fields: List[str],
    join_operator: str = "&&",
) -> str:
    """Process field conditions and join them with the specified operator."""
    valid_conditions = []
    for field, value in conditions.items():
        if field in formily_schema["properties"] and field in depend_fields:
            field_index = depend_fields.index(field)
            condition_expr = _build_condition_expression(value, field_index)
            valid_conditions.append(condition_expr)
    return join_operator.join(valid_conditions)


def _add_x_reactions(
    field: Dict[str, Any], formily_schema: Dict[str, Any], is_nested: bool = False
) -> Dict[str, Any]:
    """
    Add reactive behavior to form fields based on dependencies and requirements.

    Args:
        field: The field configuration to add reactions to
        formily_schema: The complete form schema for context
        is_nested: Whether this field is inside a nested object

    Returns:
        Dict[str, Any]: Updated field configuration with reactions
    """
    default_value_str = _get_default_value_str(field)
    state = {}

    # Add reactions to field
    x_depend_on = field.get("x-depend-on", {})
    x_required_on = field.get("x-required-on", {})

    keys = list(x_depend_on.keys()) + list(x_required_on.keys())
    depend_fields = _get_ordered_unique_keys(keys)

    # Handle visibility conditions
    if x_depend_on:
        visible_state = _process_field_conditions(
            x_depend_on, formily_schema, depend_fields
        )
        if visible_state:
            state["visible"] = f"{{{{ {visible_state} }}}}"

    # Handle requirement conditions
    if x_required_on:
        required_state = _process_field_conditions(
            x_required_on, formily_schema, depend_fields
        )
        if required_state:
            state["required"] = f"{{{{ {required_state} }}}}"

    # Add a dot prefix if this is a nested field
    if is_nested:
        depend_fields = [f".{field_name}" for field_name in depend_fields]

    # Determine if reactions should be added
    is_ignore_depend_fields = field.get("x-ignore-depend-fields", False)
    has_dependencies = depend_fields and not is_ignore_depend_fields
    has_custom_function = "x-custom-function" in field

    # Case 1: Has dependencies (x-depend-on or x-required-on)
    if has_dependencies:
        reactions = field.get("x-reactions", [])

        # Determine run script based on custom function
        if has_custom_function:
            deps_expr = ", ".join([f"$deps[{i}]" for i in range(len(depend_fields))])
            run = f"{field['x-custom-function'][0]}($self, {deps_expr})"
        else:
            run = f"$self.setValue({default_value_str})"

        reactions.append(
            {
                "dependencies": depend_fields,
                "fulfill": {
                    "state": state,
                    "run": f"{{{{ {run} }}}}",
                },
            }
        )
        field["x-reactions"] = reactions

    # Case 2: Has custom function but NO dependencies
    elif has_custom_function:
        function_name = field["x-custom-function"][0]
        existing_reactions = field.get("x-reactions", [])

        # Build run template
        run_template = f"{function_name}($self); $self.setValue({default_value_str})"

        # Only add if no non-toggle reaction exists
        if not any(r for r in existing_reactions if "target" not in r):
            new_reaction = {"fulfill": {"run": f"{{{{ {run_template} }}}}"}}
            existing_reactions.append(new_reaction)

        field["x-reactions"] = existing_reactions

    # Clean up temporary properties
    field.pop("x-depend-on", None)
    field.pop("x-required-on", None)

    return field


def convert_json_schema_to_formily(
    schema: Dict[str, Any],
    operation_config_type: Optional[str] = None,
    tooltip: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Convert a JSON Schema (draft-07) into a Formily-compatible schema.

    - Supports oneOf, allOf, if/then/else, dependencies.
    - Automatically adds x-decorator: FormItem when x-component is present.
    - Converts enum to Select component if appropriate.
    - Adds required: true to required properties.
    - Converts if/then required fields to x-reactions for required/visible.
    - Includes form groups configuration for UI rendering (if operation_config_type provided).

    Args:
        schema: JSON Schema (draft-07) object
        operation_config_type: Optional operation config type (e.g., 'NumericGeneralizationConfig').
                              If None, groups will not be included in output.
        tooltip: Optional tooltip text mapping for fields

    Returns:
        Formily-compatible schema with properties and optionally group configuration

    Example:
        >>> # With operation type (includes groups)
        >>> result = convert_json_schema_to_formily(schema, "NumericGeneralizationConfig")
        >>> result.keys()
        dict_keys(['properties', 'group'])

        >>> # Without operation type (no groups)
        >>> result = convert_json_schema_to_formily(schema)
        >>> result.keys()
        dict_keys(['properties'])
    """
    formily_schema = copy.deepcopy(schema)

    # Handle allOf
    if "allOf" in schema:
        merged = _merge_allOf(schema["allOf"])
        formily_schema = {**formily_schema, **merged}
        formily_schema.pop("allOf", None)

    # Handle properties recursively
    required_fields = formily_schema.get("required", [])
    if "properties" in schema:
        new_properties = {}
        for k, v in schema["properties"].items():
            if "x-component" in v:
                converted = convert_property(
                    k, v, required_fields, formily_schema, False, tooltip
                )
                new_properties[k] = converted
        formily_schema["properties"] = new_properties

    # Build result
    result = {
        "properties": formily_schema.get("properties", {}),
    }

    # Only include groups if operation_config_type is provided
    if operation_config_type:
        result["group"] = get_groups_with_titles(operation_config_type)

    return result


def _get_ordered_unique_keys(keys: list) -> List[str]:
    """Get unique keys from two dictionaries while preserving order."""
    seen = set()
    result = []
    for key in keys:
        if key not in seen:
            result.append(key)
            seen.add(key)
    return result
