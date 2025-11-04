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
import json
import copy
from typing import Any, Dict, List, Optional

from pamola_core.common.enum.section_name_enum import (
    SECTION_NAME_TITLE,
    SectionName,
)


def _merge_allOf(allOf_schemas: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged = {}
    merged_properties = {}
    for sub_schema in allOf_schemas:
        sub_converted = convert_json_schema_to_formily(sub_schema)
        # Merge top-level keys except properties
        for k, v in sub_converted.items():
            if k == "properties":
                if isinstance(v, dict):
                    merged_properties.update(v)
            else:
                merged[k] = v
    if merged_properties:
        merged["properties"] = merged_properties
    return merged


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

            elif "itemsTitle" in items_schema:
                items_titles = items_schema["itemsTitle"]
                for title in items_titles:
                    item_key = title.lower().replace(" ", "_")
                    field["items"]["properties"][item_key] = {
                        "type": "number",
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
            else:
                minItems = field.get("minItems", 1)
                for i in range(minItems):
                    item_key = f"value_{i+1}"
                    field["items"]["properties"][item_key] = {
                        "type": "number",
                        "title": f"Value {i+1}",
                        "x-decorator": "FormItem",
                        "x-component": items_schema["x-component"],
                        "x-component-props": {"placeholder": f"Value {i+1}"},
                    }

            field["items"]["properties"]["remove"] = {
                "type": "void",
                "x-component": "ArrayItems.Remove",
                "x-component-props": {"style": {"marginLeft": "8px"}},
            }
    else:
        field["type"] = "array"
        field["x-decorator"] = "FormItem"
        field["x-component"] = "ArrayItems"
        field["items"] = {
            "type": "object",
            "x-component": "ArrayItems.Item",
            "properties": {
                "value": {
                    "type": "string",
                    "x-decorator": "FormItem",
                    "x-component": "Input",
                },
                "remove": {
                    "type": "void",
                    "x-component": "ArrayItems.Remove",
                    "x-component-props": {"style": {"marginLeft": "8px"}},
                },
            },
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

    if "type" in prop:
        t = prop["type"]
        if t == "integer" or (isinstance(t, list) and "integer" in t):
            field["type"] = "number"
        else:
            field["type"] = t

    if "default" in field:
        field["default"] = field["default"]

    # Required
    if name in required_fields:
        field["required"] = True

    if "x-component" not in field:
        return field

    if field["x-component"] == "Select":
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

        if "oneOf" in field and all(
            isinstance(opt, dict) and "const" in opt for opt in field["oneOf"]
        ):
            field["enum"] = [
                {
                    "value": opt["const"],
                    "label": opt.get("description", str(opt["const"])),
                }
                for opt in field["oneOf"]
            ]
            field.pop("oneOf", None)

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
        field["x-component-props"] = {
            "step": 0.1,
            "precision": 1,
            "min": 0
        }

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
        field = _handle_array_items_component(field, t, formily_schema, tooltip)

    elif field["x-component"] == "DatePicker":
        field["x-decorator"] = "FormItem"

    elif field.get("x-component") == "Depend-Select":
        depend_map = field.get("x-depend-map", {})
        depend_on = depend_map.get("depend_on")
        options_map = depend_map.get("options_map", {})

        # Chuyển sang Select
        field["x-decorator"] = "FormItem"
        field["x-component"] = "Select"
        field["x-component-props"] = {
            "placeholder": f"Select {field.get('title', '').lower() or 'option'}"
        }

        field["x-reactions"] = [
            {
                "dependencies": [f".{depend_on}"],
                "fulfill": {
                    "schema": {
                        "enum": "{{ (function() { \
                            const map = " + str(options_map).replace("'", '"') + "; \
                            return map[$deps[0]] || []; \
                        })() }}"
                    },
                }
            },
            {
                "dependencies": [f".{depend_on}"],
                "fulfill": {
                    "state": {
                        "visible": "{{ !!$deps[0] }}"
                    },
                    "run": "{{ $self.setValue(undefined); }}"
                }
            }
        ]

        field.pop("x-depend-map", None)
            

    if "x-depend-on" in field or "x-required-on" in field:
        field = _add_x_reactions(field, formily_schema, is_nested)

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
        js_value = 'true' if condition_value else 'false'
        return f" $deps[{field_index}] === {js_value} "
    if condition_value == "not_null":
        return f" !!$deps[{field_index}] "
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
    depend_fields = list(
        set(
            list(field.get("x-depend-on", {}).keys())
            + list(field.get("x-required-on", {}).keys())
        )
    )

    # Handle visibility conditions
    if "x-depend-on" in field:
        visible_state = _process_field_conditions(
            field["x-depend-on"], formily_schema, depend_fields
        )
        if visible_state:
            state["visible"] = f"{{{{ {visible_state} }}}}"

    # Handle requirement conditions
    if "x-required-on" in field:
        required_state = _process_field_conditions(
            field["x-required-on"], formily_schema, depend_fields
        )
        if required_state:
            state["required"] = f"{{{{ {required_state} }}}}"

    # Add a dot prefix if this is a nested field
    if is_nested:
        depend_fields = [f".{field_name}" for field_name in depend_fields]

    if depend_fields:
        reactions = field.get("x-reactions", [])
        reactions.append(
            {
                "dependencies": depend_fields,
                "fulfill": {
                    "state": state,
                    "run": f"{{{{ $self.setValue({default_value_str}) }}}}",
                },
            }
        )
        field["x-reactions"] = reactions

    # Clean up temporary properties
    field.pop("x-depend-on", None)
    field.pop("x-required-on", None)

    return field


def convert_json_schema_to_formily(
    schema: Dict[str, Any], tooltip: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Convert a JSON Schema (draft-07) into a Formily-compatible schema.
    - Supports oneOf, allOf, if/then/else, dependencies.
    - Automatically adds x-decorator: FormItem when x-component is present.
    - Converts enum to Select component if appropriate.
    - Adds requires: true to required properties.
    - Converts if/then required fields to x-reactions for required/visible.
    """
    formily_schema = copy.deepcopy(schema)

    # Handle allOf
    if "allOf" in schema:
        merged = _merge_allOf(schema["allOf"])
        formily_schema = {**formily_schema, **merged}
        formily_schema.pop("allOf", None)

    # Handle properties recursively, pass required fields in
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

    all_groups_with_titles = [
        {"name": group.value, "title": SECTION_NAME_TITLE[group]}
        for group in SectionName
    ]
    result = {
        "properties": formily_schema.get("properties", {}),
        "group": all_groups_with_titles,
    }
    formily_schema.get("properties", {})
    return result
