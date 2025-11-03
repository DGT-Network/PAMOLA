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
from typing import Any, Dict, List
import copy

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


def _is_min_max_array(items_schema):
    return (
        isinstance(items_schema, dict)
        and items_schema.get("type") == "array"
        and items_schema.get("minItems") is not None
        and items_schema.get("maxItems") is not None
        and isinstance(items_schema.get("items"), dict)
        and items_schema["items"].get("type") == "number"
    )


def convert_property(
    name: str,
    prop: Dict[str, Any],
    required_fields: List[str] = [],
    formily_schema: Dict[str, Any] = {},
) -> Dict[str, Any]:
    field = copy.deepcopy(prop)
    field["name"] = name
    field["x-decorator"] = "FormItem"
    if "description" in prop:
        field.pop("description", None)  # Remove description if present

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
                {"type": "minimum", "message": f"{field['title']} must be at least {field['minimum']}", "minimum": field["minimum"]}
            )
        if "maximum" in field:
            field["x-validate"] = field.get("x-validate", [])
            field["x-validate"].append(
                {"type": "maximum", "message": f"{field['title']} must be at most {field['maximum']}", "maximum": field["maximum"]}
            )

    elif field["x-component"] == "ArrayItems":
        if "items" in field:
            items_schema = field["items"]
            if field["type"] == "array" or (isinstance(t, list) and "array" in t):
                if _is_min_max_array(items_schema):
                    field["type"] = "array"
                    field["x-decorator"] = "FormItem"
                    field["x-component"] = "ArrayItems"
                    field["items"] = {
                        "type": "object",
                        "x-component": "ArrayItems.Item",
                        "properties": {
                            "min": {
                                "type": "number",
                                "title": "Min value",
                                "x-decorator": "FormItem",
                                "x-component": "NumberPicker",
                                "x-component-props": {"placeholder": "Min value"},
                            },
                            "max": {
                                "type": "number",
                                "title": "Max value",
                                "x-decorator": "FormItem",
                                "x-component": "NumberPicker",
                                "x-component-props": {"placeholder": "Max value"},
                                "x-decorator-props": {"style": {"marginLeft": "8px"}},
                                "x-reactions": [
                                    {
                                        "dependencies": [".min"],
                                        "when": "{{$deps[0] !== undefined && $self.value !== undefined}}",
                                        "fulfill": {
                                            "run": "if ($self.value <= $deps[0]) { $self.setFeedback({ type: 'error', code: 'range', messages: ['Max must be greater than Min'] }) } else { $self.setFeedback({ type: 'error', code: 'range', messages: [] }) }"
                                        },
                                    }
                                ],
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
        elif field["type"] == "string":
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
                }
            }

    if "x-depend-on" in field or "x-required-on" in field:
        field = _add_x_reactions(field, formily_schema)

    # Nested object
    if field.get("type") == "object" and "properties" in field:
        nested_required = field.get("required", [])
        field["properties"] = {
            k: convert_property(k, v, nested_required)
            for k, v in field["properties"].items()
        }
    return field

def _add_x_reactions(
    field: Dict[str, Any], formily_schema: Dict[str, Any]
) -> Dict[str, Any]:

    default_value = field.get("default", None)
    if default_value is None:
        if field.get("type") == "number":
            if field.get("minimum") is not None:
                default_value_str = field.get("minimum")
            default_value_str = 0
        else:
            default_value_str = "null"
    elif isinstance(default_value, (int, float)):
        default_value_str = default_value
    else:
        default_value_str = f"'{default_value}'"
    
    state = {}

    if "x-depend-on" in field:
        visible_state = ""
        depend_on_items = field["x-depend-on"].items()
        visible_items = []
        for depend_field, depend_value in depend_on_items:
            if depend_field not in formily_schema["properties"]:
                continue
            if "not_null" in depend_value:
                visible_items.append(f" !!$form.values.{depend_field} ")
            elif isinstance(depend_value, list):
                visible_items.extend(
                    [
                        f" $form.values.{depend_field} === '{val}' "
                        for val in depend_value
                    ]
                )
            else:
                visible_items.append(f" $form.values.{depend_field} === '{depend_value}' ")

        visible_state = '&&'.join(visible_items)
        state["visible"] = f'{{{{ {visible_state} }}}}'

    if "x-required-on" in field:
        required_on_items = field["x-required-on"].items()
        required_items = []
        for required_field, required_value in required_on_items:
            if required_field not in formily_schema["properties"]:
                continue
            if "not_null" in required_value:
                required_items.append(f' !!$form.values.{required_field} ')
            elif isinstance(required_value, list):
                
                required_items.append(
                        " || ".join(
                        [
                            f" $form.values.{required_field} === '{val}' "
                            for val in required_value
                        ]
                    )
                )
            else:
                required_items.append(f" $form.values.{required_field} === '{required_value}' ")


        required_state = '&&'.join(required_items)
        state["required"] = f'{{{{ {required_state} }}}}'

    reactions = field.get("x-reactions", [])
    reactions.append(
        {
            "dependencies": [depend_field],
            "fulfill": {
                "state": state,
                "run": f"{{{{ $self.setValue({default_value_str}) }}}}",
            },
        }
    )
    field["x-reactions"] = reactions
    field.pop("x-depend-on", None)
    field.pop("x-required-on", None)

    return field

def convert_json_schema_to_formily(schema: Dict[str, Any]) -> Dict[str, Any]:
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
                converted = convert_property(k, v, required_fields, formily_schema)
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
