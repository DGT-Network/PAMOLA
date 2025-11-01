import copy
from typing import Any, Dict, List
import copy

from pamola_core.common.enum.operator_field_group import (
    OPERATOR_FIELD_GROUP_TITLE,
    OperatorFieldGroup,
)


def merge_allOf(allOf_schemas: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    name: str, prop: Dict[str, Any], required_fields: List[str] = []
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

    if field["x-component"] == "Switch":
        field["x-content"] = f"Enable {field['title']}"

    if field["x-component"] == "NumberPicker":
        min_val = field.get("minimum", 1)
        field["x-component-props"] = {"min": min_val, "step": 1}

    # Nested object
    if field.get("type") == "object" and "properties" in field:
        nested_required = field.get("required", [])
        field["properties"] = {
            k: convert_property(k, v, nested_required)
            for k, v in field["properties"].items()
        }

    if field["x-component"] == "ArrayItems":
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

    return field


def add_x_reactions_for_strategy_required(formily_schema, schema):
    # Directly handle on the original schema
    if "if" in schema and "then" in schema and "required" in schema["then"]:
        condition = schema["if"]
        then = schema.get("then", {})
        if "anyOf" in condition:
            if "properties" in then:
                any_of_list = condition["anyOf"]
                for any_of_condition in any_of_list:
                    if "required" in any_of_condition and isinstance(
                        any_of_condition["required"], list
                    ):
                        required_fields = any_of_condition["required"]
                        for required_field in required_fields:
                            reactions = formily_schema["properties"][
                                required_field
                            ].get("x-reactions", [])
                            # for required_field in schema["then"]["required"]:
                            dependencies = list(schema["then"]["required"])
                            if "properties" in then:
                                visible = ""
                                for key, value in then["properties"].items():
                                    visible = (
                                        f'{visible} && {{{{ $deps[0] === \'{value["const"]}\' }}}}'
                                        if visible
                                        else f'{{{{ $deps[0] === \'{value["const"]}\' }}}}'
                                    )
                            const_val = value["const"]
                            default_value = formily_schema["properties"][key].get(
                                "default", "null"
                            )
                            default_value_str = (
                                default_value if isinstance(default_value, (int, float)) else f"'{default_value}'"
                            )
                            reactions.append(
                                {
                                    "dependencies": dependencies,
                                    "fulfill": {
                                        "state": {
                                            # "visible": f"{{{{ {required_field} && {required_field} !== '' && {required_field} !== null }}}}"
                                            "visible": visible,
                                            # "{{!!$deps[0] && $deps[0] !== '' && $deps[0] !== null}}"
                                            # "required": f"{{{{ $deps[0] !== undefined }}}}",
                                        },
                                        "run": f"{{{{ $self.setValue({default_value_str}) }}}}",
                                    },
                                }
                            )
                            formily_schema["properties"][required_field][
                                "x-reactions"
                            ] = reactions

        elif "properties" in condition:
            for key, value in condition["properties"].items():
                if "properties" in then and key not in then["properties"]:
                    for required_field in schema["then"]["required"]:
                        if (
                            "properties" in formily_schema
                            and required_field in formily_schema["properties"]
                        ):
                            reactions = formily_schema["properties"][key].get(
                                "x-reactions", []
                            )
                            default_value = formily_schema["properties"][key].get(
                                "default", "null"
                            )
                            default_value_str = (
                                default_value if isinstance(default_value, (int, float)) else f"'{default_value}'"
                            )
                            reactions.append(
                                {
                                    "dependencies": [required_field],
                                    "fulfill": {
                                        "state": {
                                            # "visible": f"{{{{ {required_field} && {required_field} !== '' && {required_field} !== null }}}}"
                                            "visible": "{{!!$deps[0] && $deps[0] !== '' && $deps[0] !== null}}"
                                            # "required": f"{{{{ $deps[0] !== undefined }}}}",
                                        },
                                        "run": f"{{{{ $self.setValue({default_value_str}) }}}}",
                                    },
                                }
                            )
                            formily_schema["properties"][key]["x-reactions"] = reactions
                else:
                    const_val = value["const"]
                    for required_field in schema["then"]["required"]:
                        if (
                            "properties" in formily_schema
                            and required_field in formily_schema["properties"]
                        ):
                            reactions = formily_schema["properties"][
                                required_field
                            ].get("x-reactions", [])
                            reactions.append(
                                {
                                    "dependencies": [key],
                                    "fulfill": {
                                        "state": {
                                            "visible": f"{{{{ $deps[0] === '{const_val}' }}}}",
                                            "required": f"{{{{ $deps[0] === '{const_val}' }}}}",
                                        }
                                    },
                                }
                            )
                            formily_schema["properties"][required_field][
                                "x-reactions"
                            ] = reactions

        # Remove if/then/else after processing
        formily_schema.pop("if", None)
        formily_schema.pop("then", None)
        if "else" in formily_schema:
            formily_schema.pop("else", None)


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
        merged = merge_allOf(schema["allOf"])
        formily_schema = {**formily_schema, **merged}
        formily_schema.pop("allOf", None)

    # Handle properties recursively, truyền required vào
    required_fields = formily_schema.get("required", [])
    if "properties" in schema:
        new_properties = {}
        for k, v in schema["properties"].items():
            converted = convert_property(k, v, required_fields)
            if "x-component" in converted:
                new_properties[k] = converted
        formily_schema["properties"] = new_properties

    for key in ["allOf", "anyOf"]:
        if key in schema:
            for sub_schema in schema[key]:
                add_x_reactions_for_strategy_required(formily_schema, sub_schema)

    all_groups_with_titles = [
        {"name": group.value, "title": OPERATOR_FIELD_GROUP_TITLE[group]}
        for group in OperatorFieldGroup
    ]
    result = {
        "properties": formily_schema.get("properties", {}),
        "group": all_groups_with_titles,
    }
    formily_schema.get("properties", {})
    return result
