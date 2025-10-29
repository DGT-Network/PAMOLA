import copy
from pathlib import Path
import sys
from typing import Any, Dict, List, Type

from pamola_core.utils.io import write_json
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.schema_helpers.schema_generator_all import ALL_OP_CONFIGS
from pamola_core.utils.schema_helpers.schema_utils import (
    flatten_schema,
    get_filtered_schema,
    remove_none_from_enum,
    remove_none_from_enum,
)
import copy


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


def convert_property(
    name: str, prop: Dict[str, Any], required_fields: List[str] = []
) -> Dict[str, Any]:
    field = copy.deepcopy(prop)
    field["name"] = name

    def is_min_max_array(items_schema):
        return (
            isinstance(items_schema, dict)
            and items_schema.get("type") == "array"
            and items_schema.get("minItems") is not None
            and items_schema.get("maxItems") is not None
            and isinstance(items_schema.get("items"), dict)
            and items_schema["items"].get("type") == "number"
        )

    # Required
    if name in required_fields:
        field["requires"] = True

    # Enum to Select
    if "enum" in field and "x-component" not in field:
        field["x-component"] = "Select"
        field["x-component-props"] = {"getPopupContainer": "{{(node) => node?.parentElement || document.body}}"}

    # oneOf with const to Select
    if "oneOf" in field and all(isinstance(opt, dict) and "const" in opt for opt in field["oneOf"]):
        field["x-component"] = "Select"
        field["enum"] = [
            {"value": opt["const"], "label": opt.get("description", str(opt["const"]))}
            for opt in field["oneOf"]
        ]
        field.pop("oneOf", None)

    # String to Input
    if (
        field.get("type") == "string"
        or (isinstance(field.get("type"), list) and "string" in field.get("type", []))
    ) and "x-component" not in field:
        field["x-component"] = "Input"

    # Boolean to Switch
    if (
        field.get("type") == "boolean"
        or (isinstance(field.get("type"), list) and "boolean" in field.get("type", []))
    ) and "x-component" not in field:
        field["x-component"] = "Switch"
        if "title" in field:
            field["x-content"] = f"Enable {field['title']}"

    # Integer/Number to NumberPicker
    if (
        field.get("type") == "integer"
        or field.get("type") == "number"
        or (
            isinstance(field.get("type"), list)
            and ("integer" in field.get("type", []) or "number" in field.get("type", []))
        )
    ):
        field["type"] = "number"
        field["x-component"] = "NumberPicker"
        field["x-decorator"] = "FormItem"
        min_val = field.get("minimum", 1)
        field["x-component-props"] = {"min": min_val, "step": 1}
        if "default" in prop:
            field["default"] = prop["default"]

    # Add x-decorator
    if "x-component" in field and "x-decorator" not in field:
        field["x-decorator"] = "FormItem"

    # Default value mapping
    if "default" in field:
        field["default"] = field["default"]

    # Nested object
    if field.get("type") == "object" and "properties" in field:
        nested_required = field.get("required", [])
        field["properties"] = {k: convert_property(k, v, nested_required) for k, v in field["properties"].items()}

    # Handle arrays
    field_type = field.get("type")
    if ((field_type == "array") or (isinstance(field_type, list) and "array" in field_type)) and "items" in field:
        items_schema = field["items"]
        if is_min_max_array(items_schema):
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
                    }
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
        elif isinstance(items_schema, dict) and items_schema.get("type") == "string":
            # Trường hợp array các string: chuyển thành object có trường value và remove
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
                        "x-component": "Input"
                    },
                    "remove": {
                        "type": "void",
                        "x-component": "ArrayItems.Remove",
                        "x-component-props": {"style": {"marginLeft": "8px"}}
                    }
                }
            }
            field["properties"] = {
                "add": {
                    "type": "void",
                    "title": "Add Item",
                    "x-component": "ArrayItems.Addition"
                }
            }
        else:
            field["items"] = convert_json_schema_to_formily(field["items"])

    # Handle oneOf inside property
    if "oneOf" in field:
        field["x-reactions"] = [
            {
                "when": "{{ $self.value }}",
                "fulfill": {
                    "schema": {
                        "oneOf": [convert_json_schema_to_formily(option) for option in field["oneOf"]]
                    }
                },
            }
        ]
        field.pop("oneOf", None)

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
        merged = merge_allOf(schema["allOf"])
        formily_schema = {**formily_schema, **merged}
        formily_schema.pop("allOf", None)

    # Handle oneOf
    if "oneOf" in schema:
        formily_schema["x-reactions"] = [
            {
                "when": "{{ $self.value }}",
                "fulfill": {
                    "schema": {
                        "oneOf": [
                            convert_json_schema_to_formily(option)
                            for option in schema["oneOf"]
                        ]
                    }
                },
            }
        ]
        formily_schema.pop("oneOf", None)

    # Handle if/then/else for required fields (multi-condition)
    # Tìm các điều kiện kiểu: {"if": ..., "then": {"required": [...]}} trong schema hoặc trong allOf/anyOf
    def add_x_reactions_for_strategy_required(formily_schema, schema):
        # Xử lý trực tiếp trên schema gốc
        if "if" in schema and "then" in schema and "required" in schema["then"]:
            condition = schema["if"].get("properties", {})
            for key, value in condition.items():
                if "const" in value:
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
                                    "when": f"{{{{ $values.{key} }}}}",
                                    "fulfill": {
                                        "state": {
                                            "visible": f"{{{{$values.{key} === '{const_val}'}}}}",
                                            "required": f"{{{{$values.{key} === '{const_val}'}}}}",
                                        }
                                    },
                                }
                            )
                            formily_schema["properties"][required_field][
                                "x-reactions"
                            ] = reactions
            # Xóa if/then/else sau khi xử lý
            formily_schema.pop("if", None)
            formily_schema.pop("then", None)
            if "else" in formily_schema:
                formily_schema.pop("else", None)
        # Xử lý trong allOf/anyOf nếu có
        for key in ["allOf", "anyOf"]:
            if key in schema:
                for sub_schema in schema[key]:
                    add_x_reactions_for_strategy_required(formily_schema, sub_schema)

    add_x_reactions_for_strategy_required(formily_schema, schema)

    # Handle properties recursively, truyền required vào
    required_fields = formily_schema.get("required", [])
    if "properties" in schema:
        formily_schema["properties"] = {
            k: convert_property(k, v, required_fields)
            for k, v in schema["properties"].items()
        }

    # Xử lý các điều kiện trong allOf của schema gốc
    if "allOf" in schema and "properties" in formily_schema:
        for cond in schema["allOf"]:
            if (
                isinstance(cond, dict)
                and "if" in cond
                and "then" in cond
                and "required" in cond["then"]
            ):
                condition = cond["if"].get("properties", {})
                for key, value in condition.items():
                    if "const" in value:
                        const_val = value["const"]
                        for required_field in cond["then"]["required"]:
                            if required_field in formily_schema["properties"]:
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

    # Handle dependencies
    if "dependencies" in schema:
      deps = schema["dependencies"]
      for dep_field, dep_schema in deps.items():
          # Nếu là oneOf, gắn x-reactions cho tất cả các trường trong nhánh properties
          if "oneOf" in dep_schema:
              for branch in dep_schema["oneOf"]:
                  props = branch.get("properties", {})
                  prop_names = list(props.keys())
                  for idx, prop_name in enumerate(prop_names):
                      if "properties" in formily_schema and prop_name in formily_schema["properties"]:
                          prop_schema = props[prop_name]
                          if isinstance(prop_schema, dict) and prop_name != dep_field:
                              # Nếu là prop thứ 2, kiểm tra kiểu dữ liệu của prop đầu tiên
                              if idx == 1:
                                  prev_prop_name = prop_names[0]
                                  prev_prop_schema = props[prev_prop_name]
                                  prev_type = prev_prop_schema.get("type")
                                  if prev_type == "string" or (isinstance(prev_type, list) and "string" in prev_type):
                                      visible_expr = "{{!!$deps[0] && $deps[0] !== '' && $deps[0] !== null}}"
                                  elif prev_type == "array" or (isinstance(prev_type, list) and "array" in prev_type):
                                      visible_expr = "{{Array.isArray($deps[0]) && $deps[0].length > 0}}"
                                  else:
                                      visible_expr = "{{!!$deps[0]}}"
                              else:
                                  prop_type = prop_schema.get("type")
                                  if prop_type == "string" or (isinstance(prop_type, list) and "string" in prop_type):
                                      visible_expr = "{{!!$deps[0] && $deps[0] !== '' && $deps[0] !== null}}"
                                  elif prop_type == "array" or (isinstance(prop_type, list) and "array" in prop_type):
                                      visible_expr = "{{Array.isArray($deps[0]) && $deps[0].length > 0}}"
                                  else:
                                      visible_expr = "{{!!$deps[0]}}"
                              formily_schema["properties"][prop_name]["x-reactions"] = [
                                  {
                                      "dependencies": [dep_field],
                                      "fulfill": {
                                          "state": {
                                              "visible": visible_expr
                                          }
                                      }
                                  }
                              ]
          else:
              # Default: giữ nguyên logic cũ cho các dependencies khác
              formily_schema.setdefault("x-reactions", [])
              formily_schema["x-reactions"].append(
                  {
                      "when": f"{{{{ $values.{dep_field} }}}}",
                      "fulfill": {
                          "state": {"visible": True},
                          "schema": convert_json_schema_to_formily(dep_schema),
                      },
                  }
              )
      formily_schema.pop("dependencies", None)

    result = {
        "form": {"labelCol": 6, "wrapperCol": 12},
        "schema": {
            "type": "object",
            "properties": formily_schema.get("properties", {}),
        },
    }
    return result


def generate_formily_schema_json(
    config_class: Type[OperationConfig], task_dir: Path, excluded_fields: List[str] = []
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

    write_json(filtered_schema, task_dir / f"{config_class.__name__}_raw.json")

    formily_schema = convert_json_schema_to_formily(filtered_schema)

    # Use the class name as the output filename
    filename = f"{config_class.__name__}.json"

    output_path = task_dir / filename

    # Write the filtered schema to a JSON file
    path_file = write_json(formily_schema, output_path)

    # Return the path to the written JSON file
    return path_file


def generate_all_op_formily_schemas(task_dir: Path) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    for config_cls, exclude_fields in ALL_OP_CONFIGS:
        generate_formily_schema_json(config_cls, task_dir, exclude_fields)
