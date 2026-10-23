"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module:        CVPL linkage operation core schema
Package:       pamola_core.metrics.schemas
License:       BSD 3-Clause

Pure JSON Schema for ``CVPLLinkageOperation``. Semantic validation (QID names, ladder
rungs, hierarchies, ground-truth columns) is done by
``pamola_core.metrics.privacy.cvpl.CVPLConfig``; this schema fixes the shapes.
"""

from pamola_core.metrics.privacy.cvpl.config import (
    FIT_STRATEGIES,
    GT_SCOPES,
    NULL_POLICIES,
    QID_KINDS,
)
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.schemas.base_op_core_schema import BaseOperationConfig


class CVPLLinkageConfig(OperationConfig):
    """Core configuration schema of ``CVPLLinkageOperation``."""

    schema = {
        "type": "object",
        "title": "CVPL Linkage Operation Core Configuration",
        "description": "Linkage exposure of a release against its source (B -> phi -> PCA -> cosine -> tau).",
        "allOf": [
            BaseOperationConfig.schema,
            {
                "type": "object",
                "properties": {
                    "qids": {
                        "type": "array",
                        "minItems": 1,
                        "description": "Quasi-identifiers the attacker can use to restrict candidates: names or objects {name, kind, bin_width, hierarchy, generalized}.",
                        "items": {
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "kind": {"type": "string", "enum": list(QID_KINDS)},
                                        "bin_width": {"type": ["number", "null"], "exclusiveMinimum": 0},
                                        "hierarchy": {"type": ["object", "null"]},
                                        "generalized": {"type": ["boolean", "null"]},
                                    },
                                    "required": ["name"],
                                },
                            ]
                        },
                    },
                    "auxiliary": {"type": "array", "items": {"type": "string"}, "default": []},
                    "blocking_ladder": {
                        "type": ["array", "null"],
                        "maxItems": 8,
                        "items": {"type": "array", "items": {"type": "string"}},
                    },
                    "tau": {"type": "number", "exclusiveMinimum": 0, "maximum": 1, "default": 0.9},
                    "fit_strategy": {"type": "string", "enum": list(FIT_STRATEGIES), "default": "joint"},
                    "variance_retained": {"type": "number", "exclusiveMinimum": 0, "maximum": 1, "default": 0.9},
                    "min_components": {"type": "integer", "minimum": 1, "default": 3},
                    "max_components": {"type": "integer", "minimum": 1, "default": 50},
                    "max_categories": {"type": "integer", "minimum": 2, "default": 50},
                    "qid_as_features": {"type": "boolean", "default": True},
                    "null_policy": {"type": "string", "enum": list(NULL_POLICIES), "default": "equal_null"},
                    "wildcard_tokens": {"type": "array", "items": {"type": "string"}, "default": ["*"]},
                    "wildcard_token_count": {"type": "integer", "minimum": 0},
                    "wildcard_tokens_digest": {"type": "string"},
                    "direction": {"type": "string", "enum": ["source_to_release"], "default": "source_to_release"},
                    "linkage_unit": {"type": "string", "enum": ["subject"], "default": "subject"},
                    "ground_truth": {
                        "type": ["object", "null"],
                        "properties": {
                            "source_id_column": {"type": "string"},
                            "release_id_column": {"type": "string"},
                            "scope": {"type": "string", "enum": list(GT_SCOPES)},
                        },
                        "required": ["source_id_column", "release_id_column"],
                    },
                    "seed": {"type": ["integer", "null"]},
                    "max_comparisons": {"type": "integer", "minimum": 1},
                    "max_group_pairs": {"type": "integer", "minimum": 1},
                    "chunk_pairs": {"type": "integer", "minimum": 1},
                    "tie_tolerance": {"type": "number", "minimum": 0},
                    "zero_norm_tolerance": {"type": "number", "minimum": 0},
                },
                "required": ["qids"],
            },
        ],
    }
