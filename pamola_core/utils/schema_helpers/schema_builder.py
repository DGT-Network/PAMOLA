"""
PAMOLA.CORE - Auto Schema Generator for Operation Configs
--------------------------------------------------------
Module:        schema_generator_all.py
Package:       pamola_core.utils
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-10-23
License:       BSD 3-Clause

Description:
    Utility to auto-generate JSON schemas for all operation configs (fake_data, profiling, ...).
    Import and extend ALL_OP_CONFIGS as needed for other modules.

Usage:
    python -m pamola_core.utils.schema_generator_all
    # or import and call generate_all_op_schemas()
"""

import sys
from pathlib import Path
from pamola_core.anonymization.schemas.full_masking_op_tooltip import (
    FullMaskingOpTooltip,
)
from pamola_core.anonymization.schemas.numeric_op_tooltip import NumericOpTooltip
from pamola_core.anonymization.schemas.partial_masking_op_tooltip import (
    PartialMaskingOpTooltip,
)
from pamola_core.anonymization.schemas.record_op_tooltip import (
    RecordSuppressionOpTooltip,
)
from pamola_core.anonymization.schemas.uniform_numeric_op_tooltip import (
    UniformNumericNoiseOpTooltip,
)
from pamola_core.anonymization.schemas.uniform_temporal_op_tooltip import (
    UniformTemporalNoiseOpTooltip,
)
from pamola_core.fake_data.schemas.email_op_config import FakeEmailOperationConfig
from pamola_core.fake_data.schemas.email_op_config_exclude import (
    EMAIL_FAKE_EXCLUDE_FIELDS,
)
from pamola_core.fake_data.schemas.name_op_config import FakeNameOperationConfig
from pamola_core.fake_data.schemas.name_op_config_exclude import (
    NAME_FAKE_EXCLUDE_FIELDS,
)
from pamola_core.fake_data.schemas.organization_op_config import (
    FakeOrganizationOperationConfig,
)
from pamola_core.fake_data.schemas.organization_op_config_exclude import (
    ORGANIZATION_FAKE_EXCLUDE_FIELDS,
)
from pamola_core.fake_data.schemas.phone_op_config import FakePhoneOperationConfig
from pamola_core.fake_data.schemas.phone_op_config_exclude import (
    PHONE_FAKE_EXCLUDE_FIELDS,
)

from pamola_core.profiling.schemas.email_tooltip import EmailOperationTooltip
from pamola_core.profiling.schemas.identity_tooltip import IdentityAnalysisOperationTooltip
from pamola_core.profiling.schemas.mvf_tooltip import MVFAnalysisOperationTooltip
from pamola_core.profiling.schemas.numeric_tooltip import NumericOperationTooltip
from pamola_core.transformations.schemas.add_modify_fields_config import (
    AddOrModifyFieldsOperationConfig,
)
from pamola_core.transformations.schemas.add_modify_fields_config_exclude import (
    ADD_MODIFY_FIELDS_EXCLUDE_FIELDS,
)
from pamola_core.transformations.schemas.aggregate_records_op_config import (
    AggregateRecordsOperationConfig,
)
from pamola_core.transformations.schemas.aggregate_records_op_config_exclude import (
    AGGREGATE_RECORDS_EXCLUDE_FIELDS,
)
from pamola_core.transformations.schemas.clean_invalid_values_config import (
    CleanInvalidValuesOperationConfig,
)
from pamola_core.transformations.schemas.clean_invalid_values_config_exclude import (
    CLEAN_INVALID_VALUES_EXCLUDE_FIELDS,
)
from pamola_core.transformations.schemas.impute_missing_values_op_config import (
    ImputeMissingValuesConfig,
)
from pamola_core.transformations.schemas.impute_missing_values_op_config_exclude import (
    IMPUTE_MISSING_VALUES_EXCLUDE_FIELDS,
)
from pamola_core.transformations.schemas.merge_datasets_op_config import (
    MergeDatasetsOperationConfig,
)
from pamola_core.transformations.schemas.merge_datasets_op_config_exclude import (
    MERGE_DATASETS_EXCLUDE_FIELDS,
)
from pamola_core.transformations.schemas.remove_fields_config import (
    RemoveFieldsOperationConfig,
)
from pamola_core.transformations.schemas.remove_fields_config_exclude import (
    REMOVE_FIELDS_EXCLUDE_FIELDS,
)
from pamola_core.transformations.schemas.split_by_id_values_op_config import (
    SplitByIDValuesOperationConfig,
)
from pamola_core.transformations.schemas.split_by_id_values_op_config_exclude import (
    SPLIT_BY_ID_VALUES_EXCLUDE_FIELDS,
)
from pamola_core.transformations.schemas.split_fields_op_config import (
    SplitFieldsOperationConfig,
)
from pamola_core.transformations.schemas.split_fields_op_config_exclude import (
    SPLIT_FIELDS_EXCLUDE_FIELDS,
)
from pamola_core.anonymization.schemas.categorical_op_schema import (
    CategoricalGeneralizationConfig,
)
from pamola_core.anonymization.schemas.categorical_op_schema_exclude import (
    CATEGORICAL_GENERALIZATION_EXCLUDE_FIELDS,
)
from pamola_core.anonymization.schemas.datetime_op_schema import (
    DateTimeGeneralizationConfig,
)
from pamola_core.anonymization.schemas.datetime_op_schema_exclude import (
    DATETIME_GENERALIZATION_EXCLUDE_FIELDS,
)
from pamola_core.anonymization.schemas.full_masking_op_schema import FullMaskingConfig
from pamola_core.anonymization.schemas.full_masking_op_schema_exclude import (
    FULL_MASKING_EXCLUDE_FIELDS,
)
from pamola_core.anonymization.schemas.numeric_op_schema import (
    NumericGeneralizationConfig,
)
from pamola_core.anonymization.schemas.numeric_op_schema_exclude import (
    NUMERIC_GENERALIZATION_EXCLUDE_FIELDS,
)
from pamola_core.anonymization.schemas.partial_masking_op_schema import (
    PartialMaskingConfig,
)
from pamola_core.anonymization.schemas.partial_masking_op_schema_exclude import (
    PARTIAL_MASKING_EXCLUDE_FIELDS,
)
from pamola_core.anonymization.schemas.attribute_op_schema import (
    AttributeSuppressionConfig,
)
from pamola_core.anonymization.schemas.attribute_op_schema_exclude import (
    ATTRIBUTE_SUPPRESSION_EXCLUDE_FIELDS,
)
from pamola_core.anonymization.schemas.cell_op_schema import CellSuppressionConfig
from pamola_core.anonymization.schemas.cell_op_schema_exclude import CELL_EXCLUDE_FIELDS
from pamola_core.anonymization.schemas.record_op_schema import RecordSuppressionConfig
from pamola_core.anonymization.schemas.record_op_config_exclude import (
    RECORD_EXCLUDE_FIELDS,
)
from pamola_core.anonymization.schemas.uniform_numeric_op_schema import (
    UniformNumericNoiseConfig,
)
from pamola_core.anonymization.schemas.uniform_numeric_op_schema_exclude import (
    RECORD_EXCLUDE_FIELDS as UNIFORM_NUMERIC_EXCLUDE_FIELDS,
)
from pamola_core.anonymization.schemas.uniform_temporal_op_schema import (
    UniformTemporalNoiseConfig,
)
from pamola_core.anonymization.schemas.uniform_temporal_op_schema_exclude import (
    RECORD_EXCLUDE_FIELDS as UNIFORM_TEMPORAL_EXCLUDE_FIELDS,
)

from pamola_core.metrics.schemas.fidelity_ops_config import FidelityConfig
from pamola_core.metrics.schemas.fidelity_ops_config_exclude import (
    FIDELITY_EXCLUDE_FIELDS,
)
from pamola_core.metrics.schemas.privacy_ops_config import PrivacyMetricConfig
from pamola_core.metrics.schemas.privacy_ops_config_exclude import (
    PRIVACY_EXCLUDE_FIELDS,
)
from pamola_core.metrics.schemas.utility_ops_config import UtilityMetricConfig
from pamola_core.metrics.schemas.utility_ops_config_exclude import (
    UTILITY_EXCLUDE_FIELDS,
)

from pamola_core.profiling.schemas.anonymity_config import (
    KAnonymityProfilerOperationConfig,
)
from pamola_core.profiling.schemas.anonymity_config_exclude import (
    ANONYMITY_EXCLUDE_FIELDS,
)
from pamola_core.profiling.schemas.attribute_config import (
    DataAttributeProfilerOperationConfig,
)
from pamola_core.profiling.schemas.attribute_config_exclude import (
    ATTRIBUTE_EXCLUDE_FIELDS,
)
from pamola_core.profiling.schemas.categorical_config import CategoricalOperationConfig
from pamola_core.profiling.schemas.categorical_config_exclude import (
    CATEGORICAL_EXCLUDE_FIELDS,
)
from pamola_core.profiling.schemas.correlation_schema import (
    CorrelationOperationConfig,
    CorrelationMatrixOperationConfig,
)
from pamola_core.profiling.schemas.correlation_schema_exclude import (
    CORRELATION_EXCLUDE_FIELDS,
    CORRELATION_MATRIX_EXCLUDE_FIELDS,
)
from pamola_core.profiling.schemas.currency_schema import CurrencyOperationConfig
from pamola_core.profiling.schemas.currency_schema_exclude import (
    CURRENCY_EXCLUDE_FIELDS,
)
from pamola_core.profiling.schemas.date_schema import DateOperationConfig
from pamola_core.profiling.schemas.date_schema_exclude import DATE_EXCLUDE_FIELDS
from pamola_core.profiling.schemas.email_schema import EmailOperationConfig
from pamola_core.profiling.schemas.email_schema_exclude import EMAIL_EXCLUDE_FIELDS
from pamola_core.profiling.schemas.group_config import GroupAnalyzerOperationConfig
from pamola_core.profiling.schemas.group_config_exclude import GROUP_EXCLUDE_FIELDS
from pamola_core.profiling.schemas.identity_schema import (
    IdentityAnalysisOperationConfig,
)
from pamola_core.profiling.schemas.identity_schema_exclude import (
    IDENTITY_EXCLUDE_FIELDS,
)
from pamola_core.profiling.schemas.mvf_schema import MVFAnalysisOperationConfig
from pamola_core.profiling.schemas.mvf_schema_exclude import MVF_EXCLUDE_FIELDS
from pamola_core.profiling.schemas.numeric_schema import NumericOperationConfig
from pamola_core.profiling.schemas.numeric_schema_exclude import NUMERIC_EXCLUDE_FIELDS
from pamola_core.profiling.schemas.phone_config import PhoneOperationConfig
from pamola_core.profiling.schemas.phone_config_exclude import PHONE_EXCLUDE_FIELDS
from pamola_core.profiling.schemas.text_config import (
    TextSemanticCategorizerOperationConfig,
)
from pamola_core.profiling.schemas.text_config_exclude import TEXT_EXCLUDE_FIELDS
from pamola_core.anonymization.schemas.datetime_op_tooltip import DateTimeOpTooltip
from pamola_core.anonymization.schemas.categorical_op_tooltip import (
    CategoricalOpTooltip,
)
from pamola_core.anonymization.schemas.cell_op_tooltip import CellSuppressionOpTooltip
from pamola_core.anonymization.schemas.attribute_op_tooltip import AttributeSuppressionOpTooltip
from pamola_core.profiling.schemas.date_tooltip import DateOpTooltip
from pamola_core.profiling.schemas.currency_tooltip import CurrencyOpTooltip
from pamola_core.profiling.schemas.correlation_tooltip import CorrelationOpTooltip

from pamola_core.utils.schema_helpers.schema_utils import generate_schema_json

ALL_OP_CONFIGS = [
    (FakeEmailOperationConfig, EMAIL_FAKE_EXCLUDE_FIELDS, None),
    (FakeNameOperationConfig, NAME_FAKE_EXCLUDE_FIELDS, None),
    (FakeOrganizationOperationConfig, ORGANIZATION_FAKE_EXCLUDE_FIELDS, None),
    (FakePhoneOperationConfig, PHONE_FAKE_EXCLUDE_FIELDS, None),
    (AddOrModifyFieldsOperationConfig, ADD_MODIFY_FIELDS_EXCLUDE_FIELDS, None),
    (AggregateRecordsOperationConfig, AGGREGATE_RECORDS_EXCLUDE_FIELDS, None),
    (CleanInvalidValuesOperationConfig, CLEAN_INVALID_VALUES_EXCLUDE_FIELDS, None),
    (ImputeMissingValuesConfig, IMPUTE_MISSING_VALUES_EXCLUDE_FIELDS, None),
    (MergeDatasetsOperationConfig, MERGE_DATASETS_EXCLUDE_FIELDS, None),
    (RemoveFieldsOperationConfig, REMOVE_FIELDS_EXCLUDE_FIELDS, None),
    (SplitByIDValuesOperationConfig, SPLIT_BY_ID_VALUES_EXCLUDE_FIELDS, None),
    (SplitFieldsOperationConfig, SPLIT_FIELDS_EXCLUDE_FIELDS, None),
    (
        CategoricalGeneralizationConfig,
        CATEGORICAL_GENERALIZATION_EXCLUDE_FIELDS,
        CategoricalOpTooltip.as_dict(),
    ),
    (
        DateTimeGeneralizationConfig,
        DATETIME_GENERALIZATION_EXCLUDE_FIELDS,
        DateTimeOpTooltip.as_dict(),
    ),
    (
        FullMaskingConfig, 
        FULL_MASKING_EXCLUDE_FIELDS, 
        FullMaskingOpTooltip.as_dict()),
    (
        NumericGeneralizationConfig,
        NUMERIC_GENERALIZATION_EXCLUDE_FIELDS,
        NumericOpTooltip.as_dict(),
    ),
    (
        PartialMaskingConfig,
        PARTIAL_MASKING_EXCLUDE_FIELDS,
        PartialMaskingOpTooltip.as_dict(),
    ),
    (
        AttributeSuppressionConfig, 
        ATTRIBUTE_SUPPRESSION_EXCLUDE_FIELDS, 
        AttributeSuppressionOpTooltip.as_dict()),
    (
        CellSuppressionConfig, 
        CELL_EXCLUDE_FIELDS, 
        CellSuppressionOpTooltip.as_dict()),
    (
        RecordSuppressionConfig,
        RECORD_EXCLUDE_FIELDS,
        RecordSuppressionOpTooltip.as_dict(),
    ),
    (
        UniformNumericNoiseConfig,
        UNIFORM_NUMERIC_EXCLUDE_FIELDS,
        UniformNumericNoiseOpTooltip.as_dict(),
    ),
    (
        UniformTemporalNoiseConfig,
        UNIFORM_TEMPORAL_EXCLUDE_FIELDS,
        UniformTemporalNoiseOpTooltip.as_dict(),
    ),
    (FidelityConfig, FIDELITY_EXCLUDE_FIELDS, None),
    (PrivacyMetricConfig, PRIVACY_EXCLUDE_FIELDS, None),
    (UtilityMetricConfig, UTILITY_EXCLUDE_FIELDS, None),
    (KAnonymityProfilerOperationConfig, ANONYMITY_EXCLUDE_FIELDS, None),
    (DataAttributeProfilerOperationConfig, ATTRIBUTE_EXCLUDE_FIELDS, None),
    (CategoricalOperationConfig, CATEGORICAL_EXCLUDE_FIELDS, None),
    (CorrelationOperationConfig, CORRELATION_EXCLUDE_FIELDS, CorrelationOpTooltip.as_dict()),
    (CorrelationMatrixOperationConfig, CORRELATION_MATRIX_EXCLUDE_FIELDS, None),
    (CurrencyOperationConfig, CURRENCY_EXCLUDE_FIELDS, CurrencyOpTooltip.as_dict()),
    (DateOperationConfig, DATE_EXCLUDE_FIELDS, DateOpTooltip.as_dict()),
    (
        EmailOperationConfig, 
        EMAIL_EXCLUDE_FIELDS, 
        EmailOperationTooltip.as_dict()),
    (GroupAnalyzerOperationConfig, GROUP_EXCLUDE_FIELDS, None),
    (
        IdentityAnalysisOperationConfig, 
        IDENTITY_EXCLUDE_FIELDS, 
        IdentityAnalysisOperationTooltip.as_dict()),
    (
        MVFAnalysisOperationConfig, 
        MVF_EXCLUDE_FIELDS, 
        MVFAnalysisOperationTooltip.as_dict()),
    (
        NumericOperationConfig, 
        NUMERIC_EXCLUDE_FIELDS, 
        NumericOperationTooltip.as_dict()),
    (PhoneOperationConfig, PHONE_EXCLUDE_FIELDS, None),
    (TextSemanticCategorizerOperationConfig, TEXT_EXCLUDE_FIELDS, None),
]


def generate_all_op_schemas(
    task_dir: Path, generate_formily_schema: bool = True
) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    for item in ALL_OP_CONFIGS:
        config_cls, exclude_fields, tooltip = item
        generate_schema_json(
            config_cls, task_dir, exclude_fields, generate_formily_schema, tooltip
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path.cwd()
    generate_all_op_schemas(output_dir, generate_formily_schema=True)