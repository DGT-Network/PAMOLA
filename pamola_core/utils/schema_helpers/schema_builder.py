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
from pamola_core.anonymization.schemas.numeric_op_tooltip import (
    NumericGeneralizationTooltip,
)
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
from pamola_core.fake_data.schemas.email_op_schema import FakeEmailOperationConfig
from pamola_core.fake_data.schemas.email_op_schema_exclude import (
    EMAIL_FAKE_EXCLUDE_FIELDS,
)
from pamola_core.fake_data.schemas.email_op_tooltip import FakeEmailOperationTooltip
from pamola_core.fake_data.schemas.name_op_schema import FakeNameOperationConfig
from pamola_core.fake_data.schemas.name_op_schema_exclude import (
    NAME_FAKE_EXCLUDE_FIELDS,
)
from pamola_core.fake_data.schemas.name_op_tooltip import FakeNameOperationTooltip
from pamola_core.fake_data.schemas.organization_op_schema import (
    FakeOrganizationOperationConfig,
)
from pamola_core.fake_data.schemas.organization_op_schema_exclude import (
    ORGANIZATION_FAKE_EXCLUDE_FIELDS,
)
from pamola_core.fake_data.schemas.organization_op_tooltip import (
    FakeOrganizationOperationTooltip,
)
from pamola_core.fake_data.schemas.phone_op_schema import FakePhoneOperationConfig
from pamola_core.fake_data.schemas.phone_op_schema_exclude import (
    PHONE_FAKE_EXCLUDE_FIELDS,
)

from pamola_core.fake_data.schemas.phone_op_tooltip import FakePhoneOperationTooltip
from pamola_core.profiling.schemas.anonymity_tooltip import (
    KAnonymityProfilerOperationTooltip,
)
from pamola_core.profiling.schemas.attribute_tooltip import (
    DataAttributeProfilerOperationTooltip,
)
from pamola_core.profiling.schemas.categorical_tooltip import CategoricalTooltip
from pamola_core.profiling.schemas.correlation_matrix_schema import (
    CorrelationMatrixOperationConfig,
)
from pamola_core.profiling.schemas.correlation_matrix_schema_exclude import (
    CORRELATION_MATRIX_EXCLUDE_FIELDS,
)
from pamola_core.profiling.schemas.correlation_matrix_tooltip import (
    CorrelationMatrixOperationTooltip,
)
from pamola_core.profiling.schemas.email_tooltip import EmailOperationTooltip
from pamola_core.profiling.schemas.group_tooltip import GroupAnalyzerOperationTooltip
from pamola_core.profiling.schemas.identity_tooltip import (
    IdentityAnalysisOperationTooltip,
)
from pamola_core.profiling.schemas.mvf_tooltip import MVFAnalysisOperationTooltip
from pamola_core.profiling.schemas.numeric_tooltip import NumericOperationTooltip
from pamola_core.profiling.schemas.phone_tooltip import PhoneOperationTooltip
from pamola_core.profiling.schemas.text_tooltip import (
    TextSemanticCategorizerOperationTooltip,
)
from pamola_core.transformations.schemas.add_modify_fields_schema import (
    AddOrModifyFieldsOperationConfig,
)
from pamola_core.transformations.schemas.add_modify_fields_schema_exclude import (
    ADD_MODIFY_FIELDS_EXCLUDE_FIELDS,
)
from pamola_core.transformations.schemas.add_modify_fields_tooltip import (
    AddOrModifyFieldsOperationTooltip,
)
from pamola_core.transformations.schemas.aggregate_records_op_schema import (
    AggregateRecordsOperationConfig,
)
from pamola_core.transformations.schemas.aggregate_records_op_schema_exclude import (
    AGGREGATE_RECORDS_EXCLUDE_FIELDS,
)
from pamola_core.transformations.schemas.aggregate_records_op_tooltip import (
    AggregateRecordsOperationTooltip,
)
from pamola_core.transformations.schemas.clean_invalid_values_schema import (
    CleanInvalidValuesOperationConfig,
)
from pamola_core.transformations.schemas.clean_invalid_values_schema_exclude import (
    CLEAN_INVALID_VALUES_EXCLUDE_FIELDS,
)
from pamola_core.transformations.schemas.clean_invalid_values_tooltip import (
    CleanInvalidValuesOperationTooltip,
)
from pamola_core.transformations.schemas.impute_missing_values_op_schema import (
    ImputeMissingValuesConfig,
)
from pamola_core.transformations.schemas.impute_missing_values_op_schema_exclude import (
    IMPUTE_MISSING_VALUES_EXCLUDE_FIELDS,
)
from pamola_core.transformations.schemas.impute_missing_values_op_tooltip import (
    ImputeMissingValuesOperationTooltip,
)
from pamola_core.transformations.schemas.merge_datasets_op_schema import (
    MergeDatasetsOperationConfig,
)
from pamola_core.transformations.schemas.merge_datasets_op_schema_exclude import (
    MERGE_DATASETS_EXCLUDE_FIELDS,
)
from pamola_core.transformations.schemas.merge_datasets_op_tooltip import (
    MergeDatasetsOperationTooltip,
)
from pamola_core.transformations.schemas.remove_fields_op_schema import (
    RemoveFieldsOperationConfig,
)
from pamola_core.transformations.schemas.remove_fields_op_schema_exclude import (
    REMOVE_FIELDS_EXCLUDE_FIELDS,
)
from pamola_core.transformations.schemas.remove_fields_op_tooltip import (
    RemoveFieldsOperationTooltip,
)
from pamola_core.transformations.schemas.split_by_id_values_op_schema import (
    SplitByIDValuesOperationConfig,
)
from pamola_core.transformations.schemas.split_by_id_values_op_schema_exclude import (
    SPLIT_BY_ID_VALUES_EXCLUDE_FIELDS,
)
from pamola_core.transformations.schemas.split_by_id_values_op_tooltip import (
    SplitByIDValuesOperationTooltip,
)
from pamola_core.transformations.schemas.split_fields_op_schema import (
    SplitFieldsOperationConfig,
)
from pamola_core.transformations.schemas.split_fields_op_schema_exclude import (
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
from pamola_core.anonymization.schemas.record_op_schema_exclude import (
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

from pamola_core.profiling.schemas.anonymity_schema import (
    KAnonymityProfilerOperationConfig,
)
from pamola_core.profiling.schemas.anonymity_schema_exclude import (
    ANONYMITY_EXCLUDE_FIELDS,
)
from pamola_core.profiling.schemas.attribute_schema import (
    DataAttributeProfilerOperationConfig,
)
from pamola_core.profiling.schemas.attribute_schema_exclude import (
    ATTRIBUTE_EXCLUDE_FIELDS,
)
from pamola_core.profiling.schemas.categorical_schema import CategoricalOperationConfig
from pamola_core.profiling.schemas.categorical_schema_exclude import (
    CATEGORICAL_EXCLUDE_FIELDS,
)
from pamola_core.profiling.schemas.correlation_schema import (
    CorrelationOperationConfig,
)
from pamola_core.profiling.schemas.correlation_schema_exclude import (
    CORRELATION_EXCLUDE_FIELDS,
)
from pamola_core.profiling.schemas.currency_schema import CurrencyOperationConfig
from pamola_core.profiling.schemas.currency_schema_exclude import (
    CURRENCY_EXCLUDE_FIELDS,
)
from pamola_core.profiling.schemas.date_schema import DateOperationConfig
from pamola_core.profiling.schemas.date_schema_exclude import DATE_EXCLUDE_FIELDS
from pamola_core.profiling.schemas.email_schema import EmailOperationConfig
from pamola_core.profiling.schemas.email_schema_exclude import EMAIL_EXCLUDE_FIELDS
from pamola_core.profiling.schemas.group_schema import GroupAnalyzerOperationConfig
from pamola_core.profiling.schemas.group_schema_exclude import GROUP_EXCLUDE_FIELDS
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
from pamola_core.profiling.schemas.phone_schema import PhoneOperationConfig
from pamola_core.profiling.schemas.phone_schema_exclude import PHONE_EXCLUDE_FIELDS
from pamola_core.profiling.schemas.text_schema import (
    TextSemanticCategorizerOperationConfig,
)
from pamola_core.profiling.schemas.text_schema_exclude import TEXT_EXCLUDE_FIELDS
from pamola_core.anonymization.schemas.datetime_op_tooltip import DateTimeOpTooltip
from pamola_core.anonymization.schemas.categorical_op_tooltip import (
    CategoricalOpTooltip,
)
from pamola_core.anonymization.schemas.cell_op_tooltip import CellSuppressionOpTooltip
from pamola_core.anonymization.schemas.attribute_op_tooltip import (
    AttributeSuppressionOpTooltip,
)
from pamola_core.profiling.schemas.date_tooltip import DateOpTooltip
from pamola_core.profiling.schemas.currency_tooltip import CurrencyOpTooltip
from pamola_core.profiling.schemas.correlation_tooltip import CorrelationOpTooltip

from pamola_core.transformations.schemas.split_fields_op_tooltip import (
    SplitFieldsOperationTooltip,
)
from pamola_core.utils.schema_helpers.schema_utils import generate_schema_json

ALL_OP_CONFIGS = [
    # -------------- Anonymization ---------------
    (
        NumericGeneralizationConfig,
        NUMERIC_GENERALIZATION_EXCLUDE_FIELDS,
        NumericGeneralizationTooltip.as_dict(),
    ),
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
        FullMaskingOpTooltip.as_dict(),
    ),
    (
        PartialMaskingConfig,
        PARTIAL_MASKING_EXCLUDE_FIELDS,
        PartialMaskingOpTooltip.as_dict(),
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
    (
        AttributeSuppressionConfig,
        ATTRIBUTE_SUPPRESSION_EXCLUDE_FIELDS,
        AttributeSuppressionOpTooltip.as_dict(),
    ),
    (
        CellSuppressionConfig,
        CELL_EXCLUDE_FIELDS,
        CellSuppressionOpTooltip.as_dict(),
    ),
    (
        RecordSuppressionConfig,
        RECORD_EXCLUDE_FIELDS,
        RecordSuppressionOpTooltip.as_dict(),
    ),
    # -------------- Fake Data -------------------
    (
        FakeEmailOperationConfig,
        EMAIL_FAKE_EXCLUDE_FIELDS,
        FakeEmailOperationTooltip.as_dict(),
    ),
    (
        FakeNameOperationConfig,
        NAME_FAKE_EXCLUDE_FIELDS,
        FakeNameOperationTooltip.as_dict(),
    ),
    (
        FakeOrganizationOperationConfig,
        ORGANIZATION_FAKE_EXCLUDE_FIELDS,
        FakeOrganizationOperationTooltip.as_dict(),
    ),
    (
        FakePhoneOperationConfig,
        PHONE_FAKE_EXCLUDE_FIELDS,
        FakePhoneOperationTooltip.as_dict(),
    ),
    # -------------- Profiling -------------------
    (
        KAnonymityProfilerOperationConfig,
        ANONYMITY_EXCLUDE_FIELDS,
        KAnonymityProfilerOperationTooltip.as_dict(),
    ),
    (
        DataAttributeProfilerOperationConfig,
        ATTRIBUTE_EXCLUDE_FIELDS,
        DataAttributeProfilerOperationTooltip.as_dict(),
    ),
    (
        CategoricalOperationConfig,
        CATEGORICAL_EXCLUDE_FIELDS,
        CategoricalTooltip.as_dict(),
    ),
    (
        CorrelationOperationConfig,
        CORRELATION_EXCLUDE_FIELDS,
        CorrelationOpTooltip.as_dict(),
    ),
    (
        CorrelationMatrixOperationConfig,
        CORRELATION_MATRIX_EXCLUDE_FIELDS,
        CorrelationMatrixOperationTooltip.as_dict(),
    ),
    (
        CurrencyOperationConfig,
        CURRENCY_EXCLUDE_FIELDS,
        CurrencyOpTooltip.as_dict(),
    ),
    (
        DateOperationConfig,
        DATE_EXCLUDE_FIELDS,
        DateOpTooltip.as_dict(),
    ),
    (
        EmailOperationConfig,
        EMAIL_EXCLUDE_FIELDS,
        EmailOperationTooltip.as_dict(),
    ),
    (
        GroupAnalyzerOperationConfig,
        GROUP_EXCLUDE_FIELDS,
        GroupAnalyzerOperationTooltip.as_dict(),
    ),
    (
        IdentityAnalysisOperationConfig,
        IDENTITY_EXCLUDE_FIELDS,
        IdentityAnalysisOperationTooltip.as_dict(),
    ),
    (
        MVFAnalysisOperationConfig,
        MVF_EXCLUDE_FIELDS,
        MVFAnalysisOperationTooltip.as_dict(),
    ),
    (
        NumericOperationConfig,
        NUMERIC_EXCLUDE_FIELDS,
        NumericOperationTooltip.as_dict(),
    ),
    (
        PhoneOperationConfig,
        PHONE_EXCLUDE_FIELDS,
        PhoneOperationTooltip.as_dict(),
    ),
    (
        TextSemanticCategorizerOperationConfig,
        TEXT_EXCLUDE_FIELDS,
        TextSemanticCategorizerOperationTooltip.as_dict(),
    ),
    # -------------- Transformations -------------
    (
        AddOrModifyFieldsOperationConfig,
        ADD_MODIFY_FIELDS_EXCLUDE_FIELDS,
        AddOrModifyFieldsOperationTooltip.as_dict(),
    ),
    (
        RemoveFieldsOperationConfig,
        REMOVE_FIELDS_EXCLUDE_FIELDS,
        RemoveFieldsOperationTooltip.as_dict(),
    ),
    (
        AggregateRecordsOperationConfig,
        AGGREGATE_RECORDS_EXCLUDE_FIELDS,
        AggregateRecordsOperationTooltip.as_dict(),
    ),
    (
        CleanInvalidValuesOperationConfig,
        CLEAN_INVALID_VALUES_EXCLUDE_FIELDS,
        CleanInvalidValuesOperationTooltip.as_dict(),
    ),
    (
        ImputeMissingValuesConfig,
        IMPUTE_MISSING_VALUES_EXCLUDE_FIELDS,
        ImputeMissingValuesOperationTooltip.as_dict(),
    ),
    (
        MergeDatasetsOperationConfig,
        MERGE_DATASETS_EXCLUDE_FIELDS,
        MergeDatasetsOperationTooltip.as_dict(),
    ),
    (
        SplitByIDValuesOperationConfig,
        SPLIT_BY_ID_VALUES_EXCLUDE_FIELDS,
        SplitByIDValuesOperationTooltip.as_dict(),
    ),
    (
        SplitFieldsOperationConfig,
        SPLIT_FIELDS_EXCLUDE_FIELDS,
        SplitFieldsOperationTooltip.as_dict(),
    ),
    # -------------- Metrics -------------
    (FidelityConfig, FIDELITY_EXCLUDE_FIELDS, None),
    (PrivacyMetricConfig, PRIVACY_EXCLUDE_FIELDS, None),
    (UtilityMetricConfig, UTILITY_EXCLUDE_FIELDS, None),
]


def generate_all_op_schemas(
    task_dir: Path, generate_formily_schema: bool = True
) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    for item in ALL_OP_CONFIGS:
        config_cls, exclude_fields, tooltip = item
        if tooltip is None:
            pass
        try:
            generate_schema_json(
                config_cls, task_dir, exclude_fields, generate_formily_schema, tooltip
            )
        except KeyError as e:
            print(f"Skipping {config_cls.__name__} due to missing group: {e}")
            continue


if __name__ == "__main__":
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path.cwd()
    generate_all_op_schemas(output_dir, generate_formily_schema=True)
