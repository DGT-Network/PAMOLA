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
from pamola_core.fake_data.schemas.email_op_config import FakeEmailOperationConfig
from pamola_core.fake_data.schemas.email_op_config_exclude import EMAIL_FAKE_EXCLUDE_FIELDS
from pamola_core.fake_data.schemas.name_op_config import FakeNameOperationConfig
from pamola_core.fake_data.schemas.name_op_config_exclude import NAME_FAKE_EXCLUDE_FIELDS
from pamola_core.fake_data.schemas.organization_op_config import FakeOrganizationOperationConfig
from pamola_core.fake_data.schemas.organization_op_config_exclude import ORGANIZATION_FAKE_EXCLUDE_FIELDS
from pamola_core.fake_data.schemas.phone_op_config import FakePhoneOperationConfig
from pamola_core.fake_data.schemas.phone_op_config_exclude import PHONE_FAKE_EXCLUDE_FIELDS

from pamola_core.transformations.schemas.add_modify_fields_config import AddOrModifyFieldsOperationConfig
from pamola_core.transformations.schemas.add_modify_fields_config_exclude import ADD_MODIFY_FIELDS_EXCLUDE_FIELDS
from pamola_core.transformations.schemas.aggregate_records_op_config import AggregateRecordsOperationConfig
from pamola_core.transformations.schemas.aggregate_records_op_config_exclude import AGGREGATE_RECORDS_EXCLUDE_FIELDS
from pamola_core.transformations.schemas.clean_invalid_values_config import CleanInvalidValuesOperationConfig
from pamola_core.transformations.schemas.clean_invalid_values_config_exclude import CLEAN_INVALID_VALUES_EXCLUDE_FIELDS
from pamola_core.transformations.schemas.impute_missing_values_op_config import ImputeMissingValuesConfig
from pamola_core.transformations.schemas.impute_missing_values_op_config_exclude import IMPUTE_MISSING_VALUES_EXCLUDE_FIELDS
from pamola_core.transformations.schemas.merge_datasets_op_config import MergeDatasetsOperationConfig
from pamola_core.transformations.schemas.merge_datasets_op_config_exclude import MERGE_DATASETS_EXCLUDE_FIELDS
from pamola_core.transformations.schemas.remove_fields_config import RemoveFieldsOperationConfig
from pamola_core.transformations.schemas.remove_fields_config_exclude import REMOVE_FIELDS_EXCLUDE_FIELDS
from pamola_core.transformations.schemas.split_by_id_values_op_config import SplitByIDValuesOperationConfig
from pamola_core.transformations.schemas.split_by_id_values_op_config_exclude import SPLIT_BY_ID_VALUES_EXCLUDE_FIELDS
from pamola_core.transformations.schemas.split_fields_op_config import SplitFieldsOperationConfig
from pamola_core.transformations.schemas.split_fields_op_config_exclude import SPLIT_FIELDS_EXCLUDE_FIELDS


from pamola_core.anonymization.schemas.categorical_op_config import CategoricalGeneralizationConfig
from pamola_core.anonymization.schemas.categorical_op_exclude import CATEGORICAL_GENERALIZATION_EXCLUDE_FIELDS
from pamola_core.anonymization.schemas.datetime_op_config import DateTimeGeneralizationConfig
from pamola_core.anonymization.schemas.datetime_op_config_exclude import DATETIME_GENERALIZATION_EXCLUDE_FIELDS
from pamola_core.anonymization.schemas.full_masking_op_config import FullMaskingConfig
from pamola_core.anonymization.schemas.full_masking_op_config_exclude import FULL_MASKING_EXCLUDE_FIELDS
from pamola_core.anonymization.schemas.numeric_op_config import NumericGeneralizationConfig
from pamola_core.anonymization.schemas.numeric_op_config_exclude import NUMERIC_GENERALIZATION_EXCLUDE_FIELDS
from pamola_core.anonymization.schemas.partial_masking_op_config import PartialMaskingConfig
from pamola_core.anonymization.schemas.partial_masking_op_config_exclude import PARTIAL_MASKING_EXCLUDE_FIELDS
from pamola_core.anonymization.schemas.attribute_op_config import AttributeSuppressionConfig
from pamola_core.anonymization.schemas.attribute_op_config_exclude import ATTRIBUTE_EXCLUDE_FIELDS
from pamola_core.anonymization.schemas.cell_op_config import CellSuppressionConfig
from pamola_core.anonymization.schemas.cell_op_config_exclude import CELL_EXCLUDE_FIELDS
from pamola_core.anonymization.schemas.record_op_config import RecordSuppressionConfig
from pamola_core.anonymization.schemas.record_op_config_exclude import RECORD_EXCLUDE_FIELDS
from pamola_core.anonymization.schemas.uniform_numeric_op_config import UniformNumericNoiseConfig
from pamola_core.anonymization.schemas.uniform_numeric_op_config_exclude import RECORD_EXCLUDE_FIELDS as UNIFORM_NUMERIC_EXCLUDE_FIELDS
from pamola_core.anonymization.schemas.uniform_temporal_op_config import UniformTemporalNoiseConfig
from pamola_core.anonymization.schemas.uniform_temporal_op_config_exclude import RECORD_EXCLUDE_FIELDS as UNIFORM_TEMPORAL_EXCLUDE_FIELDS

from pamola_core.metrics.schemas.fidelity_ops_config import FidelityConfig
from pamola_core.metrics.schemas.fidelity_ops_config_exclude import FIDELITY_EXCLUDE_FIELDS
from pamola_core.metrics.schemas.privacy_ops_config import PrivacyMetricConfig
from pamola_core.metrics.schemas.privacy_ops_config_exclude import PRIVACY_EXCLUDE_FIELDS
from pamola_core.metrics.schemas.utility_ops_config import UtilityMetricConfig
from pamola_core.metrics.schemas.utility_ops_config_exclude import UTILITY_EXCLUDE_FIELDS

from pamola_core.profiling.schemas.anonymity_config import KAnonymityProfilerOperationConfig
from pamola_core.profiling.schemas.anonymity_config_exclude import ANONYMITY_EXCLUDE_FIELDS
from pamola_core.profiling.schemas.attribute_config import DataAttributeProfilerOperationConfig
from pamola_core.profiling.schemas.attribute_config_exclude import ATTRIBUTE_EXCLUDE_FIELDS
from pamola_core.profiling.schemas.categorical_config import CategoricalOperationConfig
from pamola_core.profiling.schemas.categorical_config_exclude import CATEGORICAL_EXCLUDE_FIELDS
from pamola_core.profiling.schemas.correlation_config import CorrelationOperationConfig, CorrelationMatrixOperationConfig
from pamola_core.profiling.schemas.correlation_config_exclude import CORRELATION_EXCLUDE_FIELDS, CORRELATION_MATRIX_EXCLUDE_FIELDS
from pamola_core.profiling.schemas.currency_config import CurrencyOperationConfig
from pamola_core.profiling.schemas.currency_config_exclude import CURRENCY_EXCLUDE_FIELDS
from pamola_core.profiling.schemas.date_config import DateOperationConfig
from pamola_core.profiling.schemas.date_config_exclude import DATE_EXCLUDE_FIELDS
from pamola_core.profiling.schemas.email_config import EmailOperationConfig
from pamola_core.profiling.schemas.email_config_exclude import EMAIL_EXCLUDE_FIELDS
from pamola_core.profiling.schemas.group_config import GroupAnalyzerOperationConfig
from pamola_core.profiling.schemas.group_config_exclude import GROUP_EXCLUDE_FIELDS
from pamola_core.profiling.schemas.identity_config import IdentityAnalysisOperationConfig
from pamola_core.profiling.schemas.identity_config_exclude import IDENTITY_EXCLUDE_FIELDS
from pamola_core.profiling.schemas.mvf_config import MVFAnalysisOperationConfig
from pamola_core.profiling.schemas.mvf_config_exclude import MVF_EXCLUDE_FIELDS
from pamola_core.profiling.schemas.numeric_config import NumericOperationConfig
from pamola_core.profiling.schemas.numeric_config_exclude import NUMERIC_EXCLUDE_FIELDS
from pamola_core.profiling.schemas.phone_config import PhoneOperationConfig
from pamola_core.profiling.schemas.phone_config_exclude import PHONE_EXCLUDE_FIELDS
from pamola_core.profiling.schemas.text_config import TextSemanticCategorizerOperationConfig
from pamola_core.profiling.schemas.text_config_exclude import TEXT_EXCLUDE_FIELDS

from pamola_core.utils.schema_utils import generate_schema_json

ALL_OP_CONFIGS = [
    (FakeEmailOperationConfig, EMAIL_FAKE_EXCLUDE_FIELDS),
    (FakeNameOperationConfig, NAME_FAKE_EXCLUDE_FIELDS),
    (FakeOrganizationOperationConfig, ORGANIZATION_FAKE_EXCLUDE_FIELDS),
    (FakePhoneOperationConfig, PHONE_FAKE_EXCLUDE_FIELDS),
    (AddOrModifyFieldsOperationConfig, ADD_MODIFY_FIELDS_EXCLUDE_FIELDS),
    (AggregateRecordsOperationConfig, AGGREGATE_RECORDS_EXCLUDE_FIELDS),
    (CleanInvalidValuesOperationConfig, CLEAN_INVALID_VALUES_EXCLUDE_FIELDS),
    (ImputeMissingValuesConfig, IMPUTE_MISSING_VALUES_EXCLUDE_FIELDS),
    (MergeDatasetsOperationConfig, MERGE_DATASETS_EXCLUDE_FIELDS),
    (RemoveFieldsOperationConfig, REMOVE_FIELDS_EXCLUDE_FIELDS),
    (SplitByIDValuesOperationConfig, SPLIT_BY_ID_VALUES_EXCLUDE_FIELDS),
    (SplitFieldsOperationConfig, SPLIT_FIELDS_EXCLUDE_FIELDS),
    (CategoricalGeneralizationConfig, CATEGORICAL_GENERALIZATION_EXCLUDE_FIELDS),
    (DateTimeGeneralizationConfig, DATETIME_GENERALIZATION_EXCLUDE_FIELDS),
    (FullMaskingConfig, FULL_MASKING_EXCLUDE_FIELDS),
    (NumericGeneralizationConfig, NUMERIC_GENERALIZATION_EXCLUDE_FIELDS),
    (PartialMaskingConfig, PARTIAL_MASKING_EXCLUDE_FIELDS),
    (AttributeSuppressionConfig, ATTRIBUTE_EXCLUDE_FIELDS),
    (CellSuppressionConfig, CELL_EXCLUDE_FIELDS),
    (RecordSuppressionConfig, RECORD_EXCLUDE_FIELDS),
    (UniformNumericNoiseConfig, UNIFORM_NUMERIC_EXCLUDE_FIELDS),
    (UniformTemporalNoiseConfig, UNIFORM_TEMPORAL_EXCLUDE_FIELDS),
    (FidelityConfig, FIDELITY_EXCLUDE_FIELDS),
    (PrivacyMetricConfig, PRIVACY_EXCLUDE_FIELDS),
    (UtilityMetricConfig, UTILITY_EXCLUDE_FIELDS),
    (KAnonymityProfilerOperationConfig, ANONYMITY_EXCLUDE_FIELDS),
    (DataAttributeProfilerOperationConfig, ATTRIBUTE_EXCLUDE_FIELDS),
    (CategoricalOperationConfig, CATEGORICAL_EXCLUDE_FIELDS),
    (CorrelationOperationConfig, CORRELATION_EXCLUDE_FIELDS),
    (CorrelationMatrixOperationConfig, CORRELATION_MATRIX_EXCLUDE_FIELDS),
    (CurrencyOperationConfig, CURRENCY_EXCLUDE_FIELDS),
    (DateOperationConfig, DATE_EXCLUDE_FIELDS),
    (EmailOperationConfig, EMAIL_EXCLUDE_FIELDS),
    (GroupAnalyzerOperationConfig, GROUP_EXCLUDE_FIELDS),
    (IdentityAnalysisOperationConfig, IDENTITY_EXCLUDE_FIELDS),
    (MVFAnalysisOperationConfig, MVF_EXCLUDE_FIELDS),
    (NumericOperationConfig, NUMERIC_EXCLUDE_FIELDS),
    (PhoneOperationConfig, PHONE_EXCLUDE_FIELDS),
    (TextSemanticCategorizerOperationConfig, TEXT_EXCLUDE_FIELDS),
]

def generate_all_op_schemas(task_dir: Path) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    for config_cls, exclude_fields in ALL_OP_CONFIGS:
        generate_schema_json(config_cls, task_dir, exclude_fields)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path.cwd()
    generate_all_op_schemas(output_dir)
