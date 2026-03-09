from pathlib import Path
from typing import Dict, Any, List, Optional
from pamola_core.common.constants import Constants
from pamola_core.utils.tasks.base_task import BaseTask
from pamola_core.utils.ops.op_result import OperationStatus

from pamola_core.utils.ops.op_data_source import DataSource

from pamola_core.utils.tasks import TaskInitializationError


from pamola_core.fake_data import (
    FakeEmailOperation,
    FakeNameOperation,
    FakeOrganizationOperation,
    FakePhoneOperation,
)
from pamola_core.profiling import (
    KAnonymityProfilerOperation,
    DataAttributeProfilerOperation,
    CategoricalOperation,
    CorrelationOperation,
    CorrelationMatrixOperation,
    DateOperation,
    EmailOperation,
    GroupAnalyzerOperation,
    IdentityAnalysisOperation,
    MVFOperation,
    NumericOperation,
    PhoneOperation,
    TextSemanticCategorizerOperation,
    CurrencyOperation,
)
from pamola_core.transformations import (
    MergeDatasetsOperation,
    ImputeMissingValuesOperation,
    AddOrModifyFieldsOperation,
    RemoveFieldsOperation,
    SplitByIDValuesOperation,
    SplitFieldsOperation,
    AggregateRecordsOperation,
    CleanInvalidValuesOperation,
)
from pamola_core.anonymization import (
    AttributeSuppressionOperation,
    RecordSuppressionOperation,
    CellSuppressionOperation,
    NumericGeneralizationOperation,
    DateTimeGeneralizationOperation,
    CategoricalGeneralizationOperation,
    UniformTemporalNoiseOperation,
    UniformNumericNoiseOperation,
    FullMaskingOperation,
    PartialMaskingOperation,
)

from pamola_core.metrics import (
    FidelityOperation,
    UtilityMetricOperation,
    PrivacyMetricOperation,
)


class TaskRunner(BaseTask):
    """
    Service for managing and executing task operations.
    Extends BaseTask to provide operation sequence management.

    Attributes:
        task_id (str): Unique identifier for the task.
        task_type (str): Type of task (e.g., 'anonymization', 'profiling').
        description (str): Task description.
        input_datasets (Dict[str, str]): Dictionary of input dataset names and paths.
        auxiliary_datasets (Dict[str, str]): Dictionary of auxiliary dataset names and paths.
        operation_configs (List[Dict[str, Any]]): List of operation configurations.
    """

    def __init__(
        self,
        task_id: str,
        task_type: str,
        description: str,
        input_datasets: Optional[Dict[str, str]] = None,
        data_types: Optional[Dict[str, Any]] = None,
        auxiliary_datasets: Optional[Dict[str, str]] = None,
        operation_configs: Optional[List[Dict[str, Any]]] = None,
        additional_options: Optional[Dict[str, Any]] = None,
        use_encryption: Optional[bool] = False,
        encryption_keys: Optional[Dict[str, str]] = None,
        task_dir: Optional[str] = None,
    ):
        """
        Initialize the task service.

        Args:
            task_id (str): Unique identifier for the task.
            task_type (str): Type of task (e.g., 'anonymization', 'profiling').
            description (str): Task description.
            input_datasets (Dict[str, str], optional): Dictionary of input dataset names and paths.
            auxiliary_datasets (Dict[str, str], optional): Dictionary of auxiliary dataset names and paths.
            operation_configs (List[Dict[str, Any]], optional): List of operation configurations.
            task_dir (str, optional): Override the task directory path instead of using the config.
        """
        super().__init__(
            task_id=task_id,
            task_type=task_type,
            description=description,
            input_datasets=input_datasets,
            auxiliary_datasets=auxiliary_datasets,
            use_encryption=use_encryption,
            encryption_keys=encryption_keys,
            task_dir=task_dir,
        )
        self.final_dataset_name = None
        self.final_dataset_path = None
        self.operation_configs = operation_configs or []
        self.lst_result = []
        self.lst_final_output = []

        self.additional_options = additional_options
        self.use_encryption = use_encryption
        self.encryption_keys = encryption_keys
        self.data_types = data_types

        self.operations_sequence: List[str] = [
            FakeEmailOperation.__name__,
            FakeNameOperation.__name__,
            FakeOrganizationOperation.__name__,
            FakePhoneOperation.__name__,
            AggregateRecordsOperation.__name__,
            ImputeMissingValuesOperation.__name__,
            AddOrModifyFieldsOperation.__name__,
            RemoveFieldsOperation.__name__,
            CleanInvalidValuesOperation.__name__,
            NumericGeneralizationOperation.__name__,
            DateTimeGeneralizationOperation.__name__,
            AttributeSuppressionOperation.__name__,
            CategoricalGeneralizationOperation.__name__,
            UniformTemporalNoiseOperation.__name__,
            UniformNumericNoiseOperation.__name__,
            RecordSuppressionOperation.__name__,
            CellSuppressionOperation.__name__,
            FullMaskingOperation.__name__,
            PartialMaskingOperation.__name__,
        ]

        self.operations_always_output_finals: List[str] = [
            MergeDatasetsOperation.__name__,
            SplitByIDValuesOperation.__name__,
            SplitFieldsOperation.__name__,
        ]

        self.function_maps: List[str] = [
            NumericGeneralizationOperation.__name__,
            DateTimeGeneralizationOperation.__name__,
            AttributeSuppressionOperation.__name__,
            CategoricalGeneralizationOperation.__name__,
            UniformTemporalNoiseOperation.__name__,
            UniformNumericNoiseOperation.__name__,
            RecordSuppressionOperation.__name__,
            CellSuppressionOperation.__name__,
            FullMaskingOperation.__name__,
            PartialMaskingOperation.__name__,
            FakeEmailOperation.__name__,
            FakeNameOperation.__name__,
            FakeOrganizationOperation.__name__,
            FakePhoneOperation.__name__,
            KAnonymityProfilerOperation.__name__,
            DataAttributeProfilerOperation.__name__,
            CategoricalOperation.__name__,
            CorrelationOperation.__name__,
            CorrelationMatrixOperation.__name__,
            DateOperation.__name__,
            EmailOperation.__name__,
            GroupAnalyzerOperation.__name__,
            IdentityAnalysisOperation.__name__,
            MVFOperation.__name__,
            NumericOperation.__name__,
            PhoneOperation.__name__,
            TextSemanticCategorizerOperation.__name__,
            CurrencyOperation.__name__,
            AggregateRecordsOperation.__name__,
            ImputeMissingValuesOperation.__name__,
            AddOrModifyFieldsOperation.__name__,
            RemoveFieldsOperation.__name__,
            MergeDatasetsOperation.__name__,
            SplitByIDValuesOperation.__name__,
            SplitFieldsOperation.__name__,
            CleanInvalidValuesOperation.__name__,
            FidelityOperation.__name__,
            UtilityMetricOperation.__name__,
            PrivacyMetricOperation.__name__,
        ]

        self.operations_not_include_field_name: List[str] = [
            KAnonymityProfilerOperation.__name__,
            DataAttributeProfilerOperation.__name__,
            CorrelationOperation.__name__,
            CorrelationMatrixOperation.__name__,
            IdentityAnalysisOperation.__name__,
            AggregateRecordsOperation.__name__,
            ImputeMissingValuesOperation.__name__,
            AddOrModifyFieldsOperation.__name__,
            RemoveFieldsOperation.__name__,
            MergeDatasetsOperation.__name__,
            SplitByIDValuesOperation.__name__,
            SplitFieldsOperation.__name__,
            CleanInvalidValuesOperation.__name__,
            FidelityOperation.__name__,
            UtilityMetricOperation.__name__,
            PrivacyMetricOperation.__name__,
        ]

    def add_to_operations_sequence(self, class_name: str) -> None:
        if class_name not in self.operations_sequence:
            self.operations_sequence.append(class_name)

    def has_function_map(self, class_name: str) -> bool:
        return class_name in self.function_maps

    def _add_operation_result(
        self,
        class_name: str,
        operation_type: str,
        operation: dict,
        scope: dict,
        parameters: dict,
        field: str = None,
    ) -> None:
        """
        Add operation result to lst_result.

        Args:
            class_name (str): Operation class name
            operation_type (str): Type of operation
            operation (dict): Operation configuration
            scope (dict): Scope configuration
            field (str, optional): Single target field
        """

        self.lst_result.append(
            {
                "parameters": parameters,
                "class_name": class_name,
                "dataset_name": operation.get("dataset_name", "main"),
                "operation_type": operation_type,
                "operation_result": "",
                "task_operation_id": operation.get("task_operation_id"),
                "task_operation_order_index": operation.get(
                    "task_operation_order_index"
                ),
                "field": f"{field}" if field is not None else "",
                "applied_scope": (
                    f"{scope.get('type')}: {field}" if field is not None else ""
                ),
            }
        )

    def configure_operations(self) -> bool:
        """
        Configure operations from the provided configurations.

        Returns:
            bool: True if operations configured successfully, False otherwise.
        """
        try:
            for operation in self.operation_configs:
                operation_type = operation.get("operation", "").lower()
                op_class = operation.get("class_name", "")
                parameters = operation.get("parameters", {})
                scope = operation.get("scope", {})
                target_fields = scope.get("target", [])

                # profiling_anonymity has output if mode in [AnalysisMode.ENRICH, AnalysisMode.BOTH]
                if op_class == KAnonymityProfilerOperation.__name__:
                    analysis_mode = parameters.get(Constants.ANALYSIS_MODE, None)

                    if analysis_mode is not None and str(analysis_mode).upper() in [
                        Constants.ENRICH,
                        Constants.BOTH,
                    ]:
                        self.add_to_operations_sequence(KAnonymityProfilerOperation.__name__)

                if not op_class or not self.has_function_map(op_class):
                    error_msg = f"Not found {op_class} operation in system to configure operation"
                    self.logger.error(error_msg)
                    raise Exception(error_msg)

                if (
                    target_fields
                    and op_class not in self.operations_not_include_field_name
                ):
                    for field in target_fields:
                        parameters["field_name"] = field
                        if not self.add_operation(op_class, **parameters):
                            error_msg = f"Failed to configure operation: {op_class} for field: {field}"
                            self.logger.error(error_msg)
                            raise Exception(error_msg)
                        self._add_operation_result(
                            op_class,
                            operation_type,
                            operation,
                            scope,
                            parameters,
                            field,
                        )
                else:
                    if not self.add_operation(op_class, **parameters):
                        error_msg = f"Failed to configure operation: {op_class}"
                        self.logger.error(error_msg)
                        raise Exception(error_msg)
                    self._add_operation_result(
                        op_class, operation_type, operation, parameters, scope
                    )

            return True

        except Exception as e:
            self.logger.exception(f"Error configuring operations: {str(e)}")
            raise

    def _run_operations(self, start_idx: int = 0) -> bool:
        """
        Run operations starting from the specified index.

        This method handles the actual execution of operations, including progress tracking,
        checkpointing, and error handling.

        Args:
            start_idx: Index of the first operation to execute

        Returns:
            True if all operations executed successfully, False otherwise
        """
        orient = self.additional_options.get(Constants.ORIENT, Constants.COLUMNS_ORIENT)
        encoding = self.additional_options.get(Constants.ENCODING, Constants.UTF_8)
        sep = self.additional_options.get(Constants.SEP, Constants.DELIMITER_COMMA)
        delimiter = self.additional_options.get(
            Constants.DELIMITER, Constants.DELIMITER_COMMA
        )
        quotechar = self.additional_options.get(
            Constants.QUOTE_CHAR, Constants.DOUBLE_QUOTE
        )

        file_paths = getattr(self.data_source, "file_paths", None)
        transformed_dataset_name = (
            isinstance(file_paths, dict)
            and Constants.TRANSFORMED_DATASET_NAME in file_paths
        )

        # Execute operations using the operation executor
        for i in range(start_idx, len(self.operations)):
            operation = self.operations[i]
            operation_name = (
                operation.name if hasattr(operation, "name") else f"Operation {i + 1}"
            )

            # Prepare operation parameters
            operation_params = self._prepare_operation_parameters(
                self.lst_result[i]["parameters"], operation
            )

            op_class_name = self.lst_result[i]["class_name"]

            # Set dataset name of operations
            if op_class_name in self.operations_sequence:
                operation_params["dataset_name"] = (
                    self.final_dataset_name
                    if self.final_dataset_name
                    else self.lst_result[i]["dataset_name"]
                )
            else:
                operation_params["dataset_name"] = self.lst_result[i]["dataset_name"]

            if transformed_dataset_name:
                operation_params[Constants.TRANSFORMED_DATASET_NAME] = (
                    Constants.TRANSFORMED_DATASET_NAME
                )

            operation_params[Constants.ORIENT] = orient
            operation_params[Constants.ENCODING] = encoding
            operation_params[Constants.SEP] = sep
            operation_params[Constants.DELIMITER] = delimiter
            operation_params[Constants.QUOTE_CHAR] = quotechar

            # Use operation executor to run the operation
            try:
                # Execute operation
                result = self.operation_executor.execute_with_retry(
                    operation=operation, params=operation_params
                )

                # Store the result
                self.results[operation_name] = result

                # Register artifacts from the operation
                if hasattr(result, "artifacts") and result.artifacts:
                    self.artifacts.extend(result.artifacts)

                    if op_class_name in self.operations_always_output_finals:
                        for artifact in result.artifacts:
                            if artifact.category == "output":
                                dataset_path = artifact.path
                                dataset_name = Path(artifact.path).name
                                self.lst_final_output.extend([dataset_path])
                                self.data_source.add_file_path(
                                    dataset_name, dataset_path
                                )
                                self.data_source.add_encryption_mode(
                                    dataset_name, dataset_path
                                )
                                if operation.use_encryption:
                                    self.data_source.add_encryption_key(
                                        dataset_name, operation.encryption_key
                                    )
                                dataset_data_type = result.metrics.get(
                                    f"data_types_{Path(artifact.path).stem}"
                                )
                                if dataset_data_type:
                                    self.data_source.add_data_type(
                                        dataset_name, dataset_data_type
                                    )

                    # Store the list result
                    if op_class_name in self.operations_sequence:
                        output_artifact = next(
                            (
                                artifact
                                for artifact in result.artifacts
                                if artifact.category == "output"
                            ),
                            None,
                        )

                        if output_artifact:
                            if self.final_dataset_path in self.lst_final_output:
                                self.lst_final_output.remove(self.final_dataset_path)
                            self.final_dataset_path = output_artifact.path
                            self.final_dataset_name = Path(output_artifact.path).name
                            self.data_source.add_file_path(
                                self.final_dataset_name, self.final_dataset_path
                            )
                            self.data_source.add_encryption_mode(
                                self.final_dataset_name, self.final_dataset_path
                            )
                            dataset_data_type = result.metrics.get(
                                f"data_types_{Path(output_artifact.path).stem}"
                            )
                            if dataset_data_type:
                                self.data_source.add_data_type(
                                    self.final_dataset_name, dataset_data_type
                                )
                            for old_final_dataset in list(
                                self.data_source.dataframes.keys()
                            ):
                                self.data_source.release_dataframe(old_final_dataset)

                            if operation.use_encryption:
                                self.data_source.add_encryption_key(
                                    self.final_dataset_name, operation.encryption_key
                                )

                            self.lst_final_output.extend([self.final_dataset_path])
                            # Correct csv_dialect again
                            encoding = Constants.UTF_8
                            delimiter = Constants.DELIMITER_COMMA
                            quotechar = Constants.DOUBLE_QUOTE

                    self.lst_result[i]["operation_result"] = result

                # Collect metrics from the operation
                if hasattr(result, "metrics") and result.metrics:
                    self.metrics[operation_name] = result.metrics

                # Check result status
                if result.status == OperationStatus.ERROR:
                    self.logger.error(
                        f"Operation {operation_name} failed: {result.error_message}"
                    )

                    # Check if we should continue on error
                    if not self.config.continue_on_error:
                        self.logger.error("Aborting task due to operation failure")
                        self.error_info = {
                            "type": "operation_error",
                            "operation": operation_name,
                            "message": result.error_message,
                        }
                        self.status = "operation_error"
                        return False

                # Save checkpoint after each operation
                if self.context_manager:
                    try:
                        self.context_manager.create_automatic_checkpoint(
                            operation_index=i, metrics=self.metrics
                        )
                    except Exception as e:
                        self.logger.warning(f"Could not create checkpoint: {e}")

            except KeyboardInterrupt:
                # Allow keyboard interrupts to propagate up the call stack
                self.logger.info("Keyboard interrupt detected, stopping task execution")
                self.error_info = {
                    "type": "keyboard_interrupt",
                    "operation": operation_name,
                    "message": "Task execution interrupted by user",
                }
                self.status = "interrupted"
                raise  # Re-raise KeyboardInterrupt to ensure it's properly handled

            except Exception as e:
                # Check for KeyboardInterrupt before general exception handling
                if isinstance(e, KeyboardInterrupt):
                    self.logger.info(
                        "Keyboard interrupt detected, stopping task execution"
                    )
                    self.error_info = {
                        "type": "keyboard_interrupt",
                        "operation": operation_name,
                        "message": "Task execution interrupted by user",
                    }
                    self.status = "interrupted"
                    raise  # Re-raise KeyboardInterrupt

                self.logger.exception(
                    f"Error executing operation {operation_name}: {str(e)}"
                )

                # Check if we should continue on error - centralized error handling
                if not self.config.continue_on_error:
                    self.logger.error("Aborting task due to operation failure")
                    self.error_info = {
                        "type": "exception",
                        "operation": operation_name,
                        "message": str(e),
                    }
                    self.status = "exception"
                    return False

        # If we get here, all operations completed
        self.status = "success"
        return True

    def _initialize_data_source(self) -> None:
        """
        Initialize the data source with input and auxiliary datasets.

        This method prepares the data source that will be used by operations by:
        1. Creating a new DataSource instance
        2. Adding all input datasets
        3. Adding all auxiliary datasets
        """
        self.data_source = DataSource()

        # Check if we have any input datasets to process
        if not self.input_datasets:
            self.logger.warning(
                "No input datasets defined for task. DataSource initialization may be incomplete."
            )
            # Don't create a progress bar for empty datasets to avoid hanging
            return

        # Process input datasets with progress tracking
        with self.progress_manager.create_operation_context(
            name="initialize_input_data",
            total=len(self.input_datasets),
            description="Initializing input datasets",
            unit="datasets",
        ) as progress:
            for name, path in self.input_datasets.items():
                try:
                    # Use directory manager to resolve and validate path
                    path_obj = self.directory_manager.normalize_and_validate_path(path)

                    # Add to data source
                    self.data_source.add_file_path(name, path_obj)
                    self.data_source.add_encryption_mode(name, path_obj)
                    self.logger.debug(f"Added input dataset: {name} from {path_obj}")

                    # Add encryption key
                    encryption_key = self.encryption_keys.get(name)
                    if encryption_key:
                        self.data_source.add_encryption_key(name, encryption_key)

                    # Add data type
                    data_type = self.data_types.get(name)
                    if data_type:
                        self.data_source.add_data_type(name, data_type)

                    # Update progress
                    progress.update(1)
                except Exception as e:
                    self.logger.error(f"Error adding input dataset '{name}': {str(e)}")
                    progress.update(1, {"status": "error"})
                    raise TaskInitializationError(
                        f"Failed to add input dataset '{name}': {str(e)}"
                    )

        # Check if we have any auxiliary datasets to process
        if not self.auxiliary_datasets:
            return  # Skip creating empty progress bar for auxiliary datasets

        # Process auxiliary datasets
        with self.progress_manager.create_operation_context(
            name="initialize_auxiliary_data",
            total=len(self.auxiliary_datasets),
            description="Initializing auxiliary datasets",
            unit="datasets",
        ) as progress:
            for name, path in self.auxiliary_datasets.items():
                try:
                    # Use directory manager to resolve and validate path
                    path_obj = self.directory_manager.normalize_and_validate_path(path)

                    # Add to data source
                    self.data_source.add_file_path(name, path_obj)
                    self.data_source.add_encryption_mode(name, path_obj)
                    self.logger.debug(
                        f"Added auxiliary dataset: {name} from {path_obj}"
                    )

                    # Add encryption key
                    encryption_key = self.encryption_keys.get(name)
                    if encryption_key:
                        self.data_source.add_encryption_key(name, encryption_key)

                    # Add data type
                    data_type = self.data_types.get(name)
                    if data_type:
                        self.data_source.add_data_type(name, data_type)

                    self.logger.debug(
                        f"Added auxiliary dataset: {name} from {path_obj}"
                    )

                    # Update progress
                    progress.update(1)
                except Exception as e:
                    self.logger.error(
                        f"Error adding auxiliary dataset '{name}': {str(e)}"
                    )
                    progress.update(1, {"status": "error"})
                    raise TaskInitializationError(
                        f"Failed to add auxiliary dataset '{name}': {str(e)}"
                    )

        # Check encryption status if enabled
        if self.use_encryption:
            self.encryption_manager.check_dataset_encryption(self.data_source)
