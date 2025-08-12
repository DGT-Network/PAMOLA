"""
Tests for the pamola_core.fake_data.commons.base module.

This module contains unit tests for the abstract base classes and concrete
implementations in the base.py module.
"""

import unittest
from unittest.mock import Mock

import pandas as pd

# Import the module to test
from pamola_core.fake_data.commons.base import (
    ResourceType,
    NullStrategy,
    FakeDataError,
    ValidationError,
    ResourceError,
    MappingError,
    BaseGenerator,
    BaseMapper,
    BaseOperation,
    FieldOperation,
    MappingStore
)


class TestEnums(unittest.TestCase):
    """Tests for the enums in the base module."""

    def test_resource_type_values(self):
        """Test that ResourceType enum has the expected values."""
        self.assertEqual(ResourceType.MEMORY.value, "memory_mb")
        self.assertEqual(ResourceType.TIME.value, "time_seconds")
        self.assertEqual(ResourceType.CPU.value, "cpu_percent")
        self.assertEqual(ResourceType.DISK.value, "disk_mb")

    def test_null_strategy_values(self):
        """Test that NullStrategy enum has the expected values."""
        self.assertEqual(NullStrategy.PRESERVE.value, "preserve")
        self.assertEqual(NullStrategy.REPLACE.value, "replace")
        self.assertEqual(NullStrategy.EXCLUDE.value, "exclude")
        self.assertEqual(NullStrategy.ERROR.value, "error")


class TestExceptions(unittest.TestCase):
    """Tests for the exception classes in the base module."""

    def test_exception_hierarchy(self):
        """Test that exceptions inherit from the correct base classes."""
        self.assertTrue(issubclass(FakeDataError, Exception))
        self.assertTrue(issubclass(ValidationError, FakeDataError))
        self.assertTrue(issubclass(ResourceError, FakeDataError))
        self.assertTrue(issubclass(MappingError, FakeDataError))

    def test_exception_instantiation(self):
        """Test that exceptions can be instantiated with messages."""
        error_msg = "Test error message"
        error = FakeDataError(error_msg)
        self.assertEqual(str(error), error_msg)

        validation_error = ValidationError("Validation failed")
        self.assertEqual(str(validation_error), "Validation failed")

        resource_error = ResourceError("Out of memory")
        self.assertEqual(str(resource_error), "Out of memory")

        mapping_error = MappingError("Mapping conflict")
        self.assertEqual(str(mapping_error), "Mapping conflict")


# Concrete implementations for testing abstract classes
class SimpleGenerator(BaseGenerator):
    """Simple implementation of BaseGenerator for testing."""

    def generate(self, count, **params):
        """Generate count random strings."""
        return [f"fake_value_{i}" for i in range(count)]

    def generate_like(self, original_value, **params):
        """Generate a fake value similar to the original."""
        return f"fake_{original_value}"

    def analyze_value(self, value):
        """Analyze the value and return its properties."""
        return {
            "type": "string",
            "length": len(str(value)),
            "has_digits": any(c.isdigit() for c in str(value))
        }


class SimpleMapper(BaseMapper):
    """Simple implementation of BaseMapper for testing."""

    def __init__(self):
        self.mapping = {}
        self.reverse_mapping = {}
        self.transitivity = {}

    def map(self, original_value, **params):
        """Map original value to synthetic one."""
        if original_value in self.mapping:
            return self.mapping[original_value]
        if params.get("force_new", False):
            synthetic = f"generated_{original_value}"
            self.add_mapping(original_value, synthetic)
            return synthetic
        return None

    def restore(self, synthetic_value):
        """Restore original value from synthetic one."""
        return self.reverse_mapping.get(synthetic_value)

    def add_mapping(self, original, synthetic, is_transitive=False):
        """Add a new mapping."""
        conflicts = self.check_conflicts(original, synthetic)
        if conflicts["has_conflicts"]:
            raise MappingError(f"Mapping conflict: {conflicts['conflict_type']}")

        self.mapping[original] = synthetic
        self.reverse_mapping[synthetic] = original
        self.transitivity[original] = is_transitive

    def check_conflicts(self, original, synthetic):
        """Check for conflicts in the mapping."""
        has_conflicts = False
        conflict_type = None
        affected_values = []

        if original in self.mapping and self.mapping[original] != synthetic:
            has_conflicts = True
            conflict_type = "original_already_mapped"
            affected_values.append(self.mapping[original])

        if synthetic in self.reverse_mapping and self.reverse_mapping[synthetic] != original:
            has_conflicts = True
            conflict_type = "synthetic_already_used"
            affected_values.append(self.reverse_mapping[synthetic])

        return {
            "has_conflicts": has_conflicts,
            "conflict_type": conflict_type,
            "affected_values": affected_values
        }


class SimpleOperation(BaseOperation):
    """Simple implementation of BaseOperation for testing."""

    name = "simple_operation"
    description = "A simple operation for testing"

    def execute(self, data_source, task_dir, reporter, **kwargs):
        """Execute the operation."""
        # Simple implementation for testing
        return {"status": "success"}


class SimpleFieldOperation(FieldOperation):
    """Simple implementation of FieldOperation for testing."""

    def __init__(self, field_name, **kwargs):
        super().__init__(field_name, **kwargs)

    def process_batch(self, batch):
        """Process a batch of data."""
        # Simple implementation: upper case the field values
        result = batch.copy()
        mask = result[self.field_name].notna()

        if self.mode == "REPLACE":
            result.loc[mask, self.field_name] = result.loc[mask, self.field_name].str.upper()
        else:  # ENRICH mode
            result[self.output_field_name] = result[self.field_name].str.upper()

        return result

    def execute(self, data_source, task_dir, reporter, **kwargs):
        """Minimal implementation for BaseOperation.execute()"""
        return {"status": "success"}


class TestBaseGenerator(unittest.TestCase):
    """Tests for the BaseGenerator abstract base class."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = SimpleGenerator()

    def test_generate_method(self):
        """Test the generate method."""
        # Generate 5 values
        values = self.generator.generate(5)
        self.assertEqual(len(values), 5)
        for i, value in enumerate(values):
            self.assertEqual(value, f"fake_value_{i}")

    def test_generate_like_method(self):
        """Test the generate_like method."""
        original = "test_value"
        synthetic = self.generator.generate_like(original)
        self.assertEqual(synthetic, "fake_test_value")

    def test_analyze_value_method(self):
        """Test the analyze_value method."""
        # Test with a simple string
        properties = self.generator.analyze_value("hello")
        self.assertEqual(properties["type"], "string")
        self.assertEqual(properties["length"], 5)
        self.assertFalse(properties["has_digits"])

        # Test with a string containing digits
        properties = self.generator.analyze_value("hello123")
        self.assertTrue(properties["has_digits"])
        self.assertEqual(properties["length"], 8)

    def test_estimate_resources(self):
        """Test the estimate_resources method."""
        # Test the default implementation
        resources = self.generator.estimate_resources(1000)
        self.assertIn(ResourceType.MEMORY.value, resources)
        self.assertIn(ResourceType.TIME.value, resources)
        self.assertEqual(resources[ResourceType.MEMORY.value], 1000 * 0.01)  # Default is 0.01 MB per record
        self.assertEqual(resources[ResourceType.TIME.value], 1000 * 0.001)  # Default is 0.001 seconds per record


class TestBaseMapper(unittest.TestCase):
    """Tests for the BaseMapper abstract base class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mapper = SimpleMapper()

    def test_map_method(self):
        """Test the map method."""
        # No mapping exists initially
        self.assertIsNone(self.mapper.map("original"))

        # Force creation of new mapping
        synthetic = self.mapper.map("original", force_new=True)
        self.assertEqual(synthetic, "generated_original")

        # Now mapping exists and should be retrieved
        self.assertEqual(self.mapper.map("original"), "generated_original")

    def test_restore_method(self):
        """Test the restore method."""
        # Add a mapping
        self.mapper.add_mapping("original", "synthetic")

        # Test restoration
        self.assertEqual(self.mapper.restore("synthetic"), "original")
        self.assertIsNone(self.mapper.restore("nonexistent"))

    def test_add_mapping_method(self):
        """Test the add_mapping method."""
        # Add a mapping
        self.mapper.add_mapping("original1", "synthetic1")
        self.assertEqual(self.mapper.map("original1"), "synthetic1")
        self.assertEqual(self.mapper.restore("synthetic1"), "original1")

        # Add a transitive mapping
        self.mapper.add_mapping("original2", "synthetic2", is_transitive=True)
        self.assertTrue(self.mapper.transitivity["original2"])

    def test_mapping_conflicts(self):
        """Test conflict detection and handling."""
        # Add a mapping
        self.mapper.add_mapping("original1", "synthetic1")

        # Test conflict: original already mapped
        with self.assertRaises(MappingError):
            self.mapper.add_mapping("original1", "synthetic2")

        # Test conflict: synthetic already used
        with self.assertRaises(MappingError):
            self.mapper.add_mapping("original2", "synthetic1")

        # No conflict case
        self.mapper.add_mapping("original3", "synthetic3")
        self.assertEqual(self.mapper.map("original3"), "synthetic3")

    def test_check_conflicts_method(self):
        """Test the check_conflicts method."""
        # Add a mapping
        self.mapper.add_mapping("original1", "synthetic1")

        # Check conflict: original already mapped to different value
        conflicts = self.mapper.check_conflicts("original1", "synthetic2")
        self.assertTrue(conflicts["has_conflicts"])
        self.assertEqual(conflicts["conflict_type"], "original_already_mapped")

        # Check conflict: synthetic already mapped from different original
        conflicts = self.mapper.check_conflicts("original2", "synthetic1")
        self.assertTrue(conflicts["has_conflicts"])
        self.assertEqual(conflicts["conflict_type"], "synthetic_already_used")

        # No conflict case
        conflicts = self.mapper.check_conflicts("original2", "synthetic2")
        self.assertFalse(conflicts["has_conflicts"])
        self.assertIsNone(conflicts["conflict_type"])

    def test_conflict_resolution_strategies(self):
        """Test the conflict resolution strategies."""
        strategies = BaseMapper.get_conflicts_resolution_strategies()
        self.assertIn("append_suffix", strategies)
        self.assertIn("use_original", strategies)
        self.assertIn("use_latest", strategies)

        # Test append_suffix strategy
        self.assertEqual(strategies["append_suffix"]("value"), "value_1")

        # Test use_original strategy
        self.assertEqual(strategies["use_original"]("original", "new"), "original")

        # Test use_latest strategy
        self.assertEqual(strategies["use_latest"]("original", "new"), "new")


class TestBaseOperation(unittest.TestCase):
    """Tests for the BaseOperation abstract base class."""

    def setUp(self):
        """Set up test fixtures."""
        self.operation = SimpleOperation()

    def test_operation_attributes(self):
        """Test the operation attributes."""
        self.assertEqual(self.operation.name, "simple_operation")
        self.assertEqual(self.operation.description, "A simple operation for testing")

    def test_execute_method(self):
        """Test the execute method."""
        # Simple test with mock parameters
        data_source = Mock()
        task_dir = Mock()
        reporter = Mock()

        result = self.operation.execute(data_source, task_dir, reporter)
        self.assertEqual(result["status"], "success")


class TestFieldOperation(unittest.TestCase):
    """Tests for the FieldOperation abstract base class."""

    def setUp(self):
        """Set up test fixtures."""
        self.operation = SimpleFieldOperation("text_field")

        # Create test data
        self.data = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "text_field": ["hello", "world", None, "test", "example"]
        })

    def test_init_parameters(self):
        """Test initialization parameters."""
        # Default params
        self.assertEqual(self.operation.field_name, "text_field")
        self.assertEqual(self.operation.mode, "REPLACE")
        self.assertIsNone(self.operation.output_field_name)
        self.assertEqual(self.operation.null_strategy, NullStrategy.PRESERVE)

        # Custom params
        op = SimpleFieldOperation(
            field_name="custom_field",
            mode="ENRICH",
            output_field_name="custom_output",
            null_strategy=NullStrategy.EXCLUDE
        )
        self.assertEqual(op.field_name, "custom_field")
        self.assertEqual(op.mode, "ENRICH")
        self.assertEqual(op.output_field_name, "custom_output")
        self.assertEqual(op.null_strategy, NullStrategy.EXCLUDE)

        # Default output field name in ENRICH mode
        op = SimpleFieldOperation(field_name="name", mode="ENRICH")
        self.assertEqual(op.output_field_name, "name_fake")

    def test_preprocess_data(self):
        """Test the preprocess_data method."""
        # Valid field
        processed = self.operation.preprocess_data(self.data)
        pd.testing.assert_frame_equal(processed, self.data)

        # Invalid field
        invalid_op = SimpleFieldOperation("nonexistent_field")
        with self.assertRaises(ValidationError):
            invalid_op.preprocess_data(self.data)

    def test_postprocess_data(self):
        """Test the postprocess_data method."""
        # Base implementation just returns the data unchanged
        processed = self.operation.postprocess_data(self.data)
        pd.testing.assert_frame_equal(processed, self.data)

    def test_handle_null_values_preserve(self):
        """Test the handle_null_values method with PRESERVE strategy."""
        # PRESERVE strategy - default
        operation = SimpleFieldOperation("text_field", null_strategy=NullStrategy.PRESERVE)
        processed = operation.handle_null_values(self.data)
        pd.testing.assert_frame_equal(processed, self.data)

    def test_handle_null_values_exclude(self):
        """Test the handle_null_values method with EXCLUDE strategy."""
        # EXCLUDE strategy
        operation = SimpleFieldOperation("text_field", null_strategy=NullStrategy.EXCLUDE)
        processed = operation.handle_null_values(self.data)
        # Should exclude the row with NULL value
        expected = self.data[self.data["text_field"].notna()].copy()
        pd.testing.assert_frame_equal(processed, expected)

    def test_handle_null_values_error(self):
        """Test the handle_null_values method with ERROR strategy."""
        # ERROR strategy
        operation = SimpleFieldOperation("text_field", null_strategy=NullStrategy.ERROR)
        with self.assertRaises(ValidationError):
            operation.handle_null_values(self.data)

    def test_handle_null_values_replace(self):
        """Test the handle_null_values method with REPLACE strategy."""
        # REPLACE strategy
        operation = SimpleFieldOperation("text_field", null_strategy=NullStrategy.REPLACE)
        processed = operation.handle_null_values(self.data)
        # Should replace NULL with empty string
        expected = self.data.copy()
        expected["text_field"] = expected["text_field"].fillna("")
        pd.testing.assert_frame_equal(processed, expected)

        # Test with numeric field
        numeric_data = pd.DataFrame({
            "id": [1, 2, 3],
            "numeric_field": [10, None, 30]
        })
        numeric_op = SimpleFieldOperation("numeric_field", null_strategy=NullStrategy.REPLACE)
        processed = numeric_op.handle_null_values(numeric_data)
        expected = numeric_data.copy()
        expected["numeric_field"] = expected["numeric_field"].fillna(0)
        pd.testing.assert_frame_equal(processed, expected)

    def test_process_batch_replace_mode(self):
        """Test the process_batch method in REPLACE mode."""
        # REPLACE mode (default)
        processed = self.operation.process_batch(self.data)

        # Should uppercase the text values
        expected = self.data.copy()
        mask = expected["text_field"].notna()
        expected.loc[mask, "text_field"] = expected.loc[mask, "text_field"].str.upper()

        pd.testing.assert_frame_equal(processed, expected)

    def test_process_batch_enrich_mode(self):
        """Test the process_batch method in ENRICH mode."""
        # ENRICH mode
        operation = SimpleFieldOperation(
            field_name="text_field",
            mode="ENRICH",
            output_field_name="text_field_upper"
        )
        processed = operation.process_batch(self.data)

        # Should add a new column with uppercase values
        expected = self.data.copy()
        expected["text_field_upper"] = expected["text_field"].str.upper()

        pd.testing.assert_frame_equal(processed, expected)


class TestMappingStore(unittest.TestCase):
    """Tests for the MappingStore class."""

    def setUp(self):
        """Set up test fixtures."""
        self.store = MappingStore()

    def test_initialization(self):
        """Test initialization of mapping store."""
        self.assertEqual(self.store.mappings, {})
        self.assertEqual(self.store.reverse_mappings, {})
        self.assertEqual(self.store.transitivity_markers, {})

    def test_add_mapping(self):
        """Test adding mappings."""
        # Add a mapping
        self.store.add_mapping("first_name", "John", "David")
        self.assertEqual(self.store.mappings["first_name"]["John"], "David")
        self.assertEqual(self.store.reverse_mappings["first_name"]["David"], "John")
        self.assertFalse(self.store.transitivity_markers["first_name"]["John"])

        # Add a transitive mapping
        self.store.add_mapping("last_name", "Smith", "Jones", True)
        self.assertEqual(self.store.mappings["last_name"]["Smith"], "Jones")
        self.assertEqual(self.store.reverse_mappings["last_name"]["Jones"], "Smith")
        self.assertTrue(self.store.transitivity_markers["last_name"]["Smith"])

    def test_get_mapping(self):
        """Test retrieving mappings."""
        # Add mappings
        self.store.add_mapping("first_name", "John", "David")
        self.store.add_mapping("first_name", "Mary", "Susan")

        # Retrieve existing mappings
        self.assertEqual(self.store.get_mapping("first_name", "John"), "David")
        self.assertEqual(self.store.get_mapping("first_name", "Mary"), "Susan")

        # Retrieve non-existent mapping
        self.assertIsNone(self.store.get_mapping("first_name", "Nonexistent"))
        self.assertIsNone(self.store.get_mapping("nonexistent_field", "Value"))

    def test_restore_original(self):
        """Test restoring original values."""
        # Add mappings
        self.store.add_mapping("first_name", "John", "David")
        self.store.add_mapping("last_name", "Smith", "Jones")

        # Restore existing mappings
        self.assertEqual(self.store.restore_original("first_name", "David"), "John")
        self.assertEqual(self.store.restore_original("last_name", "Jones"), "Smith")

        # Restore non-existent mapping
        self.assertIsNone(self.store.restore_original("first_name", "Nonexistent"))
        self.assertIsNone(self.store.restore_original("nonexistent_field", "Value"))

    def test_is_transitive(self):
        """Test checking if a mapping is transitive."""
        # Add mappings
        self.store.add_mapping("first_name", "John", "David", False)
        self.store.add_mapping("last_name", "Smith", "Jones", True)

        # Check transitivity
        self.assertFalse(self.store.is_transitive("first_name", "John"))
        self.assertTrue(self.store.is_transitive("last_name", "Smith"))

        # Check non-existent mapping
        self.assertFalse(self.store.is_transitive("first_name", "Nonexistent"))
        self.assertFalse(self.store.is_transitive("nonexistent_field", "Value"))

    def test_get_field_mappings(self):
        """Test retrieving all mappings for a field."""
        # Add mappings
        self.store.add_mapping("first_name", "John", "David")
        self.store.add_mapping("first_name", "Mary", "Susan")
        self.store.add_mapping("last_name", "Smith", "Jones")

        # Get all mappings for a field
        first_name_mappings = self.store.get_field_mappings("first_name")
        self.assertEqual(first_name_mappings, {"John": "David", "Mary": "Susan"})

        # Get mappings for non-existent field
        self.assertEqual(self.store.get_field_mappings("nonexistent"), {})

    def test_get_field_names(self):
        """Test retrieving all field names."""
        # Add mappings
        self.store.add_mapping("first_name", "John", "David")
        self.store.add_mapping("last_name", "Smith", "Jones")
        self.store.add_mapping("email", "john@example.com", "david@example.com")

        # Get all field names
        field_names = self.store.get_field_names()
        self.assertEqual(field_names, {"first_name", "last_name", "email"})

        # Empty case
        empty_store = MappingStore()
        self.assertEqual(empty_store.get_field_names(), set())

    def test_mapping_conflict_detection(self):
        """Test conflict detection when adding mappings."""
        # Add a mapping
        self.store.add_mapping("first_name", "John", "David")

        # Attempt to add conflicting mapping
        with self.assertRaises(MappingError):
            self.store.add_mapping("first_name", "John", "Different")

        # Attempt to add reverse conflict
        with self.assertRaises(MappingError):
            self.store.add_mapping("first_name", "Different", "David")

        # Non-conflicting additions should work
        self.store.add_mapping("first_name", "Mary", "Susan")
        self.store.add_mapping("last_name", "Smith", "Jones")


if __name__ == "__main__":
    unittest.main()