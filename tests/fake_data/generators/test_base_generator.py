import unittest
from pamola_core.fake_data.generators.base_generator import BaseGenerator


# SimpleBaseGenerator class inheriting from BaseGenerator with implemented methods
class SimpleBaseGenerator(BaseGenerator):
    def generate(self, **params) -> str:
        # Returns a simple generated value for testing
        return "generated_value"

    def generate_like(self, value: str, **params) -> str:
        # Returns the input value as it is
        return value


class TestBaseGenerator(unittest.TestCase):

    def test_init_with_config(self):
        # Tests initialization with a config dictionary
        config = {"key": "value"}
        generator = SimpleBaseGenerator(config=config)
        self.assertEqual(generator.config, config)

    def test_transform(self):
        # Tests transform method
        generator = SimpleBaseGenerator()
        values = ["value1", "value2", "value3"]

        # Checking if transform returns the same values since generate_like just returns the value
        transformed_values = generator.transform(values)
        self.assertEqual(transformed_values, values)

    def test_validate_with_valid_string(self):
        # Tests validate method with a valid string
        generator = SimpleBaseGenerator()
        self.assertTrue(generator.validate("valid_value"))

    def test_validate_with_invalid_string(self):
        # Tests validate method with an empty string
        generator = SimpleBaseGenerator()
        self.assertFalse(generator.validate(""))

    def test_validate_with_non_string_value(self):
        # Tests validate method with a non-string value (integer)
        generator = SimpleBaseGenerator()
        self.assertFalse(generator.validate(12345))  # Using an integer instead of a string

    def test_validate_with_empty_value(self):
        # Tests validate method with None value
        generator = SimpleBaseGenerator()
        self.assertFalse(generator.validate(None))


if __name__ == "__main__":
    unittest.main()