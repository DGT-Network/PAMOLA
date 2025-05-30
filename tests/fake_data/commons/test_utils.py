import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from pamola_core.fake_data.commons import utils
# Import the module to test
from pamola_core.fake_data.commons.utils import (
    normalize_string,
    hash_value,
    generate_deterministic_value,
    detect_language,
    format_phone_number,
    load_dictionary, detect_gender_by_dictionary, validate_email, validate_phone, find_dictionary_file, save_metrics
)
from pamola_core.utils.progress import ProgressBar


class TestStringUtilities(unittest.TestCase):
    """Tests for string manipulation utility functions."""

    def test_normalize_string(self):
        """Test the normalize_string function."""
        # Test with default parameters
        self.assertEqual(normalize_string("Hello   World"), "hello world")

        # Test with keep_case=True
        self.assertEqual(normalize_string("Hello World", keep_case=True), "Hello World")

        # Test with remove_punctuation=True
        self.assertEqual(normalize_string("Hello, World!", remove_punctuation=True), "hello world")

        # Test with both parameters
        self.assertEqual(normalize_string("Hello, World!", keep_case=True, remove_punctuation=True), "Hello World")

        # Test with multiple spaces
        self.assertEqual(normalize_string("Hello   World"), "hello world")

        # Test with leading/trailing spaces
        self.assertEqual(normalize_string("  Hello World  "), "hello world")

        # Test with empty string
        self.assertEqual(normalize_string(""), "")

        # Test with None
        self.assertEqual(normalize_string(None), "")

    def test_hash_value(self):
        """Test the hash_value function."""
        # Test with default parameters (sha256)
        value = "test_value"
        expected_hash = hashlib.sha256(value.encode()).hexdigest()
        self.assertEqual(hash_value(value), expected_hash)

        # Test with salt
        salt = "salt_value"
        salted_value = f"{value}{salt}"
        expected_hash = hashlib.sha256(salted_value.encode()).hexdigest()
        self.assertEqual(hash_value(value, salt), expected_hash)

        # Test with different algorithms
        algorithms = {
            "md5": hashlib.md5,
            "sha1": hashlib.sha1,
            "sha256": hashlib.sha256,
            "sha512": hashlib.sha512
        }

        for algo_name, algo_func in algorithms.items():
            expected_hash = algo_func(value.encode()).hexdigest()
            self.assertEqual(hash_value(value, algorithm=algo_name), expected_hash)

        # Test with invalid algorithm
        with self.assertRaises(ValueError):
            hash_value(value, algorithm="invalid_algo")

        # Test with non-string value
        non_string_value = 12345
        expected_hash = hashlib.sha256(str(non_string_value).encode()).hexdigest()
        self.assertEqual(hash_value(non_string_value), expected_hash)

        # Test with None value
        self.assertEqual(hash_value(None), "")

    def test_generate_deterministic_value(self):
        """Test the generate_deterministic_value function."""
        # Test basic functionality
        seed = "seed_value"
        result1 = generate_deterministic_value(seed)
        result2 = generate_deterministic_value(seed)

        # Same seed should produce same result
        self.assertEqual(result1, result2)

        # Length parameter should be respected
        length = 15
        result = generate_deterministic_value(seed, length=length)
        self.assertEqual(len(result), length)

        # Different seeds should produce different results
        result1 = generate_deterministic_value("seed1")
        result2 = generate_deterministic_value("seed2")
        self.assertNotEqual(result1, result2)

        # Custom character set
        chars = "ABC123"
        result = generate_deterministic_value(seed, chars=chars)
        for char in result:
            self.assertIn(char, chars)


class TestLanguageUtilities(unittest.TestCase):
    """Tests for language detection and processing utility functions."""

    def test_detect_language(self):
        """Test the detect_language function."""
        # Test Russian text
        text = (
            "Привет, меня зовут Алексей. Я работаю программистом в Москве. "
            "Моя любимая книга — 'Мастер и Маргарита'. Это тест для определения языка."
        )
        self.assertEqual(detect_language(text), "ru")

        # Test English text
        self.assertEqual(detect_language("Hello, world!"), "en")

        # Test empty string
        self.assertEqual(detect_language("", default_language = "unknown"), "unknown")

        # Test very short text (fallback to pattern matching)
        self.assertEqual(detect_language("Здравствуйте"), "ru")
        self.assertEqual(detect_language("World"), "en")

        # Test with None
        self.assertEqual(detect_language(None, default_language = "unknown"), "unknown")

        # Test numbers only
        self.assertEqual(detect_language("12345", default_language = "unknown"), "unknown")

    @patch('pamola_core.fake_data.commons.utils.language')
    def test_detect_language_with_mocked_langdetect(self, mock_langdetect):
        """Test the detect_language function with mocked langdetect."""
        # Setup mock
        mock_langdetect.detect_language.return_value = "fr"

        # Test with mock
        self.assertEqual(detect_language("Bonjour le monde"), "fr")
        mock_langdetect.detect_language.assert_called_once()

        # Test fallback when langdetect raises exception
        mock_langdetect.detect_language.side_effect = Exception("Detection failed")

        #utils.LANGUAGE_MODULE_AVAILABLE = False

        # Should fallback to pattern matching
        self.assertEqual(detect_language("Hello world"), "en")
        self.assertEqual(detect_language("Привет мир", default_language = "ru"), "ru")


class TestDictionaryUtilities(unittest.TestCase):
    def test_detect_gender_by_dictionary(self):
        """Test the detect_gender_from_name function."""
        # Test with Russian names and default gender
        self.assertEqual(detect_gender_by_dictionary("Иван", default_gender = "M"), "M")
        self.assertEqual(detect_gender_by_dictionary("Мария", default_gender = "F"), "F")

        # Test with not default gender
        self.assertEqual(detect_gender_by_dictionary("John"), None)

        # Test with dictionary
        gender_dict = {"emma": "F", "mary": "F", "john": "M", "noah": "M", "анна": "F", "екатерина": "F", "александр": "M", "иван": "M", "мария": "F"}
        self.assertEqual(detect_gender_by_dictionary("John", gender_dict), "M")
        self.assertEqual(detect_gender_by_dictionary("Mary", gender_dict), "F")

        # Test with empty string
        self.assertIsNone(detect_gender_by_dictionary(""))

        # Test with None
        self.assertIsNone(detect_gender_by_dictionary(None))

        # Test with unknown name
        self.assertIsNone(detect_gender_by_dictionary("Unknown"))

    def test_load_dictionary(self):
        # 1. From dictionary
        dict_input = {"a": "apple", "b": "banana"}
        result = load_dictionary(dict_input)
        self.assertIn("apple", result)
        self.assertIn("banana", result)

        # 2. From list
        list_input = ["apple", "banana"]
        result = load_dictionary(list_input)
        self.assertEqual(result, list_input)

        # 3. From .txt file
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as tmp_txt:
            tmp_txt.write("apple\nbanana\n")
            txt_path = tmp_txt.name
        result = load_dictionary(Path(txt_path))
        self.assertEqual(result, ["apple", "banana"])
        Path(txt_path).unlink()

        # 4. From .json file (list)
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp_json_list:
            json.dump(["apple", "banana"], tmp_json_list)
            json_list_path = tmp_json_list.name
        result = load_dictionary(Path(json_list_path))
        self.assertEqual(result, ["apple", "banana"])
        Path(json_list_path).unlink()

        # 5. From .json file (dict)
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp_json_dict:
            json.dump({"x": "xray", "y": "yellow"}, tmp_json_dict)
            json_dict_path = tmp_json_dict.name
        result = load_dictionary(Path(json_dict_path))
        self.assertIn("xray", result)
        self.assertIn("yellow", result)
        Path(json_dict_path).unlink()

        # 6. Invalid file extension
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.xml', delete=False) as tmp_xml:
            tmp_xml.write("<data>Invalid</data>")
            xml_path = tmp_xml.name
        with self.assertRaises(ValueError):
            load_dictionary(Path(xml_path))
        Path(xml_path).unlink()

        # 7. File not found
        with self.assertRaises(ValueError):
            load_dictionary("nonexistent_file.txt", base_path=Path("."))

        # 8. Cache test
        cache_input = ["cached"]
        result1 = load_dictionary(cache_input, dictionary_type="custom", cache=True)
        result2 = load_dictionary(cache_input, dictionary_type="custom", cache=True)
        self.assertEqual(result1, result2)

    def test_find_dictionary_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create dummy files
            txt_file = tmp_path / "names.txt"
            txt_file.write_text("John\nJane")

            json_file = tmp_path / "data.json"
            json_file.write_text('["a", "b"]')

            csv_file = tmp_path / "emails.csv"
            csv_file.write_text("email@example.com\ncontact@test.com")

            no_ext_file = tmp_path / "cities"
            no_ext_file.write_text("Paris\nLondon")

            # Case 1: Find .txt file
            result_txt = find_dictionary_file("names", base_dirs=[tmp_path], suffixes=['.txt'])
            self.assertEqual(result_txt, txt_file)

            # Case 2: Find .json file
            result_json = find_dictionary_file("data", base_dirs=[tmp_path], suffixes=['.txt', '.json'])
            self.assertEqual(result_json, json_file)

            # Case 3: Find .csv file
            result_csv = find_dictionary_file("emails", base_dirs=[tmp_path], suffixes=['.txt', '.json', '.csv'])
            self.assertEqual(result_csv, csv_file)

            # Case 4: Find file with no extension
            result_no_ext = find_dictionary_file("cities", base_dirs=[tmp_path], suffixes=['.txt', ''])
            self.assertEqual(result_no_ext, no_ext_file)

            # Case 5: File does not exist
            result_none = find_dictionary_file("unknown", base_dirs=[tmp_path], suffixes=['.txt', '.json', '.csv'])
            self.assertIsNone(result_none)


class TestValidateUtilities(unittest.TestCase):
    def test_validate_email(self):
        """Test the validate_email function."""
        # Test valid email
        self.assertTrue(validate_email("user@example.com"))
        self.assertTrue(validate_email("user_name@domain.co"))

        # Test invalid email
        self.assertFalse(validate_email(None))
        self.assertFalse(validate_email(""))
        self.assertFalse(validate_email("plainaddress"))
        self.assertFalse(validate_email("missingatsign.com"))
        self.assertFalse(validate_email("user@.com"))

    def test_validate_phone(self):
        """Test the validate_phone function."""
        # Test valid phone
        self.assertTrue(validate_phone("123-456-7890"))
        self.assertTrue(validate_phone("(123) 456-7890"))
        self.assertTrue(validate_phone("+1 1234567890"))

        # Test invalid email
        self.assertFalse(validate_phone(None))
        self.assertFalse(validate_phone(""))
        self.assertFalse(validate_phone("12345"))
        self.assertFalse(validate_phone("abcdefghij"))
        self.assertFalse(validate_phone("+1-123-456", "US"))
        self.assertFalse(validate_phone("912345678", "RU"))

    def test_format_phone_number(self):
        """Test the format_phone_number function."""
        # Test Russian format
        self.assertEqual(format_phone_number("79261234567", region="RU"), "+7 (926) 123-45-67")
        self.assertEqual(format_phone_number("89261234567", region="RU"), "+7 (926) 123-45-67")

        # Test US format
        self.assertEqual(format_phone_number("12025551234", region="US"), "+1 (202) 555-1234")
        self.assertEqual(format_phone_number("2025551234", region="US"), "(202) 555-1234")

        # Test with formatting=False
        self.assertEqual(format_phone_number("79261234567", region="RU", formatting=False), "79261234567")

        # Test with non-digit characters
        self.assertEqual(format_phone_number("+7 (926) 123-45-67", region="RU"), "+7 (926) 123-45-67")

        # Test with invalid format
        self.assertEqual(format_phone_number("1234", region="RU"), "+1234")

        # Test with empty string
        self.assertEqual(format_phone_number(""), "")

        # Test with None
        self.assertEqual(format_phone_number(None), "")


class TestCreateProgressBar(unittest.TestCase):
    """Tests for create progress bar utility functions."""
    def test_create_progress_bar_returns_instance(self):
        """Test that create_progress_bar returns a ProgressBar instance with correct attributes."""
        total = 100
        description = "Generating data"
        unit = "records"

        bar = utils.create_progress_bar(total=total, description=description, unit=unit)

        self.assertEqual(bar.__class__.__name__, "ProgressBar")
        self.assertIsInstance(bar, ProgressBar)
        self.assertEqual(bar.total, total)
        self.assertEqual(bar.description, description)
        self.assertEqual(bar.unit, unit)


class TestDataUtilities(unittest.TestCase):
    """Tests for data manipulation utility functions."""
    def test_save_metrics(self):
        """Test that save_metrics writes the correct JSON content to file."""
        metrics_data = {
            "field_count": 3,
            "valid_emails": 95,
            "invalid_phones": 2,
            "distribution": {"A": 30, "B": 70}
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "metrics.json"
            result_path = save_metrics(metrics_data, output_path)

            # Assert path is returned and file exists
            self.assertEqual(result_path, output_path)
            self.assertTrue(output_path.exists())

            # Read file and validate content
            with open(output_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)

            self.assertEqual(loaded, metrics_data)

    @patch("pamola_core.fake_data.commons.utils.normalize_string.cache_clear")
    def test_clear_caches(self, mock_cache_clear):
        """Test that clear_caches clears dictionary cache and string normalization cache."""
        # Pre-populate the _dictionary_cache
        utils._dictionary_cache = {"test:source": ["value1", "value2"]}

        # Call clear_caches
        utils.clear_caches()

        # Assert _dictionary_cache is now empty
        self.assertEqual(utils._dictionary_cache, {})

        # Assert normalize_string.cache_clear was called once
        mock_cache_clear.assert_called_once()


if __name__ == "__main__":
    unittest.main()