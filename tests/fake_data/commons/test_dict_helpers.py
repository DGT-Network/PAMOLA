import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pandas as pd

from pamola_core.fake_data.commons import dict_helpers



class TestFindDictionary(unittest.TestCase):

    @patch("pamola_core.fake_data.commons.dict_helpers.Path.exists")
    def test_provided_path_exists(self, mock_exists):
        mock_exists.return_value = True
        path = dict_helpers.find_dictionary(dictionary_path="some/path.txt")
        self.assertEqual(path, Path("some/path.txt"))

    @patch("pamola_core.fake_data.commons.dict_helpers.Path.exists")
    def test_provided_path_does_not_exist(self, mock_exists):
        mock_exists.return_value = False
        path = dict_helpers.find_dictionary(dictionary_path="invalid/path.txt")
        self.assertIsNone(path)

    @patch("pamola_core.fake_data.commons.dict_helpers.Path.exists", autospec=True)
    def test_convention_path_with_gender(self, mock_exists):
        def exists_side_effect(self_path):
            return self_path.name == "ru_m_first_names.txt"

        mock_exists.side_effect = exists_side_effect

        path = dict_helpers.find_dictionary(language="ru", gender="male", name_type="first_name", dict_dir="mock_dir")
        self.assertEqual(path, Path("mock_dir/ru_m_first_names.txt"))

    @patch("pamola_core.fake_data.commons.dict_helpers.Path.exists", autospec=True)
    def test_convention_path_without_gender(self, mock_exists):
        def side_effect(self_path):
            return self_path.name == "ru_first_names.txt"

        mock_exists.side_effect = side_effect

        path = dict_helpers.find_dictionary(language="ru", name_type="first_name", dict_dir="mock_dir")
        self.assertEqual(path, Path("mock_dir/ru_first_names.txt"))

    @patch("pamola_core.fake_data.commons.dict_helpers.Path.exists", autospec=True)
    def test_fallback_to_full_name(self, mock_exists):
        def side_effect(self_path):
            return self_path.name == "ru_names.txt"

        mock_exists.side_effect = side_effect

        path = dict_helpers.find_dictionary(language="ru", gender="female", name_type="last_name", dict_dir="mock_dir")
        self.assertEqual(path, Path("mock_dir/ru_names.txt"))

    @patch("pamola_core.fake_data.commons.dict_helpers.Path.exists")
    def test_not_found_returns_none(self, mock_exists):
        mock_exists.return_value = False
        path = dict_helpers.find_dictionary(language="xx", gender="unknown", name_type="nonsense", dict_dir="mock_dir")
        self.assertIsNone(path)


class TestLoadDictionaryFromText(unittest.TestCase):

    def setUp(self):
        # Clear cache before each test
        dict_helpers._dictionary_cache.clear()

    @patch("pamola_core.fake_data.commons.dict_helpers.io.read_text")
    def test_load_success_with_valid_lines(self, mock_read_text):
        # Chuẩn bị dữ liệu giả
        content = "apple\nbanana\ncarrot\n"
        mock_read_text.return_value = content

        result = dict_helpers.load_dictionary_from_text("fake_path.txt")

        self.assertEqual(result, ["apple", "banana", "carrot"])
        # Kiểm tra cache được cập nhật
        self.assertIn("fake_path.txt", dict_helpers._dictionary_cache)
        self.assertEqual(dict_helpers._dictionary_cache["fake_path.txt"], result)

    @patch("pamola_core.fake_data.commons.dict_helpers.io.read_text")
    def test_load_skips_invalid_and_too_short_lines(self, mock_read_text):
        content = "a\nvalid\ntoolongwordtoolongwordtoolongwordtoolongwordtoolongwordtoolongwordtoolongwordtoolongwordtoolongwordtoolongword\nvalid2\n"
        mock_read_text.return_value = content

        # max_length mặc định là 100, dòng toolong sẽ bị bỏ
        result = dict_helpers.load_dictionary_from_text("fake_path.txt", min_length=2, max_length=20)

        # 'a' bị loại vì < 2, toolong bị loại vì > 20
        self.assertEqual(result, ["valid", "valid2"])

    @patch("pamola_core.fake_data.commons.dict_helpers.io.read_text")
    def test_load_with_validate_pattern(self, mock_read_text):
        content = "John\nMary\nInvalid123\nO'Connor\n"
        mock_read_text.return_value = content

        # Sử dụng pattern "name" (chỉ cho phép chữ, space, apostrophe, ...)
        result = dict_helpers.load_dictionary_from_text("fake_path.txt", validate_pattern="name")

        self.assertEqual(result, ["John", "Mary", "O'Connor"])
        self.assertNotIn("Invalid123", result)

    @patch("pamola_core.fake_data.commons.dict_helpers.io.read_text", side_effect=AttributeError)
    @patch("builtins.open", new_callable=mock_open, read_data="line1\nline2\n")
    def test_fallback_to_open_on_read_text_error(self, mock_file, mock_read_text):
        # Khi io.read_text lỗi, sẽ fallback mở file bằng open
        result = dict_helpers.load_dictionary_from_text("fake_path.txt")
        self.assertEqual(result, ["line1", "line2"])
        mock_file.assert_called_once_with(Path("fake_path.txt"), 'r', encoding='utf-8')

    @patch("pamola_core.fake_data.commons.dict_helpers.io.read_text", side_effect=Exception("bad"))
    def test_exception_handling_returns_empty(self, mock_read_text):
        result = dict_helpers.load_dictionary_from_text("fake_path.txt")
        self.assertEqual(result, [])

    @patch("pamola_core.fake_data.commons.dict_helpers.io.read_text")
    def test_cache_used_if_available(self, mock_read_text):
        # Chuẩn bị cache sẵn
        dict_helpers._dictionary_cache["fake_path.txt"] = ["cached_item"]

        result = dict_helpers.load_dictionary_from_text("fake_path.txt")
        # Hàm không gọi io.read_text khi cache tồn tại
        mock_read_text.assert_not_called()
        self.assertEqual(result, ["cached_item"])


class TestLoadDictionaryWithStats(unittest.TestCase):

    @patch("pamola_core.fake_data.commons.dict_helpers.load_dictionary_from_text")
    def test_load_dictionary_with_stats_normal(self, mock_load):
        # Mock trả về danh sách items hợp lệ
        mock_load.return_value = ["Anna", "Maria", "Olga"]

        path = "fake_path.txt"
        result = dict_helpers.load_dictionary_with_stats(
            path,
            language="ru",
            gender="F",
            name_type="first_name",
            cache=True,
            validate_pattern="name"
        )

        self.assertEqual(result["items"], ["Anna", "Maria", "Olga"])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["language"], "ru")
        self.assertEqual(result["gender"], "F")
        self.assertEqual(result["name_type"], "first_name")
        self.assertEqual(result["source"], str(path))
        self.assertAlmostEqual(result["avg_length"], (4 + 5 + 4) / 3)
        self.assertEqual(result["min_length"], 4)
        self.assertEqual(result["max_length"], 5)

    @patch("pamola_core.fake_data.commons.dict_helpers.load_dictionary_from_text")
    def test_load_dictionary_with_stats_empty(self, mock_load):
        # Mock trả về danh sách rỗng
        mock_load.return_value = []

        path = Path("/some/path/dict.txt")
        result = dict_helpers.load_dictionary_with_stats(
            path,
            language="en",
            gender=None,
            name_type="last_name",
            cache=False,
            validate_pattern=None
        )

        self.assertEqual(result["items"], [])
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["language"], "en")
        self.assertIsNone(result["gender"])
        self.assertEqual(result["name_type"], "last_name")
        self.assertEqual(result["source"], str(path))
        self.assertEqual(result["avg_length"], 0)
        self.assertEqual(result["min_length"], 0)
        self.assertEqual(result["max_length"], 0)


class TestClearDictionaryCache(unittest.TestCase):

    @patch("pamola_core.fake_data.commons.dict_helpers.organizations.clear_cache")
    @patch("pamola_core.fake_data.commons.dict_helpers.addresses.clear_cache")
    @patch("pamola_core.fake_data.commons.dict_helpers.phones.clear_cache")
    @patch("pamola_core.fake_data.commons.dict_helpers.domains.clear_cache")
    @patch("pamola_core.fake_data.commons.dict_helpers.names.clear_cache")
    def test_clear_dictionary_cache(self, mock_names_clear, mock_domains_clear, mock_phones_clear, mock_addresses_clear, mock_organizations_clear):
        # Initialize fake cache with some data
        dict_helpers._dictionary_cache = {"some_key": ["some", "values"]}

        # Call the function to clear caches
        dict_helpers.clear_dictionary_cache()

        # Check that the main cache is cleared
        self.assertEqual(dict_helpers._dictionary_cache, {})

        # Verify that clear_cache was called once on each embedded dictionary
        mock_names_clear.assert_called_once()
        mock_domains_clear.assert_called_once()
        mock_phones_clear.assert_called_once()
        mock_addresses_clear.assert_called_once()
        mock_organizations_clear.assert_called_once()


class TestGetRandomItems(unittest.TestCase):

    def setUp(self):
        self.sample_dict = ["apple", "banana", "cherry", "date", "fig"]

    def test_empty_dictionary(self):
        result = dict_helpers.get_random_items([], 3)
        self.assertEqual(result, [])

    def test_zero_count(self):
        result = dict_helpers.get_random_items(self.sample_dict, 0)
        self.assertEqual(result, [])

    def test_negative_count(self):
        result = dict_helpers.get_random_items(self.sample_dict, -1)
        self.assertEqual(result, [])

    def test_count_less_than_dict_length(self):
        result = dict_helpers.get_random_items(self.sample_dict, 3, seed=42)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(set(result)), 3)  # No duplicates
        for item in result:
            self.assertIn(item, self.sample_dict)

    def test_count_equal_to_dict_length(self):
        result = dict_helpers.get_random_items(self.sample_dict, len(self.sample_dict), seed=7)
        self.assertEqual(sorted(result), sorted(self.sample_dict))

    def test_count_greater_than_dict_length(self):
        count = 10
        result = dict_helpers.get_random_items(self.sample_dict, count, seed=100)
        self.assertEqual(len(result), count)
        for item in result:
            self.assertIn(item, self.sample_dict)

    def test_different_seed_produces_different_results(self):
        result1 = dict_helpers.get_random_items(self.sample_dict, 3, seed=1)
        result2 = dict_helpers.get_random_items(self.sample_dict, 3, seed=2)
        self.assertNotEqual(result1, result2)

    def test_no_seed_randomness(self):
        result = dict_helpers.get_random_items(self.sample_dict, 3)
        self.assertEqual(len(result), 3)
        for item in result:
            self.assertIn(item, self.sample_dict)


class TestValidateDictionary(unittest.TestCase):

    def test_all_valid_entries(self):
        data = ["John", "Mary", "O'Connor", "Anna-Marie"]
        valid, stats = dict_helpers.validate_dictionary(data, dict_type="name", min_length=2, max_length=20)
        self.assertEqual(valid, data)
        self.assertEqual(stats["original_count"], 4)
        self.assertEqual(stats["valid_count"], 4)
        self.assertEqual(stats["invalid_count"], 0)
        self.assertEqual(stats["too_short_count"], 0)
        self.assertEqual(stats["too_long_count"], 0)
        self.assertEqual(stats["pattern_mismatch_count"], 0)

    def test_too_short_and_too_long(self):
        data = ["A", "ThisIsAVeryLongNameExceedingLimit"]
        valid, stats = dict_helpers.validate_dictionary(data, dict_type="name", min_length=2, max_length=10)
        self.assertEqual(valid, [])
        self.assertEqual(stats["original_count"], 2)
        self.assertEqual(stats["valid_count"], 0)
        self.assertEqual(stats["invalid_count"], 2)
        self.assertEqual(stats["too_short_count"], 1)
        self.assertEqual(stats["too_long_count"], 1)

    def test_pattern_mismatch(self):
        data = ["John", "Mary123", "O'Connor!", "Anna-Marie"]
        valid, stats = dict_helpers.validate_dictionary(data, dict_type="name")
        self.assertEqual(valid, ["John", "Anna-Marie"])
        self.assertEqual(stats["original_count"], 4)
        self.assertEqual(stats["valid_count"], 2)
        self.assertEqual(stats["invalid_count"], 2)
        self.assertEqual(stats["pattern_mismatch_count"], 2)

    def test_default_pattern_accepts_any_non_empty(self):
        data = ["", "valid", "   ", "123"]
        valid, stats = dict_helpers.validate_dictionary(data, dict_type="unknown")  # fallback to default pattern
        self.assertEqual(valid, ["valid", "   ", "123"])  # Accepts '   '
        self.assertEqual(stats["original_count"], 4)
        self.assertEqual(stats["valid_count"], 3)
        self.assertEqual(stats["too_short_count"], 1)


class TestLoadMultiDictionary(unittest.TestCase):

    @patch("pamola_core.fake_data.commons.dict_helpers.load_dictionary_from_text")
    def test_load_from_external_path_success(self, mock_load_dict):
        mock_load_dict.return_value = ["John", "Jane"]
        params = {"path": "some_path.txt"}
        result = dict_helpers.load_multi_dictionary("name", params)
        self.assertEqual(result, ["John", "Jane"])
        mock_load_dict.assert_called_once()

    @patch("pamola_core.fake_data.commons.dict_helpers.load_dictionary_from_text", side_effect=Exception("Load failed"))
    @patch("pamola_core.fake_data.commons.dict_helpers.find_dictionary", return_value=None)
    @patch("pamola_core.fake_data.commons.dict_helpers.get_embedded_dictionary", return_value=["DefaultName"])
    def test_load_from_external_path_failure_fallback_embedded(
        self, mock_get_embedded, mock_find_dict, mock_load_dict
    ):
        params = {"path": "bad_path.txt", "language": "en"}
        result = dict_helpers.load_multi_dictionary("name", params)
        self.assertEqual(result, ["DefaultName"])
        mock_get_embedded.assert_called_once_with("first_name", None, "en")

    @patch("pamola_core.fake_data.commons.dict_helpers.find_dictionary")
    @patch("pamola_core.fake_data.commons.dict_helpers.load_dictionary_from_text")
    def test_load_from_conventional_success(self, mock_load_dict, mock_find_dict):
        mock_find_dict.return_value = "dicts/en.txt"
        mock_load_dict.return_value = ["Alice", "Bob"]
        params = {"dict_dir": "dicts", "language": "en"}
        result = dict_helpers.load_multi_dictionary("name", params)
        self.assertEqual(result, ["Alice", "Bob"])
        mock_find_dict.assert_called_once()

    @patch("pamola_core.fake_data.commons.dict_helpers.get_embedded_dictionary")
    def test_fallback_to_embedded_dictionary(self, mock_get_embedded):
        mock_get_embedded.return_value = ["EmbeddedName"]
        params = {"language": "en"}
        result = dict_helpers.load_multi_dictionary("name", params)
        self.assertEqual(result, ["EmbeddedName"])
        mock_get_embedded.assert_called_once_with("first_name", None, "en")

    @patch("pamola_core.fake_data.commons.dict_helpers.domains.get_common_email_domains")
    def test_fallback_domain(self, mock_domains):
        mock_domains.return_value = ["gmail.com", "yahoo.com"]
        result = dict_helpers.load_multi_dictionary("domain", {}, fallback_to_embedded=True)
        self.assertEqual(result, ["gmail.com", "yahoo.com"])

    # Error #
    @patch("pamola_core.fake_data.commons.dict_helpers.phones.get_area_codes")
    def test_fallback_phone(self, mock_phones):
        mock_phones.return_value = ["212", "646"]
        result = dict_helpers.load_multi_dictionary("phone", {"country": "US"})
        self.assertEqual(result, ["212", "646"])

    @patch("pamola_core.fake_data.commons.dict_helpers.addresses.get_address_component")
    def test_fallback_address(self, mock_address):
        mock_address.return_value = ["Main St", "Broadway"]
        result = dict_helpers.load_multi_dictionary("address", {"country": "US", "component": "street"})
        self.assertEqual(result, ["Main St", "Broadway"])

    @patch("pamola_core.fake_data.commons.dict_helpers.organizations.get_organization_names")
    def test_fallback_organization(self, mock_orgs):
        mock_orgs.return_value = ["OpenAI", "DeepMind"]
        result = dict_helpers.load_multi_dictionary("organization", {"country": "US", "org_type": "tech"})
        self.assertEqual(result, ["OpenAI", "DeepMind"])


class TestIsMultiDictionary(unittest.TestCase):

    def create_temp_file(self, content: str) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, mode='w+', encoding='utf-8')
        tmp.write(content)
        tmp.close()
        return tmp.name

    def test_single_column(self):
        content = "John\nJane\nAlice\n"
        path = self.create_temp_file(content)
        result = dict_helpers.is_multidictionary(path)
        self.assertFalse(result)
        os.remove(path)

    def test_multi_column_comma(self):
        content = "John,Doe\nJane,Doe\n"
        path = self.create_temp_file(content)
        result = dict_helpers.is_multidictionary(path)
        self.assertTrue(result)
        os.remove(path)

    def test_multi_column_tab(self):
        content = "John\tDoe\nJane\tSmith\n"
        path = self.create_temp_file(content)
        result = dict_helpers.is_multidictionary(path)
        self.assertTrue(result)
        os.remove(path)

    def test_multi_column_pipe(self):
        content = "Alice|F\nBob|M\n"
        path = self.create_temp_file(content)
        result = dict_helpers.is_multidictionary(path)
        self.assertTrue(result)
        os.remove(path)

    def test_multi_column_semicolon(self):
        content = "A;B\nC;D\n"
        path = self.create_temp_file(content)
        result = dict_helpers.is_multidictionary(path)
        self.assertTrue(result)
        os.remove(path)

    def test_file_not_exist(self):
        result = dict_helpers.is_multidictionary("non_existent_file.txt")
        self.assertFalse(result)

    def test_empty_file(self):
        path = self.create_temp_file("")
        result = dict_helpers.is_multidictionary(path)
        self.assertFalse(result)
        os.remove(path)


class TestParseFullName(unittest.TestCase):

    def test_empty_string(self):
        result = dict_helpers.parse_full_name("")
        self.assertEqual(result, {"first_name": "", "middle_name": "", "last_name": ""})

    def test_single_name(self):
        result = dict_helpers.parse_full_name("Alice")
        self.assertEqual(result, {"first_name": "Alice", "middle_name": "", "last_name": ""})

    def test_explicit_first_last(self):
        result = dict_helpers.parse_full_name("John Smith", name_format="first_last")
        self.assertEqual(result, {"first_name": "John", "middle_name": "", "last_name": "Smith"})

    def test_explicit_last_first(self):
        result = dict_helpers.parse_full_name("Smith John", name_format="last_first")
        self.assertEqual(result, {"first_name": "John", "middle_name": "", "last_name": "Smith"})

    def test_explicit_first_middle_last(self):
        result = dict_helpers.parse_full_name("John Michael Smith", name_format="first_middle_last")
        self.assertEqual(result, {"first_name": "John", "middle_name": "Michael", "last_name": "Smith"})

    def test_russian_three_parts(self):
        result = dict_helpers.parse_full_name("Иванов Иван Иванович", language="ru")
        self.assertEqual(result, {"first_name": "Иван", "middle_name": "Иванович", "last_name": "Иванов"})

    def test_russian_two_parts(self):
        result = dict_helpers.parse_full_name("Петров Алексей", language="ru")
        self.assertEqual(result, {"first_name": "Алексей", "middle_name": "", "last_name": "Петров"})

    def test_vietnamese_three_parts(self):
        result = dict_helpers.parse_full_name("Nguyen Van A", language="vi")
        self.assertEqual(result, {"first_name": "A", "middle_name": "Van", "last_name": "Nguyen"})

    def test_vietnamese_multiple_middle(self):
        result = dict_helpers.parse_full_name("Nguyen Thi Minh Khai", language="vi")
        self.assertEqual(result, {"first_name": "Khai", "middle_name": "Thi Minh", "last_name": "Nguyen"})

    def test_default_western_three_parts(self):
        result = dict_helpers.parse_full_name("John Michael Smith", language="en")
        self.assertEqual(result, {"first_name": "John", "middle_name": "Michael", "last_name": "Smith"})

    def test_default_western_two_parts(self):
        result = dict_helpers.parse_full_name("John Smith", language="en")
        self.assertEqual(result, {"first_name": "John", "middle_name": "", "last_name": "Smith"})

    def test_default_fallback_language(self):
        result = dict_helpers.parse_full_name("John Michael Smith", language="unknown")
        self.assertEqual(result, {"first_name": "John", "middle_name": "Michael", "last_name": "Smith"})


class TestGetEmbeddedDictionary(unittest.TestCase):

    @patch("pamola_core.fake_data.commons.dict_helpers.names.get_names")
    def test_get_first_names_male_russian(self, mock_get_names):
        mock_get_names.return_value = ["Алексей", "Иван", "Дмитрий"]
        result = dict_helpers.get_embedded_dictionary(
            name_type="first_name",
            gender="M",
            language="ru"
        )
        mock_get_names.assert_called_once_with("ru", "M", "first_name")
        self.assertEqual(result, ["Алексей", "Иван", "Дмитрий"])

    @patch("pamola_core.fake_data.commons.dict_helpers.names.get_names")
    def test_get_last_names_female_english(self, mock_get_names):
        mock_get_names.return_value = ["Smith", "Johnson"]
        result = dict_helpers.get_embedded_dictionary(
            name_type="last_name",
            gender="F",
            language="en"
        )
        mock_get_names.assert_called_once_with("en", "F", "last_name")
        self.assertEqual(result, ["Smith", "Johnson"])

    @patch("pamola_core.fake_data.commons.dict_helpers.names.get_names")
    def test_get_full_names_no_gender(self, mock_get_names):
        mock_get_names.return_value = ["John Smith", "Jane Doe"]
        result = dict_helpers.get_embedded_dictionary(
            name_type="full_name",
            gender=None,
            language="en"
        )
        mock_get_names.assert_called_once_with("en", None, "full_name")
        self.assertEqual(result, ["John Smith", "Jane Doe"])

    @patch("pamola_core.fake_data.commons.dict_helpers.names.get_names")
    def test_gender_case_normalization(self, mock_get_names):
        mock_get_names.return_value = ["Мария", "Екатерина"]
        result = dict_helpers.get_embedded_dictionary(
            name_type="first_name",
            gender="f",  # lowercase, should normalize
            language="RU"
        )
        mock_get_names.assert_called_once_with("ru", "F", "first_name")
        self.assertEqual(result, ["Мария", "Екатерина"])


class TestLoadCsvDictionary(unittest.TestCase):

    def setUp(self):
        dict_helpers._dictionary_cache.clear()

    @patch("pamola_core.fake_data.commons.dict_helpers.io.read_full_csv")
    def test_load_default_column(self, mock_read_csv):
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
        mock_read_csv.return_value = df

        result = dict_helpers.load_csv_dictionary("dummy.csv")
        self.assertEqual(result, ["Alice", "Bob", "Charlie"])

    @patch("pamola_core.fake_data.commons.dict_helpers.io.read_full_csv")
    def test_load_specific_column(self, mock_read_csv):
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["Anna", "Ben", "Cara"]})
        mock_read_csv.return_value = df

        result = dict_helpers.load_csv_dictionary("dummy.csv", column_name="name")
        self.assertEqual(result, ["Anna", "Ben", "Cara"])

    @patch("pamola_core.fake_data.commons.dict_helpers.io.read_full_csv", side_effect=AttributeError)
    @patch("pamola_core.fake_data.commons.dict_helpers.pd.read_csv")
    def test_fallback_to_pandas(self, mock_pandas_read, _):
        df = pd.DataFrame({"col": ["X", "Y", "Z"]})
        mock_pandas_read.return_value = df

        result = dict_helpers.load_csv_dictionary("dummy.csv")
        self.assertEqual(result, ["X", "Y", "Z"])

    @patch("pamola_core.fake_data.commons.dict_helpers.io.read_full_csv")
    def test_column_not_found_fallback(self, mock_read_csv):
        df = pd.DataFrame({"first": ["A", "B", "C"]})
        mock_read_csv.return_value = df

        result = dict_helpers.load_csv_dictionary("dummy.csv", column_name="nonexistent")
        self.assertEqual(result, ["A", "B", "C"])  # fallback to first column

    @patch("pamola_core.fake_data.commons.dict_helpers.io.read_full_csv")
    def test_disable_cache(self, mock_read_csv):
        df = pd.DataFrame({"x": ["A", "B"]})
        mock_read_csv.return_value = df

        result = dict_helpers.load_csv_dictionary("file.csv", column_name="x", cache=False)
        self.assertEqual(result, ["A", "B"])
        self.assertNotIn("file.csv:x", dict_helpers._dictionary_cache)

    @patch("pamola_core.fake_data.commons.dict_helpers.io.read_full_csv", side_effect=Exception("read error"))
    def test_read_failure_returns_empty(self, _):
        result = dict_helpers.load_csv_dictionary("invalid.csv")
        self.assertEqual(result, [])


class TestCombineDictionaries(unittest.TestCase):

    def test_basic_combine(self):
        dicts = [["a", "bb"], ["cc", "dd"]]
        result = dict_helpers.combine_dictionaries(dicts)
        self.assertEqual(result, ["a", "bb", "cc", "dd"])

    def test_deduplication(self):
        dicts = [["a", "b", "a"], ["b", "c"]]
        result = dict_helpers.combine_dictionaries(dicts, dedup=True)
        self.assertEqual(result, ["a", "b", "c"])

    def test_no_deduplication(self):
        dicts = [["a", "b", "a"], ["b", "c"]]
        result = dict_helpers.combine_dictionaries(dicts, dedup=False)
        self.assertEqual(result, ["a", "b", "a", "b", "c"])

    def test_min_length_filter(self):
        dicts = [["a", "bb", "ccc"], ["dddd", "e"]]
        result = dict_helpers.combine_dictionaries(dicts, min_length=2)
        self.assertEqual(result, ["bb", "ccc", "dddd"])

    def test_max_length_filter(self):
        dicts = [["a", "bb", "ccc"], ["dddd", "e"]]
        result = dict_helpers.combine_dictionaries(dicts, max_length=2)
        self.assertEqual(result, ["a", "bb", "e"])

    def test_min_and_max_length_filter(self):
        dicts = [["a", "bb", "ccc", "dddd"], ["ee", "f"]]
        result = dict_helpers.combine_dictionaries(dicts, min_length=2, max_length=3)
        self.assertEqual(result, ["bb", "ccc", "ee"])

    def test_empty_input(self):
        result = dict_helpers.combine_dictionaries([])
        self.assertEqual(result, [])

    def test_empty_lists(self):
        dicts = [[], []]
        result = dict_helpers.combine_dictionaries(dicts)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()