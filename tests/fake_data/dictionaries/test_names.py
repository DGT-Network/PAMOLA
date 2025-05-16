import unittest

from pamola_core.fake_data.dictionaries import names


class TestRussianNames(unittest.TestCase):

    def setUp(self):
        # Clear the dictionary cache before each test for isolation
        names._dictionary_cache.clear()

    # ==== Tests for get_ru_male_first_names ====
    def test_get_ru_male_first_names_returns_list(self):
        result = names.get_ru_male_first_names()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(name, str) for name in result))

    def test_get_ru_male_first_names_contains_known_name(self):
        result = names.get_ru_male_first_names()
        self.assertIn("Александр", result)
        self.assertIn("Дмитрий", result)

    def test_get_ru_male_first_names_cache_is_used(self):
        names.get_ru_male_first_names()
        self.assertIn('ru_m_first', names._dictionary_cache)

        names._dictionary_cache['ru_m_first'].append("ТестМужскоеИмя")
        result = names.get_ru_male_first_names()
        self.assertIn("ТестМужскоеИмя", result)

    def test_get_ru_male_first_names_idempotent(self):
        result1 = names.get_ru_male_first_names()
        result2 = names.get_ru_male_first_names()
        self.assertEqual(result1, result2)


    # ==== Tests for get_ru_female_first_names ====
    def test_get_ru_female_first_names_returns_list(self):
        result = names.get_ru_female_first_names()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(name, str) for name in result))

    def test_get_ru_female_first_names_contains_known_name(self):
        result = names.get_ru_female_first_names()
        self.assertIn("Анастасия", result)
        self.assertIn("Екатерина", result)

    def test_get_ru_female_first_names_cache_is_used(self):
        names.get_ru_female_first_names()
        self.assertIn('ru_f_first', names._dictionary_cache)

        names._dictionary_cache['ru_f_first'].append("ТестЖенскоеИмя")
        result = names.get_ru_female_first_names()
        self.assertIn("ТестЖенскоеИмя", result)

    def test_get_ru_female_first_names_idempotent(self):
        result1 = names.get_ru_female_first_names()
        result2 = names.get_ru_female_first_names()
        self.assertEqual(result1, result2)


    # ==== Tests for get_ru_male_last_names ====
    def test_get_ru_male_last_names_returns_list(self):
        result = names.get_ru_male_last_names()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(name, str) for name in result))

    def test_get_ru_male_last_names_contains_known_name(self):
        result = names.get_ru_male_last_names()
        self.assertIn("Иванов", result)
        self.assertIn("Петров", result)

    def test_get_ru_male_last_names_cache_is_used(self):
        names.get_ru_male_last_names()
        self.assertIn('ru_m_last', names._dictionary_cache)

        names._dictionary_cache['ru_m_last'].append("ТестФамилия")
        result = names.get_ru_male_last_names()
        self.assertIn("ТестФамилия", result)

    def test_get_ru_male_last_names_idempotent(self):
        result1 = names.get_ru_male_last_names()
        result2 = names.get_ru_male_last_names()
        self.assertEqual(result1, result2)


    # ==== Tests for get_ru_female_last_names ====
    def test_get_ru_female_last_names_returns_list(self):
        result = names.get_ru_female_last_names()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(name, str) for name in result))

    def test_get_ru_female_last_names_contains_known_names(self):
        result = names.get_ru_female_last_names()
        expected_names = {"Иванова", "Петрова", "Смирнова", "Соколова"}
        self.assertTrue(expected_names.issubset(set(result)), "Some known names are missing")

    def test_get_ru_female_last_names_cache_is_used(self):
        names.get_ru_female_last_names()
        self.assertIn('ru_f_last', names._dictionary_cache)

        names._dictionary_cache['ru_f_last'].append("Соколова")
        result = names.get_ru_female_last_names()
        self.assertIn("Соколова", result)

    def test_get_ru_female_last_names_idempotent(self):
        result1 = names.get_ru_female_last_names()
        result2 = names.get_ru_female_last_names()
        self.assertEqual(result1, result2)

    # ==== Tests for get_ru_male_middle_names ====
    def test_get_ru_male_middle_names_returns_list(self):
        result = names.get_ru_male_middle_names()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(name, str) for name in result))

    def test_get_ru_male_middle_names_contains_known_names(self):
        result = names.get_ru_male_middle_names()
        expected_names = {"Ааронович", "Адамович", "Акимович", "Алмазович"}
        self.assertTrue(expected_names.issubset(set(result)), "Some known names are missing")

    def test_get_ru_male_middle_names_cache_is_used(self):
        names.get_ru_male_middle_names()
        self.assertIn('ru_m_middle', names._dictionary_cache)

        names._dictionary_cache['ru_m_middle'].append("Алмазович123")
        result = names.get_ru_male_middle_names()
        self.assertIn("Алмазович123", result)

    def test_get_ru_male_middle_names_idempotent(self):
        result1 = names.get_ru_male_middle_names()
        result2 = names.get_ru_male_middle_names()
        self.assertEqual(result1, result2)


    # ==== Tests for get_ru_female_middle_names ====
    def test_get_ru_female_middle_names_returns_list(self):
        result = names.get_ru_female_middle_names()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(name, str) for name in result))

    def test_get_ru_female_middle_names_contains_known_names(self):
        result = names.get_ru_female_middle_names()
        expected_names = {"Аароновна", "Адамовна", "Акимовна", "Алмазовна"}
        self.assertTrue(expected_names.issubset(set(result)), "Some known names are missing")

    def test_get_ru_female_middle_names_cache_is_used(self):
        names.get_ru_female_middle_names()
        self.assertIn('ru_f_middle', names._dictionary_cache)

        names._dictionary_cache['ru_f_middle'].append("Алмазовна123")
        result = names.get_ru_female_middle_names()
        self.assertIn("Алмазовна123", result)

    def test_get_ru_female_middle_names_idempotent(self):
        result1 = names.get_ru_female_middle_names()
        result2 = names.get_ru_female_middle_names()
        self.assertEqual(result1, result2)


class TestEnglishNames(unittest.TestCase):

    def setUp(self):
        # Clear the dictionary cache before each test for isolation
        names._dictionary_cache.clear()

    # ==== Tests for get_en_male_first_names ====
    def test_get_en_male_first_names_returns_list(self):
        result = names.get_en_male_first_names()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(name, str) for name in result))

    def test_get_en_male_first_names_contains_known_name(self):
        result = names.get_en_male_first_names()
        self.assertIn("James", result)
        self.assertIn("Daniel", result)

    def test_get_en_male_first_names_cache_is_used(self):
        names.get_en_male_first_names()
        self.assertIn('en_m_first', names._dictionary_cache)

        names._dictionary_cache['en_m_first'].append("James123")
        result = names.get_en_male_first_names()
        self.assertIn("James123", result)

    def test_get_en_male_first_names_idempotent(self):
        result1 = names.get_en_male_first_names()
        result2 = names.get_en_male_first_names()
        self.assertEqual(result1, result2)


    # ==== Tests for get_en_female_first_names ====
    def test_get_en_female_first_names_returns_list(self):
        result = names.get_en_female_first_names()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(name, str) for name in result))

    def test_get_en_female_first_names_contains_known_name(self):
        result = names.get_en_female_first_names()
        self.assertIn("Mary", result)
        self.assertIn("Lisa", result)

    def test_get_en_female_first_names_cache_is_used(self):
        names.get_en_female_first_names()
        self.assertIn('en_f_first', names._dictionary_cache)

        names._dictionary_cache['en_f_first'].append("Mary123")
        result = names.get_en_female_first_names()
        self.assertIn("Mary123", result)

    def test_get_en_female_first_names_idempotent(self):
        result1 = names.get_en_female_first_names()
        result2 = names.get_en_female_first_names()
        self.assertEqual(result1, result2)


    # ==== Tests for get_en_last_names ====
    def test_get_en_last_names_returns_list(self):
        result = names.get_en_last_names()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(name, str) for name in result))

    def test_get_en_last_names_contains_known_name(self):
        result = names.get_en_last_names()
        self.assertIn("Smith", result)
        self.assertIn("Thomas", result)

    def test_get_en_last_names_cache_is_used(self):
        names.get_en_last_names()
        self.assertIn('en_last', names._dictionary_cache)

        names._dictionary_cache['en_last'].append("Smith123")
        result = names.get_en_last_names()
        self.assertIn("Smith123", result)

    def test_get_en_last_names_idempotent(self):
        result1 = names.get_en_last_names()
        result2 = names.get_en_last_names()
        self.assertEqual(result1, result2)


class TestVietnameseNames(unittest.TestCase):

    def setUp(self):
        # Clear the dictionary cache before each test for isolation
        names._dictionary_cache.clear()

    # ==== Tests for get_vn_male_names ====
    def test_get_vn_male_names_returns_list(self):
        result = names.get_vn_male_names()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(name, str) for name in result))

    def test_get_vn_male_names_contains_known_name(self):
        result = names.get_vn_male_names()
        self.assertIn("Nguyễn Văn An", result)
        self.assertIn("Ngô Quang Hà", result)

    def test_get_vn_male_names_cache_is_used(self):
        names.get_vn_male_names()
        self.assertIn('vn_m_names', names._dictionary_cache)

        names._dictionary_cache['vn_m_names'].append("Nguyễn Văn An 123")
        result = names.get_vn_male_names()
        self.assertIn("Nguyễn Văn An 123", result)

    def test_get_vn_male_names_idempotent(self):
        result1 = names.get_vn_male_names()
        result2 = names.get_vn_male_names()
        self.assertEqual(result1, result2)


    # ==== Tests for get_vn_female_names ====
    def test_get_vn_female_names_returns_list(self):
        result = names.get_vn_female_names()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(name, str) for name in result))

    def test_get_vn_female_names_contains_known_name(self):
        result = names.get_vn_female_names()
        self.assertIn("Nguyễn Thị An", result)
        self.assertIn("Ngô Thị Gấm", result)

    def test_get_vn_female_names_cache_is_used(self):
        names.get_vn_female_names()
        self.assertIn('vn_f_names', names._dictionary_cache)

        names._dictionary_cache['vn_f_names'].append("Nguyễn Thị An 123")
        result = names.get_vn_female_names()
        self.assertIn("Nguyễn Thị An 123", result)

    def test_get_vn_female_names_idempotent(self):
        result1 = names.get_vn_female_names()
        result2 = names.get_vn_female_names()
        self.assertEqual(result1, result2)


class TestGetNames(unittest.TestCase):

    def test_ru_male_first_names(self):
        result = names.get_names(language="ru", gender="M", name_type="first_name")
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(name, str) for name in result))

    def test_ru_female_last_names(self):
        result = names.get_names(language="ru", gender="F", name_type="last_name")
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(name, str) for name in result))

    def test_ru_all_middle_names(self):
        result = names.get_names(language="ru", gender=None, name_type="middle_name")
        self.assertGreater(len(result), 0)
        self.assertTrue(all(isinstance(name, str) for name in result))

    def test_en_female_first_names(self):
        result = names.get_names(language="en", gender="F", name_type="first_name")
        self.assertGreater(len(result), 0)

    def test_en_last_names(self):
        result = names.get_names(language="en", name_type="last_name")
        self.assertGreater(len(result), 0)

    def test_vn_male_full_names(self):
        result = names.get_names(language="vn", gender="M", name_type="full_name")
        self.assertGreater(len(result), 0)

    def test_vn_female_full_names(self):
        result = names.get_names(language="vn", gender="F", name_type="full_name")
        self.assertGreater(len(result), 0)

    def test_vn_all_full_names(self):
        result = names.get_names(language="vn", gender=None, name_type="full_name")
        self.assertGreater(len(result), 0)

    def test_invalid_language(self):
        result = names.get_names(language="xx", gender="M", name_type="first_name")
        self.assertEqual(result, [])

    def test_invalid_name_type(self):
        result = names.get_names(language="ru", gender="M", name_type="nickname")
        self.assertEqual(result, [])

    def test_gender_case_insensitive(self):
        result_upper = names.get_names(language="ru", gender="M", name_type="first_name")
        result_lower = names.get_names(language="ru", gender="m", name_type="first_name")
        self.assertEqual(result_upper, result_lower)

    def test_language_case_insensitive(self):
        result = names.get_names(language="RU", gender="F", name_type="middle_name")
        self.assertGreater(len(result), 0)


class TestClearCache(unittest.TestCase):

    def setUp(self):
        # Set up a dummy entry in the cache before each test
        names._dictionary_cache = {}  # Reset cache before each test
        names._dictionary_cache["test_key"] = "test_value"

    def test_clear_cache_empties_dictionary_cache(self):
        # Ensure the cache initially contains the dummy key
        self.assertIn("test_key", names._dictionary_cache)
        names.clear_cache()
        # After clearing, the cache should be empty
        self.assertEqual(names._dictionary_cache, {})



if __name__ == '__main__':
    unittest.main()
