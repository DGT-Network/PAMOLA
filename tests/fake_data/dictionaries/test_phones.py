import unittest

from pamola_core.fake_data.dictionaries import phones


class TestGetCountryCodes(unittest.TestCase):

    def setUp(self):
        # Clear cache before each test to ensure isolation
        phones._dictionary_cache.clear()

    def test_returns_expected_country_codes(self):
        result = phones.get_country_codes()
        self.assertIsInstance(result, dict)
        self.assertIn("us", result)
        self.assertEqual(result["us"], "1")
        self.assertIn("vn", result)
        self.assertEqual(result["vn"], "84")

    def test_cache_is_populated(self):
        self.assertNotIn("country_codes", phones._dictionary_cache)
        phones.get_country_codes()
        self.assertIn("country_codes", phones._dictionary_cache)

    def test_cache_is_used_on_second_call(self):
        # First call populates the cache
        result1 = phones.get_country_codes()
        # Replace with a new dictionary to test if cache is used
        phones._dictionary_cache["country_codes"] = {"xx": "999"}
        result2 = phones.get_country_codes()
        self.assertEqual(result2, {"xx": "999"})


class TestGetOperatorCodes(unittest.TestCase):

    def setUp(self):
        phones._dictionary_cache.clear()

    def test_get_us_operator_codes(self):
        result = phones.get_operator_codes("us")
        self.assertIsInstance(result, list)
        self.assertIn("202", result)
        self.assertGreater(len(result), 0)

    def test_get_ru_operator_codes(self):
        result = phones.get_operator_codes("ru")
        self.assertIsInstance(result, list)
        self.assertIn("900", result)
        self.assertIn("495", result)

    def test_get_uk_operator_codes(self):
        result = phones.get_operator_codes("uk")
        self.assertIsInstance(result, list)
        self.assertIn("7400", result)
        self.assertIn("20", result)

    def test_get_ca_operator_codes(self):
        result = phones.get_operator_codes("ca")
        self.assertIn("204", result)
        self.assertIn("905", result)

    def test_get_de_operator_codes(self):
        result = phones.get_operator_codes("de")
        self.assertIn("151", result)
        self.assertIn("179", result)

    def test_get_fr_operator_codes(self):
        result = phones.get_operator_codes("fr")
        self.assertEqual(result, ["6", "7"])

    def test_unknown_country_returns_empty_list(self):
        result = phones.get_operator_codes("zz")
        self.assertEqual(result, [])

    def test_cache_is_used_on_second_call(self):
        # Populate and override cache
        phones.get_operator_codes("us")
        phones._dictionary_cache["operator_codes_us"] = ["999"]
        result = phones.get_operator_codes("us")
        self.assertEqual(result, ["999"])

    def test_cache_key_case_insensitive(self):
        result_lower = phones.get_operator_codes("us")
        result_upper = phones.get_operator_codes("US")
        self.assertEqual(result_lower, result_upper)


class TestGetPhoneFormats(unittest.TestCase):

    def setUp(self):
        # Clear cache before each test for isolation
        phones._dictionary_cache.clear()

    def test_returns_dict(self):
        result = phones.get_phone_formats()
        self.assertIsInstance(result, dict)

    def test_contains_common_format_keys(self):
        result = phones.get_phone_formats()
        self.assertIn("e164", result)
        self.assertIn("us", result)
        self.assertIn("uk", result)
        self.assertIn("ru", result)
        self.assertIn("local", result)
        self.assertIn("numeric", result)

    def test_format_strings_contain_placeholders(self):
        result = phones.get_phone_formats()
        self.assertIn("CC", result["e164"])
        self.assertIn("AAA", result["us"])
        self.assertIn("XXXX", result["us"])

    def test_cache_is_populated(self):
        self.assertNotIn("phone_formats", phones._dictionary_cache)
        phones.get_phone_formats()
        self.assertIn("phone_formats", phones._dictionary_cache)

    def test_cache_is_used_on_second_call(self):
        phones.get_phone_formats()
        phones._dictionary_cache["phone_formats"] = {"test_format": "+XX XXX"}
        result = phones.get_phone_formats()
        self.assertEqual(result, {"test_format": "+XX XXX"})


class TestGetPhoneFormatForCountry(unittest.TestCase):

    def setUp(self):
        # Clear cache before each test
        phones._dictionary_cache.clear()

    def test_known_country_us(self):
        result = phones.get_phone_format_for_country("us")
        self.assertEqual(result, "+CC (AAA) XXX-XXXX")

    def test_known_country_ru(self):
        result = phones.get_phone_format_for_country("RU")
        self.assertEqual(result, "+CC (AAA) XXX-XX-XX")

    def test_known_country_jp(self):
        result = phones.get_phone_format_for_country("jp")
        self.assertEqual(result, "+CC AA XXXX XXXX")

    def test_unknown_country_defaults_to_intl_spaces(self):
        result = phones.get_phone_format_for_country("xx")
        self.assertEqual(result, "+CC AAA XXX XXX")

    def test_case_insensitivity(self):
        lower = phones.get_phone_format_for_country("fr")
        upper = phones.get_phone_format_for_country("FR")
        self.assertEqual(lower, upper)

    def test_cache_usage(self):
        phones.get_phone_format_for_country("de")
        self.assertIn("phone_formats", phones._dictionary_cache)


class TestGetPhoneLengthRanges(unittest.TestCase):

    def setUp(self):
        # Clear the cache before each test to ensure independence
        phones._dictionary_cache.clear()

    def test_contains_expected_countries(self):
        ranges = phones.get_phone_length_ranges()
        self.assertIn("us", ranges)
        self.assertIn("cn", ranges)
        self.assertIn("default", ranges)

    def test_correct_values_for_known_countries(self):
        ranges = phones.get_phone_length_ranges()
        self.assertEqual(ranges["us"], (10, 10))
        self.assertEqual(ranges["de"], (10, 11))
        self.assertEqual(ranges["fr"], (9, 9))
        self.assertEqual(ranges["cn"], (11, 11))

    def test_default_range(self):
        ranges = phones.get_phone_length_ranges()
        self.assertEqual(ranges["default"], (9, 12))

    def test_cache_is_used(self):
        # Trigger population of cache
        ranges_first = phones.get_phone_length_ranges()
        self.assertIn("phone_lengths", phones._dictionary_cache)

        # Save the object ID to verify it’s reused
        first_id = id(phones._dictionary_cache["phone_lengths"])
        phones.get_phone_length_ranges()
        second_id = id(phones._dictionary_cache["phone_lengths"])

        self.assertEqual(first_id, second_id)


class TestClearCache(unittest.TestCase):

    def setUp(self):
        # Set up a dummy entry in the cache before each test
        phones._dictionary_cache = {}  # Reset cache before each test
        phones._dictionary_cache["test_key"] = "test_value"

    def test_clear_cache_empties_dictionary_cache(self):
        # Ensure the cache initially contains the dummy key
        self.assertIn("test_key", phones._dictionary_cache)
        phones.clear_cache()
        # After clearing, the cache should be empty
        self.assertEqual(phones._dictionary_cache, {})


if __name__ == "__main__":
    unittest.main()