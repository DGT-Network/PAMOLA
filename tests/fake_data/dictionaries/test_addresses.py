import unittest
from pamola_core.fake_data.dictionaries import addresses



class TestRussianAddresses(unittest.TestCase):

    def setUp(self):
        # Clear the cache to ensure each test starts fresh
        addresses._dictionary_cache.clear()

    def test_get_ru_street_names(self):
        streets = addresses.get_ru_street_names()
        self.assertIsInstance(streets, list)
        self.assertGreater(len(streets), 0)
        self.assertIn("Ленина", streets)
        self.assertIn("Пушкина", streets)

    def test_get_ru_cities(self):
        cities = addresses.get_ru_cities()
        self.assertIsInstance(cities, list)
        self.assertGreater(len(cities), 0)
        self.assertIn("Москва", cities)
        self.assertIn("Новосибирск", cities)

    def test_get_ru_regions(self):
        regions = addresses.get_ru_regions()
        self.assertIsInstance(regions, list)
        self.assertGreater(len(regions), 0)
        self.assertIn("Московская область", regions)
        self.assertIn("Республика Татарстан", regions)

    def test_get_ru_postal_codes(self):
        postcodes = addresses.get_ru_postal_codes()
        self.assertIsInstance(postcodes, dict)
        self.assertIn("Москва", postcodes)
        self.assertIsInstance(postcodes["Москва"], list)
        self.assertIn("101000", postcodes["Москва"])

    def test_caching_behavior(self):
        # Call each function once to populate the cache
        addresses.get_ru_street_names()
        addresses.get_ru_cities()
        addresses.get_ru_regions()
        addresses.get_ru_postal_codes()

        # Ensure cache keys exist
        self.assertIn("ru_streets", addresses._dictionary_cache)
        self.assertIn("ru_cities", addresses._dictionary_cache)
        self.assertIn("ru_regions", addresses._dictionary_cache)
        self.assertIn("ru_postcodes", addresses._dictionary_cache)

        # Check that the data returned is reused (same object)
        streets_id = id(addresses.get_ru_street_names())
        self.assertEqual(streets_id, id(addresses._dictionary_cache["ru_streets"]))


class TestEnglishAddresses(unittest.TestCase):

    def setUp(self):
        # Ensure the dictionary cache is cleared before each test
        addresses._dictionary_cache.clear()

    def test_get_us_street_names(self):
        streets = addresses.get_us_street_names()
        self.assertIsInstance(streets, list)
        self.assertGreater(len(streets), 0)
        self.assertIn("Main Street", streets)
        self.assertIn("Broadway", streets)

    def test_get_us_cities(self):
        cities = addresses.get_us_cities()
        self.assertIsInstance(cities, list)
        self.assertGreater(len(cities), 0)
        self.assertIn("New York", cities)
        self.assertIn("Los Angeles", cities)

    def test_get_us_states(self):
        states = addresses.get_us_states()
        self.assertIsInstance(states, dict)
        self.assertGreater(len(states), 0)
        self.assertEqual(states["California"], "CA")
        self.assertEqual(states["Texas"], "TX")
        self.assertIn("New York", states)

    def test_get_us_zip_codes(self):
        zipcodes = addresses.get_us_zip_codes()
        self.assertIsInstance(zipcodes, dict)
        self.assertIn("New York", zipcodes)
        self.assertIsInstance(zipcodes["New York"], list)
        self.assertIn("10001", zipcodes["New York"])
        self.assertIn("90001", zipcodes["Los Angeles"])

    def test_us_address_caching(self):
        # Call each function to populate the cache
        addresses.get_us_street_names()
        addresses.get_us_cities()
        addresses.get_us_states()
        addresses.get_us_zip_codes()

        # Ensure the cached keys exist
        self.assertIn("us_streets", addresses._dictionary_cache)
        self.assertIn("us_cities", addresses._dictionary_cache)
        self.assertIn("us_states", addresses._dictionary_cache)
        self.assertIn("us_zipcodes", addresses._dictionary_cache)

        # Check that the returned object is reused from the cache
        self.assertIs(addresses.get_us_states(), addresses._dictionary_cache["us_states"])


class TestVietnameseAddresses(unittest.TestCase):

    def setUp(self):
        # Clear the cache before each test to ensure isolation
        addresses._dictionary_cache.clear()

    def test_get_vn_street_names(self):
        streets = addresses.get_vn_street_names()
        self.assertIsInstance(streets, list)
        self.assertGreater(len(streets), 0)
        self.assertIn("Đường Lê Lợi", streets)
        self.assertIn("Đường Nguyễn Huệ", streets)

    def test_get_vn_cities(self):
        cities = addresses.get_vn_cities()
        self.assertIsInstance(cities, list)
        self.assertGreater(len(cities), 0)
        self.assertIn("Hà Nội", cities)
        self.assertIn("Hồ Chí Minh", cities)

    def test_get_vn_districts(self):
        districts = addresses.get_vn_districts()
        self.assertIsInstance(districts, dict)
        self.assertIn("Hà Nội", districts)
        self.assertIn("Ba Đình", districts["Hà Nội"])
        self.assertIn("Hồ Chí Minh", districts)
        self.assertIn("Quận 1", districts["Hồ Chí Minh"])

    def test_get_vn_postal_codes(self):
        postcodes = addresses.get_vn_postal_codes()
        self.assertIsInstance(postcodes, dict)
        self.assertIn("Hà Nội", postcodes)
        self.assertEqual(postcodes["Hà Nội"], "100000")
        self.assertEqual(postcodes["Hồ Chí Minh"], "700000")
        self.assertEqual(postcodes.get("Cần Thơ"), "900000")

    def test_vn_address_caching(self):
        # Call each function to trigger cache population
        addresses.get_vn_street_names()
        addresses.get_vn_cities()
        addresses.get_vn_districts()
        addresses.get_vn_postal_codes()

        # Verify that all expected keys are present in the cache
        self.assertIn("vn_streets", addresses._dictionary_cache)
        self.assertIn("vn_cities", addresses._dictionary_cache)
        self.assertIn("vn_districts", addresses._dictionary_cache)
        self.assertIn("vn_postcodes", addresses._dictionary_cache)

        # Confirm cache reuse
        self.assertIs(addresses.get_vn_districts(), addresses._dictionary_cache["vn_districts"])


class TestGetAddressComponent(unittest.TestCase):

    def test_get_address_component_street(self):
        # Test for street names for different countries
        ru_streets = addresses.get_address_component("RU", "street")
        us_streets = addresses.get_address_component("US", "street")
        vn_streets = addresses.get_address_component("VN", "street")

        # Test Russia (RU) streets
        self.assertIn("Ленина", ru_streets)
        self.assertIn("Пушкина", ru_streets)

        # Test US (US) streets
        self.assertIn("Main Street", us_streets)
        self.assertIn("Oak Street", us_streets)

        # Test Vietnam (VN) streets
        self.assertIn("Đường Lê Lợi", vn_streets)
        self.assertIn("Đường Nguyễn Huệ", vn_streets)

    def test_get_address_component_city(self):
        # Test for city names for different countries
        ru_cities = addresses.get_address_component("RU", "city")
        us_cities = addresses.get_address_component("US", "city")
        vn_cities = addresses.get_address_component("VN", "city")

        # Test Russia (RU) cities
        self.assertIn("Москва", ru_cities)
        self.assertIn("Санкт-Петербург", ru_cities)

        # Test US (US) cities
        self.assertIn("New York", us_cities)
        self.assertIn("Los Angeles", us_cities)

        # Test Vietnam (VN) cities
        self.assertIn("Hà Nội", vn_cities)
        self.assertIn("Hồ Chí Minh", vn_cities)

    def test_get_address_component_region(self):
        # Test for region names for different countries
        ru_regions = addresses.get_address_component("RU", "region")
        us_regions = addresses.get_address_component("US", "region")
        vn_regions = addresses.get_address_component("VN", "region")

        # Test Russia (RU) regions
        self.assertIn("Московская область", ru_regions)
        self.assertIn("Ленинградская область", ru_regions)

        # Test US (US) regions
        self.assertIn("Alabama", us_regions)
        self.assertIn("California", us_regions)

        # Test Vietnam (VN) regions (should return empty list as not implemented)
        self.assertEqual(vn_regions, [])

    def test_get_address_component_invalid_component_type(self):
        # Test for invalid component type
        invalid_street = addresses.get_address_component("RU", "invalid_component")
        self.assertEqual(invalid_street, [])

        invalid_city = addresses.get_address_component("US", "invalid_component")
        self.assertEqual(invalid_city, [])

    def test_get_address_component_invalid_country_code(self):
        # Test for invalid country code
        invalid_ru = addresses.get_address_component("ZZ", "street")
        self.assertEqual(invalid_ru, [])

        invalid_us = addresses.get_address_component("ZZ", "city")
        self.assertEqual(invalid_us, [])

        invalid_vn = addresses.get_address_component("ZZ", "region")
        self.assertEqual(invalid_vn, [])


class TestGetPostalCodeForCity(unittest.TestCase):

    def test_get_postal_code_for_city_ru(self):
        # Test for postal codes in Russia (RU)
        ru_postal_code = addresses.get_postal_code_for_city("RU", "Москва")
        self.assertEqual(ru_postal_code, "101000")

        ru_postal_code = addresses.get_postal_code_for_city("RU", "Санкт-Петербург")
        self.assertEqual(ru_postal_code, "190000")

        # Test with a city that doesn't exist in the list
        ru_postal_code_invalid = addresses.get_postal_code_for_city("RU", "Неизвестный Город")
        self.assertEqual(ru_postal_code_invalid, "")

    def test_get_postal_code_for_city_us(self):
        # Test for postal codes in the US (US)
        us_postal_code = addresses.get_postal_code_for_city("US", "New York")
        self.assertEqual(us_postal_code, "10001")

        us_postal_code = addresses.get_postal_code_for_city("US", "Los Angeles")
        self.assertEqual(us_postal_code, "90001")

        # Test with a city that doesn't exist in the list
        us_postal_code_invalid = addresses.get_postal_code_for_city("US", "Unknown City")
        self.assertEqual(us_postal_code_invalid, "")

    def test_get_postal_code_for_city_vn(self):
        # Test for postal codes in Vietnam (VN)
        vn_postal_code = addresses.get_postal_code_for_city("VN", "Hà Nội")
        self.assertEqual(vn_postal_code, "100000")

        vn_postal_code = addresses.get_postal_code_for_city("VN", "Hồ Chí Minh")
        self.assertEqual(vn_postal_code, "700000")

        # Test with a city that doesn't exist in the list
        vn_postal_code_invalid = addresses.get_postal_code_for_city("VN", "Unknown City")
        self.assertEqual(vn_postal_code_invalid, "")

    def test_get_postal_code_for_city_invalid_country_code(self):
        # Test with an invalid country code
        invalid_postal_code = addresses.get_postal_code_for_city("ZZ", "New York")
        self.assertEqual(invalid_postal_code, "")

    def test_get_postal_code_for_city_invalid_city(self):
        # Test with an invalid city name (valid country code but invalid city)
        invalid_postal_code = addresses.get_postal_code_for_city("US", "Nonexistent City")
        self.assertEqual(invalid_postal_code, "")


class TestClearCache(unittest.TestCase):

    def setUp(self):
        # Set up a dummy entry in the cache before each test
        addresses._dictionary_cache = {}  # Reset cache before each test
        addresses._dictionary_cache["test_key"] = "test_value"

    def test_clear_cache_empties_dictionary_cache(self):
        # Ensure the cache initially contains the dummy key
        self.assertIn("test_key", addresses._dictionary_cache)
        addresses.clear_cache()
        # After clearing, the cache should be empty
        self.assertEqual(addresses._dictionary_cache, {})


if __name__ == "__main__":
    unittest.main()