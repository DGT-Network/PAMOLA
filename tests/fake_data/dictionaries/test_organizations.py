import unittest
from pamola_core.fake_data.dictionaries import organizations



class TestGetEducationalInstitutions(unittest.TestCase):

    def setUp(self):
        # Clear the cache before each test to ensure test independence
        organizations._dictionary_cache.clear()

    def test_us_institutions(self):
        result = organizations.get_educational_institutions("US")
        self.assertGreater(len(result), 0)
        self.assertIn("Harvard University", result)

    def test_ru_institutions(self):
        result = organizations.get_educational_institutions("RU")
        self.assertGreater(len(result), 0)
        self.assertIn("МГТУ им. Баумана", result)

    def test_gb_institutions(self):
        result = organizations.get_educational_institutions("GB")
        self.assertGreater(len(result), 0)
        self.assertIn("University of Oxford", result)

    def test_unknown_country_code_returns_empty_list(self):
        result = organizations.get_educational_institutions("ZZ")
        self.assertEqual(result, [])

    def test_lowercase_country_code(self):
        result_upper = organizations.get_educational_institutions("GB")
        result_lower = organizations.get_educational_institutions("gb")
        self.assertEqual(result_upper, result_lower)

    def test_cache_usage(self):
        # First call should populate the cache
        organizations.get_educational_institutions("US")
        self.assertIn("US_edu", organizations._dictionary_cache)
        # Cache should not be overwritten on subsequent calls
        before = id(organizations._dictionary_cache["US_edu"])
        organizations.get_educational_institutions("US")
        after = id(organizations._dictionary_cache["US_edu"])
        self.assertEqual(before, after)


class TestGetBusinessOrganizations(unittest.TestCase):

    def setUp(self):
        # Clear the cache before each test to ensure test independence
        organizations._dictionary_cache.clear()

    def test_us_all_industries(self):
        result = organizations.get_business_organizations("US")
        self.assertGreater(len(result), 0)
        self.assertIn("Acme Technologies", result)
        self.assertIn("Heritage Financial", result)
        self.assertIn("Evergreen Stores", result)
        self.assertIn("Wellness Medical Center", result)

    def test_us_tech_industry(self):
        result = organizations.get_business_organizations("US", "tech")
        self.assertIn("Acme Technologies", result)
        self.assertNotIn("Heritage Financial", result)

    def test_us_unknown_industry_returns_empty_list(self):
        result = organizations.get_business_organizations("US", "unknown")
        self.assertEqual(result, [])

    def test_ru_all_industries(self):
        result = organizations.get_business_organizations("RU")
        self.assertGreater(len(result), 0)
        self.assertIn("ИнфоТех", result)
        self.assertIn("Финансовый Альянс", result)
        self.assertIn("Торговый Дом", result)

    def test_ru_tech_industry(self):
        result = organizations.get_business_organizations("RU", "tech")
        self.assertIn("ИнфоТех", result)
        self.assertNotIn("Финансовый Альянс", result)

    def test_unknown_country_code_returns_empty_list(self):
        result = organizations.get_business_organizations("ZZ")
        self.assertEqual(result, [])

    def test_lowercase_country_code(self):
        result_upper = organizations.get_business_organizations("US", "tech")
        result_lower = organizations.get_business_organizations("us", "tech")
        self.assertEqual(result_upper, result_lower)

    def test_cache_usage(self):
        organizations.get_business_organizations("US")
        self.assertIn("US_business", organizations._dictionary_cache)
        before = id(organizations._dictionary_cache["US_business"])
        organizations.get_business_organizations("US")
        after = id(organizations._dictionary_cache["US_business"])
        self.assertEqual(before, after)


class TestGetGovernmentOrganizations(unittest.TestCase):

    def setUp(self):
        # Clear the cache before each test to ensure test independence
        organizations._dictionary_cache.clear()

    def test_us_government_organizations(self):
        result = organizations.get_government_organizations("US")
        self.assertGreater(len(result), 0)
        self.assertIn("Department of State", result)
        self.assertIn("Environmental Protection Agency", result)

    def test_ru_government_organizations(self):
        result = organizations.get_government_organizations("RU")
        self.assertGreater(len(result), 0)
        self.assertIn("Министерство иностранных дел", result)
        self.assertIn("Федеральная налоговая служба", result)

    def test_unknown_country_code_returns_empty_list(self):
        result = organizations.get_government_organizations("ZZ")
        self.assertEqual(result, [])

    def test_lowercase_country_code(self):
        result_upper = organizations.get_government_organizations("US")
        result_lower = organizations.get_government_organizations("us")
        self.assertEqual(result_upper, result_lower)

    def test_cache_usage(self):
        organizations.get_government_organizations("RU")
        self.assertIn("RU_gov", organizations._dictionary_cache)
        before = id(organizations._dictionary_cache["RU_gov"])
        organizations.get_government_organizations("RU")
        after = id(organizations._dictionary_cache["RU_gov"])
        self.assertEqual(before, after)


class TestGetOrganizationNames(unittest.TestCase):

    def setUp(self):
        # Clear cache to ensure test isolation
        organizations._dictionary_cache.clear()

    def test_get_education_organizations_us(self):
        result = organizations.get_organization_names("US", "education")
        self.assertGreater(len(result), 0)
        self.assertIn("Harvard University", result)

    def test_get_business_organizations_us_tech(self):
        result = organizations.get_organization_names("US", "business", "tech")
        self.assertGreater(len(result), 0)
        self.assertIn("Acme Technologies", result)

    def test_get_business_organizations_ru_retail(self):
        result = organizations.get_organization_names("RU", "business", "retail")
        self.assertGreater(len(result), 0)
        self.assertIn("Торговый Дом", result)

    def test_get_government_organizations_ru(self):
        result = organizations.get_organization_names("RU", "government")
        self.assertGreater(len(result), 0)
        self.assertIn("Министерство иностранных дел", result)

    def test_get_organizations_with_lowercase_country_code(self):
        upper = organizations.get_organization_names("US", "education")
        lower = organizations.get_organization_names("us", "education")
        self.assertEqual(upper, lower)

    def test_get_business_all_industries(self):
        result = organizations.get_organization_names("US", "business")
        self.assertGreater(len(result), 20)  # Because it includes multiple industries

    def test_get_unknown_org_type_returns_empty(self):
        result = organizations.get_organization_names("US", "nonprofit")
        self.assertEqual(result, [])

    def test_get_unknown_country_returns_empty(self):
        result = organizations.get_organization_names("XX", "education")
        self.assertEqual(result, [])


class TestClearCache(unittest.TestCase):

    def setUp(self):
        # Set up a dummy entry in the cache before each test
        organizations._dictionary_cache = {}  # Reset cache before each test
        organizations._dictionary_cache["test_key"] = "test_value"

    def test_clear_cache_empties_dictionary_cache(self):
        # Ensure the cache initially contains the dummy key
        self.assertIn("test_key", organizations._dictionary_cache)
        organizations.clear_cache()
        # After clearing, the cache should be empty
        self.assertEqual(organizations._dictionary_cache, {})


if __name__ == "__main__":
    unittest.main()