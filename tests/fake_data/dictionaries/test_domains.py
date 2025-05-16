import unittest
from pamola_core.fake_data.dictionaries import domains



class TestGetCommonEmailDomains(unittest.TestCase):

    def setUp(self):
        # Clear cache before each test
        domains._dictionary_cache.clear()

    def test_get_common_email_domains_returns_expected_list(self):
        expected_domains = [
            "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
            "mail.ru", "yandex.ru", "protonmail.com", "icloud.com", "zoho.com",
            "gmx.com", "tutanota.com", "fastmail.com", "mailchimp.com", "inbox.com"
        ]
        result = domains.get_common_email_domains()
        self.assertEqual(result, expected_domains)

    def test_get_common_email_domains_uses_cache(self):
        # First call populates the cache
        result1 = domains.get_common_email_domains()

        # Modify the cache manually
        domains._dictionary_cache['email_domains'].append('fake.com')

        # Second call should reflect the modified cached result
        result2 = domains.get_common_email_domains()

        self.assertIn('fake.com', result2)
        self.assertEqual(result2, result1)  # Same reference


class TestGetBusinessEmailDomains(unittest.TestCase):

    def setUp(self):
        # Clear the cache before each test to avoid side effects
        domains._dictionary_cache.clear()

    def test_get_business_email_domains_returns_expected_list(self):
        expected_domains = [
            "company.com", "enterprise.com", "corp.com", "business.com", "firm.com",
            "inc.com", "llc.com", "agency.com", "consulting.com", "solutions.com",
            "international.com", "global.com", "group.com", "partners.com", "associates.com"
        ]
        result = domains.get_business_email_domains()
        self.assertEqual(result, expected_domains)

    def test_get_business_email_domains_uses_cache(self):
        # First call populates the cache
        result1 = domains.get_business_email_domains()

        # Modify the cached data directly
        domains._dictionary_cache['business_domains'].append('fakebiz.com')

        # Second call should return the cached version with the added domain
        result2 = domains.get_business_email_domains()

        self.assertIn('fakebiz.com', result2)
        self.assertEqual(result2, result1)  # Cached list is reused


class TestGetEducationalEmailDomains(unittest.TestCase):

    def setUp(self):
        # Clear the cache to isolate each test
        domains._dictionary_cache.clear()

    def test_get_educational_email_domains_returns_expected_list(self):
        expected_domains = [
            "edu", "ac.uk", "edu.au", "edu.cn", "ac.jp",
            "harvard.edu", "stanford.edu", "mit.edu", "oxford.ac.uk", "cambridge.ac.uk",
            "berkeley.edu", "columbia.edu", "princeton.edu", "yale.edu", "caltech.edu"
        ]
        result = domains.get_educational_email_domains()
        self.assertEqual(result, expected_domains)

    def test_get_educational_email_domains_uses_cache(self):
        # Call once to populate cache
        _ = domains.get_educational_email_domains()

        # Modify cached value
        domains._dictionary_cache['educational_domains'].append('test.edu')

        # Call again, should return modified cached version
        result = domains.get_educational_email_domains()
        self.assertIn('test.edu', result)
        self.assertEqual(result, domains._dictionary_cache['educational_domains'])


class TestGetUsernamePrefixes(unittest.TestCase):

    def setUp(self):
        # Clear cache before each test to isolate behavior
        domains._dictionary_cache.clear()

    def test_get_username_prefixes_returns_expected_list(self):
        expected_prefixes = [
            "user", "member", "client", "customer", "student",
            "professor", "doctor", "employee", "staff", "admin",
            "support", "help", "info", "contact", "service"
        ]
        result = domains.get_username_prefixes()
        self.assertEqual(result, expected_prefixes)

    def test_get_username_prefixes_uses_cache(self):
        # Prime the cache
        _ = domains.get_username_prefixes()
        # Modify the cache
        domains._dictionary_cache['username_prefixes'].append("testprefix")
        # Call again to verify it uses cached value
        result = domains.get_username_prefixes()
        self.assertIn("testprefix", result)
        self.assertEqual(result, domains._dictionary_cache['username_prefixes'])


class TestGetTLDsByCountry(unittest.TestCase):

    def setUp(self):
        # Clear the cache before each test
        domains._dictionary_cache.clear()

    def test_get_tlds_by_country_returns_expected_mapping(self):
        expected_mapping = {
            "ru": "ru",
            "us": "com",
            "uk": "co.uk",
            "ca": "ca",
            "au": "com.au",
            "de": "de",
            "fr": "fr",
            "jp": "jp",
            "cn": "cn",
            "br": "com.br",
            "in": "in",
            "it": "it",
            "es": "es",
            "nl": "nl",
            "se": "se"
        }
        result = domains.get_tlds_by_country()
        self.assertEqual(result, expected_mapping)

    def test_get_tlds_by_country_uses_cache(self):
        # Call the function to populate cache
        domains.get_tlds_by_country()
        # Modify cache directly
        domains._dictionary_cache['country_tlds']['xx'] = 'custom'
        # Ensure the function now returns modified data
        result = domains.get_tlds_by_country()
        self.assertIn('xx', result)
        self.assertEqual(result['xx'], 'custom')


class TestGetDomainByCountry(unittest.TestCase):

    def setUp(self):
        domains._dictionary_cache.clear()

    def test_valid_country_code(self):
        domains._dictionary_cache['country_tlds'] = {
            "ru": "ru",
            "us": "com"
        }
        result = domains.get_domain_by_country("ru")
        expected = ["mail.ru", "email.ru", "inbox.ru", "webmail.ru"]
        self.assertEqual(result, expected)

    def test_valid_country_code_us(self):
        domains._dictionary_cache['country_tlds'] = {
            "us": "com"
        }
        result = domains.get_domain_by_country("us")
        expected = ["mail.com", "email.com", "inbox.com", "webmail.com"]
        self.assertEqual(result, expected)

    def test_invalid_country_code_returns_empty_list(self):
        domains._dictionary_cache['country_tlds'] = {
            "ru": "ru",
            "us": "com"
        }
        result = domains.get_domain_by_country("xyz")
        self.assertEqual(result, [])

    def test_country_code_is_case_insensitive(self):
        domains._dictionary_cache['country_tlds'] = {
            "ru": "ru"
        }
        result = domains.get_domain_by_country("RU")
        expected = ["mail.ru", "email.ru", "inbox.ru", "webmail.ru"]
        self.assertEqual(result, expected)


class TestClearCache(unittest.TestCase):

    def setUp(self):
        # Set up a dummy entry in the cache before each test
        domains._dictionary_cache = {}  # Reset cache before each test
        domains._dictionary_cache["test_key"] = "test_value"

    def test_clear_cache_empties_dictionary_cache(self):
        # Ensure the cache initially contains the dummy key
        self.assertIn("test_key", domains._dictionary_cache)
        domains.clear_cache()
        # After clearing, the cache should be empty
        self.assertEqual(domains._dictionary_cache, {})



if __name__ == "__main__":
    unittest.main()