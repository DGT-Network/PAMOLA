import unittest
from unittest.mock import patch

from pamola_core.fake_data.dictionaries import domains
from pamola_core.fake_data.generators.email import EmailGenerator



class TestEmailGeneratorInit(unittest.TestCase):

    def test_init_with_default_config(self):
        # Testing initialization with default config
        config = {}
        email_generator = EmailGenerator(config=config)

        # Check default values for attributes
        self.assertEqual(email_generator.domains, [])
        self.assertIsNone(email_generator.format)
        self.assertEqual(email_generator.format_ratio, {})
        self.assertTrue(email_generator.validate_source)
        self.assertEqual(email_generator.handle_invalid_email, 'generate_new')
        self.assertIsNone(email_generator.nicknames_dict)
        self.assertEqual(email_generator.max_length, 254)
        self.assertEqual(email_generator.max_local_part_length, 64)
        self.assertEqual(email_generator.max_domain_length, 255)
        self.assertEqual(email_generator.separator_options, [".", "_", "-", ""])
        self.assertEqual(email_generator.number_suffix_probability, 0.4)
        self.assertEqual(email_generator.preserve_domain_ratio, 0.5)
        self.assertEqual(email_generator.business_domain_ratio, 0.2)
        self.assertIsInstance(email_generator._domain_list, list)  # Check if _domain_list is a list
        self.assertEqual(email_generator._common_domains, domains.get_common_email_domains())
        self.assertEqual(email_generator._business_domains, domains.get_business_email_domains())
        self.assertEqual(email_generator._educational_domains, domains.get_educational_email_domains())
        self.assertIsInstance(email_generator._nicknames, list)  # Check if _nicknames is a list
        self.assertIsNone(email_generator.prgn_generator)  # PRGN generator should be None
        self.assertIsNotNone(email_generator.email_pattern)  # Check that email_pattern is set
        self.assertEqual(email_generator.allowed_local_chars, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_+")

    def test_init_with_custom_config(self):
        # Testing initialization with a custom config
        config = {
            'domains': ['example.com', 'test.org'],
            'format': 'custom_format',
            'format_ratio': {'custom_format': 0.8},
            'validate_source': False,
            'handle_invalid_email': 'ignore',
            'nicknames_dict': None,
            'max_length': 200,
            'max_local_part_length': 50,
            'max_domain_length': 100,
            'separator_options': ['_', '.'],
            'number_suffix_probability': 0.5,
            'preserve_domain_ratio': 0.7,
            'business_domain_ratio': 0.3,
            'key': 'some_key'
        }
        email_generator = EmailGenerator(config=config)

        # Check custom values for attributes
        self.assertEqual(email_generator.domains, ['example.com', 'test.org'])
        self.assertEqual(email_generator.format, 'custom_format')
        self.assertEqual(email_generator.format_ratio, {'custom_format': 0.8})
        self.assertFalse(email_generator.validate_source)
        self.assertEqual(email_generator.handle_invalid_email, 'ignore')
        self.assertIsNone(email_generator.nicknames_dict)
        self.assertEqual(email_generator.max_length, 200)
        self.assertEqual(email_generator.max_local_part_length, 50)
        self.assertEqual(email_generator.max_domain_length, 100)
        self.assertEqual(email_generator.separator_options, ['_', '.'])
        self.assertEqual(email_generator.number_suffix_probability, 0.5)
        self.assertEqual(email_generator.preserve_domain_ratio, 0.7)
        self.assertEqual(email_generator.business_domain_ratio, 0.3)
        self.assertIsNotNone(email_generator.prgn_generator)  # PRGN generator should be initialized
        self.assertEqual(email_generator.prgn_generator.global_seed, 1600873849)  # Check PRGN seed value

        # Check if domains are categorized
        self.assertIsInstance(email_generator._domain_list, list)
        self.assertIsInstance(email_generator._common_domains, list)
        self.assertIsInstance(email_generator._business_domains, list)
        self.assertIsInstance(email_generator._educational_domains, list)
        self.assertIsInstance(email_generator._nicknames, list)  # Check if _nicknames is a list

    def test_init_with_invalid_key(self):
        # Testing initialization with an invalid key
        config = {
            'key': ''
        }
        email_generator = EmailGenerator(config=config)

        # Check if PRGN generator is None when key is invalid
        self.assertIsNone(email_generator.prgn_generator)

    def test_init_with_no_domains(self):
        # Testing initialization with no domains specified
        config = {
            'domains': []
        }
        email_generator = EmailGenerator(config=config)

        # Check if the domain list is empty
        self.assertEqual(email_generator.domains, [])
        self.assertIsInstance(email_generator._domain_list, list)  # Check if _domain_list is still a list


class TestEmailValidation(unittest.TestCase):
    def setUp(self):
        self.generator = EmailGenerator(config={})

    def test_valid_email(self):
        self.assertTrue(self.generator.validate_email("user.name@example.com"))

    def test_empty_email(self):
        self.assertFalse(self.generator.validate_email(""))

    def test_non_string_email(self):
        self.assertFalse(self.generator.validate_email(123))  # int
        self.assertFalse(self.generator.validate_email(None))  # NoneType

    def test_invalid_format_missing_at(self):
        self.assertFalse(self.generator.validate_email("username.example.com"))

    def test_invalid_format_double_at(self):
        self.assertFalse(self.generator.validate_email("user@@example.com"))

    def test_too_long_email(self):
        long_email = "a" * 245 + "@example.com"  # >254 characters
        self.assertFalse(self.generator.validate_email(long_email))

    def test_local_part_too_long(self):
        long_local = "a" * 65
        email = f"{long_local}@example.com"
        self.assertFalse(self.generator.validate_email(email))

    def test_domain_too_long(self):
        long_domain = "a" * 256
        email = f"user@{long_domain}.com"
        self.assertFalse(self.generator.validate_email(email))

    def test_domain_without_dot(self):
        self.assertFalse(self.generator.validate_email("user@domaincom"))

    def test_domain_starts_with_dash(self):
        self.assertFalse(self.generator.validate_email("user@-domain.com"))

    def test_domain_ends_with_dash(self):
        self.assertFalse(self.generator.validate_email("user@domain-.com"))

    def test_valid_edge_case(self):
        email = "a@b.co"
        self.assertTrue(self.generator.validate_email(email))  # Valid minimal email


class TestExtractDomain(unittest.TestCase):
    def setUp(self):
        self.generator = EmailGenerator(config={})

    def test_valid_email(self):
        self.assertEqual(self.generator.extract_domain("user@example.com"), "example.com")

    def test_email_with_subdomain(self):
        self.assertEqual(self.generator.extract_domain("admin@mail.service.org"), "mail.service.org")

    def test_empty_string(self):
        self.assertIsNone(self.generator.extract_domain(""))

    def test_none_input(self):
        self.assertIsNone(self.generator.extract_domain(None))

    def test_non_string_input(self):
        self.assertIsNone(self.generator.extract_domain(123))
        self.assertIsNone(self.generator.extract_domain(['a@b.com']))

    def test_missing_at_symbol(self):
        self.assertIsNone(self.generator.extract_domain("user.example.com"))

    def test_multiple_at_symbols(self):
        self.assertIsNone(self.generator.extract_domain("user@@example.com"))

    def test_email_with_trailing_spaces(self):
        self.assertEqual(self.generator.extract_domain(" user@example.com ".strip()), "example.com")

    def test_email_with_plus_sign(self):
        self.assertEqual(self.generator.extract_domain("user+tag@example.com"), "example.com")


class TestParseEmailFormat(unittest.TestCase):

    def setUp(self):
        # Mock top 5 domains
        self.generator = EmailGenerator(config={})
        self.generator._domain_list = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'live.com']

    def test_invalid_email_none(self):
        self.assertEqual(self.generator.parse_email_format(None), "unknown")

    def test_invalid_email_not_string(self):
        self.assertEqual(self.generator.parse_email_format(12345), "unknown")

    def test_invalid_email_no_at_symbol(self):
        self.assertEqual(self.generator.parse_email_format("invalid.email.com"), "unknown")

    def test_name_surname_format(self):
        email = "john_smith@gmail.com"
        self.assertEqual(self.generator.parse_email_format(email), "name_surname")

    def test_surname_name_format(self):
        email = "smith_john@gmail.com"
        self.assertEqual(self.generator.parse_email_format(email), "surname_name")

    def test_name_surname_with_numeric_suffix(self):
        email = "john_smith_99@gmail.com"
        self.assertEqual(self.generator.parse_email_format(email), "name_surname")

    def test_nickname_with_digits(self):
        email = "coolguy123@yahoo.com"
        self.assertEqual(self.generator.parse_email_format(email), "nickname")

    def test_nickname_short_length(self):
        email = "jd@outlook.com"
        self.assertEqual(self.generator.parse_email_format(email), "nickname")

    def test_existing_domain(self):
        email = "randomname@live.com"
        self.assertEqual(self.generator.parse_email_format(email), "existing_domain")

    def test_unknown_format(self):
        email = "longlocalpartthatdoesntmatchany@unknowncustomdomain.com"
        self.assertEqual(self.generator.parse_email_format(email), "unknown")


class TestEmailGeneratorGenerate(unittest.TestCase):

    def setUp(self):
        self.generator = EmailGenerator(config={})

    @patch.object(EmailGenerator, '_generate_email')
    def test_generate_multiple_emails(self, mock_generate_email):
        # Setup mock return
        mock_generate_email.side_effect = lambda **kwargs: f"{kwargs.get('first_name', 'john')}.{kwargs.get('last_name', 'doe')}@example.com"

        result = self.generator.generate(3, first_name="alice", last_name="smith", domain="example.com")

        self.assertEqual(len(result), 3)
        self.assertTrue(all(email.endswith("@example.com") for email in result))
        self.assertEqual(mock_generate_email.call_count, 3)

    @patch.object(EmailGenerator, '_generate_email')
    def test_generate_with_default_params(self, mock_generate_email):
        mock_generate_email.return_value = "default@example.com"

        result = self.generator.generate(2)

        self.assertEqual(result, ["default@example.com", "default@example.com"])
        mock_generate_email.assert_called_with()

    @patch.object(EmailGenerator, '_generate_email')
    def test_generate_with_original_email(self, mock_generate_email):
        mock_generate_email.return_value = "test@original.com"

        result = self.generator.generate(1, original_email="some@original.com")

        self.assertEqual(result, ["test@original.com"])
        mock_generate_email.assert_called_once_with(original_email="some@original.com")

    def test_generate_zero_count(self):
        result = self.generator.generate(0)
        self.assertEqual(result, [])


class TestEmailGeneratorGenerateLike(unittest.TestCase):

    def setUp(self):
        self.generator = EmailGenerator(config={
            'validate_source': True,
            'handle_invalid_email': 'generate_new',
            'domains': ['synthetic.com']
        })

    @patch.object(EmailGenerator, 'validate_email', return_value=True)
    @patch.object(EmailGenerator, 'extract_domain', return_value='original.com')
    @patch.object(EmailGenerator, '_generate_from_name_components', return_value='john.smith')
    def test_generate_like_existing_domain_format_valid_email(self, mock_gen_local, mock_extract, mock_validate):
        result = self.generator.generate_like(
            "someone@original.com",
            first_name="John",
            last_name="Smith",
            format="existing_domain"
        )
        self.assertEqual(result, "john.smith@original.com")
        mock_validate.assert_called_once()
        mock_extract.assert_called_once_with("someone@original.com")
        mock_gen_local.assert_called_once_with("John", "Smith", "name_surname")

    @patch.object(EmailGenerator, 'validate_email', return_value=True)
    @patch.object(EmailGenerator, '_generate_email', return_value='generated@synthetic.com')
    def test_generate_like_non_existing_format_valid_email(self, mock_generate, mock_validate):
        result = self.generator.generate_like(
            "person@something.com",
            first_name="Ann",
            last_name="Lee",
            format="nickname"
        )
        self.assertEqual(result, 'generated@synthetic.com')
        mock_generate.assert_called_once()

    @patch.object(EmailGenerator, 'validate_email', return_value=False)
    @patch.object(EmailGenerator, '_generate_email', return_value='fallback@synthetic.com')
    def test_generate_like_invalid_email_generate_new(self, mock_generate, mock_validate):
        self.generator.handle_invalid_email = "generate_new"
        result = self.generator.generate_like("not-an-email", first_name="Sam", last_name="Park")
        self.assertEqual(result, "fallback@synthetic.com")
        mock_generate.assert_called_once()

    @patch.object(EmailGenerator, '_generate_email', return_value='default@synthetic.com')
    def test_generate_like_invalid_email_generate_with_default_domain(self, mock_generate):
        self.generator.handle_invalid_email = "generate_with_default_domain"
        self.generator.validate_source = True

        with patch.object(EmailGenerator, 'validate_email', return_value=False):
            result = self.generator.generate_like("xxx", first_name="Ana", last_name="Diaz")
            self.assertEqual(result, "default@synthetic.com")
            self.assertEqual(mock_generate.call_args[1]["domain"], "synthetic.com")

    def test_generate_like_invalid_email_keep_empty(self):
        self.generator.handle_invalid_email = "keep_empty"
        self.generator.validate_source = True

        with patch.object(EmailGenerator, 'validate_email', return_value=False):
            result = self.generator.generate_like("invalid_email")
            self.assertEqual(result, "")

    @patch.object(EmailGenerator, '_parse_full_name', return_value={"first_name": "Lara", "last_name": "Croft"})
    @patch.object(EmailGenerator, '_generate_email', return_value='lara.croft@synthetic.com')
    def test_generate_like_with_full_name_parsing(self, mock_generate, mock_parse):
        result = self.generator.generate_like("lara@tomb.com", full_name="Lara Croft", name_format="FL")
        self.assertEqual(result, "lara.croft@synthetic.com")
        mock_parse.assert_called_once_with("Lara Croft", "FL")
        mock_generate.assert_called_once()


class TestEmailGeneratorTransform(unittest.TestCase):

    def setUp(self):
        self.generator = EmailGenerator(config={
            'validate_source': True,
            'handle_invalid_email': 'generate_new',
            'domains': ['synthetic.com']
        })

    @patch.object(EmailGenerator, 'generate_like', return_value='generated@synthetic.com')
    def test_transform_valid_values(self, mock_generate_like):
        values = ["user1@domain.com", "user2@domain.com"]
        params = {"first_name": "John", "last_name": "Doe", "format": "nickname"}

        result = self.generator.transform(values, **params)

        mock_generate_like.assert_any_call("user1@domain.com", first_name="John", last_name="Doe", format="nickname")
        mock_generate_like.assert_any_call("user2@domain.com", first_name="John", last_name="Doe", format="nickname")

        self.assertEqual(result, ['generated@synthetic.com', 'generated@synthetic.com'])
        self.assertEqual(mock_generate_like.call_count, 2)

    @patch.object(EmailGenerator, 'generate_like', return_value='generated2@synthetic.com')
    def test_transform_empty_values(self, mock_generate_like):
        values = []
        params = {"first_name": "Jane", "last_name": "Smith"}

        result = self.generator.transform(values, **params)

        # Check that an empty list returns an empty list
        self.assertEqual(result, [])
        mock_generate_like.assert_not_called()

    @patch.object(EmailGenerator, 'generate_like', side_effect=lambda value, **params: f"generated_{value.split('@')[0]}@synthetic.com")
    def test_transform_with_side_effect(self, mock_generate_like):
        values = ["example1@domain.com", "example2@domain.com"]
        params = {"format": "existing_domain"}

        result = self.generator.transform(values, **params)

        # Check that the result is returned in the correct format.
        self.assertEqual(result, ["generated_example1@synthetic.com", "generated_example2@synthetic.com"])
        mock_generate_like.assert_any_call("example1@domain.com", format="existing_domain")
        mock_generate_like.assert_any_call("example2@domain.com", format="existing_domain")
        self.assertEqual(mock_generate_like.call_count, 2)

    @patch.object(EmailGenerator, 'generate_like', return_value='generated_with_default@synthetic.com')
    def test_transform_invalid_email(self, mock_generate_like):
        values = ["invalid_email1", "invalid_email2"]
        params = {"handle_invalid_email": "generate_new"}

        result = self.generator.transform(values, **params)

        # Check that invalid emails are handled correctly.
        self.assertEqual(result, ['generated_with_default@synthetic.com', 'generated_with_default@synthetic.com'])
        mock_generate_like.assert_any_call("invalid_email1", handle_invalid_email="generate_new")
        mock_generate_like.assert_any_call("invalid_email2", handle_invalid_email="generate_new")
        self.assertEqual(mock_generate_like.call_count, 2)

    @patch.object(EmailGenerator, 'generate_like', return_value='generated@synthetic.com')
    def test_transform_multiple_params(self, mock_generate_like):
        values = ["user1@domain.com", "user2@domain.com"]
        params = {
            "first_name": "Alice",
            "last_name": "Brown",
            "format": "surname_name",
            "local_format": "nickname"
        }

        result = self.generator.transform(values, **params)

        # Check that the values are generated correctly with all parameters
        self.assertEqual(result, ['generated@synthetic.com', 'generated@synthetic.com'])

        # Check that `generate_like` was called at least once with "user1@domain.com"
        mock_generate_like.assert_any_call("user1@domain.com", first_name="Alice", last_name="Brown",
                                           format="surname_name", local_format="nickname")

        # Check that `generate_like` was called at least once with "user2@domain.com"
        mock_generate_like.assert_any_call("user2@domain.com", first_name="Alice", last_name="Brown",
                                           format="surname_name", local_format="nickname")

        self.assertEqual(mock_generate_like.call_count, 2)


class TestEmailValidator(unittest.TestCase):

    @patch.object(EmailGenerator, 'validate_email', return_value=True)
    def test_validate_valid_email(self, mock_validate_email):
        # Test a valid email address
        validator = EmailGenerator(config={})
        value = "valid_email@domain.com"

        result = validator.validate(value)

        self.assertTrue(result)
        mock_validate_email.assert_called_once_with(value)

    @patch.object(EmailGenerator, 'validate_email', return_value=False)
    def test_validate_invalid_email(self, mock_validate_email):
        # Test an invalid email address
        validator = EmailGenerator(config={})
        value = "invalid_email.com"

        result = validator.validate(value)

        self.assertFalse(result)
        mock_validate_email.assert_called_once_with(value)


if __name__ == '__main__':
    unittest.main()