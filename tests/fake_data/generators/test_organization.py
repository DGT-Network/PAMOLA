import unittest
from unittest.mock import Mock, patch
from pamola_core.fake_data.generators.organization import OrganizationGenerator



class TestOrganizationGeneratorInit(unittest.TestCase):

    def test_init_with_default_config(self):
        # Test case for default values if no config is provided for certain fields
        default_config = {}
        org_gen_default = OrganizationGenerator(config=default_config)

        # Verify default values
        self.assertEqual(org_gen_default.organization_type, 'general')
        self.assertEqual(org_gen_default.dictionaries, {})
        self.assertEqual(org_gen_default.prefixes_dict, {})
        self.assertEqual(org_gen_default.suffixes_dict, {})
        self.assertEqual(org_gen_default.add_prefix_probability, 0.3)
        self.assertEqual(org_gen_default.add_suffix_probability, 0.5)
        self.assertEqual(org_gen_default.region, 'en')
        self.assertTrue(org_gen_default.preserve_type)  # Default is True
        self.assertIsNone(org_gen_default.industry)  # No industry provided
        self.assertIsNone(org_gen_default.prgn_generator)  # No key provided, should be None

    def test_init_with_custom_config(self):
        # Test case for custom values in the config
        custom_config = {
            'organization_type': 'tech',
            'dictionaries': {'en': ['Company', 'Corp']},
            'prefixes': {'en': ['Tech', 'Innovative']},
            'suffixes': {'en': ['LLC', 'Inc']},
            'add_prefix_probability': 0.4,
            'add_suffix_probability': 0.6,
            'region': 'us',
            'preserve_type': False,
            'industry': 'Software',
            'key': 'some-key'
        }
        org_gen_custom = OrganizationGenerator(config=custom_config)

        # Verify custom values
        self.assertEqual(org_gen_custom.organization_type, 'tech')
        self.assertEqual(org_gen_custom.dictionaries, {'en': ['Company', 'Corp']})
        self.assertEqual(org_gen_custom.prefixes_dict, {'en': ['Tech', 'Innovative']})
        self.assertEqual(org_gen_custom.suffixes_dict, {'en': ['LLC', 'Inc']})
        self.assertEqual(org_gen_custom.add_prefix_probability, 0.4)
        self.assertEqual(org_gen_custom.add_suffix_probability, 0.6)
        self.assertEqual(org_gen_custom.region, 'us')
        self.assertFalse(org_gen_custom.preserve_type)  # Set to False in config
        self.assertEqual(org_gen_custom.industry, 'Software')
        self.assertIsNotNone(org_gen_custom.prgn_generator)  # Should be initialized with the key


class TestDetectOrganizationType(unittest.TestCase):

    def setUp(self):
        # Set up the OrganizationGenerator with mock patterns
        self.config = {}
        self.org_gen = OrganizationGenerator(config=self.config)

        # Mock the _type_patterns to simulate different types
        self.org_gen._type_patterns = {
            'tech': Mock(search=Mock(return_value=False)),  # Set 'tech' to not match initially
            'health': Mock(search=Mock(return_value=False)),  # Set 'health' to not match initially
            'education': Mock(search=Mock(return_value=False)),  # Set 'education' to not match initially
        }

    def test_detect_organization_type_tech(self):
        # Test case where the organization type matches 'tech'
        org_name = 'Tech Innovations LLC'
        self.org_gen._type_patterns['tech'].search.return_value = True  # Simulate a match for 'tech'
        detected_type = self.org_gen.detect_organization_type(org_name)
        self.assertEqual(detected_type, 'tech')

    def test_detect_organization_type_health(self):
        # Test case where the organization type matches 'health'
        org_name = 'HealthCorp'
        self.org_gen._type_patterns['health'].search.return_value = True  # Simulate a match for 'health'
        detected_type = self.org_gen.detect_organization_type(org_name)
        self.assertEqual(detected_type, 'health')

    def test_detect_organization_type_default(self):
        # Test case where no pattern matches, should default to 'general'
        org_name = 'Unknown Organization'
        detected_type = self.org_gen.detect_organization_type(org_name)
        self.assertEqual(detected_type, 'general')

    def test_detect_organization_type_empty_name(self):
        # Test case for empty organization name, should default to 'general'
        org_name = ''
        detected_type = self.org_gen.detect_organization_type(org_name)
        self.assertEqual(detected_type, 'general')

    def test_detect_organization_type_none(self):
        # Test case for None as organization name, should default to 'general'
        org_name = None
        detected_type = self.org_gen.detect_organization_type(org_name)
        self.assertEqual(detected_type, 'general')


class TestOrganizationGeneratorGenerateOrgName(unittest.TestCase):
    def setUp(self):
        self.org_gen = OrganizationGenerator()
        self.org_gen.organization_type = 'general'
        self.org_gen.region = 'en'
        self.org_gen._org_names = {
            'general': {
                'en': ['GenOrg1', 'GenOrg2']
            },
            'tech': {
                'en': ['TechOrg1', 'TechOrg2']
            },
            'industry': {
                'en': ['IndustryOrg1']
            },
            'finance': {
                'us': ['FinanceOrgUS1'],
                'en': ['FinanceOrgEN1']
            }
        }
        self.org_gen.prgn_generator = None  # Default no PRGN

    def test_default_type_and_region(self):
        name = self.org_gen.generate_organization_name()
        self.assertIn(name, self.org_gen._org_names['general']['en'])

    def test_industry_type_with_industry_set(self):
        self.org_gen.organization_type = 'industry'
        self.org_gen.industry = 'finance'
        self.org_gen._org_names['finance'] = {'en': ['FinanceOrg1', 'FinanceOrg2']}
        name = self.org_gen.generate_organization_name()
        self.assertIn(name, self.org_gen._org_names['finance']['en'])

    def test_fallback_to_general_same_region(self):
        # No 'tech' orgs in 'us', fallback to general in 'us' if exists
        self.org_gen._org_names['general']['us'] = ['GeneralUS1']
        name = self.org_gen.generate_organization_name('tech', 'us')
        self.assertIn(name, self.org_gen._org_names['general']['us'])

    def test_fallback_to_type_en_region(self):
        # Empty finance 'us' to force fallback
        self.org_gen._org_names['finance']['us'] = []

        name = self.org_gen.generate_organization_name('finance', 'us')
        self.assertIn(name, self.org_gen._org_names['finance']['en'])

    def test_fallback_to_general_en(self):
        # Remove requested and fallback regions to force final fallback
        self.org_gen._org_names = {
            'general': {
                'en': ['GenOrgFallback']
            }
        }
        name = self.org_gen.generate_organization_name('nonexistent_type', 'nonexistent_region')
        self.assertIn(name, self.org_gen._org_names['general']['en'])

    def test_fallback_to_random_generic(self):
        # Empty _org_names to force generic fallback
        self.org_gen._org_names = {}
        name = self.org_gen.generate_organization_name('nonexistent', 'nonexistent')
        self.assertTrue(name.startswith("Organization "))
        self.assertTrue(name[13:].isdigit())
        self.assertEqual(len(name), len("Organization ") + 4)

    def test_prgn_generator_deterministic_selection(self):
        mock_rng = Mock()
        mock_rng.randint.return_value = 1
        self.org_gen.prgn_generator = Mock()
        self.org_gen.prgn_generator.get_random_by_value.return_value = mock_rng

        name = self.org_gen.generate_organization_name('general', 'en')
        expected_name = self.org_gen._org_names['general']['en'][1]  # index 1 per mock

        self.assertEqual(name, expected_name)
        self.org_gen.prgn_generator.get_random_by_value.assert_called_once_with(
            'general_en',
            salt='org-name-selection'
        )
        mock_rng.randint.assert_called_once_with(0, len(self.org_gen._org_names['general']['en']) - 1)


class TestOrganizationGeneratorAddPrefix(unittest.TestCase):

    def setUp(self):
        self.org_gen = OrganizationGenerator(config={})
        self.org_gen.organization_type = "tech"
        self.org_gen.region = "en"
        self.org_gen.industry = "finance"
        self.org_gen._prefixes = {
            "tech": {"en": ["Global", "NextGen"]},
            "finance": {"en": ["Fin", "Capital"]},
            "general": {
                "en": ["The", "United"],
                "fr": ["Le", "Les"]
            }
        }

    def test_add_prefix_exact_match(self):
        result = self.org_gen.add_prefix("Solutions", org_type="tech", region="en")
        self.assertIn(result.split()[0], ["Global", "NextGen"])

    def test_add_prefix_with_fallback_to_general_region(self):
        result = self.org_gen.add_prefix("Systems", org_type="unknown", region="fr")
        self.assertIn(result.split()[0], ["Le", "Les"])

    def test_add_prefix_with_fallback_to_type_en(self):
        result = self.org_gen.add_prefix("Analytics", org_type="finance", region="unknown")
        self.assertIn(result.split()[0], ["Fin", "Capital"])

    def test_add_prefix_with_fallback_to_general_en(self):
        result = self.org_gen.add_prefix("Group", org_type="unknown", region="unknown")
        self.assertIn(result.split()[0], ["The", "United"])

    def test_add_prefix_with_industry_override(self):
        self.org_gen.organization_type = "industry"
        result = self.org_gen.add_prefix("Holdings", region="en")
        self.assertIn(result.split()[0], ["Fin", "Capital"])

    def test_add_prefix_with_no_prefix_available(self):
        self.org_gen._prefixes = {}  # Clear prefixes
        result = self.org_gen.add_prefix("CoreTech", org_type="tech", region="en")
        self.assertEqual(result, "CoreTech")

    def test_add_prefix_with_prgn_generator(self):
        prgn_mock = Mock()
        prgn_mock.get_random_by_value.return_value.randint.return_value = 1  # Always select 2nd item
        self.org_gen.prgn_generator = prgn_mock

        result = self.org_gen.add_prefix("Logistics", org_type="tech", region="en")
        self.assertEqual(result, "NextGen Logistics")

        prgn_mock.get_random_by_value.assert_called_once_with("Logistics", salt="org-prefix-selection")

    def test_add_prefix_appends_space_if_missing(self):
        self.org_gen._prefixes["tech"]["en"] = ["TechLeader"]  # no space at end
        result = self.org_gen.add_prefix("Networks", org_type="tech", region="en")
        self.assertTrue(result.startswith("TechLeader "))
        self.assertEqual(result, "TechLeader Networks")


class TestOrganizationGeneratorAddSuffix(unittest.TestCase):

    def setUp(self):
        self.org_gen = OrganizationGenerator(config={})
        self.org_gen.organization_type = "tech"
        self.org_gen.region = "en"
        self.org_gen.industry = "finance"
        self.org_gen._suffixes = {
            "tech": {"en": ["Inc", "Technologies"]},
            "finance": {"en": ["Capital", "Group"]},
            "general": {
                "en": ["LLC", "Corp"],
                "fr": ["SARL", "Groupe"]
            }
        }

    def test_add_suffix_exact_match(self):
        result = self.org_gen.add_suffix("Quantum", org_type="tech", region="en")
        self.assertTrue(any(result.endswith(f" {sfx}") for sfx in ["Inc", "Technologies"]))

    def test_add_suffix_with_fallback_to_general_region(self):
        result = self.org_gen.add_suffix("Delta", org_type="unknown", region="fr")
        self.assertTrue(any(result.endswith(f" {sfx}") for sfx in ["SARL", "Groupe"]))

    def test_add_suffix_with_fallback_to_type_en(self):
        result = self.org_gen.add_suffix("Apex", org_type="finance", region="unknown")
        self.assertTrue(any(result.endswith(f" {sfx}") for sfx in ["Capital", "Group"]))

    def test_add_suffix_with_fallback_to_general_en(self):
        result = self.org_gen.add_suffix("Omega", org_type="unknown", region="unknown")
        self.assertTrue(any(result.endswith(f" {sfx}") for sfx in ["LLC", "Corp"]))

    def test_add_suffix_with_industry_override(self):
        self.org_gen.organization_type = "industry"
        result = self.org_gen.add_suffix("Vector", region="en")
        self.assertTrue(any(result.endswith(f" {sfx}") for sfx in ["Capital", "Group"]))

    def test_add_suffix_with_no_suffix_available(self):
        self.org_gen._suffixes = {}  # No suffixes defined
        result = self.org_gen.add_suffix("Nimbus", org_type="tech", region="en")
        self.assertEqual(result, "Nimbus")

    def test_add_suffix_with_prgn_generator(self):
        prgn_mock = Mock()
        prgn_mock.get_random_by_value.return_value.randint.return_value = 1  # Always select 2nd suffix
        self.org_gen.prgn_generator = prgn_mock

        result = self.org_gen.add_suffix("Echo", org_type="tech", region="en")
        self.assertEqual(result, "Echo Technologies")
        prgn_mock.get_random_by_value.assert_called_once_with("Echo", salt="org-suffix-selection")

    def test_add_suffix_appends_space_if_missing(self):
        self.org_gen._suffixes["tech"]["en"] = ["SecureTech"]  # No space prefix
        result = self.org_gen.add_suffix("Nova", org_type="tech", region="en")
        self.assertTrue(result.endswith(" SecureTech"))


class TestValidateOrganizationName(unittest.TestCase):

    def setUp(self):
        self.org_gen = OrganizationGenerator(config={})

    def test_valid_name(self):
        self.assertTrue(self.org_gen.validate_organization_name("TechCorp"))

    def test_valid_name_with_numbers_and_symbols(self):
        self.assertTrue(self.org_gen.validate_organization_name("X-123 Group"))

    def test_empty_string(self):
        self.assertFalse(self.org_gen.validate_organization_name(""))

    def test_none_value(self):
        self.assertFalse(self.org_gen.validate_organization_name(None))

    def test_non_string_input(self):
        self.assertFalse(self.org_gen.validate_organization_name(1234))
        self.assertFalse(self.org_gen.validate_organization_name(['Company']))
        self.assertFalse(self.org_gen.validate_organization_name({'name': 'Org'}))

    def test_too_short_name(self):
        self.assertFalse(self.org_gen.validate_organization_name("A"))

    def test_name_with_no_alpha_characters(self):
        self.assertFalse(self.org_gen.validate_organization_name("123456"))
        self.assertFalse(self.org_gen.validate_organization_name("!@#$%"))

    def test_name_with_mixed_content(self):
        self.assertTrue(self.org_gen.validate_organization_name("123 Corp!"))


class TestOrganizationGeneratorGenerate(unittest.TestCase):

    def setUp(self):
        self.org_gen = OrganizationGenerator(config={})
        self.org_gen.organization_type = "general"
        self.org_gen.region = "en"
        self.org_gen.add_prefix_probability = 1.0
        self.org_gen.add_suffix_probability = 1.0
        self.org_gen.generate_organization_name = Mock(return_value="BaseOrg")
        self.org_gen.add_prefix = Mock(return_value="Prefix BaseOrg")
        self.org_gen.add_suffix = Mock(return_value="BaseOrg Suffix")

    def test_generate_count_correct(self):
        result = self.org_gen.generate(5)
        self.assertEqual(len(result), 5)

    def test_generate_with_prefix_and_suffix_enabled(self):
        result = self.org_gen.generate(1, add_prefix=True, add_suffix=True)
        self.org_gen.generate_organization_name.assert_called_once_with("general", "en")
        self.org_gen.add_prefix.assert_called_once()
        self.org_gen.add_suffix.assert_called_once()
        self.assertEqual(result[0], "BaseOrg Suffix")

    def test_generate_with_prefix_and_suffix_disabled(self):
        result = self.org_gen.generate(1, add_prefix=False, add_suffix=False)
        self.org_gen.add_prefix.assert_not_called()
        self.org_gen.add_suffix.assert_not_called()
        self.assertEqual(result[0], "BaseOrg")

    def test_generate_with_random_prefix_suffix_probabilities(self):
        self.org_gen.add_prefix_probability = 0.5
        self.org_gen.add_suffix_probability = 0.7
        with patch("random.random", side_effect=[0.4, 0.6]):
            result = self.org_gen.generate(1)
            self.org_gen.add_prefix.assert_called_once()
            self.org_gen.add_suffix.assert_called_once()
            self.assertEqual(result[0], "BaseOrg Suffix")

    def test_generate_with_custom_org_type_and_region(self):
        result = self.org_gen.generate(1, organization_type="nonprofit", region="fr")
        self.org_gen.generate_organization_name.assert_called_once_with("nonprofit", "fr")

    def test_generate_multiple_calls(self):
        result = self.org_gen.generate(3, add_prefix=True, add_suffix=True)
        self.assertEqual(len(result), 3)
        self.assertEqual(self.org_gen.generate_organization_name.call_count, 3)
        self.assertEqual(self.org_gen.add_prefix.call_count, 3)
        self.assertEqual(self.org_gen.add_suffix.call_count, 3)


class TestOrganizationGeneratorGenerateLike(unittest.TestCase):

    def setUp(self):
        self.org_gen = OrganizationGenerator(config={
            "use_mapping": False,
            "preserve_type": True,
            "add_prefix_probability": 1.0,  # Always add prefix
            "add_suffix_probability": 1.0   # Always add suffix
        })
        self.org_gen._org_names = {
            "tech": {"en": ["BaseTech"]},
            "general": {"en": ["FallbackOrg"]}
        }
        self.org_gen.organization_type = "tech"
        self.org_gen.region = "en"

    def test_generate_like_empty_input(self):
        # Should return an empty string if input is empty
        result = self.org_gen.generate_like("")
        self.assertEqual(result, "")

    def test_generate_like_with_type_preservation(self):
        # Should detect org_type and region, generate name, and apply prefix and suffix
        self.org_gen.detect_organization_type = Mock(return_value="tech")
        self.org_gen._determine_region_from_name = Mock(return_value="en")
        self.org_gen.generate_organization_name = Mock(return_value="BaseTech")
        self.org_gen.add_prefix = Mock(return_value="Global BaseTech")
        self.org_gen.add_suffix = Mock(return_value="Global BaseTech Group")

        result = self.org_gen.generate_like("OriginalTechCorp")
        self.assertEqual(result, "Global BaseTech Group")

    def test_generate_like_with_param_override(self):
        # Should respect explicitly passed organization_type and region
        self.org_gen.generate_organization_name = Mock(return_value="BaseTech")
        self.org_gen.add_prefix = Mock(return_value="New BaseTech")
        self.org_gen.add_suffix = Mock(return_value="New BaseTech Inc.")

        result = self.org_gen.generate_like("AnyOrg", organization_type="tech", region="en")
        self.assertEqual(result, "New BaseTech Inc.")

    def test_generate_like_with_prgn_controlled_prefix_suffix(self):
        # Should add prefix/suffix based on PRGN-controlled randomness
        mock_rng = Mock()
        mock_rng.random.side_effect = [0.1, 0.05]  # Values less than probability → triggers add

        self.org_gen.prgn_generator = Mock()
        self.org_gen.prgn_generator.get_random_by_value.side_effect = lambda val, salt: mock_rng

        self.org_gen.detect_organization_type = Mock(return_value="tech")
        self.org_gen._determine_region_from_name = Mock(return_value="en")
        self.org_gen.generate_organization_name = Mock(return_value="BaseTech")
        self.org_gen.add_prefix = Mock(return_value="Super BaseTech")
        self.org_gen.add_suffix = Mock(return_value="Super BaseTech Corp")

        result = self.org_gen.generate_like("OriginalOrg")
        self.assertEqual(result, "Super BaseTech Corp")

    def test_generate_like_without_prefix_suffix(self):
        # Should generate base name without any prefix/suffix
        self.org_gen.add_prefix_probability = 0.0
        self.org_gen.add_suffix_probability = 0.0

        self.org_gen.generate_organization_name = Mock(return_value="CleanOrg")

        result = self.org_gen.generate_like("OriginalOrg")
        self.assertEqual(result, "CleanOrg")


class TestOrganizationGeneratorTransform(unittest.TestCase):
    def setUp(self):
        self.org_gen = OrganizationGenerator(config={})
        # Mock generate_like to control output
        self.org_gen.generate_like = Mock(side_effect=lambda v, **p: f"Transformed-{v}")

    def test_transform_basic(self):
        input_values = ["Org1", "Org2", "Org3"]
        result = self.org_gen.transform(input_values)
        expected = ["Transformed-Org1", "Transformed-Org2", "Transformed-Org3"]
        self.assertEqual(result, expected)
        # Verify generate_like called correct number of times with correct arguments
        self.assertEqual(self.org_gen.generate_like.call_count, len(input_values))
        for call_arg, original in zip(self.org_gen.generate_like.call_args_list, input_values):
            self.assertEqual(call_arg.args[0], original)

    def test_transform_with_params(self):
        input_values = ["OrgA", "OrgB"]
        params = {"organization_type": "nonprofit", "region": "fr"}
        result = self.org_gen.transform(input_values, **params)
        expected = [f"Transformed-{v}" for v in input_values]
        self.assertEqual(result, expected)
        # Verify params are passed correctly to generate_like
        for call_arg in self.org_gen.generate_like.call_args_list:
            for key, val in params.items():
                self.assertIn(key, call_arg.kwargs)
                self.assertEqual(call_arg.kwargs[key], val)


class TestOrganizationGeneratorValidate(unittest.TestCase):
    def setUp(self):
        self.org_gen = OrganizationGenerator(config={})
        # Mock validate_organization_name to control behavior
        self.org_gen.validate_organization_name = lambda v: v == "ValidOrg"

    def test_validate_valid_value(self):
        self.assertTrue(self.org_gen.validate("ValidOrg"))

    def test_validate_invalid_value(self):
        self.assertFalse(self.org_gen.validate("InvalidOrg"))

    def test_validate_empty_string(self):
        self.assertFalse(self.org_gen.validate(""))

    def test_validate_none_value(self):
        self.assertFalse(self.org_gen.validate(None))


if __name__ == '__main__':
    unittest.main()