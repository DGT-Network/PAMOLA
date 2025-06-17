import random
import unittest
from unittest.mock import patch, Mock
from pamola_core.fake_data.generators.name import NameGenerator



class TestNameGeneratorInit(unittest.TestCase):

    def test_init_with_default_config(self):
        # Initialize with an empty config (config = {})
        config = {}

        # Create the instance
        generator = NameGenerator(config)

        # Test the instance variables are correctly set to their default values
        self.assertEqual(generator.language, "en")  # Default value is "en"
        self.assertIsNone(generator.gender)  # Default value is None
        self.assertEqual(generator.format, "FL")  # Default value is "FL"
        self.assertFalse(generator.use_faker)  # Default value is False
        self.assertEqual(generator.case, "title")  # Default value is "title"
        self.assertFalse(generator.gender_from_name)  # Default value is False
        self.assertEqual(generator.f_m_ratio, 0.5)  # Default value is 0.5
        self.assertEqual(generator.dictionaries, {})  # Default value is empty dictionary
        self.assertIsNone(generator.prgn_generator)  # Default value is None
        self.assertIsNone(generator.mapping_store)  # Default value is None
        self.assertFalse(generator.use_mapping)  # Default value is False

    def test_init_with_custom_config(self):
        # Testing initialization with a custom config
        config = {
            'language': 'ru',
            'gender': 'female',
            'format': 'LF',
            'use_faker': True,
            'case': 'upper',
            'gender_from_name': True,
            'f_m_ratio': 0.8,
            'dictionaries': {'first_names': ['Anna'], 'last_names': ['Ivanova']},
            'key': 'my_secret_key',
            'context_salt': 'salt_value',
            'mapping_store': Mock(),
            'use_mapping': True
        }

        generator = NameGenerator(config)

        # Validate all custom values
        self.assertEqual(generator.language, 'ru')
        self.assertEqual(generator.gender, 'female')
        self.assertEqual(generator.format, 'LF')
        self.assertEqual(generator.case, 'upper')
        self.assertTrue(generator.gender_from_name)
        self.assertEqual(generator.f_m_ratio, 0.8)
        self.assertEqual(generator.dictionaries, {'first_names': ['Anna'], 'last_names': ['Ivanova']})
        self.assertTrue(generator.use_mapping)
        self.assertEqual(generator.mapping_store, config['mapping_store'])


class TestNameGeneratorGenerate(unittest.TestCase):

    def setUp(self):
        self.generator = NameGenerator(config={
            'language': 'en',
            'gender': 'M',
            'format': 'FL',
            'f_m_ratio': 0.5,
            'use_faker': False  # Disable Faker for default testing
        })

    @patch.object(NameGenerator, 'generate_full_name')
    def test_generate_default_behavior(self, mock_generate_full_name):
        # Test basic name generation with default settings
        mock_generate_full_name.return_value = "John Doe"
        result = self.generator.generate(3)

        self.assertEqual(result, ["John Doe", "John Doe", "John Doe"])
        self.assertEqual(mock_generate_full_name.call_count, 3)

    @patch.object(NameGenerator, 'generate_full_name')
    def test_generate_with_overridden_gender_and_language(self, mock_generate_full_name):
        # Test name generation with overridden gender and language
        mock_generate_full_name.return_value = "Anna Ivanova"
        result = self.generator.generate(2, gender="F", language="ru", format="LF")

        self.assertEqual(result, ["Anna Ivanova", "Anna Ivanova"])
        mock_generate_full_name.assert_any_call("F", "ru", "LF")

    @patch.object(NameGenerator, 'generate_full_name')
    def test_generate_with_seed_produces_consistent_results(self, mock_generate_full_name):
        # Ensure reproducibility when seed is provided
        def side_effect(gender, language, format_str):
            return f"Name_{gender}"

        mock_generate_full_name.side_effect = side_effect

        gen1 = self.generator.generate(2, seed=42, gender=None)
        random.seed(None)  # Reset the random state
        gen2 = self.generator.generate(2, seed=42, gender=None)

        self.assertEqual(gen1, gen2)

    @patch.object(NameGenerator, '_generate_with_faker')
    def test_generate_uses_faker_when_enabled(self, mock_faker_generate):
        # Test that _generate_with_faker is used when Faker is enabled
        mock_faker_generate.return_value = ["Fake Name 1", "Fake Name 2"]

        generator = NameGenerator(config={
            'use_faker': True,
            'language': 'en'
        })

        generator.faker = Mock()  # Use Mock instead of MagicMock
        result = generator.generate(2)

        mock_faker_generate.assert_called_once_with(2, None, 'en', 'FL')
        self.assertEqual(result, ["Fake Name 1", "Fake Name 2"])


class TestNameGeneratorGenerateLike(unittest.TestCase):

    def setUp(self):
        self.generator = NameGenerator(config={
            "use_mapping": False,
            "gender_from_name": False
        })

    def test_generate_like_empty_value(self):
        result = self.generator.generate_like("")
        self.assertEqual(result, "")

    def test_generate_like_with_mapping_store(self):
        mapping_store = Mock()
        mapping_store.get_mapping.return_value = "Mapped Name"
        generator = NameGenerator(config={
            "use_mapping": True,
            "mapping_store": mapping_store
        })

        result = generator.generate_like("Original Name", field_name="name")
        self.assertEqual(result, "Mapped Name")
        mapping_store.get_mapping.assert_called_once_with("name", "Original Name")

    def test_generate_like_with_prgn(self):
        prgn_mock = Mock()
        prgn_mock.select_from_list.side_effect = [
            "SyntheticFirst", "SyntheticLast"  # First and last name
        ]
        generator = NameGenerator(config={
            "use_mapping": False,
            "key": "test-key"  # To trigger PRNGenerator
        })
        generator.prgn_generator = prgn_mock
        generator._first_names_male = {"en": ["SyntheticFirst"]}
        generator._last_names = {"en": ["SyntheticLast"]}
        generator.gender = "M"
        generator.language = "en"
        generator.format = "FL"

        result = generator.generate_like("John Doe")
        self.assertEqual(result, "Syntheticfirst Syntheticlast")  # Updated expectation
        self.assertEqual(prgn_mock.select_from_list.call_count, 2)

    def test_generate_like_fallback_to_generate_full_name(self):
        fallback_mock = Mock(return_value="Fallback Full Name")
        generator = NameGenerator(config={
            "use_mapping": False
        })
        generator.prgn_generator = None  # Disable PRGN
        generator.generate_full_name = fallback_mock
        generator.gender = "F"
        generator.language = "en"
        generator.format = "FL"

        result = generator.generate_like("Jane")
        self.assertEqual(result, "Fallback Full Name")
        fallback_mock.assert_called_once_with("F", "en", "FL")


class TestNameGeneratorGenerateFirstName(unittest.TestCase):

    def test_generate_first_name_from_dict_male(self):
        generator = NameGenerator(config={})
        generator._first_names_male = {"en": ["Alex", "Bob"]}
        generator._first_names_female = {}
        generator.use_faker = False
        generator.language = "en"
        generator._apply_case = lambda name: name  # Bỏ qua casing

        with patch("random.choice", return_value="Bob"):
            name = generator.generate_first_name(gender="M")
            self.assertEqual(name, "Bob")

    def test_generate_first_name_from_dict_female(self):
        generator = NameGenerator(config={})
        generator._first_names_female = {"en": ["Alice", "Emma"]}
        generator._first_names_male = {}
        generator.use_faker = False
        generator.language = "en"
        generator._apply_case = lambda name: name

        with patch("random.choice", return_value="Emma"):
            name = generator.generate_first_name(gender="F")
            self.assertEqual(name, "Emma")

    def test_generate_first_name_fallback_to_en(self):
        generator = NameGenerator(config={})
        generator._first_names_male = {"en": ["John"]}
        generator._first_names_female = {}
        generator.use_faker = False
        generator.language = "ru"
        generator._apply_case = lambda name: name

        name = generator.generate_first_name(gender="M", language="xx")  # Unsupported lang
        self.assertEqual(name, "John")

    def test_generate_first_name_with_fallback_placeholder(self):
        generator = NameGenerator(config={})
        generator._first_names_male = {}
        generator._first_names_female = {}
        generator.use_faker = False
        generator.language = "en"
        generator._apply_case = lambda name: name

        name = generator.generate_first_name(gender="F")
        self.assertEqual(name, "Jane")

        name2 = generator.generate_first_name(gender="M")
        self.assertEqual(name2, "John")

    def test_generate_first_name_with_faker_male(self):
        faker_mock = Mock()
        faker_mock.locale = "en"
        faker_mock.first_name_male.return_value = "David"

        generator = NameGenerator(config={"use_faker": True})
        generator.use_faker = True
        generator.faker = faker_mock
        generator._get_faker_locale = lambda lang: "en"
        generator._apply_case = lambda name: name

        name = generator.generate_first_name(gender="M", language="en")
        self.assertEqual(name, "David")
        faker_mock.first_name_male.assert_called_once()

    def test_generate_first_name_with_faker_default(self):
        faker_mock = Mock()
        faker_mock.locale = "en"
        faker_mock.first_name.return_value = "Taylor"

        generator = NameGenerator(config={"use_faker": True})
        generator.use_faker = True
        generator.faker = faker_mock
        generator._get_faker_locale = lambda lang: "en"
        generator._apply_case = lambda name: name

        name = generator.generate_first_name(gender=None, language="en")
        self.assertEqual(name, "Taylor")
        faker_mock.first_name.assert_called_once()

    def test_generate_first_name_random_gender(self):
        generator = NameGenerator(config={})
        generator.f_m_ratio = 1.0  # Force female
        generator._first_names_female = {"en": ["Emily"]}
        generator._first_names_male = {}
        generator.language = "en"
        generator.use_faker = False
        generator._apply_case = lambda name: name

        with patch("random.choice", return_value="Emily"):
            name = generator.generate_first_name(gender=None)
            self.assertEqual(name, "Emily")


class TestNameGeneratorGenerateLastName(unittest.TestCase):

    def test_generate_last_name_from_dict(self):
        generator = NameGenerator(config={})
        generator._last_names = {"en": ["Taylor", "Brown"]}
        generator.language = "en"
        generator.use_faker = False
        generator._apply_case = lambda name: name

        with patch("random.choice", return_value="Taylor"):
            result = generator.generate_last_name()
            self.assertEqual(result, "Taylor")

    def test_generate_last_name_fallback_to_en(self):
        generator = NameGenerator(config={})
        generator._last_names = {
            "en": ["Johnson"]
        }
        generator.language = "xx"  # Unsupported language
        generator.use_faker = False
        generator._apply_case = lambda name: name

        result = generator.generate_last_name()
        self.assertEqual(result, "Johnson")

    def test_generate_last_name_with_fallback_placeholder(self):
        generator = NameGenerator(config={})
        generator._last_names = {}
        generator.language = "en"
        generator.use_faker = False
        generator._apply_case = lambda name: name

        result = generator.generate_last_name()
        self.assertEqual(result, "Smith")

    def test_generate_last_name_with_faker_male(self):
        faker_mock = Mock()
        faker_mock.locale = "en"
        faker_mock.last_name_male.return_value = "Williams"

        generator = NameGenerator(config={"use_faker": True})
        generator.use_faker = True
        generator.faker = faker_mock
        generator._get_faker_locale = lambda lang: "en"
        generator._apply_case = lambda name: name

        result = generator.generate_last_name(gender="M", language="en")
        self.assertEqual(result, "Williams")
        faker_mock.last_name_male.assert_called_once()

    def test_generate_last_name_with_faker_female(self):
        faker_mock = Mock()
        faker_mock.locale = "en"
        faker_mock.last_name_female.return_value = "Watson"

        generator = NameGenerator(config={"use_faker": True})
        generator.use_faker = True
        generator.faker = faker_mock
        generator._get_faker_locale = lambda lang: "en"
        generator._apply_case = lambda name: name

        result = generator.generate_last_name(gender="F", language="en")
        self.assertEqual(result, "Watson")
        faker_mock.last_name_female.assert_called_once()

    def test_generate_last_name_with_faker_default(self):
        faker_mock = Mock()
        faker_mock.locale = "en"
        faker_mock.last_name.return_value = "Parker"

        generator = NameGenerator(config={"use_faker": True})
        generator.use_faker = True
        generator.faker = faker_mock
        generator._get_faker_locale = lambda lang: "en"
        generator._apply_case = lambda name: name

        result = generator.generate_last_name(gender=None, language="en")
        self.assertEqual(result, "Parker")
        faker_mock.last_name.assert_called_once()


class TestNameGeneratorGenerateMiddleName(unittest.TestCase):

    def test_generate_middle_name_from_dict_male(self):
        generator = NameGenerator(config={})
        generator._middle_names_male = {"ru": ["Ivanovich"]}
        generator._middle_names_female = {}
        generator.language = "ru"
        generator.use_faker = False
        generator._apply_case = lambda x: x

        with patch("random.choice", return_value="Ivanovich"):
            result = generator.generate_middle_name(gender="M", language="ru")
            self.assertEqual(result, "Ivanovich")

    def test_generate_middle_name_from_dict_female(self):
        generator = NameGenerator(config={})
        generator._middle_names_female = {"ru": ["Ivanovna"]}
        generator._middle_names_male = {}
        generator.language = "ru"
        generator.use_faker = False
        generator._apply_case = lambda x: x

        with patch("random.choice", return_value="Ivanovna"):
            result = generator.generate_middle_name(gender="F", language="ru")
            self.assertEqual(result, "Ivanovna")

    def test_generate_middle_name_random_gender(self):
        generator = NameGenerator(config={})
        generator._middle_names_female = {"ru": ["Annaevna"]}
        generator._middle_names_male = {}
        generator.language = "ru"
        generator.use_faker = False
        generator._apply_case = lambda x: x
        generator.f_m_ratio = 1.0  # Force female gender

        with patch("random.choice", return_value="Annaevna"):
            result = generator.generate_middle_name(gender=None, language="ru")
            self.assertEqual(result, "Annaevna")

    def test_generate_middle_name_fallback_empty(self):
        generator = NameGenerator(config={})
        generator._middle_names_female = {}
        generator._middle_names_male = {}
        generator.language = "ru"
        generator.use_faker = False
        generator._apply_case = lambda x: x

        result = generator.generate_middle_name(gender="M", language="ru")
        self.assertEqual(result, "")

    def test_generate_middle_name_with_faker_male(self):
        faker_mock = Mock()
        faker_mock.locale = "ru"
        faker_mock.middle_name_male.return_value = "Petrovich"

        generator = NameGenerator(config={"use_faker": True})
        generator.language = "ru"
        generator.use_faker = True
        generator.faker = faker_mock
        generator._get_faker_locale = lambda lang: "ru"
        generator._apply_case = lambda x: x

        result = generator.generate_middle_name(gender="M", language="ru")
        self.assertEqual(result, "Petrovich")
        faker_mock.middle_name_male.assert_called_once()

    def test_generate_middle_name_with_faker_female(self):
        faker_mock = Mock()
        faker_mock.locale = "ru"
        faker_mock.middle_name_female.return_value = "Petrovna"

        generator = NameGenerator(config={"use_faker": True})
        generator.language = "ru"
        generator.use_faker = True
        generator.faker = faker_mock
        generator._get_faker_locale = lambda lang: "ru"
        generator._apply_case = lambda x: x

        result = generator.generate_middle_name(gender="F", language="ru")
        self.assertEqual(result, "Petrovna")
        faker_mock.middle_name_female.assert_called_once()


class TestNameGeneratorGenerateFullName(unittest.TestCase):

    def setUp(self):
        self.generator = NameGenerator(config={})
        self.generator._normalize_language = lambda x: x
        self.generator._parse_format_case = lambda fmt: (fmt, "title")
        self.generator._format_name = lambda f, m, l, fmt, case: "_".join(filter(None, [f, m, l]))

    def test_generate_full_name_custom_methods_fl_format(self):
        self.generator.use_faker = False
        self.generator.generate_first_name = lambda gender, lang: "John"
        self.generator.generate_last_name = lambda gender, lang: "Smith"
        self.generator.generate_middle_name = lambda gender, lang: "Middle"  # Should not be used

        result = self.generator.generate_full_name(gender="M", language="en", format_str="FL")
        self.assertEqual(result, "John_Smith")

    def test_generate_full_name_custom_methods_fml_format(self):
        self.generator.use_faker = False
        self.generator.generate_first_name = lambda gender, lang: "Jane"
        self.generator.generate_middle_name = lambda gender, lang: "Anna"
        self.generator.generate_last_name = lambda gender, lang: "Doe"

        result = self.generator.generate_full_name(gender="F", language="en", format_str="FML")
        self.assertEqual(result, "Jane_Anna_Doe")

    def test_generate_full_name_faker_male_ru_with_middle(self):
        faker_mock = Mock()
        faker_mock.locale = "ru"
        faker_mock.first_name_male.return_value = "Ivan"
        faker_mock.last_name_male.return_value = "Petrov"
        faker_mock.middle_name_male.return_value = "Ivanovich"

        self.generator.use_faker = True
        self.generator.faker = faker_mock
        self.generator._get_faker_locale = lambda lang: "ru"
        self.generator._format_name = lambda f, m, l, fmt, case: f"{f} {m} {l}"

        result = self.generator.generate_full_name(gender="M", language="ru", format_str="FML")
        self.assertEqual(result, "Ivan Ivanovich Petrov")

    def test_generate_full_name_faker_female_ru_without_middle(self):
        faker_mock = Mock()
        faker_mock.locale = "ru"
        faker_mock.first_name_female.return_value = "Anna"
        faker_mock.last_name_female.return_value = "Ivanova"
        faker_mock.middle_name_female.return_value = "Petrovna"  # Should not be used

        self.generator.use_faker = True
        self.generator.faker = faker_mock
        self.generator._get_faker_locale = lambda lang: "ru"
        self.generator._format_name = lambda f, m, l, fmt, case: f"{f} {l}"

        result = self.generator.generate_full_name(gender="F", language="ru", format_str="FL")
        self.assertEqual(result, "Anna Ivanova")

    def test_generate_full_name_faker_neutral_gender_with_middle(self):
        faker_mock = Mock()
        faker_mock.locale = "ru"
        faker_mock.first_name.return_value = "Alex"
        faker_mock.last_name.return_value = "Smirnov"
        faker_mock.middle_name_male.return_value = "Alexandrovich"

        self.generator.use_faker = True
        self.generator.faker = faker_mock
        self.generator._get_faker_locale = lambda lang: "ru"
        self.generator._format_name = lambda f, m, l, fmt, case: f"{f} {m} {l}"

        result = self.generator.generate_full_name(gender=None, language="ru", format_str="FML")
        self.assertEqual(result, "Alex Alexandrovich Smirnov")


class TestNameGeneratorDetectGender(unittest.TestCase):

    def setUp(self):
        # Mocking the NameGenerator class
        self.generator = NameGenerator(config={})
        self.generator._normalize_language = lambda x: x
        self.generator.parse_full_name = lambda name, lang: {"first_name": name.split()[0]}  # Mock parse_full_name
        self.generator._first_names_male = {
            "en": ["John", "Michael", "David"]
        }
        self.generator._first_names_female = {
            "en": ["Mary", "Jessica", "Emily"]
        }

    def test_detect_gender_male(self):
        # Test male name
        result = self.generator.detect_gender("John Doe", language="en")
        self.assertEqual(result, "M")

    def test_detect_gender_female(self):
        # Test female name
        result = self.generator.detect_gender("Mary Jane", language="en")
        self.assertEqual(result, "F")

    def test_detect_gender_no_match(self):
        # Test name that doesn't match male or female
        result = self.generator.detect_gender("Alex Smith", language="en")
        self.assertEqual(result, None)

    def test_detect_gender_empty_name(self):
        # Test empty name
        result = self.generator.detect_gender("", language="en")
        self.assertEqual(result, None)

    def test_detect_gender_no_first_name(self):
        # Test when parse_full_name returns an empty first_name
        self.generator.parse_full_name = lambda name, lang: {"first_name": ""}
        result = self.generator.detect_gender("", language="en")
        self.assertEqual(result, None)

    def test_detect_gender_non_english_language(self):
        # Test name detection in a non-English language (e.g., Russian)
        self.generator._first_names_male["ru"] = ["Иван", "Дмитрий"]
        self.generator._first_names_female["ru"] = ["Мария", "Анна"]

        result = self.generator.detect_gender("Иван Иванов", language="ru")
        self.assertEqual(result, "M")

        result = self.generator.detect_gender("Мария Иванова", language="ru")
        self.assertEqual(result, "F")

    def test_detect_gender_fallback_to_english(self):
        # Test fallback to English when name is in a non-English language
        self.generator._first_names_male["es"] = ["Juan"]
        self.generator._first_names_female["es"] = ["Maria"]

        result = self.generator.detect_gender("Juan Pérez", language="es")
        self.assertEqual(result, "M")

        result = self.generator.detect_gender("Maria Gómez", language="es")
        self.assertEqual(result, "F")

    def test_detect_gender_no_name_found_in_all_languages(self):
        # Test when name is not found in any language
        result = self.generator.detect_gender("UnkownName", language="es")
        self.assertEqual(result, None)


class TestNameGeneratorParseFullName(unittest.TestCase):

    def setUp(self):
        # Mocking the NameGenerator class
        self.generator = NameGenerator(config={})
        self.generator._normalize_language = lambda x: x
        self.generator.language = "en"

    def test_parse_full_name_empty(self):
        # Test case for empty name
        result = self.generator.parse_full_name("")
        self.assertEqual(result, {"first_name": "", "middle_name": "", "last_name": ""})

    def test_parse_full_name_single_name(self):
        # Test case for a name with only one part
        result = self.generator.parse_full_name("John")
        self.assertEqual(result, {"first_name": "John", "middle_name": "", "last_name": ""})

    def test_parse_full_name_two_parts_western(self):
        # Test case for a name with two parts (Western languages)
        result = self.generator.parse_full_name("John Doe")
        self.assertEqual(result, {"first_name": "John", "middle_name": "", "last_name": "Doe"})

    def test_parse_full_name_two_parts_russian(self):
        # Test case for a name with two parts (Russian language)
        self.generator.language = "ru"
        result = self.generator.parse_full_name("Иван Иванов")
        self.assertEqual(result, {"first_name": "Иванов", "middle_name": "", "last_name": "Иван"})

    def test_parse_full_name_three_parts_western(self):
        # Test case for a name with three parts (Western languages)
        result = self.generator.parse_full_name("John Michael Doe")
        self.assertEqual(result, {"first_name": "John", "middle_name": "Michael", "last_name": "Doe"})

    def test_parse_full_name_three_parts_russian(self):
        # Test case for a name with three parts (Russian language)
        self.generator.language = "ru"
        result = self.generator.parse_full_name("Иван Дмитрий Иванов")
        self.assertEqual(result, {'first_name': 'Дмитрий', 'last_name': 'Иван', 'middle_name': 'Иванов'})

    def test_parse_full_name_three_parts_vietnamese(self):
        # Test case for a name with three parts (Vietnamese language)
        self.generator.language = "vn"
        result = self.generator.parse_full_name("Nguyễn Văn A")
        self.assertEqual(result, {"first_name": "A", "middle_name": "Văn", "last_name": "Nguyễn"})

    def test_parse_full_name_more_than_three_parts(self):
        # Test case for a name with more than three parts
        result = self.generator.parse_full_name("John Michael David Doe")
        self.assertEqual(result, {"first_name": "John", "middle_name": "Michael David", "last_name": "Doe"})

    def test_parse_full_name_with_middle_name(self):
        # Test case for a name with a middle name
        result = self.generator.parse_full_name("John Michael David Doe")
        self.assertEqual(result, {"first_name": "John", "middle_name": "Michael David", "last_name": "Doe"})

    def test_parse_full_name_fallback_to_english(self):
        # Test case for fallback to English if the name is not found in the specified language
        self.generator.language = "es"
        result = self.generator.parse_full_name("Juan Carlos Pérez")
        self.assertEqual(result, {"first_name": "Juan", "middle_name": "Carlos", "last_name": "Pérez"})


if __name__ == "__main__":
    unittest.main()