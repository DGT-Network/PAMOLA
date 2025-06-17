import re
import unittest
from unittest.mock import patch, Mock
from pamola_core.fake_data.generators.phone import PhoneGenerator



class TestPhoneGeneratorInit(unittest.TestCase):

    @patch('pamola_core.fake_data.dictionaries.phones.get_phone_formats', return_value={'us': ['+1 (###) ###-####']})
    def test_init_with_default_config(self, mock_get_formats):
        config = {}

        generator = PhoneGenerator(config)

        # Instead of asserting empty, check structure and common entries
        self.assertIsInstance(generator.country_codes, dict)
        self.assertIn('1', generator.country_codes)  # US
        self.assertIn('44', generator.country_codes)  # UK
        self.assertGreater(generator.country_codes['1'], 0)

        self.assertIsNone(generator.operator_codes_dict)
        self.assertIsNone(generator.format)
        self.assertTrue(generator.validate_source)  # Default is True
        self.assertEqual(generator.handle_invalid_phone, 'generate_new')  # Default value
        self.assertEqual(generator.default_country, 'us')  # Default value
        self.assertTrue(generator.preserve_country_code)  # Default value
        self.assertFalse(generator.preserve_operator_code)  # Default value
        self.assertEqual(generator.region, 'us')  # Uses default_country if not set
        self.assertIsNotNone(generator._phone_formats)  # From patched return
        self.assertIsInstance(generator.phone_pattern.pattern, str)
        self.assertIsInstance(generator.digits_pattern.pattern, str)
        self.assertIsNone(generator.prgn_generator)  # Default is None

    @patch.object(PhoneGenerator, '_process_country_codes')
    def test_init_with_custom_config(self, mock_process_country_codes):
        # Mock the return value of _process_country_codes
        mock_process_country_codes.return_value = {'1': 1.0}

        config = {
            'country_codes': {"us": 1.0},
            'operator_codes_dict': None,
            'format': '+1 (XXX) XXX-XXXX',
            'validate_source': False,
            'handle_invalid_phone': 'generate_new',
            'default_country': 'us',
            'preserve_country_code': False,
            'preserve_operator_code': True,
            'region': 'us',
            'key': 'test_seed'
        }

        generator = PhoneGenerator(config)

        # Assertions
        self.assertEqual(generator.country_codes, {'1': 1.0})
        self.assertIsNone(generator.operator_codes_dict)
        self.assertEqual(generator.format, '+1 (XXX) XXX-XXXX')
        self.assertFalse(generator.validate_source)
        self.assertEqual(generator.handle_invalid_phone, 'generate_new')
        self.assertEqual(generator.default_country, 'us')
        self.assertFalse(generator.preserve_country_code)
        self.assertTrue(generator.preserve_operator_code)
        self.assertEqual(generator.region, 'us')
        self.assertIsNotNone(generator._phone_formats)  # Assuming phones.get_phone_formats() returns something
        self.assertIsNotNone(generator.phone_pattern)
        self.assertIsNotNone(generator.digits_pattern)
        self.assertIsNotNone(generator.prgn_generator)

        # Ensure the mock was called correctly
        mock_process_country_codes.assert_called_once_with({"us": 1.0})


class TestPhoneGeneratorValidatePhone(unittest.TestCase):
    def setUp(self):
        self.generator = PhoneGenerator({})
        # Ensure regex patterns are initialized
        self.generator.phone_pattern = re.compile(r'^\+?[0-9\s()-.]+$')
        self.generator.digits_pattern = re.compile(r'[0-9]+')

    def test_valid_phone_numbers(self):
        # Valid phone number formats
        valid_phones = [
            '+1 (123) 456-7890',
            '123-456-7890',
            '+44 20 7946 0958',
            '0812345678',
            '(123)4567890',
            '+911234567890'
        ]
        for phone in valid_phones:
            with self.subTest(phone=phone):
                self.assertTrue(self.generator.validate_phone(phone))

    def test_invalid_phone_empty_or_non_string(self):
        # Invalid when phone is None, not a string, or empty
        self.assertFalse(self.generator.validate_phone(None))
        self.assertFalse(self.generator.validate_phone(1234567890))  # Not a string
        self.assertFalse(self.generator.validate_phone(""))  # Empty string

    def test_invalid_format(self):
        # Invalid due to unsupported characters
        self.assertFalse(self.generator.validate_phone("abc-123-xyz"))
        self.assertFalse(self.generator.validate_phone("123*456#7890"))

    def test_too_short_phone(self):
        # Invalid: less than 7 digits
        self.assertFalse(self.generator.validate_phone("123456"))

    def test_too_long_phone(self):
        # Invalid: more than 15 digits
        self.assertFalse(self.generator.validate_phone("+1234567890123456"))


class TestPhoneGeneratorExtractCountryCode(unittest.TestCase):

    def setUp(self):
        self.generator = PhoneGenerator({})
        self.generator.digits_pattern = re.compile(r'[0-9]+')

        # Giả lập _operator_codes cho các test
        self.generator._operator_codes = {
            '49': ['30', '40'],   # Germany
            '91': ['22', '33'],   # India
            '81': ['3'],          # Japan
            '33': ['1']           # France
        }

    def test_none_or_non_string(self):
        self.assertIsNone(self.generator.extract_country_code(None))
        self.assertIsNone(self.generator.extract_country_code(1234567890))
        self.assertIsNone(self.generator.extract_country_code(""))

    def test_us_canada_code(self):
        self.assertEqual(self.generator.extract_country_code("12345678901"), '1')
        self.assertEqual(self.generator.extract_country_code("+11234567890"), '1')

    def test_uk_code(self):
        self.assertEqual(self.generator.extract_country_code("447123456789"), '44')
        self.assertEqual(self.generator.extract_country_code("+447123456789"), '44')

    def test_russia_code(self):
        self.assertEqual(self.generator.extract_country_code("71234567890"), '7')
        self.assertEqual(self.generator.extract_country_code("+71234567890"), '7')

    def test_common_operator_codes(self):
        self.assertEqual(self.generator.extract_country_code("491234567890"), '49')
        self.assertEqual(self.generator.extract_country_code("911234567890"), '91')
        self.assertEqual(self.generator.extract_country_code("811234567890"), '81')
        self.assertEqual(self.generator.extract_country_code("331234567890"), '33')

    def test_plus_prefix_operator_codes(self):
        self.assertEqual(self.generator.extract_country_code("+491234567890"), '49')
        self.assertEqual(self.generator.extract_country_code("+911234567890"), '91')
        self.assertEqual(self.generator.extract_country_code("+811234567890"), '81')
        self.assertEqual(self.generator.extract_country_code("+331234567890"), '33')

    def test_no_matching_country_code(self):
        self.assertIsNone(self.generator.extract_country_code("88888888888"))
        self.assertIsNone(self.generator.extract_country_code("+88888888888"))


class TestPhoneGeneratorExtractOperatorCode(unittest.TestCase):

    def setUp(self):
        self.generator = PhoneGenerator({})
        self.generator.digits_pattern = re.compile(r'[0-9]+')

        # Thiết lập giả _operator_codes
        self.generator._operator_codes = {
            '1': ['212', '213'],       # US
            '44': ['20', '121'],       # UK
            '91': ['22', '33'],        # India
        }

        # Patch extract_country_code khi cần
        self.generator.extract_country_code = lambda phone: '1' if phone.startswith('1') else '91'

    def test_none_or_invalid_input(self):
        self.assertIsNone(self.generator.extract_operator_code(None))
        self.assertIsNone(self.generator.extract_operator_code(12345))
        self.assertIsNone(self.generator.extract_operator_code(""))

    def test_valid_operator_with_country_code(self):
        phone = "12122223333"  # country: '1', operator: '212'
        self.assertEqual(self.generator.extract_operator_code(phone, country_code='1'), '212')

        phone2 = "912233445566"  # country: '91', operator: '22'
        self.assertEqual(self.generator.extract_operator_code(phone2, country_code='91'), '22')

    def test_valid_operator_without_country_code(self):
        phone = "12132223333"  # Fallback extract_country_code returns '1'
        self.assertEqual(self.generator.extract_operator_code(phone), '213')

    def test_no_matching_operator_code(self):
        phone = "19999999999"  # no operator '999' in ['212', '213']
        self.assertIsNone(self.generator.extract_operator_code(phone, country_code='1'))

    def test_country_code_not_in_operator_list(self):
        phone = "33123456789"
        self.assertIsNone(self.generator.extract_operator_code(phone, country_code='33'))

    def test_country_code_needs_detection_but_not_found(self):
        # Override extract_country_code to return None
        self.generator.extract_country_code = lambda phone: None
        self.assertIsNone(self.generator.extract_operator_code("99999999999"))

    def test_operator_code_match_prefers_longest_first(self):
        self.generator._operator_codes['44'] = ['2', '20', '201']
        phone = "442013456789"
        self.assertEqual(self.generator.extract_operator_code(phone, country_code='44'), '201')


class TestPhoneGeneratorGenerateCountryCode(unittest.TestCase):

    def setUp(self):
        self.config = {
            'preserve_country_code': True,
            'default_country': 'us'
        }
        self.generator = PhoneGenerator(self.config)
        self.generator._get_country_code_for_region = Mock(return_value='1')  # e.g., for US
        self.generator.country_codes = {'1': 0.6, '44': 0.4}

    def test_preserve_original_country_code(self):
        # Nếu cấu hình preserve và có original, trả về original
        result = self.generator.generate_country_code(original_country_code='44')
        self.assertEqual(result, '44')

    def test_return_default_country_if_no_country_codes(self):
        self.generator.country_codes = {}
        result = self.generator.generate_country_code()
        self.generator._get_country_code_for_region.assert_called_with('us')
        self.assertEqual(result, '1')

    @patch('random.choices')
    def test_random_weighted_choice_without_prgn(self, mock_choices):
        # PRGN không tồn tại, dùng random.choices
        self.generator.preserve_country_code = False
        self.generator.prgn_generator = None
        mock_choices.return_value = ['44']

        result = self.generator.generate_country_code()
        mock_choices.assert_called_once_with(['1', '44'], weights=[0.6, 0.4], k=1)
        self.assertEqual(result, '44')

    def test_prgn_deterministic_selection(self):
        self.generator.preserve_country_code = False

        # Giả PRGN generator
        mock_rng = Mock()
        mock_rng.random.return_value = 0.5  # sẽ rơi vào '1' nếu weight đủ
        mock_prgn = Mock()
        mock_prgn.get_random_by_value.return_value = mock_rng
        self.generator.prgn_generator = mock_prgn

        result = self.generator.generate_country_code(original_country_code='test_value')
        mock_prgn.get_random_by_value.assert_called_with('test_value', salt="country-code-selection")
        self.assertIn(result, ['1', '44'])  # vì phụ thuộc vào ngưỡng tích lũy
        self.assertEqual(result, '1')  # 0.5 <= 0.6

    def test_prgn_fallback_to_last_if_random_above_total(self):
        self.generator.preserve_country_code = False

        # Setup PRGN trả về giá trị random lớn hơn tổng weights (giả sử lỗi chuẩn hóa)
        mock_rng = Mock()
        mock_rng.random.return_value = 1.1
        mock_prgn = Mock()
        mock_prgn.get_random_by_value.return_value = mock_rng
        self.generator.prgn_generator = mock_prgn

        result = self.generator.generate_country_code()
        self.assertEqual(result, '44')  # fallback to last

    def test_preserve_is_false_and_original_provided(self):
        self.generator.preserve_country_code = False
        self.generator.prgn_generator = None

        with patch('random.choices', return_value=['1']) as mock_choices:
            result = self.generator.generate_country_code(original_country_code='44')
            self.assertEqual(result, '1')
            mock_choices.assert_called_once()


class TestPhoneGeneratorGenerateOperatorCode(unittest.TestCase):

    def setUp(self):
        self.config = {
            'preserve_operator_code': True,
        }
        self.generator = PhoneGenerator(self.config)
        self.generator._operator_codes = {
            '1': ['202', '303', '415'],
            '44': ['20', '121'],
        }

    def test_preserve_valid_operator_code(self):
        result = self.generator.generate_operator_code('1', original_operator_code='202')
        self.assertEqual(result, '202')

    def test_preserve_invalid_operator_code(self):
        # preserve=True nhưng operator không hợp lệ
        result = self.generator.generate_operator_code('1', original_operator_code='999')
        self.assertIn(result, self.generator._operator_codes['1'])  # fallback to random or PRGN

    def test_no_operator_codes_for_country(self):
        result = self.generator.generate_operator_code('99')
        self.assertIsNone(result)

    def test_empty_operator_codes_for_country(self):
        self.generator._operator_codes['99'] = []
        result = self.generator.generate_operator_code('99')
        self.assertIsNone(result)

    @patch('random.choice')
    def test_random_selection_without_prgn(self, mock_choice):
        self.generator.preserve_operator_code = False
        self.generator.prgn_generator = None
        mock_choice.return_value = '303'

        result = self.generator.generate_operator_code('1', original_operator_code='202')
        mock_choice.assert_called_once_with(['202', '303', '415'])
        self.assertEqual(result, '303')

    def test_deterministic_selection_with_prgn(self):
        self.generator.preserve_operator_code = False

        mock_rng = Mock()
        mock_rng.randint.return_value = 1  # Will select index 1
        mock_prgn = Mock()
        mock_prgn.get_random_by_value.return_value = mock_rng
        self.generator.prgn_generator = mock_prgn

        result = self.generator.generate_operator_code('1', original_operator_code='202')
        mock_prgn.get_random_by_value.assert_called_with('202', salt="operator-code-selection")
        self.assertEqual(result, '303')

    def test_deterministic_selection_with_country_as_seed(self):
        self.generator.preserve_operator_code = False

        mock_rng = Mock()
        mock_rng.randint.return_value = 1  # Fix: use valid index
        mock_prgn = Mock()
        mock_prgn.get_random_by_value.return_value = mock_rng
        self.generator.prgn_generator = mock_prgn

        self.generator._operator_codes = {
            '44': ['20', '121']
        }

        result = self.generator.generate_operator_code('44')
        mock_prgn.get_random_by_value.assert_called_with('44', salt="operator-code-selection")
        self.assertEqual(result, '121')


class TestPhoneGeneratorFormatPhone(unittest.TestCase):

    def setUp(self):
        self.generator = PhoneGenerator(config={})
        self.generator._get_country_name_for_code = Mock(return_value='us')

    @patch("pamola_core.fake_data.dictionaries.phones.get_phone_format_for_country", return_value="+CC (AAA) XXX-XXXX")
    def test_format_with_all_placeholders(self, mock_get_format):
        formatted = self.generator.format_phone(
            country_code='1',
            operator_code='415',
            number='1234567'
        )
        self.assertEqual(formatted, "+1 (415) 123-4567")

    @patch("pamola_core.fake_data.dictionaries.phones.get_phone_format_for_country", return_value="+CC (AAA) XXX-XXXX")
    def test_format_without_operator_code(self, mock_get_format):
        formatted = self.generator.format_phone(
            country_code='1',
            operator_code=None,
            number='4151234567'
        )
        self.assertEqual(formatted, "+1 (415) 123-4567")

    def test_format_with_custom_template(self):
        template = "+CC-AA-XXXXXXX"
        formatted = self.generator.format_phone(
            country_code='91',
            operator_code='22',
            number='1234567',
            format_template=template
        )
        self.assertEqual(formatted, "+91-22-1234567")

    def test_format_with_short_operator_code_padding(self):
        template = "+CC-AAAA-XXXX"
        formatted = self.generator.format_phone(
            country_code='44',
            operator_code='12',
            number='3456',
            format_template=template
        )
        # Expect padded operator_code -> "1200"
        self.assertEqual(formatted, "+44-1200-3456")

    def test_format_with_long_operator_code_truncating(self):
        template = "+CC-AAA-XXXX"
        formatted = self.generator.format_phone(
            country_code='49',
            operator_code='1234',
            number='5678',
            format_template=template
        )
        self.assertEqual(formatted, "+49-1234-5678")

    @patch("pamola_core.fake_data.dictionaries.phones.get_phone_format_for_country", return_value=None)
    def test_format_fallback_to_default(self, mock_get_format):
        formatted = self.generator.format_phone(
            country_code='61',
            operator_code='2',
            number='12345678'
        )
        self.assertTrue(formatted.startswith("+61"))
        self.assertNotIn("X", formatted)

    def test_fill_random_digits_when_insufficient_number(self):
        self.generator._generate_random_digits = Mock(return_value="9")  # 1 digit
        template = "+CC-AAA-XXX-XXX-X"
        formatted = self.generator.format_phone(
            country_code='1',
            operator_code='202',
            number='123456',
            format_template=template
        )
        self.assertTrue(formatted.startswith("+1-202-123-456-"))
        self.assertEqual(len(formatted.split("-")[-1]), 1)  # single 'X' filled with 1 digit

    def test_format_without_placeholders_returns_as_is(self):
        template = "STATIC FORMAT"
        formatted = self.generator.format_phone(
            country_code='1',
            operator_code='212',
            number='1234567',
            format_template=template
        )
        self.assertEqual(formatted, "STATIC FORMAT")


class TestPhoneGeneratorGeneratePhoneNumber(unittest.TestCase):
    def setUp(self):
        self.generator = PhoneGenerator()
        self.generator.format_phone = Mock(return_value="+1-202-1234567")
        self.generator._generate_random_digits = Mock(return_value="1234567")
        self.generator._determine_phone_length = Mock(return_value=10)
        self.generator.extract_country_code = Mock(return_value='1')
        self.generator.extract_operator_code = Mock(return_value='202')
        self.generator.generate_country_code = Mock(return_value='1')
        self.generator.generate_operator_code = Mock(return_value='202')

    def test_generate_with_all_params(self):
        phone = self.generator.generate_phone_number(country_code='1', operator_code='202', original_number='1234567')
        self.assertEqual(phone, "+1-202-1234567")
        self.generator.format_phone.assert_called_with('1', '202', '1234567')

    def test_generate_without_country_operator(self):
        phone = self.generator.generate_phone_number()
        self.generator.generate_country_code.assert_called_once()
        self.generator.generate_operator_code.assert_called_once()
        self.generator._generate_random_digits.assert_called_once()
        self.generator.format_phone.assert_called_once()
        self.assertEqual(phone, "+1-202-1234567")

    def test_generate_with_original_number(self):
        phone = self.generator.generate_phone_number(original_number='+1-202-555-1234')
        self.generator.extract_country_code.assert_called_with('+1-202-555-1234')
        self.generator.extract_operator_code.assert_called_with('+1-202-555-1234', '1')
        self.generator.generate_country_code.assert_called_with('1')
        self.generator.generate_operator_code.assert_called_with('1', '202')
        self.assertEqual(phone, "+1-202-1234567")

    def test_generate_without_operator_code(self):
        self.generator.generate_operator_code = Mock(return_value=None)
        self.generator._generate_random_digits = Mock(return_value="1234567")
        self.generator.format_phone = Mock(return_value="+1-None-1234567")

        phone = self.generator.generate_phone_number(country_code='1')

        self.generator.format_phone.assert_called_with("1", None, "1234567")
        self.assertEqual(phone, "+1-None-1234567")

    def test_generate_with_zero_length(self):
        self.generator._determine_phone_length = Mock(return_value=3)
        self.generator.generate_operator_code = Mock(return_value="123")
        self.generator._generate_random_digits = Mock(return_value="")

        # Giả lập output của format_phone để phản ánh tham số được truyền
        self.generator.format_phone = Mock(return_value="+1-123-")

        phone = self.generator.generate_phone_number(country_code="1", original_number="anything")

        self.generator.format_phone.assert_called_with("1", "123", "")
        self.assertEqual(phone, "+1-123-")


class TestPhoneGeneratorGenerate(unittest.TestCase):
    def setUp(self):
        self.generator = PhoneGenerator()

    def test_generate_count(self):
        self.generator._generate_phone = Mock(return_value="+1-202-1234567")
        phones = self.generator.generate(5)
        self.assertEqual(len(phones), 5)
        self.assertTrue(all(p == "+1-202-1234567" for p in phones))

    def test_generate_with_params(self):
        self.generator._generate_phone = Mock(return_value="+44-20-7654321")
        phones = self.generator.generate(3, country_code='44', operator_code='20', format="+CC-AAA-XXXXXXX")
        self.assertEqual(len(phones), 3)
        self.generator._generate_phone.assert_called_with(
            country_code='44',
            operator_code='20',
            format="+CC-AAA-XXXXXXX"
        )

    def test_generate_with_zero(self):
        phones = self.generator.generate(0)
        self.assertEqual(phones, [])

    def test_generate_calls_generate_phone_each_time(self):
        self.generator._generate_phone = Mock(side_effect=[
            "+1-202-0000001",
            "+1-202-0000002",
            "+1-202-0000003",
        ])
        phones = self.generator.generate(3)
        self.assertEqual(phones, [
            "+1-202-0000001",
            "+1-202-0000002",
            "+1-202-0000003"
        ])

    def test_generate_handles_exceptions_gracefully(self):
        def faulty_call(**kwargs):
            raise ValueError("Generation failed")

        self.generator._generate_phone = Mock(side_effect=faulty_call)
        with self.assertRaises(ValueError):
            self.generator.generate(2)


class TestPhoneGeneratorGenerateLike(unittest.TestCase):
    def setUp(self):
        self.generator = PhoneGenerator()
        self.generator.validate_phone = Mock()
        self.generator.extract_country_code = Mock()
        self.generator.extract_operator_code = Mock()
        self.generator.generate_phone_number = Mock()
        self.generator._get_country_code_for_region = Mock()
        self.generator.digits_pattern = Mock()

    def test_empty_original_value_returns_empty(self):
        result = self.generator.generate_like("")
        self.assertEqual(result, "")

        result = self.generator.generate_like(None)
        self.assertEqual(result, "")

    def test_valid_phone_preserve_country_and_operator(self):
        self.generator.validate_source = True
        self.generator.preserve_country_code = True
        self.generator.preserve_operator_code = True
        self.generator.validate_phone.return_value = True
        self.generator.extract_country_code.return_value = "1"
        self.generator.extract_operator_code.return_value = "202"
        self.generator.generate_phone_number.return_value = "+1-202-5555555"

        result = self.generator.generate_like("+1-202-1234567")
        self.generator.generate_phone_number.assert_called_with("1", "202", "+1-202-1234567")
        self.assertEqual(result, "+1-202-5555555")

    def test_valid_phone_no_preserve(self):
        self.generator.validate_source = True
        self.generator.preserve_country_code = False
        self.generator.preserve_operator_code = False
        self.generator.validate_phone.return_value = True
        self.generator.extract_country_code.return_value = "1"
        self.generator.extract_operator_code.return_value = "202"
        self.generator.generate_phone_number.return_value = "+99-999-9999999"

        result = self.generator.generate_like("+1-202-1234567")
        self.generator.generate_phone_number.assert_called_with(None, None, "+1-202-1234567")
        self.assertEqual(result, "+99-999-9999999")

    def test_invalid_phone_keep_empty(self):
        self.generator.validate_source = True
        self.generator.validate_phone.return_value = False
        self.generator.handle_invalid_phone = "keep_empty"

        result = self.generator.generate_like("invalid-phone")
        self.assertEqual(result, "")

    def test_invalid_phone_generate_with_default_country(self):
        self.generator.validate_source = True
        self.generator.validate_phone.return_value = False
        self.generator.handle_invalid_phone = "generate_with_default_country"
        self.generator.default_country = "US"
        self.generator._get_country_code_for_region.return_value = "1"
        self.generator.generate_phone_number.return_value = "+1-999-9999999"

        result = self.generator.generate_like("invalid-phone")
        self.generator._get_country_code_for_region.assert_called_with("US")
        self.generator.generate_phone_number.assert_called_with("1")
        self.assertEqual(result, "+1-999-9999999")

    def test_invalid_phone_generate_new(self):
        self.generator.validate_source = True
        self.generator.validate_phone.return_value = False
        self.generator.handle_invalid_phone = "generate_new"
        self.generator.generate_phone_number.return_value = "+99-888-0000000"

        result = self.generator.generate_like("invalid-phone")
        self.generator.generate_phone_number.assert_called_with()
        self.assertEqual(result, "+99-888-0000000")

    def test_validation_disabled_and_digits_found(self):
        self.generator.validate_source = False
        self.generator.digits_pattern.search.return_value = True
        self.generator.extract_country_code.return_value = "44"
        self.generator.extract_operator_code.return_value = "20"
        self.generator.preserve_country_code = True
        self.generator.preserve_operator_code = True
        self.generator.generate_phone_number.return_value = "+44-20-7654321"

        result = self.generator.generate_like("some phone text 44 20 7654321")
        self.generator.generate_phone_number.assert_called_with("44", "20", "some phone text 44 20 7654321")
        self.assertEqual(result, "+44-20-7654321")


class TestPhoneGeneratorTransform(unittest.TestCase):

    def setUp(self):
        self.generator = PhoneGenerator()
        self.generator.generate_like = Mock()

    def test_transform_multiple_values(self):
        input_values = ["+1-202-1234567", "invalid", "", "+44-20-7654321"]
        expected_outputs = ["+1-202-7654321", "", "", "+44-20-8888888"]

        # Thiết lập mock return value cho mỗi lần gọi
        self.generator.generate_like.side_effect = expected_outputs

        result = self.generator.transform(input_values)

        self.assertEqual(result, expected_outputs)
        self.assertEqual(self.generator.generate_like.call_count, len(input_values))

        for original, call in zip(input_values, self.generator.generate_like.call_args_list):
            self.assertEqual(call.kwargs, {})  # Không có params

    def test_transform_with_additional_params(self):
        input_values = ["+1-202-1234567", "+49-89-123456"]
        params = {"region": "US", "format": "+CC (AAA) XXX-XXXX"}
        self.generator.generate_like.side_effect = ["custom1", "custom2"]

        result = self.generator.transform(input_values, **params)

        self.assertEqual(result, ["custom1", "custom2"])
        for call in self.generator.generate_like.call_args_list:
            self.assertEqual(call.kwargs, params)

    def test_transform_empty_list(self):
        result = self.generator.transform([])
        self.assertEqual(result, [])
        self.generator.generate_like.assert_not_called()


class TestPhoneGeneratorValidate(unittest.TestCase):

    def setUp(self):
        self.generator = PhoneGenerator()
        self.generator.validate_phone = Mock()

    def test_validate_returns_true_for_valid_input(self):
        self.generator.validate_phone.return_value = True
        result = self.generator.validate("+1-202-1234567")
        self.assertTrue(result)
        self.generator.validate_phone.assert_called_once_with("+1-202-1234567")

    def test_validate_returns_false_for_invalid_input(self):
        self.generator.validate_phone.return_value = False
        result = self.generator.validate("invalid-phone")
        self.assertFalse(result)
        self.generator.validate_phone.assert_called_once_with("invalid-phone")

    def test_validate_empty_string(self):
        self.generator.validate_phone.return_value = False
        result = self.generator.validate("")
        self.assertFalse(result)
        self.generator.validate_phone.assert_called_once_with("")

    def test_validate_none(self):
        self.generator.validate_phone.return_value = False
        result = self.generator.validate(None)
        self.assertFalse(result)
        self.generator.validate_phone.assert_called_once_with(None)


if __name__ == "__main__":
    unittest.main()