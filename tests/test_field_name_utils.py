import unittest
from utils.field_name_utils import generate_field_name
class TestGenerateFieldName(unittest.TestCase):

    def test_generate_field_name_basic(self):
        # Main Features
        result = generate_field_name("SEX;AGE;INCOME;BRAND", prefix="C_", letters_count=2)
        self.assertEqual(result, "C_SEAGINBR")

    def test_generate_field_name_limit(self):
        # Length Limitation
        result = generate_field_name("SEX;AGE;INCOME;BRAND", prefix="C_", letters_count=2, max_length=10)
        self.assertEqual(result, "C_SEAGINBR")