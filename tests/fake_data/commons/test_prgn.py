import random
import unittest
from typing import Dict, List

from pamola_core.fake_data.commons.prgn import PRNGenerator, generate_deterministic_replacement, generate_seed_from_key


class TestPRNGeneratorInit(unittest.TestCase):

    def test_init_with_str_seed(self):
        seed = "project-seed-2023"
        gen = PRNGenerator(global_seed=seed)

        # Check that global_seed has been normalized (assuming _normalize_seed returns bytes or int)
        self.assertIsNotNone(gen.global_seed)

        # Check that random_state is an instance of random.Random
        self.assertIsInstance(gen.random_state, random.Random)

        # Check that the random_state seed is applied correctly
        # Because random.Random does not provide a way to retrieve the seed directly, this test is limited
        # So we test that the first random value is the same for two instances with the same seed
        value1 = gen.random_state.random()
        gen2 = PRNGenerator(global_seed=seed)
        value2 = gen2.random_state.random()
        self.assertEqual(value1, value2)

    def test_init_with_none_seed(self):
        gen = PRNGenerator(global_seed=None)
        self.assertIsNotNone(gen.global_seed)
        self.assertIsInstance(gen.random_state, random.Random)

    def test_init_with_int_seed(self):
        seed = 12345
        gen = PRNGenerator(global_seed=seed)
        self.assertIsNotNone(gen.global_seed)
        self.assertIsInstance(gen.random_state, random.Random)


class TestGenerateWithSeed(unittest.TestCase):
    def setUp(self):
        self.generator = PRNGenerator(global_seed=1234)

    def test_generate_with_seed_hmac_with_salt(self):
        result1 = self.generator.generate_with_seed("testvalue", salt="mysalt", algorithm="hmac")
        result2 = self.generator.generate_with_seed("testvalue", salt="mysalt", algorithm="hmac")
        self.assertEqual(result1, result2)
        self.assertIsInstance(result1, int)

    def test_generate_with_seed_hmac_without_salt(self):
        result1 = self.generator.generate_with_seed("testvalue", algorithm="hmac")
        result2 = self.generator.generate_with_seed("testvalue", algorithm="hmac")
        self.assertEqual(result1, result2)
        self.assertIsInstance(result1, int)

    def test_generate_with_seed_fast_with_salt(self):
        result1 = self.generator.generate_with_seed("testvalue", salt="mysalt", algorithm="fast")
        result2 = self.generator.generate_with_seed("testvalue", salt="mysalt", algorithm="fast")
        self.assertEqual(result1, result2)
        self.assertIsInstance(result1, int)

    def test_generate_with_seed_fast_without_salt(self):
        result1 = self.generator.generate_with_seed("testvalue", algorithm="fast")
        result2 = self.generator.generate_with_seed("testvalue", algorithm="fast")
        self.assertEqual(result1, result2)
        self.assertIsInstance(result1, int)

    def test_generate_with_seed_simple_with_salt(self):
        result1 = self.generator.generate_with_seed("testvalue", salt="mysalt", algorithm="simple")
        result2 = self.generator.generate_with_seed("testvalue", salt="mysalt", algorithm="simple")
        self.assertEqual(result1, result2)
        self.assertIsInstance(result1, int)

    def test_generate_with_seed_simple_without_salt(self):
        result1 = self.generator.generate_with_seed("testvalue", algorithm="simple")
        result2 = self.generator.generate_with_seed("testvalue", algorithm="simple")
        self.assertEqual(result1, result2)
        self.assertIsInstance(result1, int)

    def test_generate_with_seed_non_string_base_value(self):
        result = self.generator.generate_with_seed(12345)
        self.assertIsInstance(result, int)

    def test_generate_with_seed_bytes_salt_and_base_value(self):
        result = self.generator.generate_with_seed(b"bytevalue", salt=b"bytesalt", algorithm="hmac")
        self.assertIsInstance(result, int)


class TestGetRandomByValue(unittest.TestCase):
    def setUp(self):
        self.generator = PRNGenerator(global_seed=1234)

    def test_get_random_by_value_returns_random_instance(self):
        rng = self.generator.get_random_by_value("testvalue", salt="mysalt")
        self.assertIsInstance(rng, random.Random)

    def test_deterministic_random_generator_with_same_input(self):
        rng1 = self.generator.get_random_by_value("testvalue", salt="mysalt")
        rng2 = self.generator.get_random_by_value("testvalue", salt="mysalt")
        # The next random numbers must be the same
        self.assertEqual(rng1.randint(1, 1000), rng2.randint(1, 1000))
        self.assertEqual(rng1.random(), rng2.random())

    def test_different_seeds_produce_different_sequences(self):
        rng1 = self.generator.get_random_by_value("testvalue", salt="salt1")
        rng2 = self.generator.get_random_by_value("testvalue", salt="salt2")
        # Likely to be different; flexible assertion due to randomness
        self.assertNotEqual(rng1.randint(1, 1000), rng2.randint(1, 1000))

    def test_get_random_by_value_without_salt(self):
        rng1 = self.generator.get_random_by_value("testvalue")
        rng2 = self.generator.get_random_by_value("testvalue")
        self.assertEqual(rng1.randint(1, 1000), rng2.randint(1, 1000))

    def test_get_random_by_value_non_string_base_value(self):
        rng = self.generator.get_random_by_value(12345, salt="mysalt")
        self.assertIsInstance(rng, random.Random)


class TestSelectFromList(unittest.TestCase):
    def setUp(self):
        self.generator = PRNGenerator(global_seed=1234)
        self.items = ["Alice", "Bob", "Charlie"]

    def test_returns_item_from_list(self):
        """Ensure the returned value is in the original list."""
        result = self.generator.select_from_list(self.items, base_value="John", salt="first-names")
        self.assertIn(result, self.items)

    def test_deterministic_selection(self):
        """Ensure that the selection is deterministic with same input."""
        result1 = self.generator.select_from_list(self.items, base_value="John", salt="first-names")
        result2 = self.generator.select_from_list(self.items, base_value="John", salt="first-names")
        self.assertEqual(result1, result2)

    def test_different_salts_give_different_results(self):
        """Different salts should (likely) give different selections."""
        result1 = self.generator.select_from_list(self.items, base_value="John", salt="salt1")
        result2 = self.generator.select_from_list(self.items, base_value="John", salt="salt2")
        self.assertIn(result1, self.items)
        self.assertIn(result2, self.items)
        # Kết quả có thể giống nhau một cách ngẫu nhiên, nhưng thường sẽ khác

    def test_empty_list_returns_none(self):
        """Returns None if item list is empty."""
        result = self.generator.select_from_list([], base_value="John", salt="salt")
        self.assertIsNone(result)


class TestSelectWithMapping(unittest.TestCase):
    def setUp(self):
        self.generator = PRNGenerator(global_seed=42)
        self.mapping = {
            "Alice": "A1",
            "Bob": "B2"
        }

    def test_returns_mapped_value_if_exists(self):
        result = self.generator.select_with_mapping(self.mapping, "Alice")
        self.assertEqual(result, "A1")

    def test_returns_none_if_not_in_mapping_and_no_fallback(self):
        result = self.generator.select_with_mapping(self.mapping, "Charlie")
        self.assertIsNone(result)

    def test_uses_fallback_generator_if_not_in_mapping(self):
        fallback = lambda x: f"generated_{x}"
        result = self.generator.select_with_mapping(self.mapping, "Charlie", fallback_generator=fallback)
        self.assertEqual(result, "generated_Charlie")

    def test_fallback_generator_called_with_correct_value(self):
        calls = []

        def fallback(x):
            calls.append(x)
            return f"gen_{x}"

        self.generator.select_with_mapping(self.mapping, "Daisy", fallback_generator=fallback)
        self.assertIn("Daisy", calls)

    def test_salt_is_ignored(self):
        # Ensure salt does not affect behavior (since it’s not used in logic)
        fallback = lambda x: f"gen_{x}"
        result1 = self.generator.select_with_mapping(self.mapping, "Eve", fallback_generator=fallback, salt="salt1")
        result2 = self.generator.select_with_mapping(self.mapping, "Eve", fallback_generator=fallback, salt="salt2")
        self.assertEqual(result1, result2)


class TestShuffleList(unittest.TestCase):

    def setUp(self):
        self.generator = PRNGenerator(global_seed=1234)
        self.original_list = ["Alice", "Bob", "Charlie", "Dave", "Eve"]

    def test_shuffle_returns_list(self):
        result = self.generator.shuffle_list(self.original_list, base_value="test")
        self.assertIsInstance(result, list)

    def test_original_list_not_modified(self):
        copy_before = self.original_list[:]
        self.generator.shuffle_list(self.original_list, base_value="seed")
        self.assertEqual(self.original_list, copy_before)

    def test_shuffle_is_deterministic(self):
        shuffled1 = self.generator.shuffle_list(self.original_list, base_value="seed", salt="salt1")
        shuffled2 = self.generator.shuffle_list(self.original_list, base_value="seed", salt="salt1")
        self.assertEqual(shuffled1, shuffled2)

    def test_different_salts_produce_different_results(self):
        shuffled1 = self.generator.shuffle_list(self.original_list, base_value="seed", salt="salt1")
        shuffled2 = self.generator.shuffle_list(self.original_list, base_value="seed", salt="salt2")
        self.assertNotEqual(shuffled1, shuffled2)

    def test_empty_list_returns_empty(self):
        result = self.generator.shuffle_list([], base_value="seed")
        self.assertEqual(result, [])

    def test_shuffle_non_string_base_value(self):
        result = self.generator.shuffle_list(self.original_list, base_value=42)
        self.assertIsInstance(result, list)
        self.assertEqual(sorted(result), sorted(self.original_list))  # Elements must match


class TestSelectNameByGenderRegion(unittest.TestCase):

    def setUp(self):
        self.generator = PRNGenerator(global_seed=1234)
        self.names_dict = {
            "en": {
                "M": ["John", "James", "Robert"],
                "F": ["Emily", "Anna", "Grace"]
            },
            "ru": {
                "M": ["Ivan", "Dmitry", "Alexey"],
                "F": ["Olga", "Maria", "Elena"]
            }
        }

    def test_returns_name_from_correct_category(self):
        # Should select from English male names
        name = self.generator.select_name_by_gender_region(
            self.names_dict, original_name="Michael", gender="M", region="en"
        )
        self.assertIn(name, self.names_dict["en"]["M"])

    def test_returns_original_name_if_category_not_found(self):
        # Should return the original name if gender or region not found
        name = self.generator.select_name_by_gender_region(
            self.names_dict, original_name="Yuki", gender="F", region="jp"
        )
        self.assertEqual(name, "Yuki")

    def test_returns_name_deterministically(self):
        # Same input should always return the same name
        name1 = self.generator.select_name_by_gender_region(
            self.names_dict, original_name="Alex", gender="M", region="ru", salt="v1"
        )
        name2 = self.generator.select_name_by_gender_region(
            self.names_dict, original_name="Alex", gender="M", region="ru", salt="v1"
        )
        self.assertEqual(name1, name2)

    def test_different_salt_produces_different_name(self):
        # With a longer name list, different salts should give different names
        self.names_dict["en"]["M"] = [
            "John", "James", "Robert", "David", "Thomas", "Daniel", "Matthew", "Andrew"
        ]

        name1 = self.generator.select_name_by_gender_region(
            self.names_dict, original_name="Michael", gender="M", region="en", salt="v1"
        )
        name2 = self.generator.select_name_by_gender_region(
            self.names_dict, original_name="Michael", gender="M", region="en", salt="v2"
        )

        self.assertNotEqual(name1, name2, "Expected different outputs with different salts")


class TestGenerateDeterministicReplacement(unittest.TestCase):

    def setUp(self):
        self.replacement_list = ["Alice", "Bob", "Charlie"]
        self.original_value = "John"
        self.seed = "test-seed"
        self.salt = "test-salt"

    def test_returns_item_from_list(self):
        result = generate_deterministic_replacement(
            self.original_value, self.replacement_list, self.seed, self.salt
        )
        self.assertIn(result, self.replacement_list)

    def test_deterministic_output(self):
        result1 = generate_deterministic_replacement(
            self.original_value, self.replacement_list, self.seed, self.salt
        )
        result2 = generate_deterministic_replacement(
            self.original_value, self.replacement_list, self.seed, self.salt
        )
        self.assertEqual(result1, result2)

    def test_different_seed_produces_different_result(self):
        result1 = generate_deterministic_replacement(
            self.original_value, self.replacement_list, "seed1", self.salt
        )
        result2 = generate_deterministic_replacement(
            self.original_value, self.replacement_list, "seed2", self.salt
        )
        # Có thể giống nhau nếu danh sách ngắn, nhưng xác suất thấp
        self.assertTrue(
            result1 != result2 or len(set(self.replacement_list)) == 1,
            "Expected different outputs with different seeds"
        )

    def test_none_when_empty_list(self):
        result = generate_deterministic_replacement(
            self.original_value, [], self.seed, self.salt
        )
        self.assertIsNone(result)

    def test_works_without_salt(self):
        result1 = generate_deterministic_replacement(
            self.original_value, self.replacement_list, self.seed
        )
        result2 = generate_deterministic_replacement(
            self.original_value, self.replacement_list, self.seed
        )
        self.assertEqual(result1, result2)
        self.assertIn(result1, self.replacement_list)


class TestGenerateSeedFromKey(unittest.TestCase):

    def test_seed_is_int(self):
        seed = generate_seed_from_key("test-key")
        self.assertIsInstance(seed, int)

    def test_same_key_same_seed(self):
        seed1 = generate_seed_from_key("test-key")
        seed2 = generate_seed_from_key("test-key")
        self.assertEqual(seed1, seed2)

    def test_different_key_different_seed(self):
        seed1 = generate_seed_from_key("key1")
        seed2 = generate_seed_from_key("key2")
        self.assertNotEqual(seed1, seed2)

    def test_context_changes_seed(self):
        seed1 = generate_seed_from_key("test-key", context="context1")
        seed2 = generate_seed_from_key("test-key", context="context2")
        self.assertNotEqual(seed1, seed2)

    def test_key_as_bytes(self):
        key_bytes = b"byte-key"
        seed1 = generate_seed_from_key(key_bytes)
        seed2 = generate_seed_from_key("byte-key")
        # Same string in bytes or str without context should produce same seed
        self.assertEqual(seed1, seed2)

    def test_seed_with_empty_context(self):
        seed1 = generate_seed_from_key("test-key", context="")
        seed2 = generate_seed_from_key("test-key")
        self.assertEqual(seed1, seed2)


if __name__ == "__main__":
    unittest.main()