import pickle
import unittest
from io import BytesIO
from pathlib import Path
from time import sleep
from unittest.mock import patch, Mock, mock_open
import pandas as pd
from pamola_core.fake_data.commons.mapping_store import MappingStore



class TestMappingStoreInit(unittest.TestCase):

    def test_initial_state(self):
        store = MappingStore()
        self.assertEqual(store.mappings, {})
        self.assertEqual(store.reverse_mappings, {})
        self.assertEqual(store.transitivity_markers, {})
        self.assertEqual(store.metadata["version"], "1.0")
        self.assertEqual(store.metadata["fields"], {})
        self.assertIsNone(store.metadata["created_at"])
        self.assertIsNone(store.metadata["updated_at"])


class TestMappingStoreAddMapping(unittest.TestCase):

    def test_add_mapping_creates_field_entry(self):
        store = MappingStore()
        store.add_mapping("email", "user@example.com", "anon123@example.com", is_transitive=True)

        self.assertIn("email", store.mappings)
        self.assertEqual(store.mappings["email"]["user@example.com"], "anon123@example.com")

        self.assertIn("email", store.reverse_mappings)
        self.assertEqual(store.reverse_mappings["email"]["anon123@example.com"], "user@example.com")

        self.assertTrue(store.transitivity_markers["email"]["user@example.com"])

        self.assertEqual(store.metadata["fields"]["email"]["count"], 1)
        self.assertEqual(store.metadata["fields"]["email"]["type"], "str")

        self.assertIsNotNone(store.metadata["created_at"])
        self.assertIsNotNone(store.metadata["updated_at"])

    def test_add_multiple_mappings_same_field(self):
        store = MappingStore()
        store.add_mapping("phone", "123", "abc")
        sleep(0.01)  # Ensure timestamps differ for updated_at
        store.add_mapping("phone", "456", "def")

        self.assertEqual(len(store.mappings["phone"]), 2)
        self.assertEqual(store.metadata["fields"]["phone"]["count"], 2)

        created = store.metadata["created_at"]
        updated = store.metadata["updated_at"]
        self.assertIsNotNone(created)
        self.assertIsNotNone(updated)
        self.assertNotEqual(created, updated)  # updated_at should be newer than created_at

    def test_transitive_default(self):
        store = MappingStore()
        store.add_mapping("name", "Alice", "User1")
        self.assertFalse(store.transitivity_markers["name"]["Alice"])


class TestMappingStoreUpdateMapping(unittest.TestCase):

    def test_update_mapping_success(self):
        store = MappingStore()
        store.add_mapping("email", "user@example.com", "anon123@example.com")
        old_updated_at = store.metadata["updated_at"]

        sleep(0.01)  # Ensure timestamp will change
        result = store.update_mapping("email", "user@example.com", "anon456@example.com")

        self.assertTrue(result)
        self.assertEqual(store.mappings["email"]["user@example.com"], "anon456@example.com")
        self.assertEqual(store.reverse_mappings["email"]["anon456@example.com"], "user@example.com")
        self.assertNotIn("anon123@example.com", store.reverse_mappings["email"])
        self.assertNotEqual(store.metadata["updated_at"], old_updated_at)

    def test_update_mapping_nonexistent_field_or_key(self):
        store = MappingStore()

        result1 = store.update_mapping("email", "user@example.com", "anon456@example.com")
        self.assertFalse(result1)

        store.add_mapping("email", "user@example.com", "anon123@example.com")
        result2 = store.update_mapping("email", "nonexistent@example.com", "anon456@example.com")
        self.assertFalse(result2)

    def test_update_transitivity_flag(self):
        store = MappingStore()
        store.add_mapping("email", "user1@example.com", "shared@example.com")
        store.add_mapping("email", "user2@example.com", "anon456@example.com")

        # Transitivity should initially be False for both
        self.assertFalse(store.transitivity_markers["email"]["user1@example.com"])
        self.assertFalse(store.transitivity_markers["email"]["user2@example.com"])

        # Now update user2 to use the same synthetic value
        result = store.update_mapping("email", "user2@example.com", "shared@example.com")

        self.assertTrue(result)
        self.assertEqual(store.mappings["email"]["user2@example.com"], "shared@example.com")

        # Only user2's transitivity marker is updated
        self.assertTrue(store.transitivity_markers["email"]["user2@example.com"])
        self.assertFalse(store.transitivity_markers["email"]["user1@example.com"])  # Still unchanged


class TestMappingStoreGetMapping(unittest.TestCase):

    def test_get_mapping_exists(self):
        store = MappingStore()
        store.add_mapping("email", "user@example.com", "anon123@example.com")
        result = store.get_mapping("email", "user@example.com")
        self.assertEqual(result, "anon123@example.com")

    def test_get_mapping_field_not_exists(self):
        store = MappingStore()
        result = store.get_mapping("nonexistent_field", "user@example.com")
        self.assertIsNone(result)

    def test_get_mapping_original_not_exists(self):
        store = MappingStore()
        store.add_mapping("email", "user@example.com", "anon123@example.com")
        result = store.get_mapping("email", "unknown@example.com")
        self.assertIsNone(result)


class TestMappingStoreRestoreOriginal(unittest.TestCase):

    def test_restore_original_field_not_exist(self):
        store = MappingStore()
        self.assertIsNone(store.restore_original("nonexistent_field", "anon@example.com"))

    def test_restore_original_synthetic_exists(self):
        store = MappingStore()
        store.add_mapping("email", "user@example.com", "anon@example.com")
        self.assertEqual(store.restore_original("email", "anon@example.com"), "user@example.com")

    def test_restore_original_synthetic_not_found(self):
        store = MappingStore()
        store.add_mapping("email", "user@example.com", "anon@example.com")
        self.assertIsNone(store.restore_original("email", "not_found@example.com"))


class TestMappingStoreIsTransitive(unittest.TestCase):

    def test_is_transitive_field_not_exist(self):
        store = MappingStore()
        self.assertFalse(store.is_transitive("email", "user@example.com"))

    def test_is_transitive_original_not_in_transitivity_markers(self):
        store = MappingStore()
        store.add_mapping("email", "user1@example.com", "anon1@example.com", is_transitive=False)
        self.assertFalse(store.is_transitive("email", "nonexistent@example.com"))

    def test_is_transitive_mapping_marked_transitive(self):
        store = MappingStore()
        store.add_mapping("email", "user2@example.com", "anon2@example.com", is_transitive=True)
        self.assertTrue(store.is_transitive("email", "user2@example.com"))

    def test_is_transitive_mapping_not_transitive(self):
        store = MappingStore()
        store.add_mapping("email", "user1@example.com", "anon1@example.com", is_transitive=False)
        self.assertFalse(store.is_transitive("email", "user1@example.com"))


class TestMappingStoreMarkAsTransitive(unittest.TestCase):

    def test_mark_as_transitive_field_not_exist(self):
        store = MappingStore()
        result = store.mark_as_transitive("email", "user@example.com")
        self.assertFalse(result)

    def test_mark_as_transitive_successful(self):
        store = MappingStore()
        store.add_mapping("email", "user@example.com", "anon@example.com", is_transitive=False)

        self.assertFalse(store.is_transitive("email", "user@example.com"))
        result = store.mark_as_transitive("email", "user@example.com")
        self.assertTrue(result)
        self.assertTrue(store.is_transitive("email", "user@example.com"))

    def test_mark_as_transitive_updates_timestamp(self):
        store = MappingStore()
        store.add_mapping("email", "user@example.com", "anon@example.com", is_transitive=False)
        old_updated_at = store.metadata["updated_at"]
        sleep(0.01)  # Ensure updated_at will change
        store.mark_as_transitive("email", "user@example.com")
        self.assertNotEqual(store.metadata["updated_at"], old_updated_at)


class TestMappingStoreRemoveMapping(unittest.TestCase):

    def test_remove_mapping_field_not_exist(self):
        store = MappingStore()
        result = store.remove_mapping("nonexistent", "value")
        self.assertFalse(result)

    def test_remove_mapping_original_not_exist(self):
        store = MappingStore()
        store.add_mapping("email", "user@example.com", "anon@example.com")
        result = store.remove_mapping("email", "nonexistent@example.com")
        self.assertFalse(result)

    def test_remove_mapping_successful(self):
        store = MappingStore()
        store.add_mapping("email", "user@example.com", "anon@example.com", is_transitive=True)
        result = store.remove_mapping("email", "user@example.com")
        self.assertTrue(result)

    def test_remove_mapping_cleanup(self):
        store = MappingStore()
        store.add_mapping("email", "user@example.com", "anon@example.com", is_transitive=True)
        store.remove_mapping("email", "user@example.com")

        self.assertNotIn("user@example.com", store.mappings["email"])
        self.assertNotIn("anon@example.com", store.reverse_mappings["email"])
        self.assertNotIn("user@example.com", store.transitivity_markers["email"])
        self.assertEqual(store.metadata["fields"]["email"]["count"], 0)

    def test_remove_mapping_updates_timestamp(self):
        store = MappingStore()
        store.add_mapping("email", "user@example.com", "anon@example.com")
        old_updated_at = store.metadata["updated_at"]
        sleep(0.01)  # Ensure timestamp will differ
        store.remove_mapping("email", "user@example.com")
        self.assertNotEqual(store.metadata["updated_at"], old_updated_at)


class TestMappingStoreGetFieldMappings(unittest.TestCase):

    def test_get_field_mappings_field_not_exist(self):
        store = MappingStore()
        # Check the case when the field does not exist, should return an empty dict
        self.assertEqual(store.get_field_mappings("nonexistent_field"), {})

    def test_get_field_mappings_field_exists(self):
        store = MappingStore()
        store.add_mapping("email", "user@example.com", "anon@example.com")
        store.add_mapping("email", "user2@example.com", "anon2@example.com")

        expected_mappings = {
            "user@example.com": "anon@example.com",
            "user2@example.com": "anon2@example.com"
        }
        # Check the case when the field exists and has mappings
        self.assertEqual(store.get_field_mappings("email"), expected_mappings)

    def test_get_field_mappings_field_with_empty_mappings(self):
        store = MappingStore()
        store.add_mapping("email", "user@example.com", "anon@example.com")
        # Remove all mappings for the "email" field
        store.remove_mapping("email", "user@example.com")
        # Check the case when the field exists but has no mappings
        self.assertEqual(store.get_field_mappings("email"), {})


class TestMappingStoreStats(unittest.TestCase):

    def setUp(self):
        self.store = MappingStore()

    def test_get_field_stats_field_not_exist(self):
        # Field không tồn tại trong metadata
        expected_stats = {
            "count": 0,
            "type": None,
            "transitive_count": 0
        }
        self.assertEqual(self.store.get_field_stats("nonexistent_field"), expected_stats)

    def test_get_field_stats_field_exists_no_transitives(self):
        # Tồn tại field, không có transitive
        self.store.add_mapping("email", "user@example.com", "anon@example.com")
        expected_stats = {
            "count": 1,
            "type": "str",
            "transitive_count": 0
        }
        self.assertEqual(self.store.get_field_stats("email"), expected_stats)

    def test_get_field_stats_field_exists_with_transitives(self):
        # Tồn tại field có mix giữa transitive và không transitive
        self.store.add_mapping("email", "user@example.com", "anon@example.com", is_transitive=True)
        self.store.add_mapping("email", "user2@example.com", "anon2@example.com", is_transitive=False)
        expected_stats = {
            "count": 2,
            "type": "str",
            "transitive_count": 1
        }
        self.assertEqual(self.store.get_field_stats("email"), expected_stats)

    def test_get_field_stats_field_with_no_mappings(self):
        # Tồn tại field nhưng không có mappings nào
        self.store.metadata["fields"]["email"] = {"count": 0, "type": "str"}
        expected_stats = {
            "count": 0,
            "type": "str",
            "transitive_count": 0
        }
        self.assertEqual(self.store.get_field_stats("email"), expected_stats)

    def test_get_all_stats_empty_store(self):
        # Check the case when the store is empty
        expected_stats = {
            "total_fields": 0,
            "total_mappings": 0,
            "created_at": self.store.metadata["created_at"],
            "updated_at": self.store.metadata["updated_at"],
            "fields": {}
        }
        self.assertEqual(self.store.get_all_stats(), expected_stats)

    def test_get_all_stats_with_mappings(self):
        self.store.add_mapping("email", "user@example.com", "anon@example.com")
        self.store.add_mapping("phone", "12345", "98765")
        # Check the case when the store has multiple fields with mappings
        expected_stats = {
            "total_fields": 2,
            "total_mappings": 2,
            "created_at": self.store.metadata["created_at"],
            "updated_at": self.store.metadata["updated_at"],
            "fields": {
                "email": {
                    "count": 1,
                    "type": "str",
                    "transitive_count": 0
                },
                "phone": {
                    "count": 1,
                    "type": "str",
                    "transitive_count": 0
                }
            }
        }
        self.assertEqual(self.store.get_all_stats(), expected_stats)

    def test_get_all_stats_with_multiple_mappings_and_transitive(self):
        self.store.add_mapping("email", "user@example.com", "anon@example.com", is_transitive=True)
        self.store.add_mapping("email", "user2@example.com", "anon2@example.com", is_transitive=False)
        self.store.add_mapping("phone", "12345", "98765")

        # Check the case when the store has multiple fields with mappings, including transitive mappings
        expected_stats = {
            "total_fields": 2,
            "total_mappings": 3,
            "created_at": self.store.metadata["created_at"],
            "updated_at": self.store.metadata["updated_at"],
            "fields": {
                "email": {
                    "count": 2,
                    "type": "str",
                    "transitive_count": 1
                },
                "phone": {
                    "count": 1,
                    "type": "str",
                    "transitive_count": 0
                }
            }
        }
        self.assertEqual(self.store.get_all_stats(), expected_stats)

    def test_get_all_stats_no_fields(self):
        self.store.mappings.clear()  # Ensure no fields are present
        # Check the case when no fields exist in the store
        expected_stats = {
            "total_fields": 0,
            "total_mappings": 0,
            "created_at": self.store.metadata["created_at"],
            "updated_at": self.store.metadata["updated_at"],
            "fields": {}
        }
        self.assertEqual(self.store.get_all_stats(), expected_stats)


class TestMappingStoreClear(unittest.TestCase):

    def setUp(self):
        self.store = MappingStore()

    def test_clear_field_existing_field(self):
        # Add some mappings
        self.store.add_mapping("email", "user@example.com", "anon@example.com")
        self.store.add_mapping("phone", "12345", "98765")

        # Clear "email" field
        self.store.clear_field("email")

        # Check that the "email" field has been cleared
        self.assertNotIn("email", self.store.mappings)
        self.assertNotIn("email", self.store.reverse_mappings)
        self.assertNotIn("email", self.store.transitivity_markers)
        self.assertNotIn("email", self.store.metadata["fields"])

        # Check other fields are unaffected
        self.assertIn("phone", self.store.mappings)
        self.assertIn("phone", self.store.reverse_mappings)
        self.assertIn("phone", self.store.transitivity_markers)
        self.assertIn("phone", self.store.metadata["fields"])

    def test_clear_field_non_existent_field(self):
        # Try clearing a non-existent field
        self.store.clear_field("non_existent_field")

        # Ensure no changes are made to the store
        self.assertNotIn("non_existent_field", self.store.mappings)
        self.assertNotIn("non_existent_field", self.store.reverse_mappings)
        self.assertNotIn("non_existent_field", self.store.transitivity_markers)
        self.assertNotIn("non_existent_field", self.store.metadata["fields"])

    def test_clear_field_updates_timestamp(self):
        # Add a mapping to ensure there's a field
        self.store.add_mapping("email", "user@example.com", "anon@example.com")

        # Store the old timestamp
        old_updated_at = self.store.metadata["updated_at"]

        sleep(0.01)  # Ensure time passes

        # Clear the "email" field
        self.store.clear_field("email")

        # Check if the timestamp was updated
        self.assertNotEqual(self.store.metadata["updated_at"], old_updated_at)

    def test_clear_field_empty_store(self):
        # Clear a field when no fields exist
        self.store.clear_field("email")

        # Ensure the store is still empty
        self.assertEqual(self.store.mappings, {})
        self.assertEqual(self.store.reverse_mappings, {})
        self.assertEqual(self.store.transitivity_markers, {})
        self.assertEqual(self.store.metadata["fields"], {})

    def test_clear_all_removes_all_data_and_updates_timestamp(self):
        # Add multiple mappings
        self.store.add_mapping("email", "user1@example.com", "anon1@example.com")
        self.store.add_mapping("phone", "12345", "abcde")

        # Ensure mappings exist
        self.assertIn("email", self.store.mappings)
        self.assertIn("phone", self.store.mappings)
        self.assertIn("email", self.store.reverse_mappings)
        self.assertIn("phone", self.store.reverse_mappings)
        self.assertIn("email", self.store.transitivity_markers)
        self.assertIn("phone", self.store.transitivity_markers)
        self.assertIn("email", self.store.metadata["fields"])
        self.assertIn("phone", self.store.metadata["fields"])

        # Store old timestamp
        old_updated_at = self.store.metadata["updated_at"]
        sleep(0.01)

        # Clear all data
        self.store.clear_all()

        # Verify all internal dictionaries are empty
        self.assertEqual(self.store.mappings, {})
        self.assertEqual(self.store.reverse_mappings, {})
        self.assertEqual(self.store.transitivity_markers, {})
        self.assertEqual(self.store.metadata["fields"], {})

        # Verify updated_at changed
        self.assertNotEqual(self.store.metadata["updated_at"], old_updated_at)


class TestMappingStoreJson(unittest.TestCase):

    def setUp(self):
        self.store = MappingStore()

    @patch('pamola_core.utils.io.write_json')
    def test_save_json_without_io_module(self, mock_write_json):
        self.store.add_mapping("email", "user@example.com", "anon@example.com")

        output_path = Path("/fake/path/mapping.json")

        # Call save_json without providing io_module
        self.store.save_json(output_path)

        # Assert write_json from hhr_io was called correctly
        mock_write_json.assert_called_once()
        args, kwargs = mock_write_json.call_args
        self.assertIn("email", args[0]["mappings"])
        self.assertEqual(args[1], output_path)

    def test_save_json_with_io_module(self):
        self.store.add_mapping("phone", "123", "xyz")

        mock_io_module = Mock()
        output_path = "/custom/path/data.json"

        self.store.save_json(output_path, io_module=mock_io_module)

        mock_io_module.write_json.assert_called_once()
        args, kwargs = mock_io_module.write_json.call_args
        self.assertIn("phone", args[0]["mappings"])
        self.assertEqual(args[1], output_path)

    @patch("pamola_core.utils.io.read_json")
    def test_load_json_without_io_module(self, mock_read_json):
        sample_data = {
            "mappings": {
                "email": [
                    {"original": "a@example.com", "synthetic": "anon@example.com", "is_transitive": True}
                ]
            },
            "metadata": {
                "fields": {"email": {"count": 1, "type": "string"}},
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z"
            }
        }

        mock_read_json.return_value = sample_data
        self.store.load_json("dummy_path.json")

        # Assert write_json from hhr_io was called correctly
        mock_read_json.assert_called_once()

        self.assertEqual(self.store.mappings["email"]["a@example.com"], "anon@example.com")
        self.assertTrue(self.store.is_transitive("email", "a@example.com"))

    def test_load_json_with_io_module(self):
        mock_io = Mock()
        sample_data = {
            "mappings": {
                "phone": [
                    {"original": "12345", "synthetic": "xxxxx", "is_transitive": False}
                ]
            },
            "metadata": {
                "fields": {"phone": {"count": 1, "type": "string"}},
                "created_at": "2024-03-01T00:00:00Z",
                "updated_at": "2024-03-02T00:00:00Z"
            }
        }

        mock_io.read_json.return_value = sample_data
        self.store.load_json("another_path.json", io_module=mock_io)

        self.assertEqual(self.store.mappings["phone"]["12345"], "xxxxx")
        self.assertFalse(self.store.is_transitive("phone", "12345"))
        self.assertEqual(self.store.metadata["fields"]["phone"]["count"], 1)

    def test_update_from_json_with_overwrite(self):
        mock_io = Mock()
        mock_io.read_json.return_value = {
            "mappings": {
                "email": [
                    {"original": "a@example.com", "synthetic": "syn_a@example.com"},
                    {"original": "b@example.com", "synthetic": "syn_b@example.com", "is_transitive": True}
                ],
                "phone": [
                    {"original": "+1-202-555-0100", "synthetic": "+1-999-555-0001"},
                    {"original": "+1-202-555-0101", "synthetic": "+1-999-555-0002"}
                ]
            }
        }

        result = self.store.update_from_json(path="fake_path.json", overwrite_existing=True, io_module=mock_io)

        self.assertEqual(result, {"email": 2, "phone": 2})
        self.assertIn("a@example.com", self.store.mappings["email"])
        self.assertEqual(self.store.mappings["email"]["b@example.com"], "syn_b@example.com")
        self.assertTrue(self.store.transitivity_markers["email"]["b@example.com"])

        self.assertIn("+1-202-555-0100", self.store.mappings["phone"])
        self.assertEqual(self.store.mappings["phone"]["+1-202-555-0101"], "+1-999-555-0002")
        self.assertEqual(self.store.reverse_mappings["phone"]["+1-999-555-0001"], "+1-202-555-0100")

    def test_update_from_json_without_overwrite(self):
        # Setup initial mapping that should not be overwritten
        self.store.add_mapping("phone", "+1-202-555-0100", "+1-999-555-0001")

        mock_io = Mock()
        mock_io.read_json.return_value = {
            "mappings": {
                "phone": [
                    {"original": "+1-202-555-0100", "synthetic": "+1-999-555-0009"},  # Should NOT overwrite
                    {"original": "+1-202-555-0102", "synthetic": "+1-999-555-0003"}   # Should be added
                ]
            }
        }

        result = self.store.update_from_json(path="fake_path.json", overwrite_existing=False, io_module=mock_io)

        self.assertEqual(result, {"phone": 1})
        self.assertEqual(self.store.mappings["phone"]["+1-202-555-0100"], "+1-999-555-0001")  # Unchanged
        self.assertEqual(self.store.mappings["phone"]["+1-202-555-0102"], "+1-999-555-0003")  # New

    def test_update_from_json_with_field_filtering(self):
        mock_io = Mock()
        mock_io.read_json.return_value = {
            "mappings": {
                "email": [
                    {"original": "c@example.com", "synthetic": "syn_c@example.com"}
                ],
                "phone": [
                    {"original": "+1-202-555-0103", "synthetic": "+1-999-555-0004"}
                ]
            }
        }

        result = self.store.update_from_json(
            path="fake_path.json",
            overwrite_existing=True,
            fields_to_update=["email"],  # Only update 'email'
            io_module=mock_io
        )

        self.assertEqual(result, {"email": 1})
        self.assertIn("c@example.com", self.store.mappings["email"])
        self.assertNotIn("phone", self.store.mappings)  # Should not exist


class TestMappingStorePickle(unittest.TestCase):

    @patch("pamola_core.utils.io.ensure_directory")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_pickle_writes_correct_data(self, mock_file, mock_ensure_dir):
        # Arrange
        store = MappingStore()
        store.mappings = {"email": {"a@example.com": "syn_a@example.com"}}
        store.reverse_mappings = {"email": {"syn_a@example.com": "a@example.com"}}
        store.transitivity_markers = {"email": {"a@example.com": False}}
        store.metadata = {"version": "1.0", "fields": {}, "created_at": None, "updated_at": None}

        test_path = "fake_path/store.pkl"

        # Act
        with patch("pickle.dump") as mock_pickle_dump:
            store.save_pickle(test_path)

        # Assert directory creation
        mock_ensure_dir.assert_called_once_with(Path(test_path).parent)

        # Assert file was opened in binary write mode
        mock_file.assert_called_once_with(Path(test_path), "wb")

        # Assert pickle.dump was called with correct data
        expected_data = {
            "mappings": store.mappings,
            "reverse_mappings": store.reverse_mappings,
            "transitivity_markers": store.transitivity_markers,
            "metadata": store.metadata,
        }
        mock_pickle_dump.assert_called_once()
        args, kwargs = mock_pickle_dump.call_args
        self.assertEqual(args[0], expected_data)  # check pickle.dump(data, f)

    @patch("builtins.open", new_callable=mock_open)
    def test_load_pickle_loads_correct_data(self, mock_file):
        # Arrange
        store = MappingStore()
        mocked_data = {
            "mappings": {"phone": {"0901234567": "+84901234567"}},
            "reverse_mappings": {"phone": {"+84901234567": "0901234567"}},
            "transitivity_markers": {"phone": {"0901234567": False}},
            "metadata": {"version": "1.0", "fields": {}, "created_at": None, "updated_at": None},
        }
        pickled_bytes = pickle.dumps(mocked_data)
        mock_file.return_value = BytesIO(pickled_bytes)

        with patch("pickle.load", return_value=mocked_data) as mock_pickle_load:
            # Act
            store.load_pickle("fake_path/store.pkl")

        # Assert
        mock_pickle_load.assert_called_once()
        self.assertEqual(store.mappings, mocked_data["mappings"])
        self.assertEqual(store.reverse_mappings, mocked_data["reverse_mappings"])
        self.assertEqual(store.transitivity_markers, mocked_data["transitivity_markers"])
        self.assertEqual(store.metadata, mocked_data["metadata"])


class TestMappingStoreDataFrame(unittest.TestCase):

    def setUp(self):
        self.store = MappingStore()

    def test_to_dataframe_empty_field(self):
        # Field does not exist → should return empty DataFrame
        df = self.store.to_dataframe("email")
        self.assertTrue(df.empty)
        self.assertListEqual(list(df.columns), ["original", "synthetic", "is_transitive"])

    def test_to_dataframe_with_data(self):
        # Add mappings and verify the returned DataFrame
        self.store.add_mapping("email", "a@example.com", "anon_a@example.com", is_transitive=True)
        self.store.add_mapping("email", "b@example.com", "anon_b@example.com", is_transitive=False)

        df = self.store.to_dataframe("email")
        self.assertEqual(len(df), 2)
        self.assertIn("original", df.columns)
        self.assertIn("synthetic", df.columns)
        self.assertIn("is_transitive", df.columns)

        # Verify specific row values
        record = df[df["original"] == "a@example.com"].iloc[0]
        self.assertEqual(record["synthetic"], "anon_a@example.com")
        self.assertTrue(record["is_transitive"])

    def test_from_dataframe_adds_mappings(self):
        # Prepare input DataFrame
        df = pd.DataFrame([
            {"original": "u1@example.com", "synthetic": "anon1@example.com", "is_transitive": True},
            {"original": "u2@example.com", "synthetic": "anon2@example.com", "is_transitive": False},
        ])

        added = self.store.from_dataframe(df, "email")
        self.assertEqual(added, 2)

        # Verify the loaded data
        self.assertEqual(self.store.mappings["email"]["u1@example.com"], "anon1@example.com")
        self.assertTrue(self.store.transitivity_markers["email"]["u1@example.com"])
        self.assertFalse(self.store.transitivity_markers["email"]["u2@example.com"])

    def test_from_dataframe_without_overwrite(self):
        # Existing mapping
        self.store.add_mapping("email", "u1@example.com", "anon_old@example.com", is_transitive=False)

        # DataFrame with a duplicate key
        df = pd.DataFrame([
            {"original": "u1@example.com", "synthetic": "anon_new@example.com", "is_transitive": True},
            {"original": "u2@example.com", "synthetic": "anon2@example.com", "is_transitive": False},
        ])

        added = self.store.from_dataframe(df, "email", overwrite_existing=False)
        self.assertEqual(added, 1)  # Only u2@example.com should be added

        self.assertEqual(self.store.mappings["email"]["u1@example.com"], "anon_old@example.com")
        self.assertEqual(self.store.mappings["email"]["u2@example.com"], "anon2@example.com")

    def test_from_dataframe_with_custom_column_names(self):
        # Use custom column names
        df = pd.DataFrame([
            {"src": "u1", "dst": "v1", "flag": True},
            {"src": "u2", "dst": "v2", "flag": False}
        ])

        added = self.store.from_dataframe(df, "usernames",
                                          original_col="src",
                                          synthetic_col="dst",
                                          transitive_col="flag")
        self.assertEqual(added, 2)
        self.assertEqual(self.store.mappings["usernames"]["u1"], "v1")
        self.assertTrue(self.store.transitivity_markers["usernames"]["u1"])


class TestMappingStoreCSV(unittest.TestCase):

    def setUp(self):
        self.store = MappingStore()
        self.store.add_mapping("email", "user@example.com", "anon@example.com", is_transitive=True)
        self.store.add_mapping("phone", "0123456789", "0987654321")

    @patch("pamola_core.utils.io.write_dataframe_to_csv")
    def test_save_csv_writes_expected_dataframe(self, mock_write_csv):
        path = Path("/fake/path/mappings.csv")

        # Act
        self.store.save_csv(path)

        # Assert
        mock_write_csv.assert_called_once()
        called_df = mock_write_csv.call_args[0][0]
        self.assertIsInstance(called_df, pd.DataFrame)
        self.assertEqual(set(called_df.columns),
                         {"field_name", "original", "synthetic", "is_transitive", "original_type"})
        self.assertEqual(len(called_df), 2)  # 2 mappings

    @patch("pamola_core.utils.io.read_full_csv")
    def test_load_csv_reads_and_restores_mappings(self, mock_read_csv):
        df = pd.DataFrame([
            {
                "field_name": "email",
                "original": "user@example.com",
                "synthetic": "anon@example.com",
                "is_transitive": True,
                "original_type": "str"
            },
            {
                "field_name": "phone",
                "original": "123",
                "synthetic": "456",
                "is_transitive": False,
                "original_type": "int"
            }
        ])
        mock_read_csv.return_value = df

        store = MappingStore()
        result = store.load_csv(Path("/fake/path/mappings.csv"))

        self.assertEqual(result["email"], 1)
        self.assertEqual(result["phone"], 1)
        self.assertEqual(store.mappings["email"]["user@example.com"], "anon@example.com")
        self.assertEqual(store.mappings["phone"][123], "456")
        self.assertTrue(store.transitivity_markers["email"]["user@example.com"])

    @patch("pamola_core.utils.io.read_full_csv")
    def test_load_csv_respects_fields_to_load(self, mock_read_csv):
        df = pd.DataFrame([
            {
                "field_name": "email",
                "original": "a@example.com",
                "synthetic": "x@example.com",
                "is_transitive": False,
                "original_type": "str"
            },
            {
                "field_name": "phone",
                "original": "999",
                "synthetic": "888",
                "is_transitive": False,
                "original_type": "str"
            }
        ])
        mock_read_csv.return_value = df

        store = MappingStore()
        result = store.load_csv(Path("any.csv"), fields_to_load=["email"])

        self.assertEqual(result["email"], 1)
        self.assertNotIn("phone", result)
        self.assertIn("email", store.mappings)
        self.assertNotIn("phone", store.mappings)


class TestMappingStoreSaveLoad(unittest.TestCase):

    def setUp(self):
        self.store = MappingStore()
        self.store.add_mapping("email", "user@example.com", "anon@example.com", is_transitive=True)
        self.store.add_mapping("phone", "0123456789", "0987654321")

    @patch("pamola_core.fake_data.commons.mapping_store.MappingStore.save_json")
    @patch("pamola_core.fake_data.commons.mapping_store.MappingStore.save_pickle")
    @patch("pamola_core.fake_data.commons.mapping_store.MappingStore.save_csv")
    def test_save_calls_correct_method(self, mock_save_csv, mock_save_pickle, mock_save_json):
        path = Path("/fake/path/mappings.json")

        # Test JSON format
        self.store.save(path, format="json")
        mock_save_json.assert_called_once()
        mock_save_pickle.assert_not_called()
        mock_save_csv.assert_not_called()

        # Test Pickle format
        self.store.save(path, format="pickle")
        mock_save_pickle.assert_called_once()
        mock_save_json.assert_called_once()  # Called in the previous test, need to reset for next run
        mock_save_csv.assert_not_called()

        # Test CSV format
        self.store.save(path, format="csv")
        mock_save_csv.assert_called_once()
        mock_save_json.assert_called_once()  # Called in the previous tests
        mock_save_pickle.assert_called_once()

    @patch("pamola_core.fake_data.commons.mapping_store.MappingStore.load_json")
    @patch("pamola_core.fake_data.commons.mapping_store.MappingStore.load_pickle")
    @patch("pamola_core.fake_data.commons.mapping_store.MappingStore.load_csv")
    def test_load_calls_correct_method(self, mock_load_csv, mock_load_pickle, mock_load_json):
        path = Path("/fake/path/mappings.json")

        # Test JSON format loading
        self.store.load(path, format="json")
        mock_load_json.assert_called_once()
        mock_load_pickle.assert_not_called()
        mock_load_csv.assert_not_called()

        # Test Pickle format loading
        path = Path("/fake/path/mappings.pkl")
        self.store.load(path, format="pickle")
        mock_load_pickle.assert_called_once()
        mock_load_json.assert_called_once()  # Called in the previous test, reset for next run
        mock_load_csv.assert_not_called()

        # Test CSV format loading
        path = Path("/fake/path/mappings.csv")
        self.store.load(path, format="csv")
        mock_load_csv.assert_called_once()
        mock_load_json.assert_called_once()  # Called in the previous tests
        mock_load_pickle.assert_called_once()

    @patch("pamola_core.fake_data.commons.mapping_store.MappingStore.load_pickle")
    @patch("pamola_core.fake_data.commons.mapping_store.MappingStore.load_csv")
    @patch("pamola_core.fake_data.commons.mapping_store.MappingStore.load_json")
    def test_load_infer_format_from_extension(self, mock_load_json, mock_load_csv, mock_load_pickle):
        store = MappingStore()

        # JSON file
        path_json = Path("/fake/path/mappings.json")
        store.load(path_json)
        mock_load_json.assert_called_once_with(path_json)

        # Pickle file
        path_pickle = Path("/fake/path/mappings.pkl")
        store.load(path_pickle)
        mock_load_pickle.assert_called_once_with(path_pickle)

        # CSV file
        path_csv = Path("/fake/path/mappings.csv")
        store.load(path_csv)
        mock_load_csv.assert_called_once_with(path_csv, True, None)

        # Đảm bảo đúng định dạng
        self.assertEqual(path_json.suffix, ".json")
        self.assertIn(path_pickle.suffix, [".pkl", ".pickle"])
        self.assertEqual(path_csv.suffix, ".csv")

    def test_save_invalid_format(self):
        path = Path("/fake/path/mappings.xyz")
        with self.assertRaises(ValueError):
            self.store.save(path, format="xyz")

    def test_load_invalid_format(self):
        path = Path("/fake/path/mappings.xyz")
        with self.assertRaises(ValueError):
            self.store.load(path, format="xyz")


if __name__ == "__main__":
    unittest.main()