"""
Tests for the utils.nlp.cache module.

This module contains unit tests for the abstract base classes and concrete
implementations in the cache.py module.
"""
import unittest
from unittest.mock import patch, mock_open, MagicMock
import tempfile
import os
import time
from collections import OrderedDict

# Mocks or sample values (replace these with real ones from your environment)
CACHE_ENABLED = True
MAX_CACHE_SIZE = 3
DEFAULT_CACHE_TTL = 2  # short for testing
POLICY_LRU = 'lru'
POLICY_LFU = 'lfu'
POLICY_FIFO = 'fifo'
POLICY_TTL = 'ttl'
POLICY_TLRU = 'tlru'

# Assuming MemoryCache is imported from your module
from pamola_core.utils.nlp.cache import (
    MemoryCache,
    FileCache,
    ModelCache,
    get_cache,
    cache_function,
    detect_file_encoding,
    _file_cache,
    _model_cache,
    _memory_cache
)



class TestMemoryCache(unittest.TestCase):
    """Tests for the MemoryCache class."""

    def test_set_get(self):
        cache = MemoryCache(max_size=3, ttl=5, policy=POLICY_LRU)
        cache.set("key1", "value1")
        self.assertEqual(cache.get("key1"), "value1")

    def test_expiration(self):
        cache = MemoryCache(ttl=1, policy=POLICY_LRU)
        cache.set("key1", "value1")
        time.sleep(1.5)
        self.assertIsNone(cache.get("key1"))

    def test_eviction_lru(self):
        cache = MemoryCache(max_size=2, policy=POLICY_LRU)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # Access 'a' to make it recently used
        cache.set("c", 3)  # Should evict 'b'
        self.assertIn("a", cache._cache)
        self.assertNotIn("b", cache._cache)

    def test_eviction_lfu(self):
        cache = MemoryCache(max_size=2, policy=POLICY_LFU)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # Hit 'a' once
        cache.get("a")  # Hit 'a' again
        cache.set("c", 3)  # Should evict 'b' (least frequently used)
        self.assertIn("a", cache._cache)
        self.assertNotIn("b", cache._cache)

    def test_eviction_fifo(self):
        cache = MemoryCache(max_size=2, policy=POLICY_FIFO)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # Should evict 'a'
        self.assertNotIn("a", cache._cache)
        self.assertIn("b", cache._cache)
        self.assertIn("c", cache._cache)

    def test_eviction_ttl_policy(self):
        cache = MemoryCache(max_size=2, policy=POLICY_TTL)
        cache.set("a", 1)
        time.sleep(0.5)
        cache.set("b", 2)
        cache.set("c", 3)  # Should evict 'a' (oldest timestamp)
        self.assertNotIn("a", cache._cache)

    def test_eviction_tlru_with_expired(self):
        cache = MemoryCache(max_size=2, ttl=1, policy=POLICY_TLRU)
        cache.set("a", 1)
        time.sleep(1.2)
        cache.set("b", 2)
        cache.set("c", 3)  # Should evict 'a' because it’s expired
        self.assertNotIn("a", cache._cache)

    def test_delete_and_clear(self):
        cache = MemoryCache(policy=POLICY_LRU)
        cache.set("a", 1)
        cache.set("b", 2)
        self.assertTrue(cache.delete("a"))
        self.assertFalse(cache.delete("z"))  # non-existent
        cache.clear()
        self.assertEqual(len(cache._cache), 0)

    def test_stats(self):
        cache = MemoryCache(policy=POLICY_LRU)
        cache.set("a", 1)
        _ = cache.get("a")  # hit
        _ = cache.get("b")  # miss
        stats = cache.get_stats()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertEqual(stats['size'], 1)



class TestFileCache(unittest.TestCase):
    """Tests for the FileCache class."""

    def setUp(self):
        self.cache = FileCache(max_size=MAX_CACHE_SIZE)
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.write(b"initial content")
        self.temp_file.flush()
        self.temp_file_path = self.temp_file.name

    def tearDown(self):
        self.temp_file.close()
        if os.path.exists(self.temp_file_path):
            os.unlink(self.temp_file_path)

    def test_set_and_get_valid_entry(self):
        self.cache.set("file1", "data", file_path=self.temp_file_path)
        result = self.cache.get("file1")
        self.assertEqual(result, "data")

    def test_get_invalid_when_file_modified(self):
        self.cache.set("file1", "data", file_path=self.temp_file_path)
        # Modify file to change mtime
        time.sleep(1)
        with open(self.temp_file_path, "a") as f:
            f.write("update")
        result = self.cache.get("file1")
        self.assertIsNone(result)

    def test_is_valid_true(self):
        self.cache.set("file1", "data", file_path=self.temp_file_path)
        self.assertTrue(self.cache.is_valid("file1"))

    def test_is_valid_false_on_modification(self):
        self.cache.set("file1", "data", file_path=self.temp_file_path)
        time.sleep(1)
        with open(self.temp_file_path, "a") as f:
            f.write("update")
        self.assertFalse(self.cache.is_valid("file1"))

    def test_eviction(self):
        # Add more than max_size
        self.cache.set("key1", "val1", file_path=self.temp_file_path)
        time.sleep(0.5)
        with tempfile.NamedTemporaryFile(delete=False) as f2:
            f2.write(b"test")
            f2.flush()
            file2 = f2.name
        self.cache.set("key2", "val2", file_path=file2)
        time.sleep(0.5)
        with tempfile.NamedTemporaryFile(delete=False) as f3:
            f3.write(b"test")
            f3.flush()
            file3 = f3.name
        self.cache.set("key3", "val3", file_path=file3)
        time.sleep(0.5)
        with tempfile.NamedTemporaryFile(delete=False) as f4:
            f4.write(b"test")
            f4.flush()
            file4 = f4.name
        self.cache.set("key4", "val4", file_path=file4) # should evict key1

        self.assertNotIn("key1", self.cache._cache)
        self.assertEqual(len(self.cache._cache), MAX_CACHE_SIZE)

        os.unlink(file2)
        os.unlink(file3)

    def test_delete(self):
        self.cache.set("file1", "data", file_path=self.temp_file_path)
        self.assertTrue(self.cache.delete("file1"))
        self.assertFalse(self.cache.delete("file1"))  # Already deleted

    def test_clear(self):
        self.cache.set("file1", "data", file_path=self.temp_file_path)
        self.cache.clear()
        self.assertEqual(len(self.cache._cache), 0)
        self.assertEqual(len(self.cache._file_paths), 0)

    def test_stats_tracking(self):
        self.cache.set("file1", "data", file_path=self.temp_file_path)
        _ = self.cache.get("file1")  # hit
        _ = self.cache.get("missing")  # miss
        stats = self.cache.get_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["file_tracked"], 1)



class DummyModel:
    pass

class TestModelCache(unittest.TestCase):
    """Tests for the ModelCache class."""

    def setUp(self):
        self.cache = ModelCache(max_size=3, memory_threshold=0.75, check_memory=True)

    def test_set_and_get_model(self):
        model = DummyModel()
        self.cache.set("model1", model)
        result = self.cache.get("model1")
        self.assertIs(result, model)

    def test_cache_hit_and_miss(self):
        model = DummyModel()
        self.cache.set("model1", model)
        _ = self.cache.get("model1")  # hit
        _ = self.cache.get("missing")  # miss
        stats = self.cache.get_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)

    def test_eviction_by_max_size(self):
        for i in range(3):
            self.cache.set(f"model{i}", DummyModel())
        self.cache.set("model3", DummyModel())  # should trigger eviction
        stats = self.cache.get_stats()
        self.assertEqual(stats["evictions"], 1)
        self.assertEqual(len(self.cache._cache), 3)

    def test_model_metadata_and_info(self):
        self.cache.set("model1", DummyModel(), metadata={"source": "huggingface"})
        info = self.cache.get_model_info("model1")
        self.assertIn("source", info)
        self.assertTrue(info["loaded"])
        self.assertIn("last_used", info)

        all_info = self.cache.get_model_info()
        self.assertIn("model1", all_info)

    def test_delete_and_clear(self):
        self.cache.set("model1", DummyModel())
        deleted = self.cache.delete("model1")
        self.assertTrue(deleted)
        self.assertFalse(self.cache.delete("model1"))

        self.cache.set("model2", DummyModel())
        self.cache.clear()
        self.assertEqual(len(self.cache._cache), 0)
        self.assertEqual(len(self.cache._metadata), 0)

    @patch("psutil.virtual_memory")
    def test_memory_pressure_triggers_eviction(self, mock_memory):
        mock_memory.return_value.percent = 90  # Simulate 90% memory usage
        self.cache.set("model1", DummyModel())
        self.cache.set("model2", DummyModel())
        self.cache.set("model3", DummyModel())
        # Add model4 which should trigger memory pressure
        self.cache.set("model4", DummyModel())
        stats = self.cache.get_stats()
        self.assertGreaterEqual(stats["memory_evictions"], 1)

    @patch("psutil.virtual_memory")
    def test_no_eviction_when_memory_below_threshold(self, mock_memory):
        mock_memory.return_value.percent = 50  # Simulate 50% usage
        for i in range(3):
            self.cache.set(f"model{i}", DummyModel())
        stats = self.cache.get_stats()
        self.assertEqual(stats["memory_evictions"], 0)

    def test_memory_check_disabled(self):
        cache = ModelCache(max_size=2, check_memory=False)
        with patch.object(cache, "_is_memory_pressure") as mock_check:
            cache.set("a", DummyModel())
            cache.set("b", DummyModel())
            cache.set("c", DummyModel())  # Should evict without checking memory
            mock_check.assert_not_called()



class DummyCache:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ttl=None):
        self.store[key] = value


class TestGlobalCacheUtilities(unittest.TestCase):

    def test_get_cache_returns_correct_instance(self):
        self.assertIs(get_cache("file"), _file_cache)
        self.assertIs(get_cache("model"), _model_cache)
        self.assertIs(get_cache("memory"), _memory_cache)
        self.assertIs(get_cache("unknown"), _memory_cache)

    def test_cache_function_decorator_caches_result(self):
        dummy_cache = DummyCache()

        with patch("pamola_core.utils.nlp.cache.get_cache", return_value=dummy_cache):

            call_counter = {"count": 0}

            @cache_function(ttl=10, cache_type='memory')
            def compute(x):
                call_counter["count"] += 1
                return x * 2

            result1 = compute(5)
            result2 = compute(5)

            self.assertEqual(result1, 10)
            self.assertEqual(result2, 10)
            self.assertEqual(call_counter["count"], 1)  # Second call hits cache

    @patch("builtins.open", new_callable=mock_open, read_data=b'test content')
    @patch("pamola_core.utils.nlp.cache.get_cache", return_value=DummyCache())
    def test_detect_file_encoding_with_chardet(self, mock_cache, mock_file):
        with patch("pamola_core.utils.nlp.cache.chardet.detect", return_value={'encoding': 'utf-8'}) as mock_detect:
            encoding = detect_file_encoding("fake.txt")
            self.assertEqual(encoding, "utf-8")
            self.assertIn("file_encoding:fake.txt", mock_cache.return_value.store)

    @patch("pamola_core.utils.nlp.cache.get_cache", return_value=DummyCache())
    def test_detect_file_encoding_with_import_error(self, mock_cache):
        with patch.dict("sys.modules", {"chardet": None}):
            encoding = detect_file_encoding("nonexistent.txt", fallback_encoding="latin1")
            self.assertEqual(encoding, "latin1")

    @patch("pamola_core.utils.nlp.cache.get_cache", return_value=DummyCache())
    def test_detect_file_encoding_cached(self, mock_cache):
        mock_cache.return_value.set("file_encoding:cached.txt", "utf-8")
        encoding = detect_file_encoding("cached.txt")
        self.assertEqual(encoding, "utf-8")



if __name__ == "__main__":
    unittest.main()
