import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest import mock
from pamola_core.profiling.commons import text_utils
import tempfile
import shutil
import os

class DummyLogger:
    def __init__(self):
        self.infos = []
        self.debugs = []
        self.warnings = []
    def info(self, msg):
        self.infos.append(msg)
    def debug(self, msg):
        self.debugs.append(msg)
    def warning(self, msg):
        self.warnings.append(msg)

@pytest.fixture(autouse=True)
def patch_logger(monkeypatch):
    dummy = DummyLogger()
    monkeypatch.setattr(text_utils, "logger", dummy)
    return dummy

class TestAnalyzeNullAndEmpty:
    def test_valid_case(self):
        df = pd.DataFrame({"col": ["a", "", None, "   ", "b"]})
        result = text_utils.analyze_null_and_empty(df, "col")
        assert result["total_records"] == 5
        assert result["null_values"]["count"] == 1
        assert result["empty_strings"]["count"] == 2
        assert result["whitespace_strings"]["count"] == 1
        assert result["actual_data"]["count"] == 1

    def test_edge_empty_df(self):
        df = pd.DataFrame({"col": []})
        result = text_utils.analyze_null_and_empty(df, "col")
        assert result["total_records"] == 0
        for k in ["null_values", "empty_strings", "whitespace_strings", "actual_data"]:
            assert result[k]["count"] == 0
            assert result[k]["percentage"] == 0

    def test_chunking(self, monkeypatch):
        called = {}
        def fake_chunk(df, field, chunk):
            called["called"] = True
            return {"total_records": 1, "null_values": {"count": 0, "percentage": 0},
                    "empty_strings": {"count": 0, "percentage": 0},
                    "whitespace_strings": {"count": 0, "percentage": 0},
                    "actual_data": {"count": 1, "percentage": 100}}
        monkeypatch.setattr(text_utils, "analyze_null_and_empty_in_chunks", fake_chunk)
        df = pd.DataFrame({"col": ["a"]*11})
        text_utils.analyze_null_and_empty(df, "col", chunk_size=10)
        assert called["called"]

    def test_invalid_field(self):
        df = pd.DataFrame({"col": [1, 2, 3]})
        with pytest.raises(KeyError):
            text_utils.analyze_null_and_empty(df, "not_a_col")

class TestAnalyzeNullAndEmptyInChunks:
    def test_valid_case(self):
        df = pd.DataFrame({"col": ["a", "", None, "   ", "b"]*10})
        result = text_utils.analyze_null_and_empty_in_chunks(df, "col", chunk_size=7)
        assert result["total_records"] == 50
        assert result["null_values"]["count"] == 10
        assert result["empty_strings"]["count"] == 20
        assert result["whitespace_strings"]["count"] == 10
        assert result["actual_data"]["count"] == 10

    def test_edge_empty_df(self):
        df = pd.DataFrame({"col": []})
        result = text_utils.analyze_null_and_empty_in_chunks(df, "col", chunk_size=3)
        assert result["total_records"] == 0
        for k in ["null_values", "empty_strings", "whitespace_strings", "actual_data"]:
            assert result[k]["count"] == 0
            assert result[k]["percentage"] == 0

    def test_invalid_field(self):
        df = pd.DataFrame({"col": [1, 2, 3]})
        with pytest.raises(KeyError):
            text_utils.analyze_null_and_empty_in_chunks(df, "not_a_col", chunk_size=2)

class TestCalculateLengthStats:
    def test_valid_case(self):
        texts = ["abc", "de", "fghij"]
        result = text_utils.calculate_length_stats(texts)
        assert result["min"] == 2
        assert result["max"] == 5
        assert result["mean"] > 0
        assert result["median"] in [2, 3, 5]
        assert "length_distribution" in result

    def test_empty_list(self):
        result = text_utils.calculate_length_stats([])
        assert result["min"] == 0
        assert result["max"] == 0
        assert result["mean"] == 0
        assert result["median"] == 0
        assert result["std"] == 0
        assert result["length_distribution"] == {}

    def test_max_texts_sampling(self):
        texts = [str(i) for i in range(100)]
        result = text_utils.calculate_length_stats(texts, max_texts=10)
        assert result["min"] >= 1
        assert result["max"] <= 100

    def test_cache(self, tmp_path, monkeypatch):
        called = {}
        def fake_load(key, dir, op):
            called["called"] = True
            return {"min": 1, "max": 2, "mean": 1.5, "median": 1, "std": 0.5, "length_distribution": {}}
        monkeypatch.setattr(text_utils, "load_cached_result", fake_load)
        result = text_utils.calculate_length_stats(["a", "bb"], use_cache=True, cache_key="k", cache_dir=tmp_path)
        assert called["called"]
        assert result["min"] == 1

    def test_cache_save(self, tmp_path, monkeypatch):
        monkeypatch.setattr(text_utils, "load_cached_result", lambda *a, **k: None)
        called = {}
        def fake_save(res, key, dir, op):
            called["called"] = True
            return True
        monkeypatch.setattr(text_utils, "save_cached_result", fake_save)
        result = text_utils.calculate_length_stats(["a", "bb"], use_cache=True, cache_key="k", cache_dir=tmp_path)
        assert called["called"]

class TestChunkTexts:
    def test_valid_case(self):
        texts = [str(i) for i in range(10)]
        chunks = text_utils.chunk_texts(texts, 3)
        assert len(chunks) == 4
        assert all(isinstance(c, list) for c in chunks)

    def test_empty(self):
        assert text_utils.chunk_texts([], 3) == []

    def test_chunk_size_larger_than_list(self):
        texts = ["a", "b"]
        chunks = text_utils.chunk_texts(texts, 10)
        assert chunks == [["a", "b"]]

class TestProcessTextsInChunks:
    def test_valid_case(self):
        def proc(chunk, **kwargs):
            return [x.upper() for x in chunk]
        with mock.patch("pamola_core.utils.nlp.base.batch_process", side_effect=lambda texts, func, chunk_size, **kwargs: func(texts, **kwargs)):
            result = text_utils.process_texts_in_chunks(["a", "b"], proc, chunk_size=2)
            assert result == ["A", "B"]

class TestMergeAnalysisResults:
    def test_valid_case(self):
        r1 = {"total_records": 2, "null_count": 1, "empty_count": 1, "category_distribution": {"a": 1}, "aliases_distribution": {"x": 2},
              "null_values": {"count": 1, "percentage": 50}, "empty_strings": {"count": 1, "percentage": 50}, "whitespace_strings": {"count": 0, "percentage": 0}, "actual_data": {"count": 0, "percentage": 0}}
        r2 = {"total_records": 3, "null_count": 0, "empty_count": 2, "category_distribution": {"a": 2}, "aliases_distribution": {"x": 1},
              "null_values": {"count": 0, "percentage": 0}, "empty_strings": {"count": 2, "percentage": 66.7}, "whitespace_strings": {"count": 1, "percentage": 33.3}, "actual_data": {"count": 0, "percentage": 0}}
        merged = text_utils.merge_analysis_results([r1, r2])
        assert merged["total_records"] == 5
        assert merged["null_count"] == 1
        assert merged["empty_count"] == 3
        assert merged["category_distribution"]["a"] == 3
        assert merged["aliases_distribution"]["x"] == 3
        assert merged["null_values"]["percentage"] >= 0

    def test_empty(self):
        assert text_utils.merge_analysis_results([]) == {}

    def test_single(self):
        r = {"total_records": 1, "null_count": 0, "empty_count": 0}
        assert text_utils.merge_analysis_results([r]) == r

class TestCalculateWordFrequencies:
    def test_valid_case(self, monkeypatch):
        monkeypatch.setattr(text_utils, "nlp_calculate_word_frequencies", lambda texts, stop_words=None, min_word_length=3: {"hello": 2, "world": 1})
        result = text_utils.calculate_word_frequencies(["hello world", "hello"])
        assert result["hello"] >= 2
        assert "world" in result

    def test_stop_words(self, monkeypatch):
        monkeypatch.setattr(text_utils, "nlp_calculate_word_frequencies", lambda texts, stop_words=None, min_word_length=3: {"quick": 1, "brown": 1, "fox": 1})
        result = text_utils.calculate_word_frequencies(["the quick brown fox"], stop_words={"the"})
        assert "the" not in result

    def test_min_word_length(self, monkeypatch):
        monkeypatch.setattr(text_utils, "nlp_calculate_word_frequencies", lambda texts, stop_words=None, min_word_length=4: {"abcd": 1})
        result = text_utils.calculate_word_frequencies(["a ab abc abcd"], min_word_length=4)
        assert "abcd" in result
        assert "abc" not in result

    def test_chunking(self, monkeypatch):
        def fake_proc(texts, stop_words=None, min_word_length=3, max_words=None):
            return {"foo": 1}
        monkeypatch.setattr(text_utils, "nlp_calculate_word_frequencies", fake_proc)
        monkeypatch.setattr(text_utils, "process_texts_in_chunks", lambda texts, func, chunk_size, **kwargs: [fake_proc(texts, **kwargs), fake_proc(texts, **kwargs)])
        result = text_utils.calculate_word_frequencies(["a", "b"]*10, chunk_size=2)
        assert result["foo"] == 2

class TestCalculateTermFrequencies:
    def test_valid_case(self, monkeypatch):
        monkeypatch.setattr(text_utils, "nlp_calculate_term_frequencies", lambda texts, language="auto", stop_words=None, min_word_length=3: {"hello": 2, "world": 1})
        result = text_utils.calculate_term_frequencies(["hello world", "hello"])
        assert result["hello"] >= 2
        assert "world" in result

    def test_stop_words(self, monkeypatch):
        monkeypatch.setattr(text_utils, "nlp_calculate_term_frequencies", lambda texts, language="auto", stop_words=None, min_word_length=3: {"quick": 1, "brown": 1, "fox": 1})
        result = text_utils.calculate_term_frequencies(["the quick brown fox"], stop_words={"the"})
        assert "the" not in result

    def test_min_word_length(self, monkeypatch):
        monkeypatch.setattr(text_utils, "nlp_calculate_term_frequencies", lambda texts, language="auto", stop_words=None, min_word_length=4: {"abcd": 1})
        result = text_utils.calculate_term_frequencies(["a ab abc abcd"], min_word_length=4)
        assert "abcd" in result
        assert "abc" not in result

    def test_chunking(self, monkeypatch):
        def fake_proc(texts, language="auto", stop_words=None, min_word_length=3, max_terms=None):
            return {"foo": 1}
        monkeypatch.setattr(text_utils, "nlp_calculate_term_frequencies", fake_proc)
        monkeypatch.setattr(text_utils, "process_texts_in_chunks", lambda texts, func, chunk_size, **kwargs: [fake_proc(texts, **kwargs), fake_proc(texts, **kwargs)])
        result = text_utils.calculate_term_frequencies(["a", "b"]*10, chunk_size=2)
        assert result["foo"] == 2

class TestGetCacheKeyForTexts:
    def test_valid_case(self):
        key = text_utils.get_cache_key_for_texts(["a", "b"], "op", {"x": 1})
        assert key.startswith("text_utils_op_")

    def test_empty_texts(self):
        key = text_utils.get_cache_key_for_texts([], "op")
        assert key.startswith("text_utils_op_")

    def test_long_texts(self):
        texts = [str(i) for i in range(200)]
        key = text_utils.get_cache_key_for_texts(texts, "op")
        assert key.startswith("text_utils_op_")

class TestLoadCachedResult:
    def test_valid_case(self, tmp_path, monkeypatch):
        d = {"a": 1}
        file = tmp_path / "text_utils_k_length_stats.json"
        file.write_text("{\"a\": 1}")
        monkeypatch.setattr(text_utils, "read_json", lambda f: d)
        monkeypatch.setattr(Path, "exists", lambda self: True)
        result = text_utils.load_cached_result("k", tmp_path, "length_stats")
        assert result == d

    def test_file_not_exists(self, tmp_path):
        result = text_utils.load_cached_result("k", tmp_path, "length_stats")
        assert result is None

    def test_read_json_exception(self, tmp_path, monkeypatch):
        file = tmp_path / "text_utils_k_length_stats.json"
        file.write_text("bad json")
        monkeypatch.setattr(text_utils, "read_json", lambda f: (_ for _ in ()).throw(Exception("fail")))
        result = text_utils.load_cached_result("k", tmp_path, "length_stats")
        assert result is None

class TestSaveCachedResult:
    def test_valid_case(self, tmp_path, monkeypatch):
        monkeypatch.setattr(text_utils, "ensure_directory", lambda d: None)
        monkeypatch.setattr(text_utils, "write_json", lambda d, f: True)
        result = text_utils.save_cached_result({"a": 1}, "k", tmp_path, "length_stats")
        assert result is True

    def test_write_json_exception(self, tmp_path, monkeypatch):
        monkeypatch.setattr(text_utils, "ensure_directory", lambda d: None)
        monkeypatch.setattr(text_utils, "write_json", lambda d, f: (_ for _ in ()).throw(Exception("fail")))
        result = text_utils.save_cached_result({"a": 1}, "k", tmp_path, "length_stats")
        assert result is False

class TestDetectTextType:
    def test_job(self):
        assert text_utils.detect_text_type("Senior Engineer") == "job"
        assert text_utils.detect_text_type("Менеджер проекта") == "job"
    def test_organization(self):
        assert text_utils.detect_text_type("Acme Inc") == "organization"
        assert text_utils.detect_text_type("университет") == "organization"
    def test_transaction(self):
        assert text_utils.detect_text_type("Payment for invoice") == "transaction"
        assert text_utils.detect_text_type("оплата счета") == "transaction"
    def test_skill(self):
        assert text_utils.detect_text_type("Python programming") == "skill"
        assert text_utils.detect_text_type("программирование на Python") == "skill"
    def test_generic(self):
        assert text_utils.detect_text_type("Random text") == "generic"
        assert text_utils.detect_text_type("") == "generic"

class TestSuggestEntityType:
    def test_valid_case(self):
        texts = ["Senior Engineer", "Acme Inc", "Payment for invoice", "Python programming"]*30
        result = text_utils.suggest_entity_type(texts, sample_size=10)
        assert result in {"job", "organization", "transaction", "skill", "generic"}

    def test_empty(self):
        assert text_utils.suggest_entity_type([]) == "generic"

    def test_all_generic(self):
        texts = ["foo", "bar", "baz"]
        assert text_utils.suggest_entity_type(texts) == "generic"

    def test_sample_size(self):
        texts = ["job engineer"]*200
        result = text_utils.suggest_entity_type(texts, sample_size=5)
        assert result == "job"

if __name__ == "__main__":
    pytest.main()
