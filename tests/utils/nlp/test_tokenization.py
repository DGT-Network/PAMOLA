"""
Tests for the utils.nlp.tokenization module.

This module contains unit tests for the abstract base classes and concrete
implementations in the tokenization.py module.
"""
import unittest
from unittest.mock import patch, MagicMock
from pamola_core.utils.nlp.tokenization import (
    NGramExtractor,
    LemmatizerRegistry,
    SimpleTokenizer,
    NLTKTokenizer,
    SpacyTokenizer,
    TransformersTokenizer,
    TokenizerFactory,
    TextProcessor,
    lemmatize,
    normalize_text,
    normalize_tokens,
    tokenize,
    calculate_word_frequencies,
    calculate_term_frequencies,
    tokenize_and_lemmatize,
    extract_ngrams,
    extract_character_ngrams,
    batch_tokenize,
    batch_tokenize_and_lemmatize
)



class TestNGramExtractor(unittest.TestCase):

    def test_character_ngrams_basic(self):
        extractor = NGramExtractor(n=3, pad_text=False)
        result = extractor.extract_character_ngrams("test")
        self.assertEqual(result, ["tes", "est"])

    def test_character_ngrams_with_padding(self):
        extractor = NGramExtractor(n=3, pad_text=True, pad_char="_")
        result = extractor.extract_character_ngrams("hi")
        self.assertEqual(result, ["__h", "_hi", "hi_", "i__"])

    def test_token_ngrams_basic(self):
        extractor = NGramExtractor(n=2)
        result = extractor.extract_token_ngrams(["this", "is", "a", "test"])
        self.assertEqual(result, [["this", "is"], ["is", "a"], ["a", "test"]])

    def test_token_ngrams_with_stopwords(self):
        extractor = NGramExtractor(n=2, skip_stopwords=True, language="en", stopwords={"is"})
        result = extractor.extract_token_ngrams(["this", "is", "a", "test"])
        self.assertEqual(result, [["this", "a"], ["a", "test"]])

    def test_unique_ngrams(self):
        extractor = NGramExtractor(n=2)
        result = extractor.get_unique_ngrams("test", is_tokens=False)
        self.assertIsInstance(result, set)
        self.assertIn("te", result)

    def test_ngram_counts(self):
        extractor = NGramExtractor(n=2)
        result = extractor.count_ngrams("hello", is_tokens=False)
        self.assertEqual(result.get("he"), 1)
        self.assertEqual(result.get("el"), 1)



class TestLemmatizerRegistry(unittest.TestCase):

    def test_register_and_get(self):
        dummy_lemmatizer = object()
        LemmatizerRegistry.register("xx", dummy_lemmatizer, overwrite=True)
        self.assertIs(LemmatizerRegistry.get("xx"), dummy_lemmatizer)

    def test_has_lemmatizer(self):
        LemmatizerRegistry.register("en", "dummy", overwrite=True)
        self.assertTrue(LemmatizerRegistry.has_lemmatizer("en"))

    def test_remove_lemmatizer(self):
        LemmatizerRegistry.register("de", "dummy", overwrite=True)
        removed = LemmatizerRegistry.remove("de")
        self.assertTrue(removed)
        self.assertIsNone(LemmatizerRegistry.get("de"))

    def test_clear_registry(self):
        LemmatizerRegistry.register("fr", "dummy", overwrite=True)
        LemmatizerRegistry.clear()
        self.assertFalse(LemmatizerRegistry.has_lemmatizer("fr"))



class TestSimpleTokenizer(unittest.TestCase):

    def test_tokenize_basic(self):
        tokenizer = SimpleTokenizer()
        result = tokenizer.tokenize("This is test.")
        self.assertEqual(result, ["this", "is", "test"])

    def test_tokenize_with_punctuation(self):
        tokenizer = SimpleTokenizer()
        result = tokenizer.tokenize("hello, world! test-case")
        self.assertIn("test", result)
        self.assertNotIn("-", result)



class TestNLTKTokenizer(unittest.TestCase):

    @patch("pamola_core.utils.nlp.base.DependencyManager.get_module")
    def test_nltk_tokenize_success(self, mock_get_module):
        # Simulate nltk.word_tokenize
        mock_nltk = MagicMock()
        mock_nltk.tokenize.word_tokenize = lambda text, language="english": text.split()
        mock_get_module.return_value = mock_nltk

        tokenizer = NLTKTokenizer(language="en")
        tokens = tokenizer.tokenize("This is test.")
        self.assertEqual(tokens, ["this", "is", "test"])

    @patch("pamola_core.utils.nlp.base.DependencyManager.get_module", return_value=None)
    def test_nltk_fallback(self, mock_get_module):
        tokenizer = NLTKTokenizer(language="en")
        tokens = tokenizer.tokenize("Fallback test.")
        self.assertIn("fallback", tokens)


class TestSpacyTokenizer(unittest.TestCase):

    @patch("pamola_core.utils.nlp.base.DependencyManager.get_module")
    def test_spacy_tokenize_success(self, mock_get_module):
        # Fake spaCy doc object
        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = [MagicMock(text="hello"), MagicMock(text="world")]
        mock_nlp = MagicMock(return_value=mock_doc)

        mock_spacy = MagicMock()
        mock_spacy.load.return_value = mock_nlp
        mock_get_module.return_value = mock_spacy

        tokenizer = SpacyTokenizer(language="en")
        tokens = tokenizer.tokenize("hello world")
        self.assertEqual(tokens, ["hello", "world"])

    @patch("pamola_core.utils.nlp.base.DependencyManager.get_module", return_value=None)
    def test_spacy_fallback(self, mock_get_module):
        tokenizer = SpacyTokenizer(language="en")
        tokens = tokenizer.tokenize("Fallback test.")
        self.assertIn("fallback", tokens)



class TestTransformersTokenizer(unittest.TestCase):

    @patch("pamola_core.utils.nlp.base.DependencyManager.get_module")
    @patch("transformers.AutoTokenizer")
    def test_transformers_tokenize_success(self, mock_auto_tokenizer_class, mock_get_module):
        # Simulate tokenizer.tokenize()
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.return_value = ["[CLS]", "hello", "world", "[SEP]"]
        mock_auto_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_get_module.return_value = MagicMock()

        tokenizer = TransformersTokenizer(language="en")
        tokens = tokenizer.tokenize("hello world", add_special_tokens=True)
        self.assertIn("hello", tokens)

    @patch("pamola_core.utils.nlp.base.DependencyManager.get_module", return_value=None)
    def test_transformers_fallback(self, mock_get_module):
        tokenizer = TransformersTokenizer(language="en")
        tokens = tokenizer.tokenize("Fallback test.")
        self.assertIn("fallback", tokens)



class TestTokenizerFactory(unittest.TestCase):

    def test_create_simple_tokenizer(self):
        tok = TokenizerFactory.create_tokenizer("simple")
        self.assertIsInstance(tok, SimpleTokenizer)

    def test_create_tokenizer_auto_fallback(self):
        # Should fallback to SimpleTokenizer if no other libs available
        tok = TokenizerFactory.create_tokenizer("auto", language="en", no_cache=True)
        self.assertTrue(hasattr(tok, "tokenize"))

    def test_cache_reuse(self):
        tok1 = TokenizerFactory.create_tokenizer("simple", language="en")
        tok2 = TokenizerFactory.create_tokenizer("simple", language="en")
        self.assertIs(tok1, tok2)  # Should be same instance due to caching



class TestTextProcessor(unittest.TestCase):

    @patch("pamola_core.utils.nlp.tokenization.TokenizerFactory.create_tokenizer")
    def test_tokenize(self, mock_create_tokenizer):
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.return_value = ["hello", "world"]
        mock_create_tokenizer.return_value = mock_tokenizer

        tp = TextProcessor(language="en")
        tokens = tp.tokenize("Hello World!")

        self.assertEqual(tokens, ["hello", "world"])
        mock_tokenizer.tokenize.assert_called_once()

    @patch("pamola_core.utils.nlp.tokenization.lemmatize")
    @patch("pamola_core.utils.nlp.tokenization.TokenizerFactory.create_tokenizer")
    def test_lemmatize(self, mock_create_tokenizer, mock_lemmatize):
        mock_create_tokenizer.return_value = MagicMock()
        mock_lemmatize.return_value = ["hello", "world"]

        tp = TextProcessor(language="en")
        lemmas = tp.lemmatize(["helloing", "worlds"])

        self.assertEqual(lemmas, ["hello", "world"])
        mock_lemmatize.assert_called_once()

    @patch("pamola_core.utils.nlp.tokenization.lemmatize")
    @patch("pamola_core.utils.nlp.tokenization.TokenizerFactory.create_tokenizer")
    def test_tokenize_and_lemmatize(self, mock_create_tokenizer, mock_lemmatize):
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.return_value = ["jumped", "over"]
        mock_create_tokenizer.return_value = mock_tokenizer
        mock_lemmatize.return_value = ["jump", "over"]

        tp = TextProcessor(language="en")
        result = tp.tokenize_and_lemmatize("Jumped over.")

        self.assertEqual(result, ["jump", "over"])

    @patch("pamola_core.utils.nlp.tokenization.NGramExtractor")
    @patch("pamola_core.utils.nlp.tokenization.lemmatize")
    @patch("pamola_core.utils.nlp.tokenization.TokenizerFactory.create_tokenizer")
    def test_process_text_all_features(self, mock_create_tokenizer, mock_lemmatize, mock_ngram_extractor):
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.return_value = ["the", "quick", "brown"]
        mock_create_tokenizer.return_value = mock_tokenizer

        mock_lemmatize.return_value = ["the", "quick", "brown"]

        mock_extractor = MagicMock()
        mock_extractor.extract_token_ngrams_as_strings.side_effect = [
            ["the quick", "quick brown"],
            ["the quick brown"]
        ]
        mock_ngram_extractor.return_value = mock_extractor

        tp = TextProcessor(language="en")
        result = tp.process_text(
            "The quick brown fox",
            lemmatize_tokens=True,
            extract_ngrams_flag=True,
            ngram_sizes=[2, 3]
        )

        self.assertEqual(result["tokens"], ["the", "quick", "brown"])
        self.assertEqual(result["lemmas"], ["the", "quick", "brown"])
        self.assertIn(2, result["ngrams"])
        self.assertIn(3, result["ngrams"])

    @patch("pamola_core.utils.nlp.language.detect_language")
    @patch("pamola_core.utils.nlp.tokenization.lemmatize")
    @patch("pamola_core.utils.nlp.tokenization.TokenizerFactory.create_tokenizer")
    def test_process_text_with_language_detection(self, mock_create_tokenizer, mock_lemmatize, mock_detect_lang):
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.return_value = ["bonjour", "le", "monde"]
        mock_create_tokenizer.return_value = mock_tokenizer
        mock_lemmatize.return_value = ["bonjour", "le", "monde"]
        mock_detect_lang.return_value = "fr"

        tp = TextProcessor(language=None)
        result = tp.process_text("Bonjour le monde")

        self.assertEqual(result["detected_language"], "fr")
        self.assertEqual(result["tokens"], ["bonjour", "le", "monde"])

    @patch("pamola_core.utils.nlp.tokenization.TokenizerFactory.create_tokenizer")
    def test_process_empty_text(self, mock_create_tokenizer):
        tp = TextProcessor(language="en")
        result = tp.process_text("")

        self.assertEqual(result["tokens"], [])
        self.assertEqual(result["token_count"], 0)

class TestFunctions(unittest.TestCase):

    @patch("pamola_core.utils.nlp.tokenization_helpers.load_synonym_dictionary")
    @patch("pamola_core.utils.nlp.tokenization._LEMMATIZERS", new_callable=dict)
    def test_lemmatize_with_custom_dict(self, mock_lemmatizers, mock_load_dict):
        tokens = ["cars", "ran"]
        mock_load_dict.return_value = {"car": ["cars"], "run": ["ran"]}
        result = lemmatize(tokens, language="en", dict_sources="dummy_dict")
        self.assertEqual(result,  ["car", "run"])

    def test_normalize_text_all_options(self):
        text = "Hello, WORLD! 123"
        result = normalize_text(text, remove_punctuation=True, lowercase=True, remove_digits=True)
        self.assertEqual(result, "hello world")

    @patch("pamola_core.utils.nlp.tokenization_helpers.load_synonym_dictionary")
    def test_normalize_tokens_with_synonyms(self, mock_load_dict):
        tokens = ["fast", "speedy"]
        mock_load_dict.return_value = {"quick": ["fast", "speedy"]}
        result = normalize_tokens(tokens, synonym_sources="dummy_src")
        self.assertEqual(result, ["quick", "quick"])

    @patch("pamola_core.utils.nlp.tokenization.TokenizerFactory.create_tokenizer")
    def test_tokenize_with_factory(self, mock_factory):
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.return_value = ["this", "is", "test"]
        mock_factory.return_value = mock_tokenizer
        result = tokenize("This is a test.")
        self.assertEqual(result, ["this", "is", "test"])

    @patch("pamola_core.utils.nlp.tokenization.tokenize")
    def test_calculate_word_frequencies(self, mock_tokenize):
        mock_tokenize.side_effect = lambda text, **kwargs: text.split()
        texts = ["apple orange", "apple banana"]
        result = calculate_word_frequencies(texts)
        self.assertEqual(result, {"apple": 2, "orange": 1, "banana": 1})

    @patch("pamola_core.utils.nlp.tokenization.tokenize_and_lemmatize")
    def test_calculate_term_frequencies(self, mock_tokenize_lemmas):
        mock_tokenize_lemmas.side_effect = lambda text, **kwargs: text.lower().split()
        texts = ["Jumped over", "Jump over the wall"]
        result = calculate_term_frequencies(texts)
        self.assertEqual(result["jump"], 1)
        self.assertEqual(result["over"], 2)

    @patch("pamola_core.utils.nlp.tokenization.lemmatize")
    @patch("pamola_core.utils.nlp.tokenization.tokenize")
    @patch("pamola_core.utils.nlp.tokenization._get_or_detect_language")
    def test_tokenize_and_lemmatize_pipeline(self, mock_lang, mock_tokenize, mock_lemmatize):
        mock_lang.return_value = "en"
        mock_tokenize.return_value = ["cars"]
        mock_lemmatize.return_value = ["car"]
        result = tokenize_and_lemmatize("Cars", language=None)
        self.assertEqual(result, ["car"])

    @patch("pamola_core.utils.nlp.tokenization.NGramExtractor")
    def test_extract_ngrams_as_strings(self, mock_extractor_cls):
        mock_extractor = MagicMock()
        mock_extractor.extract_token_ngrams_as_strings.return_value = ["new york", "york city"]
        mock_extractor_cls.return_value = mock_extractor
        tokens = ["new", "york", "city"]
        result = extract_ngrams(tokens, n=2)
        self.assertEqual(result, ["new york", "york city"])

    def test_extract_character_ngrams_default(self):
        text = "hello"
        expected = ["__h", "_he", "hel", "ell", "llo", "lo_", "o__"]
        result = extract_character_ngrams(text, n=3)
        self.assertEqual(result, expected)

    def test_extract_character_ngrams_uppercase_preserved(self):
        text = "HELLO"
        result = extract_character_ngrams(text, n=2, lowercase=False)
        self.assertEqual(result, ["_H", "HE", "EL", "LL", "LO", "O_"])

    def test_extract_character_ngrams_short_input(self):
        text = "hi"
        result = extract_character_ngrams(text, n=3)
        self.assertEqual(result, ["__h", "_hi", "hi_", "i__"])

    def test_extract_character_ngrams_exact_fit(self):
        text = "cat"
        result = extract_character_ngrams(text, n=3)
        self.assertEqual(result, ["__c", "_ca", "cat", "at_", "t__"])

    @patch("pamola_core.utils.nlp.tokenization.batch_process")
    def test_batch_tokenize_calls_batch_process(self, mock_batch):
        mock_batch.return_value = [["hello", "world"]]
        result = batch_tokenize(["Hello world"])
        self.assertEqual(result, [["hello", "world"]])
        mock_batch.assert_called_once()

    def test_batch_tokenize_and_lemmatize_calls_batch_process_correctly(self):
        sample_texts = ["Running fast", "Jumped high"]
        expected_output = [["run", "fast"], ["jump", "high"]]

        with patch("pamola_core.utils.nlp.tokenization.batch_process") as mock_batch:
            mock_batch.return_value = expected_output

            result = batch_tokenize_and_lemmatize(
                texts=sample_texts,
                language="en",
                min_length=2,
                config_sources=None,
                lemma_dict_sources=None,
                processes=2,
                tokenizer_type="nltk"
            )

            mock_batch.assert_called_once()
            # Ensure the callable being passed is correct
            _, kwargs = mock_batch.call_args
            self.assertEqual(kwargs["language"], "en")
            self.assertEqual(kwargs["min_length"], 2)
            self.assertEqual(kwargs["tokenizer_type"], "nltk")

            self.assertEqual(result, expected_output)




if __name__ == "__main__":
    unittest.main()
