"""
Tests for the utils.nlp.language module.

This module contains unit tests for the abstract base classes and concrete
implementations in the language.py module.
"""
import unittest

from pamola_core.utils.nlp.language import (
    normalize_language_code,
    detect_language,
    detect_language_with_confidence,
    detect_mixed_language,
    get_primary_language,
    is_multilingual,
    analyze_language_structure,
    is_cyrillic,
    is_latin,
    detect_languages
)

class TestLanguageDetection(unittest.TestCase):

    def test_normalize_language_code(self):
        self.assertEqual(normalize_language_code("EN"), "en")
        self.assertEqual(normalize_language_code("en-US"), "en")
        self.assertEqual(normalize_language_code("eng"), "en")
        self.assertEqual(normalize_language_code("fr_ca"), "fr")
        self.assertEqual(normalize_language_code(""), "en")
        self.assertEqual(normalize_language_code("xyz"), "xyz")

    def test_detect_language_simple(self):
        self.assertEqual(detect_language("This is an English sentence."), "en")
        self.assertEqual(detect_language("Это русское предложение."), "ru")
        self.assertEqual(detect_language("Ceci est une phrase française."), "fr")

    def test_detect_language_with_confidence(self):
        lang, conf = detect_language_with_confidence("This is English text.")
        self.assertEqual(lang, "en")
        self.assertGreater(conf, 0.5)

        lang, conf = detect_language_with_confidence("")
        self.assertEqual(lang, "en")
        self.assertEqual(conf, 0.0)

    def test_detect_mixed_language(self):
        text = "This is English. Это по-русски. C'est en français."
        proportions = detect_mixed_language(text)
        self.assertIsInstance(proportions, dict)
        self.assertGreaterEqual(len(proportions), 2)

    def test_get_primary_language(self):
        text = "Bonjour tout le monde. Ceci est un texte en français. Hello world."
        primary = get_primary_language(text, threshold=0.4)
        self.assertIn(primary, ["fr", "en"])

    def test_is_multilingual(self):
        self.assertTrue(is_multilingual("Hello world. Привет мир. Bonjour le monde."))
        self.assertFalse(is_multilingual("Just an English sentence."))

    def test_analyze_language_structure(self):
        analysis = analyze_language_structure("This is English. Это по-русски.")
        self.assertIn("primary_language", analysis)
        self.assertIn("is_multilingual", analysis)
        self.assertIn("language_proportions", analysis)
        self.assertIsInstance(analysis["script_info"], dict)

    def test_is_cyrillic_and_latin(self):
        self.assertTrue(is_cyrillic("Привет"))
        self.assertFalse(is_cyrillic("Hello"))
        self.assertTrue(is_latin("Hello"))
        self.assertFalse(is_latin("Привет"))

    def test_detect_languages_bulk(self):
        texts = [
            "Hello, how are you?",
            "Bonjour, comment ça va?",
            "Hola, ¿cómo estás?",
            "Hallo, wie geht's?",
            "Привет, как дела?"
        ]
        result = detect_languages(texts)
        self.assertIsInstance(result, dict)
        self.assertGreaterEqual(len(result), 2)
        self.assertAlmostEqual(sum(result.values()), 1.0, places=2)



if __name__ == '__main__':
    unittest.main()
