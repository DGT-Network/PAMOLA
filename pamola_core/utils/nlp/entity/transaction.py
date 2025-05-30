"""
Transaction purpose entity extractor.

This module provides functionality for extracting purposes and categories
from transaction descriptions, useful for financial data analysis.
"""

import logging
import re
from typing import Optional, Tuple

from pamola_core.utils.nlp.entity.base import BaseEntityExtractor, EntityMatchResult
from pamola_core.utils.nlp.model_manager import NLPModelManager

# Configure logger
logger = logging.getLogger(__name__)

# NLP model manager instance
nlp_model_manager = NLPModelManager()


class TransactionPurposeExtractor(BaseEntityExtractor):
    """
    Extractor for transaction purposes and financial categories.
    """

    def __init__(self, **kwargs):
        """
        Initialize the transaction purpose extractor.

        Additional parameters:
        -----------
        remove_account_numbers : bool
            Whether to remove account numbers and IDs from text
        remove_dates : bool
            Whether to remove dates from text
        """
        super().__init__(**kwargs)

        # Transaction-specific parameters
        self.remove_account_numbers = kwargs.get('remove_account_numbers', True)
        self.remove_dates = kwargs.get('remove_dates', True)

        # Common transaction categories
        self.transaction_categories = {
            'payment': [
                'payment', 'платеж', 'оплата', 'pay', 'invoice', 'счет', 'purchase',
                'покупка', 'fee', 'комиссия'
            ],
            'salary': [
                'salary', 'зарплата', 'wage', 'payroll', 'compensation', 'comp',
                'заработная плата', 'аванс', 'расчет', 'вознаграждение'
            ],
            'transfer': [
                'transfer', 'перевод', 'переброска', 'remittance', 'wire', 'перевести',
                'moved', 'send', 'отправка', 'перечисление'
            ],
            'refund': [
                'refund', 'возврат', 'cashback', 'reimbursement', 'repayment', 'return',
                'возмещение', 'компенсация'
            ],
            'withdrawal': [
                'withdrawal', 'снятие', 'cash', 'наличные', 'atm', 'банкомат', 'terminal',
                'терминал', 'касса'
            ],
            'deposit': [
                'deposit', 'взнос', 'внесение', 'вложение', 'депозит', 'savings',
                'сбережения', 'investment', 'инвестиция'
            ],
            'subscription': [
                'subscription', 'подписка', 'recurring', 'periodic', 'ежемесячно',
                'monthly', 'годовой', 'annual', 'service', 'сервис'
            ],
            'utility': [
                'utility', 'жкх', 'utilities', 'electric', 'water', 'газ', 'электричество',
                'water', 'вода', 'utility bill', 'счет за коммунальные услуги'
            ]
        }

        # ID and date patterns
        self.account_pattern = re.compile(r'[0-9]{4,}|[A-Z0-9]{6,}')
        self.date_pattern = re.compile(r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{1,2}\s+[a-zA-Z]{3,}\s+\d{2,4}')

    def _get_entity_type(self) -> str:
        """
        Get the entity type for this extractor.

        Returns:
        --------
        str
            Entity type string
        """
        return "transaction"

    def _extract_with_ner(self, text: str, normalized_text: str, language: str) -> Optional[EntityMatchResult]:
        """
        Extract transaction purposes using patterns and NER.

        Parameters:
        -----------
        text : str
            Original text
        normalized_text : str
            Normalized text
        language : str
            Language of the text

        Returns:
        --------
        EntityMatchResult or None
            Match result if found, None otherwise
        """
        try:
            # Clean the text if needed
            cleaned_text = self._clean_transaction_text(normalized_text)

            # First try to match transaction categories directly
            category, confidence = self._determine_transaction_category(cleaned_text)

            if category and confidence >= self.min_confidence:
                return EntityMatchResult(
                    original_text=text,
                    normalized_text=normalized_text,
                    category=f"TRANSACTION_{category.upper()}",
                    alias=f"transaction_{category.lower()}",
                    domain="Finance",
                    level=1,
                    seniority="Any",
                    confidence=confidence,
                    method="pattern",
                    language=language,
                    conflicts=[]
                )

            # If pattern matching fails, try NER if available
            try:
                # Try to use a specialized transaction purpose extractor if available
                extractor = nlp_model_manager.get_entity_extractor("financial", language)

                if extractor:
                    entities = extractor.extract_entities([text], language)

                    if entities and 'transactions' in entities and entities['transactions']:
                        # Sort by confidence/count
                        transactions = sorted(entities['transactions'],
                                              key=lambda x: x.get('count', 0),
                                              reverse=True)

                        if transactions:
                            purpose = transactions[0].get('text', '')

                            # Create match result
                            return EntityMatchResult(
                                original_text=text,
                                normalized_text=normalized_text,
                                category="NER_TRANSACTION",
                                alias="transaction_general",
                                domain="Finance",
                                level=1,
                                seniority="Any",
                                confidence=0.6,
                                method="ner",
                                language=language,
                                conflicts=[]
                            )
            except Exception as e:
                logger.debug(f"Error using specialized transaction extractor: {e}")

            # As a last resort, try to use general NER
            model = nlp_model_manager.get_model('spacy', language)
            if model:
                doc = model(text)

                # Extract potential purposes or organizations
                entities = [ent.text for ent in doc.ents
                            if ent.label_ in ('ORG', 'PRODUCT', 'EVENT', 'FAC')]

                if entities:
                    # Create match result with the first entity
                    return EntityMatchResult(
                        original_text=text,
                        normalized_text=normalized_text,
                        category="NER_ORGANIZATION",
                        alias="transaction_organization",
                        domain="Finance",
                        level=1,
                        seniority="Any",
                        confidence=0.5,
                        method="ner",
                        language=language,
                        conflicts=[]
                    )

            # No matches found
            return None

        except Exception as e:
            logger.error(f"Error in transaction purpose extraction: {e}")
            return None

    def _clean_transaction_text(self, text: str) -> str:
        """
        Clean transaction text by removing irrelevant information.

        Parameters:
        -----------
        text : str
            Text to clean

        Returns:
        --------
        str
            Cleaned text
        """
        # Make a copy of the original text
        cleaned = text

        # Remove account numbers and IDs if requested
        if self.remove_account_numbers:
            cleaned = self.account_pattern.sub('', cleaned)

        # Remove dates if requested
        if self.remove_dates:
            cleaned = self.date_pattern.sub('', cleaned)

        # Remove extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned

    def _determine_transaction_category(self, text: str) -> Tuple[str, float]:
        """
        Determine the transaction category from text.

        Parameters:
        -----------
        text : str
            Transaction text

        Returns:
        --------
        Tuple[str, float]
            Category name and confidence score
        """
        text_lower = text.lower()

        # Count matches by category
        category_scores = {}

        for category, terms in self.transaction_categories.items():
            # Count matches
            matches = 0
            for term in terms:
                if term in text_lower:
                    matches += 1

            # Calculate score based on matches and term count
            if matches > 0:
                score = matches / len(terms)
                category_scores[category] = score

        # If no categories matched
        if not category_scores:
            return "general", 0.0

        # Find the category with the highest score
        best_category = max(category_scores.items(), key=lambda x: x[1])
        category_name = best_category[0]
        confidence = best_category[1]

        return category_name, confidence