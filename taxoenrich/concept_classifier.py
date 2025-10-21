"""
Модуль для классификации релевантности концептов в таксономии.

Предназначен для фильтрации узлов таксономии перед их добавлением в контекст
LLM-агента, что позволяет избежать переполнения контекста при работе с узлами,
имеющими большое количество гипонимов.
"""

import json
from typing import Optional, Dict, Any, List
import anthropic
import openai


class ConceptClassifier:
    """
    Классификатор для определения релевантности концептов в таксономии.
    
    Принимает на вход текст с выделенной сущностью и информацию о концепте,
    предсказывает вероятность того, что искомое понятие находится ниже
    по дереву таксономии от данного концепта.
    
    Attributes:
        model_name (str): Название используемой LLM модели
        api_key (str): API ключ для доступа к модели
        provider (str): Провайдер LLM ('anthropic', 'openai')
        temperature (float): Температура генерации (0.0 для детерминированности)
        max_tokens (int): Максимальное количество токенов в ответе
    """
    
    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        provider: str = "anthropic",
        temperature: float = 0.0,
        max_tokens: int = 100
    ):
        """
        Инициализация классификатора.
        
        Args:
            model_name: Название модели (по умолчанию Claude 3.5 Sonnet)
            api_key: API ключ (если None, берется из переменных окружения)
            provider: Провайдер API ('anthropic' или 'openai')
            temperature: Температура для генерации (рекомендуется 0.0)
            max_tokens: Максимальное количество токенов в ответе
        """
        self.model_name = model_name
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if self.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=api_key)
        elif self.provider == "openai":
            self.client = openai.OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'anthropic' or 'openai'")
    
    def _construct_prompt(
        self,
        text: str,
        entity: str,
        concept_info: Dict[str, Any]
    ) -> str:
        """
        Формирование промпта для классификации.
        
        Args:
            text: Исходный текст с контекстом
            entity: Выделенная сущность для поиска гиперонима
            concept_info: Информация о концепте (name, words, definition, hyponyms и т.д.)
        
        Returns:
            Сформированный промпт для LLM
        """
        concept_name = concept_info.get('name', 'Unknown')
        concept_words = concept_info.get('words', [])
        concept_definition = concept_info.get('definition', '')
        concept_hyponyms = concept_info.get('hyponyms', [])
        
        # Формируем описание концепта
        concept_description = f"**Концепт**: {concept_name}\n"
        
        if concept_words:
            concept_description += f"**Синонимы**: {', '.join(concept_words[:10])}\n"
        
        if concept_definition:
            concept_description += f"**Определение**: {concept_definition}\n"
        
        if concept_hyponyms:
            hyponyms_preview = concept_hyponyms[:10]
            if len(concept_hyponyms) > 10:
                hyponyms_preview.append(f"... и еще {len(concept_hyponyms) - 10}")
            concept_description += f"**Примеры гипонимов**: {', '.join(hyponyms_preview)}\n"
        
        prompt = f"""Задача: определить, может ли искомое понятие для сущности находиться ниже по дереву таксономии от данного концепта.

**Текст с контекстом**:
{text}

**Сущность для классификации**: {entity}

{concept_description}

**Вопрос**: Может ли гипероним (более общее понятие) для сущности "{entity}" находиться в поддереве данного концепта "{concept_name}" или ниже по иерархии?

Ответь только одним числом:
- 1 - если потенциально искомое понятие может быть ниже по дереву от этого концепта
- 0 - если точно нет, этот концепт и его поддерево не релевантны

Ответ (только 0 или 1):"""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """
        Вызов LLM для получения предсказания.
        
        Args:
            prompt: Промпт для модели
        
        Returns:
            Текстовый ответ модели
        """
        if self.provider == "anthropic":
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text.strip()
        
        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
    
    def _parse_prediction(self, response: str) -> int:
        """
        Парсинг ответа модели.
        
        Args:
            response: Ответ от LLM
        
        Returns:
            0 или 1
        """
        # Убираем лишние символы и пробелы
        response = response.strip().strip('.')
        
        # Пытаемся извлечь первую цифру 0 или 1
        for char in response:
            if char == '0':
                return 0
            elif char == '1':
                return 1
        
        # Если не нашли явного ответа, пытаемся понять по ключевым словам
        response_lower = response.lower()
        if any(word in response_lower for word in ['нет', 'не релевант', 'no', 'not relevant']):
            return 0
        elif any(word in response_lower for word in ['да', 'релевант', 'yes', 'relevant', 'может']):
            return 1
        
        # По умолчанию возвращаем 0 (консервативный подход)
        return 0
    
    def predict(
        self,
        text: str,
        entity: str,
        concept_info: Dict[str, Any]
    ) -> int:
        """
        Предсказание релевантности концепта.
        
        Args:
            text: Исходный текст с контекстом использования сущности
            entity: Выделенная сущность для поиска гиперонима
            concept_info: Информация о концепте, должна содержать:
                - name (str): Название концепта
                - words (list[str], optional): Синонимы
                - definition (str, optional): Определение
                - hyponyms (list[str], optional): Список гипонимов
        
        Returns:
            0 - концепт не релевантен
            1 - концепт потенциально релевантен
        
        Example:
            >>> classifier = ConceptClassifier()
            >>> text = "Сегодня я купил новый **смартфон** в магазине"
            >>> entity = "смартфон"
            >>> concept = {
            ...     'name': 'устройство',
            ...     'words': ['прибор', 'аппарат', 'устройство'],
            ...     'definition': 'техническое приспособление',
            ...     'hyponyms': ['компьютер', 'телефон', 'планшет']
            ... }
            >>> prediction = classifier.predict(text, entity, concept)
            >>> print(prediction)  # Ожидается 1
        """
        prompt = self._construct_prompt(text, entity, concept_info)
        response = self._call_llm(prompt)
        prediction = self._parse_prediction(response)
        
        return prediction
    
    def predict_batch(
        self,
        text: str,
        entity: str,
        concepts: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Пакетное предсказание для нескольких концептов.
        
        Args:
            text: Исходный текст с контекстом
            entity: Выделенная сущность
            concepts: Список информации о концептах
        
        Returns:
            Список предсказаний (0 или 1) для каждого концепта
        
        Example:
            >>> classifier = ConceptClassifier()
            >>> text = "Я читаю интересную **книгу** о истории"
            >>> entity = "книгу"
            >>> concepts = [
            ...     {'name': 'объект', 'words': ['вещь', 'предмет']},
            ...     {'name': 'издание', 'words': ['публикация', 'печатное издание']}
            ... ]
            >>> predictions = classifier.predict_batch(text, entity, concepts)
        """
        predictions = []
        for concept_info in concepts:
            prediction = self.predict(text, entity, concept_info)
            predictions.append(prediction)
        
        return predictions
    
    def filter_concepts(
        self,
        text: str,
        entity: str,
        concepts: List[Dict[str, Any]],
        return_scores: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Фильтрация концептов, оставляя только релевантные.
        
        Args:
            text: Исходный текст с контекстом
            entity: Выделенная сущность
            concepts: Список концептов для фильтрации
            return_scores: Если True, добавляет поле 'relevance_score' к каждому концепту
        
        Returns:
            Отфильтрованный список концептов (где prediction == 1)
        
        Example:
            >>> classifier = ConceptClassifier()
            >>> text = "В парке я видел красивого **лебедя**"
            >>> entity = "лебедя"
            >>> concepts = [
            ...     {'name': 'животное', 'id': '1'},
            ...     {'name': 'машина', 'id': '2'},
            ...     {'name': 'птица', 'id': '3'}
            ... ]
            >>> filtered = classifier.filter_concepts(text, entity, concepts)
            >>> # Ожидается, что останутся только 'животное' и 'птица'
        """
        filtered_concepts = []
        
        for concept_info in concepts:
            prediction = self.predict(text, entity, concept_info)
            
            if prediction == 1:
                if return_scores:
                    concept_copy = concept_info.copy()
                    concept_copy['relevance_score'] = prediction
                    filtered_concepts.append(concept_copy)
                else:
                    filtered_concepts.append(concept_info)
        
        return filtered_concepts