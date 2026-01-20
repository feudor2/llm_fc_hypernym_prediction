from openai import AsyncOpenAI
from pydantic import BaseModel

from logging import Logger
from typing import Optional
from pprint import pformat

from utils.utils import read_prompts, display_message

class ContextAnalyserRequest(BaseModel):
    senses: list[list[str]]
    context: Optional[str] = None
    temperature: float = 0.5
    top_p: float = 0.95
    max_tokens: int = 300

class ContextAnalyser:
    def __init__(self, client: AsyncOpenAI, model_name: str, logger: Optional[Logger] = None):
        self.client = client
        self.model_name = model_name
        self.logger = logger
        self.prompts = read_prompts('context_analyser_', logger)
        msg = f'ContextAnalyser initialized'
        display_message(msg, self.logger, 10)
        msg = f'Found ContextAnalyser prompts: {list(self.prompts.keys())}'
        display_message(msg, self.logger, 20)

    async def __call__(self, request: ContextAnalyserRequest):
        kwargs = request.model_dump()
        msg = f'Context analysis started: {pformat(kwargs)}'
        display_message(msg, self.logger, 10)
        response = await self._execute(**kwargs)
        try:
            result = response.content
            msg = f'Context analysis finished successfuly. Result: {result}'
            display_message(msg, self.logger, 20)
            return self._analyse_result(result, request)
        except AttributeError as e:
            msg = f'Context analysis failed for context {pformat(request.context)} with error "{e}"'
            display_message(msg, self.logger, 40)
            return None

    async def _execute(self, senses, context, temperature=0.0, top_p=0.0, max_tokens=300, **kwargs) -> int:
        try:
            if '<predict_kb>' not in context:
                raise Exception('Tag not found in context')
            formatted_senses = self._prepare_senses(senses)
            if not formatted_senses:
                raise ValueError('Empty senses')
            messages = []
            system_prompt = self.prompts.get('context_analyser_system')
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            user_prompt = self.prompts.get('context_analyser_user') 
            if  user_prompt:
                messages.append({"role": "user", "content": user_prompt.format(context=context, senses=formatted_senses)})
            else:
                msg = f'User prompt "context_analyser_user" not found'
                display_message(msg, self.logger, 40)
                raise Exception(msg)
            
            msg = f'Created messages: {messages}'
            display_message(msg, self.logger, 10)

            response_obj = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens
            )
            return response_obj.choices[0].message
        
        except Exception as e:
            msg = f'Context analysis failed for context {pformat(context)} with error "{e}"'
            display_message(msg, self.logger, 40)

    def _prepare_senses(self, senses):
        template = "{i}. {senses}"
        lines = [template.format(i=i, senses=", ".join(words)) for i, words in enumerate(senses, 1)]
        return '\n'.join(lines)

    def _analyse_result(self, result, request):
        try:
            request.senses[int(result) - 1]
            return int(result) - 1
        except Exception as e:
            msg = f'Failed to extract sense from {result} with error "{e}"'
            display_message(msg, self.logger, 40)
            return None