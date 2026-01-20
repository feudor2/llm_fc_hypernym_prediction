from openai import AsyncOpenAI
from pydantic import BaseModel

from logging import Logger
from typing import Optional
from pprint import pformat

from utils.utils import read_prompts, display_message

class InterpreterRequest(BaseModel):
    context: Optional[str] = None
    temperature: float = 0.5
    top_p: float = 0.95
    max_tokens: int = 300

class Interpreter:
    def __init__(self, client: AsyncOpenAI, model_name: str, logger: Optional[Logger] = None):
        self.client = client
        self.model_name = model_name
        self.logger = logger
        self.prompts = read_prompts('interpreter_', logger)
        msg = f'Interpreter initialized'
        display_message(msg, self.logger, 10)
        msg = f'Found interpreter prompts: {list(self.prompts.keys())}'
        display_message(msg, self.logger, 20)

    async def __call__(self, request: InterpreterRequest):
        kwargs = request.model_dump()
        msg = f'Interpretation started: {pformat(kwargs)}'
        display_message(msg, self.logger, 10)
        response = await self._execute(**kwargs)
        try:
            result = response.content
            msg = f'Interpretation finished successfuly. Result: {result}'
            display_message(msg, self.logger, 20)
            return result
        except AttributeError as e:
            msg = f'Interpretation failed for context {pformat(request.context)} with error "{e}"'
            display_message(msg, self.logger, 40)
            return None

    async def _execute(self, context, temperature=0.0, top_p=0.0, max_tokens=300, **kwargs) -> int:
        try:
            messages = []
            system_prompt = self.prompts.get('interpreter_system')
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            user_prompt = self.prompts.get('interpreter_user') 
            if  user_prompt:
                messages.append({"role": "user", "content": user_prompt.format(context=context)})
            else:
                msg = f'User prompt "interpreter_user" not found'
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
            msg = f'Interpretation failed for context {pformat(context)} with error "{e}"'
            display_message(msg, self.logger, 40)