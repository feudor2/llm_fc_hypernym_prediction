import asyncio

from openai import OpenAI
from pydantic import BaseModel

from logging import Logger
from typing import Optional
from pprint import pformat

from utils.utils import read_prompts, delayed_call, display_message

class RerankerRequest(BaseModel):
    target: str
    candidates: tuple
    sense: Optional[str] = None
    temperature: float = 0.5
    top_p: float = 0.95
    threshold: float = 0.5
    rel: str = 'hyponym'

class Reranker:
    def __init__(self, client: OpenAI, model_name: str, logger: Optional[Logger] = None):
        self.client = client
        self.model_name = model_name
        self.logger = logger
        self.prompts = read_prompts('rerank_', logger)
        msg = f'Reranker initialized'
        display_message(msg, self.logger, 10)
        msg = f'Found reranker prompts: {list(self.prompts.keys())}'
        display_message(msg, self.logger, 20)

    async def __call__(self, request: RerankerRequest):
        kwargs = request.model_dump()
        msg = f'Reranking started: {pformat(kwargs)}'
        display_message(msg, self.logger, 20)
        del kwargs['candidates']

        awaitables = []
        for i, candidate in enumerate(request.candidates):
            awaitable = delayed_call(
                delay=i*0.01,
                func=self._rerank,
                args=(candidate,),
                kwargs=kwargs
            )
            awaitables.append(awaitable)

        responses = await asyncio.gather(*awaitables)
        scored = []
        for c, response in zip(request.candidates, responses):
            try:
                r = response.content
                scored.append((c, float(r) / 5.0))
                msg = f'Score {r} predicted for candidate {c}'
                display_message(msg, self.logger, 10)
            except ValueError:
                try:
                    scored.append((c, float(r)[0] / 5.0))
                    msg = f'Score {r[0]} extracted from response {r} for candidate {c}'
                    display_message(msg, self.logger, 10)
                except ValueError:
                    scored.append((c, 0.0))
                    msg = f'Invalid LLM response for candidate {c}: {r}'
                    display_message(msg, self.logger, 30)
                except TypeError:
                    scored.append((c, 0.0))
                    msg = f'Empty response for candidate {c}: {r}'
                    display_message(msg, self.logger, 30)

        return [{'candidate': c, 'score': score} for c, score in sorted(scored, reverse=True, key=lambda x: x[1]) if score >= request.threshold]
        

    async def _rerank(self, candidate, target=None, sense=None, temperature=0.0, top_p=0.0, rel='hyponym', **kwargs) -> int:
        try:
            # Обработка пустого контекста
            if not sense:
                sense = 'Не задано'
            
            messages = []
            if self.prompts.get('rerank_system') or self.prompts.get(f'rerank_system_{rel}'):
                system_prompt_name = 'rerank_system' if 'rerank_system' in self.prompts else f'rerank_system_{rel}'
                messages.append({"role": "system", "content": self.prompts[system_prompt_name]})
            if self.prompts.get('rerank_user') or self.prompts.get(f'rerank_user_{rel}'):
                user_prompt_name = 'rerank_user' if 'rerank_user' in self.prompts else f'rerank_user_{rel}'
                messages.append({"role": "user", "content": self.prompts[user_prompt_name].format(target=target, candidate=candidate, sense=sense)})
            else:
                msg = f'User prompt "rerank_user" or "rerank_user_{rel}" not found'
                display_message(msg, self.logger, 40)
                raise Exception(msg)
            
            msg = f'Created messages: {messages}'
            display_message(msg, self.logger, 10)

            response_obj = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=5,
            )
            return response_obj.choices[0].message
        
        except Exception as e:
            msg = f'Reranking failed for candidate {candidate} with error "{e}"'
            display_message(msg, self.logger, 40)