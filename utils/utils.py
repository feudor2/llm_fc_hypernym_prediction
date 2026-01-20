import os
import asyncio

from pathlib import Path
from typing import Optional, Dict
from logging import Logger

def display_message(msg: str, logger: Optional[Logger] = None, log_level: int = None):
    if logger is not None:
        logger.log(log_level, msg)
    else:
        print(msg)

def read_prompt(prompt_name: str, logger: Optional[Logger] = None) -> str:
    """
    Функция для чтения промпта
    
    :param prompt_name: имя промпта
    :type prompt_name: str
    :param logger: логер
    :type prompt_name: logging.Logger
    :return: текст промпта
    :rtype: str
    """
    prompt_dir = 'prompts/'
    prompt_path = f'{prompt_dir}{prompt_name}.md'
    if os.path.exists(prompt_path):
        msg = f'Prompt `{prompt_name}` is available.'
        display_message(msg, logger, 10)
        with open(prompt_path, 'r', encoding='utf8') as fin:
            return fin.read()
    else:
        msg = f'Prompt `{prompt_name}` does not exist in `{prompt_dir}`.'
        display_message(msg, logger, 30)


def read_prompts(prefix: str, logger: Optional[Logger] = None) -> Dict[str, str]:
    prompt_dir = Path('prompts/')
    prompts = dict()
    for entry in prompt_dir.glob(f'{prefix}*'):
        prompts[entry.stem] = read_prompt(entry.stem)
    if not prompts:
        msg = f'No prompts found with prefix "{prefix}"'
        display_message(msg, logger, 30)
    return prompts
    
async def delayed_call(delay, func, args=(), kwargs={}):
    await asyncio.sleep(delay)
    return await func(*args, **kwargs)

__all__ = ['read_prompt', 'read_prompts', 'delayed_call', 'display_message']