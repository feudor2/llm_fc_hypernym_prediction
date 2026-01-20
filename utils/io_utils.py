import os
import json
import yaml
import aiofiles
import asyncio

from typing import Dict, Any, Optional

def scan_directory(directory_path: str, extension: str = '.htm'):
    files = []
    with os.scandir(directory_path) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(extension):
                files.append(entry.path)
    return files

def save_txt(path: str, text: str):
    with open(path, 'w', encoding='utf8') as f:
        f.write(text)

def save_json(path: str, _dict: Dict[Any, Any]):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(_dict, f, ensure_ascii=False)

def read_json(path):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)

async def aread_txt(path, encoding='utf8'):
    async with aiofiles.open(path, 'r', encoding=encoding) as f:
        return await f.read()

def read_txt(path, encoding='utf8'):
    with open(path, 'r', encoding=encoding) as f:
        return f.read()

async def aread_json(path):
    async with aiofiles.open(path, 'r', encoding='utf8') as f:
        content = await f.read()
        return json.loads(content)

async def aread_json_semaphore(path, semaphore: Optional[asyncio.Semaphore] = None):
    if semaphore is not None:
        async with semaphore:
            async with aiofiles.open(path, 'r', encoding='utf8') as f:
                content = await f.read()
                return json.loads(content)
    else:
        return await aread_json(path)

def read_yaml(path: str):
    with open(path) as f:
        try:
            return yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise exc
          

async def asave_json(path: str, content: Dict[Any, Any]):
    async with aiofiles.open(path, 'w', encoding='utf8') as f:
        await f.write(json.dumps(content, ensure_ascii=False))

__all__ = ["scan_directory", "save_json", "read_json", "aread_json", "save_txt", "aread_txt", "aread_json_semaphore"]

if __name__ == '__main__':
    pass