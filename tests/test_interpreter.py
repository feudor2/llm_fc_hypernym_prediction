import os
import asyncio
import logging
import numpy as np
import pandas as pd

from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv

from openai import AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from utils.utils import delayed_call
from utils.io_utils import aread_txt
from prompt_functions.interpreter import Interpreter, InterpreterRequest

load_dotenv()

# Initialize client
oclient = AsyncOpenAI(api_key=os.environ['API_KEY'], base_url=os.environ['BASE_URL'])

# Initialize logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_dir = 'logs'

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
)
logger = logging.getLogger()
for handler in logger.handlers[:]:
    handler.close()
    logger.removeHandler(handler)
handlers = [
    logging.StreamHandler(),
    logging.FileHandler(f"{log_dir}/{datetime.now().strftime('%Y%m%d')}_interpreter.log")
]
for handler in handlers:
    logger.addHandler(handler)



async def prepare_dataset(corpus_path: str, n_texts: 10):
    tasks = []
    filenames = os.listdir(corpus_path)[:n_texts]
    for i, filepath in enumerate(filenames):
        task = delayed_call(
            delay=i*0.01,
            func=aread_txt,
            args=(os.path.join(corpus_path, filepath),),
            kwargs={}
        )
        tasks.append(task)

    responses = await tqdm_asyncio.gather(*tasks, desc='Reading contexts')
    words = [filename.split('_')[0] for filename in filenames]
    return [pair for pair in zip(words, responses)]



async def run_test(interpreter: Interpreter, requests: list[InterpreterRequest], batch_size: int = 1):
    async def interprete(request: InterpreterRequest, semaphore: asyncio.Semaphore):
        async with semaphore:
            return await interpreter(request)

    semaphore = asyncio.Semaphore(batch_size)
    tasks = [interprete(request, semaphore) for request in requests]
    results = await tqdm_asyncio.gather(*tasks, desc='Running tests')
    return results 
    


async def main():
    dataset_path = 'corpus/private'
    contexts = await prepare_dataset(dataset_path, 100)
    interpreter = Interpreter(oclient, model_name=os.environ['MODEL_NAME'], logger=logger)
    requests = [InterpreterRequest(context=pair[1]) for pair in contexts]
    results = await run_test(interpreter, requests, batch_size=10)
    results = pd.DataFrame({'word': [pair[0] for pair in contexts], 'sense': results})
    results.to_csv('test_results/interpreter_results.tsv', sep='\t')


if __name__ == "__main__":
    asyncio.run(main())