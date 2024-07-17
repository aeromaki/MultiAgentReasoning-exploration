from langchain_openai import ChatOpenAI
from utils import load_api_key
from .ModelOption import ModelOption


OPTION = ModelOption(
    "GPT35",
    lambda: ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=load_api_key())
)