from dataclasses import dataclass
from typing import Callable, Optional
from langchain_core.language_models import BaseLanguageModel


@dataclass
class ModelOption:
    name: str
    model: Callable[[], BaseLanguageModel]