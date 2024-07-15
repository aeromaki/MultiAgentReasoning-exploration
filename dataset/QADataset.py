from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from datasets.arrow_dataset import Dataset as ArrowDataset


class QADataset(ABC):
    CACHE_DIR = "./.cache"

    def __init__(self) -> None:
        self.dataset = self.load_dataset()

    def __getitem__(self, key: int) -> tuple[str, dict]:
        return (self.convert_row(self.dataset[key]), self.dataset[key])

    @classmethod
    @abstractmethod
    def load_dataset(cls) -> ArrowDataset:
        pass

    @staticmethod
    @abstractmethod
    def convert_row(row: dict) -> str:
        pass