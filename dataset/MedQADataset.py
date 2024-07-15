from typing import Optional
from datasets import load_dataset
from .QADataset import ArrowDataset, QADataset


class MedQADataset(QADataset):
    HF_DATASET_ID: str = "aeromaki/MedQA"
    HF_DATASET_NAME: Optional[str] = None
    HF_DATASET_SPLIT: str = "train"

    @classmethod
    def load_dataset(cls) -> ArrowDataset:
        return load_dataset(
            cls.HF_DATASET_ID,
            name=cls.HF_DATASET_NAME,
            split=cls.HF_DATASET_SPLIT,
            cache_dir=cls.CACHE_DIR
        )

    @staticmethod
    def convert_row(row: dict) -> str:
        str_question = f"Question: {row['question']}"
        str_options = "\n".join(f"{i}. {row['options'][i]}" for i in ['A', 'B', 'C', 'D', 'E'])

        body = str_question + "\n\n" + str_options
        return body