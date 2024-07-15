from dataclasses import dataclass
from typing import Callable, Optional
from langchain.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from dataset import QADataset


FormatOut = Callable[[dict, Optional[str], str], dict]


@dataclass
class DatasetOption:
    dataset: type[QADataset]
    template: PromptTemplate
    examples: list[dict]
    parse_result: Callable[[str, dict], Optional[str]]
    format_out: FormatOut

    @property
    def name(self) -> str:
        return self.dataset.__name__[:-7]

    def create_fewshot_template(self, n_shot: int) -> FewShotPromptTemplate:
        example_prompt = PromptTemplate.from_template("{question}\n\n{answer}\n")
        return FewShotPromptTemplate(
            examples=self.examples[:n_shot],
            example_prompt=example_prompt,
            suffix="{question}",
            input_variables=["question"]
        )