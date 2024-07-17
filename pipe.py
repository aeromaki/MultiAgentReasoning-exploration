from typing import Callable, Optional
from tqdm import tqdm
from options import DatasetOption, ModelOption, FormatOut
from dataset import QADataset
import os
import json


Invoke = Callable[[str, dict], tuple[str, Optional[str]]]


def _create_pipe(
    dataset_option: DatasetOption,
    model_option: ModelOption,
    n_shot: int = 3
) -> tuple[
    QADataset,
    Invoke,
    FormatOut
]:
    chain = dataset_option.template | model_option.model()
    if n_shot > 0:
        chain = dataset_option.create_fewshot_template(n_shot) | chain
    dataset = dataset_option.dataset()

    def invoke(str_row: str, row: dict) -> tuple[str, Optional[str]]:
        result = chain.invoke({ "question": str_row })
        if not isinstance(result, str):
            result = result.content
        return result, dataset_option.parse_result(result, row)

    format_out = dataset_option.format_out

    return dataset, invoke, format_out


def run_inference(
    dataset_option: DatasetOption,
    model_option: ModelOption,
    n_shot: int,
    n_row: int
) -> list[dict]:
    dataset, invoke, format_out = _create_pipe(dataset_option, model_option, n_shot)

    results = []

    try:
        for i in tqdm(range(min(n_row, len(dataset.dataset)))):
            str_row, row = dataset[i]
            output, pred = invoke(str_row, row)

            result = format_out(row, pred, output)
            results.append(result)
    finally:
        return results


def run(
    dataset_option: DatasetOption,
    model_option: ModelOption,
    n_shot: int = 3,
    n_row: int = 200
) -> None:
    results = run_inference(dataset_option, model_option, n_shot, n_row)

    filename = f"results/{dataset_option.name}_{model_option.name}_{n_shot}shot_{len(results)}.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    json.dump(results, open(filename, 'w'))