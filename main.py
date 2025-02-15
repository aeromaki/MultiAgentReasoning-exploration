import argparse
from options import *
from pipe import run


dataset_options = { i.name.lower(): i for i in [MEDQA] }
model_options = { i.name.lower(): i for i in [OPENAI_GPT35, HF_MEERKAT_8B, HF_MEERKAT_70B] }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="medqa", help="|".join(dataset_options.keys()))
    parser.add_argument("-m", "--model", type=str, required=True, help="|".join(model_options.keys()))
    parser.add_argument("-s", "--num_shot", type=int, default=3, help="0 ~ 3")
    parser.add_argument("-n", "--num_row", type=int, default=200)
    return parser


if __name__ == "__main__":
    args = create_parser().parse_args()

    dop = dataset_options[args.dataset.lower()]
    mop = model_options[args.model.lower()]
    n_shot = args.num_shot
    n_row = args.num_row

    run(dop, mop, n_shot=n_shot, n_row=n_row)