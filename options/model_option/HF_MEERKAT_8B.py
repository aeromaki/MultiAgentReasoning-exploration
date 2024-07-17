from hf import create_hf_chain
from .ModelOption import ModelOption


OPTION = ModelOption(
    "Meerkat8B",
    lambda: create_hf_chain("dmis-lab/llama-3-meerkat-8b-v1.0")
)