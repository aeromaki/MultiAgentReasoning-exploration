from hf import create_hf_chain
from .ModelOption import ModelOption


OPTION = ModelOption(
    "Meerkat70B",
    lambda: create_hf_chain("dmis-lab/llama-3-meerkat-70b-v1.0")
)