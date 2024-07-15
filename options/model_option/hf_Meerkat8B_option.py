from hf import create_hf_chain
from .ModelOption import ModelOption


hf_Meerkat8B_option = ModelOption(
    "Meerkat8B",
    lambda: create_hf_chain("dmis-lab/llama-3-meerkat-8b-v1.0")
)