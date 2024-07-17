from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline


CACHE_DIR = "./.cache"


def create_hf_chain(model_name: str, max_new_tokens: int = 512) -> HuggingFacePipeline:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR, device_map="auto")

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens, device_map="auto")

    return HuggingFacePipeline(pipeline=pipe)