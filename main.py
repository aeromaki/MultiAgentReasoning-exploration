from options import *
from pipe import run


dataset_options = { i.name.lower(): i for i in [medQA_option] }
model_options = { i.name.lower(): i for i in [openAI_GPT35_option] }

# run(medQA_option, openAI_GPT35_option, n_row=200)