from typing import Optional
from langchain.prompts import PromptTemplate
from .DatasetOption import DatasetOption
from dataset import MedQADataset


_MEDQA_TEMPLATE = """"You are a medical doctor taking the US Medical Licensing Examination. \
You need to demonstrate your understanding of basic and clinical science, medical knowledge, \
and mechanisms underlying health, disease, patient care, and modes of therapy. Show your ability \
to apply the knowledge essential for medical practice. Base your answer on the current and standard practices \
referenced in medical guidelines. \
Create a logical progression of tail questions and combine the answers to select an answer from the options here:\n\n{body}"""
MEDQA_TEMPLATE = PromptTemplate.from_template(_MEDQA_TEMPLATE)

MEDQA_EXAMPLES = [
    {
        "question": 'Question: A 36-year-old software professional consults a physician to discuss his concerns about small-vessel vasculitis as his mother and sister both have autoimmune small-vessel vasculitides. He has read about vasculitides and recently he came across an article which stated that an analgesic that he often uses for relief from a headache can cause small-vessel vasculitis. Due to his positive family history, he is especially concerned about his risk of developing small-vessel vasculitis. Which of the following clinical presentations is most likely to occur in this man?\n\nA. Absence of pulses in the upper extremity\nB. Infarction of an internal organ\nC. Stroke\nD. Aneurysm of an artery\nE. Palpable purpura',
        "answer": """\
First, let's review the basics of small-vessel vasculitis. Small-vessel vasculitis involves inflammation of the small blood vessels, typically affecting the skin, kidneys, lungs, and nervous system. It can present with a variety of symptoms, including palpable purpura, which are small red or purple spots on the skin due to bleeding underneath the skin.
In this case, the patient is concerned about his risk of developing small-vessel vasculitis due to his family history and potential exposure to a medication that may trigger vasculitis. Given his concerns and history, the most likely clinical presentation in this man would be E. Palpable purpura. This is a common skin manifestation of small-vessel vasculitis and would be a key symptom to watch for in this patient.
Answer: E. Palpable purpura"""
    },
    {
        "question": "Question: A 55-year-old woman with type 2 diabetes mellitus presents to her physician with intermittent nausea for the past 2 months. Her symptoms are exacerbated within one hour after eating. She has no other history of a serious illness. She takes metformin and injects insulin. Her vitals are normal. Abdominal examination is normal. An ECG shows normal sinus rhythm with no evidence of ischemia. Hemoglobin A1c is 7%. A gastric emptying scan shows 60% of her meal in the stomach 75 minutes after eating. Which of the following is the most appropriate pharmacotherapy at this time?\n\nA. Dimenhydrinate\nB. Lorazepam\nC. Metoclopramide\nD. Octreotide\nE. Ondansetron",
        "answer": """\
First, what is the most likely cause of the patient's symptoms based on her presentation and diagnostic findings?
Answer: Gastroparesis, which is delayed gastric emptying commonly seen in patients with diabetes mellitus.
Next, what is the first-line pharmacotherapy for gastroparesis?
Answer: Metoclopramide, a prokinetic agent that enhances gastric emptying and improves symptoms of gastroparesis.
Therefore, the most appropriate pharmacotherapy for this patient at this time is metoclopramide.
Answer: C. Metoclopramide"""
    },
    {
        "question": "Question: A 56-year-old female presents for initial evaluation by a rheumatologist with a chief complaint of back and joint pain. She says that she has been having mild pain for years, but that the pain has become worse over the course of the last 6 months. She clarifies that the pain is most severe in the mornings just after waking up but seems to improve throughout the day. She also notices that her mouth feels dry and she has difficulty eating dry food such as crackers. Finally, she has the sensation of having bits of sand in her eyes. She denies any past medical history or medication use. Serology for which of the following would most likely be positive in this patient?\n\nA. Anti-smooth muscle antibody\nB. Anti-centromere antibody\nC. Anti-cyclic citrullinated peptide (CCP) antibody\nD. Anti-Jo1 and anti-Mi2 antibodies\nE. Anti-Ro and anti-La antibodies",
        "answer": """\
To determine the serology most likely to be positive in this patient, we need to consider the presenting symptoms and clinical presentation. The patient's chief complaint of back and joint pain, along with dry mouth, difficulty eating dry food, and sensation of having bits of sand in her eyes, are suggestive of Sjögren's syndrome.
Sjögren's syndrome is an autoimmune condition characterized by dry eyes and mouth, joint pain, and fatigue. Serology for Sjögren's syndrome typically includes testing for anti-Ro and anti-La antibodies. These antibodies are commonly found in patients with Sjögren's syndrome and are associated with the autoimmune response that targets the salivary and lacrimal glands.
Therefore, the correct answer is anti-Ro and anti-La antibodies.
Answer: E. Anti-Ro and anti-La antibodies"""
    }
]

def MEDQA_PARSE_RESULT(result: str, row: dict) -> Optional[str]:
    for pred in [f"{k}. {v}" for k, v in row["options"].items()]:
        if pred in result:
            return pred[0]
    return None

def MEDQA_FORMAT_OUT(row: dict, pred: Optional[str], output: str) -> dict:
    return {
        'question': row['question'],
        'options': row['options'],
        'output': output,
        'prediction': pred,
        'label': row['answer_idx']
    }


medQA_option = DatasetOption(
    MedQADataset,
    MEDQA_TEMPLATE,
    MEDQA_EXAMPLES,
    MEDQA_PARSE_RESULT,
    MEDQA_FORMAT_OUT
)