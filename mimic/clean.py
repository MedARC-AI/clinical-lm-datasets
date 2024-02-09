import pandas as pd

from note_filter import FILTERED_NOTES_FN
from utils import clean_mimic

from vllm import LLM, SamplingParams

# from zensols.nlp import FeatureToken
# from zensols.mimic import Section
# from zensols.mimicsid import PredictedNote, ApplicationFactory
# from zensols.mimicsid.pred import SectionPredictor

MAX_GEN_TOKENS = 4096
INSTRUCTION = """
Instruction: You are tasked with cleaning a clinical note such that it can be used as high-quality training data for a clinical LLM.

The rules are:
1) Section headers should be in all-caps (e.g., "MEDICATIONS:") and go on their own line which starts with "##".
2) Each sentence should have its own line and be syntactically correct.
3) Include every detail about the patient's history, diagnoses, treatments, medications, and care plan.
4) Remove all empty sections, non-medical text, and sentences which contain mostly numbers.
"""


if __name__ == '__main__':
    # section_predictor = ApplicationFactory.section_predictor()

    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_GEN_TOKENS)
    model = LLM(
        'mistralai/Mixtral-8x7B-Instruct-v0.1',
        dtype='auto',
        tensor_parallel_size=8,
        gpu_memory_utilization=0.9,
    )

    notes = pd.read_csv(FILTERED_NOTES_FN)
    # notes['TEXT'] = notes['TEXT'].apply(clean_mimic)
    
    for record in notes.to_dict('records'):
        text = clean_mimic(record['TEXT'])

        prompt = INSTRUCTION.strip() + '\n\n' + f'# NOTE:\n\n{text}]\n\n# CLEAN NOTE:\n\n'

        outputs = model.generate(prompt, sampling_params)
        print(outputs)
        print(outputs[0].text)
        # sec_obj = section_predictor.predict([text])[0]

        # Sections should have more than 3 tokens
        # Sections should have more words than non-words (numbers, punctuation)
