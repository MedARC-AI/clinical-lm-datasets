import pandas as pd

from note_filter import FILTERED_NOTES_FN
from utils import clean_mimic


from zensols.nlp import FeatureToken
from zensols.mimic import Section
from zensols.mimicsid import PredictedNote, ApplicationFactory
from zensols.mimicsid.pred import SectionPredictor


if __name__ == '__main__':
    section_predictor = ApplicationFactory.section_predictor()

    notes = pd.read_csv(FILTERED_NOTES_FN)
    notes['TEXT'] = notes['TEXT'].apply(clean_mimic)
    for record in notes.to_dict('records'):
        text = record['TEXT']
        sec_obj = section_predictor.predict([text])[0]
        print('here')

        # Sections should have more than 3 tokens
        # Sections should have more words than non-words (numbers, punctuation)