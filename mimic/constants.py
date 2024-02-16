import os

MIMIC_DIR = '/weka/home-griffin/clinical_pile/mimic'

MIMIC_III_NOTES = os.path.join(MIMIC_DIR, 'mimic_iii_carevue_subset_notes.csv')
MIMIC_IV_DSUMS = os.path.join(MIMIC_DIR, 'mimic_iv_discharge.csv')
MIMIC_IV_REPORTS = os.path.join(MIMIC_DIR, 'mimic_iv_radiology.csv')
