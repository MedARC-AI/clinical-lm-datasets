import os

MIMIC_DIR = '/weka/home-griffin/clinical_pile/mimic'

MIMIC_III_NOTES = os.path.join(MIMIC_DIR, 'NOTEEVENTS.csv')
MIMIC_IV_DSUMS = os.path.join(MIMIC_DIR, 'discharge.csv')
MIMIC_IV_REPORTS = os.path.join(MIMIC_DIR, 'radiology_reports.csv')
