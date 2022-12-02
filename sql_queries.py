#%% SQlite connection
import sqlite3
import pandas as pd


#%% Constants
DB_PATH = r"C:\Users\steve\Downloads\ZLocal\mimic-iii-clinical-database-1.4\mimic3.db"
CONN = sqlite3.connect(DB_PATH) # Start connection
C = CONN.cursor() # For if you want to execute SQL commands directly via C.execute("<sql code>")

QUERY_TEST = """
SELECT admissions.SUBJECT_ID, admissions.HADM_ID, admissions.DIAGNOSIS, diagnoses_icd.ICD9_CODE, d_icd_diagnoses.LONG_TITLE
FROM (admissions INNER JOIN diagnoses_icd ON admissions.HADM_ID = diagnoses_icd.HADM_ID) INNER JOIN d_icd_diagnoses ON diagnoses_icd.ICD9_CODE = d_icd_diagnoses.ICD9_CODE
WHERE (((diagnoses_icd.ICD9_CODE)="3484" Or (diagnoses_icd.ICD9_CODE)="14496" Or (diagnoses_icd.ICD9_CODE) Between "80000" And "80199" Or (diagnoses_icd.ICD9_CODE) Between "80310" And "80499" Or (diagnoses_icd.ICD9_CODE) Between "85100" And "85419" Or (diagnoses_icd.ICD9_CODE)="V8001"));
"""

QUERY = """
SELECT admissions.SUBJECT_ID, admissions.HADM_ID, admissions.ADMITTIME, admissions.DISCHTIME, admissions.DEATHTIME, admissions.DIAGNOSIS, diagnoses_icd.ICD9_CODE, chartevents.ICUSTAY_ID, chartevents.CHARTTIME, chartevents.STORETIME, d_items.LABEL, chartevents.VALUE, chartevents.VALUE, chartevents.VALUENUM, chartevents.VALUEUOM, d_items.CATEGORY, d_items.PARAM_TYPE
FROM ((diagnoses_icd INNER JOIN admissions ON diagnoses_icd.HADM_ID = admissions.HADM_ID) INNER JOIN d_icd_diagnoses ON diagnoses_icd.ICD9_CODE = d_icd_diagnoses.ICD9_CODE) INNER JOIN (chartevents INNER JOIN d_items ON chartevents.ITEMID = d_items.ITEMID) ON admissions.HADM_ID = chartevents.HADM_ID
WHERE (((diagnoses_icd.ICD9_CODE) Between "80000" And "80199" Or (diagnoses_icd.ICD9_CODE) Between "80300" And "80499" Or (diagnoses_icd.ICD9_CODE) Between "8500" And "8509" Or (diagnoses_icd.ICD9_CODE) Between "85000" And "85419" Or (diagnoses_icd.ICD9_CODE) Between "9501" And "9503" Or (diagnoses_icd.ICD9_CODE)="95901") AND (LOWER(d_items.LABEL) = "surgery" Or LOWER(d_items.LABEL) = "Surgery"));
""" # tbi v2, chartevents for surgery
 
QUERY = """
SELECT admissions.SUBJECT_ID, admissions.HADM_ID, admissions.ADMITTIME, admissions.DISCHTIME, admissions.DEATHTIME, admissions.DIAGNOSIS, diagnoses_icd.ICD9_CODE, chartevents.ICUSTAY_ID, chartevents.CHARTTIME, chartevents.STORETIME, d_items.LABEL, chartevents.VALUE, chartevents.VALUE, chartevents.VALUENUM, chartevents.VALUEUOM, d_items.CATEGORY, d_items.PARAM_TYPE
FROM ((diagnoses_icd INNER JOIN admissions ON diagnoses_icd.HADM_ID = admissions.HADM_ID) INNER JOIN d_icd_diagnoses ON diagnoses_icd.ICD9_CODE = d_icd_diagnoses.ICD9_CODE) INNER JOIN (chartevents INNER JOIN d_items ON chartevents.ITEMID = d_items.ITEMID) ON admissions.HADM_ID = chartevents.HADM_ID
WHERE (((diagnoses_icd.ICD9_CODE) Between "80000" And "80199" Or (diagnoses_icd.ICD9_CODE) Between "80300" And "80499" Or (diagnoses_icd.ICD9_CODE) Between "8500" And "8509" Or (diagnoses_icd.ICD9_CODE) Between "85000" And "85419" Or (diagnoses_icd.ICD9_CODE) Between "9501" And "9503" Or (diagnoses_icd.ICD9_CODE)="95901") AND (LOWER(d_items.LABEL) Like "%gcs%"));
""" # tbi v2, chartevents for GCS


#%% Execution

df_init = pd.read_sql(sql=QUERY, con=CONN)
df_init.to_excel(F"data/tbi2_admit_chevents.xlsx", index=False) # Exclude index column
