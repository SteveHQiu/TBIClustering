#%% Imports
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
SELECT admissions.SUBJECT_ID, admissions.HADM_ID, d_items.LABEL, chartevents.CHARTTIME, chartevents.ITEMID, chartevents.VALUE, chartevents.VALUENUM, chartevents.VALUEUOM, d_items.UNITNAME, d_items.CATEGORY
FROM (admissions INNER JOIN diagnoses_icd ON admissions.HADM_ID = diagnoses_icd.HADM_ID) INNER JOIN (chartevents INNER JOIN d_items ON chartevents.ITEMID = d_items.ITEMID) ON diagnoses_icd.HADM_ID = chartevents.HADM_ID
WHERE (((LOWER(d_items.LABEL) Like "%parin%") Or (LOWER(d_items.LABEL) Like "%oban%")) AND ((diagnoses_icd.ICD9_CODE)="3484" Or (diagnoses_icd.ICD9_CODE)="14496" Or (diagnoses_icd.ICD9_CODE) Between "80000" And "80199" Or (diagnoses_icd.ICD9_CODE) Between "80310" And "80499" Or (diagnoses_icd.ICD9_CODE) Between "85100" And "85419" Or (diagnoses_icd.ICD9_CODE)="V8001"));
"""
#%% Execution 

df = pd.read_sql(sql=QUERY, con=CONN)
df.to_excel("data/chartevents_test.xlsx", index=False) # Exclude index column

