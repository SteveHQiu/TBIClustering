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
SELECT admissions.SUBJECT_ID, admissions.HADM_ID, d_items.LABEL, chartevents.CHARTTIME, chartevents.ITEMID, chartevents.VALUE, chartevents.VALUENUM, chartevents.VALUEUOM, d_items.UNITNAME, d_items.CATEGORY
FROM (admissions INNER JOIN diagnoses_icd ON admissions.HADM_ID = diagnoses_icd.HADM_ID) INNER JOIN (chartevents INNER JOIN d_items ON chartevents.ITEMID = d_items.ITEMID) ON diagnoses_icd.HADM_ID = chartevents.HADM_ID
WHERE (((LOWER(d_items.LABEL) Like "%parin%") Or (LOWER(d_items.LABEL) Like "%oban%")) AND ((diagnoses_icd.ICD9_CODE)="3484" Or (diagnoses_icd.ICD9_CODE)="14496" Or (diagnoses_icd.ICD9_CODE) Between "80000" And "80199" Or (diagnoses_icd.ICD9_CODE) Between "80310" And "80499" Or (diagnoses_icd.ICD9_CODE) Between "85100" And "85419" Or (diagnoses_icd.ICD9_CODE)="V8001"));
"""
#%% Execution 

df = pd.read_sql(sql=QUERY, con=CONN)
df.to_excel("data/chartevents_test.xlsx", index=False) # Exclude index column

#%% 

import pandas as pd
from pandas import DataFrame
from icd_to_comorbidities import getComorbGroups

#%%

df: DataFrame = pd.read_excel("data/tbi_admission_comorbidities_age.xlsx")

#%% Merging of raw admission entries into single entries grouped by subject ID,

df1: DataFrame = df.sort_values(["SUBJECT_ID", "SEQ_NUM"]) \
    .groupby(["SUBJECT_ID", "GENDER", "DOB"])["ICD9_CODE"] \
    .apply(",".join) \
    .reset_index(name = "COMORB") # Output 

df2: DataFrame = df.groupby(["SUBJECT_ID"])["ADMITTIME"].max()
    
df3: DataFrame = df.groupby(["SUBJECT_ID"])["DOD"].max()

df1: DataFrame = df1.merge(df2, on="SUBJECT_ID", how="left") \
    .merge(df3, on="SUBJECT_ID", how="left")
    
# Data format processing 
df1["DOB"] = pd.to_datetime(df1["DOB"]).dt.date # .dt.date conversion needed to prevent overflow during addition 
df1["ADMITTIME"] = pd.to_datetime(df1["ADMITTIME"]).dt.date # .dt.date conversion needed to prevent overflow during addition 
df1["AGE"] = df1.apply(lambda e: (e["ADMITTIME"] - e["DOB"]).days/365, axis=1) # Need .apply() method by bypass overflow addition error as the dates are scrambled and some are 300+ years in the future
df1["COMORB"] = [code_list.split(",") for code_list in df1["COMORB"]] # Convert ICD code string into list 

#%%
df_comorb = getComorbGroups(df1, "SUBJECT_ID", ["COMORB"])
# df_comorb.to_excel("data/merged.xlsx")

#%%
df_final = df1.merge(df_comorb, on="SUBJECT_ID", how="left")
df_final.to_excel("data/merged_output_elix.xlsx")
#%%
