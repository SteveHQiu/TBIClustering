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

df_init = pd.read_sql(sql=QUERY, con=CONN)
df_init.to_excel("data/chartevents_test.xlsx", index=False) # Exclude index column

#%% DF Processing
import os
import pandas as pd
from pandas import DataFrame
from icd_to_comorbidities import getComorbGroups

#%%
DF_PATH = "data/tbi_admit_icd.xlsx"
ROOT_NAME = os.path.splitext(DF_PATH)[0]
df_init: DataFrame = pd.read_excel(DF_PATH)

#%% Merging of raw admission entries into single entries grouped by subject ID,

df_icds: DataFrame = df_init.sort_values(["SUBJECT_ID", "SEQ_NUM"]) \
    .groupby(["SUBJECT_ID", "GENDER", "DOB"])["ICD9_CODE"] \
    .apply(",".join) \
    .reset_index(name = "comorb") # Gather ICD9 codes for each subject and store them in list

df_admit: DataFrame = df_init.groupby(["SUBJECT_ID"])["ADMITTIME"].max() # Get most recent admission time
    
df_death: DataFrame = df_init.groupby(["SUBJECT_ID"])["DOD"].max() # Get last DOD in case if there are 

df_icds: DataFrame = df_icds.merge(df_admit, on="SUBJECT_ID", how="left") \
    .merge(df_death, on="SUBJECT_ID", how="left") # Merge most recent admission and most recent date of death into comorb df
    
# Data format processing and adding of age
df_icds["DOB"] = pd.to_datetime(df_icds["DOB"]).dt.date # .dt.date conversion needed to prevent overflow during addition 
df_icds["ADMITTIME"] = pd.to_datetime(df_icds["ADMITTIME"]).dt.date 
df_icds["age"] = df_icds.apply(lambda row: (row["ADMITTIME"] - row["DOB"]).days/365, axis=1) # Need .apply() method by bypass overflow addition error as the dates are scrambled and some are 300+ years in the future
df_icds["age"] = df_icds.apply(lambda row: row["age"] if row["age"] < 90 else 90, axis=1) # Merge all ages above 90 to 90 since MIMIC-III shifts all ages above 89 to 300
max_age = max(df_icds["age"])
df_icds["age_normalized"] = df_icds.apply(lambda row: row["age"]/max_age, axis=1) # Generate normalized age by dividing by max age in group 
df_icds["comorb"] = [code_list.split(",") for code_list in df_icds["comorb"]] # Convert ICD code string into list 

df_comorb = getComorbGroups(df_icds, "SUBJECT_ID", ["comorb"]) # Retrieve elixhauser comorbidity groups


df_final = df_icds.merge(df_comorb, on="SUBJECT_ID", how="left")
df_final.to_excel(F"{ROOT_NAME}_age_elix.xlsx")


#%% Getting df for population info
col_origin = ["congestive_heart_failure", "cardiac_arrhythmia", "valvular_disease", "pulmonary_circulation_disorder", "peripheral_vascular_disorder", "hypertension_uncomplicated", "hypertension_complicated", "paralysis", "other_neurological_disorder", "chronic_pulmonary_disease", "diabetes_uncomplicated", "diabetes_complicated", "hypothyroidism", "renal_failure", "liver_disease", "peptic_ulcer_disease_excluding_bleeding", "aids_hiv", "lymphoma", "metastatic_cancer", "solid_tumor_wo_metastasis", "rheumatoid_arhritis", "coagulopathy", "obesity", "weight_loss", "fluid_and_electrolyte_disorders", "blood_loss_anemia", "deficiency_anemia", "alcohol_abuse", "drug_abuse", "psychoses", "depression"]
df_raw_info = pd.read_excel(F"{ROOT_NAME}_age_elix.xlsx")
#%%
df1 = DataFrame()
df1["num_comorbs"] = df_raw_info[col_origin].sum(axis=1)
