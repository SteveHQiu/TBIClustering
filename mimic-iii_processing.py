#%%

import pandas as pd
from icd_to_comorbidities import getComorbGroups

#%%

df = pd.read_excel("data/tbi_admission_comorbidities_age.xlsx")

#%% Merging of raw admission entries into single entries grouped by subject ID,

df1 = df.sort_values(["SUBJECT_ID", "SEQ_NUM"]) \
    .groupby(["SUBJECT_ID", "GENDER", "DOB"])["ICD9_CODE"] \
    .apply(",".join) \
    .reset_index(name = "COMORB") # Output 

df2 = df.groupby(["SUBJECT_ID"])["ADMITTIME"].max()
    
df3 = df.groupby(["SUBJECT_ID"])["DOD"].max()

df1 = df1.merge(df2, on="SUBJECT_ID", how="left") \
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
# %%
