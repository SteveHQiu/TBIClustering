#%% DF Processing
import os
import seaborn as sns
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from icd_to_comorbidities import getComorbGroups


#%%
DF_PATH = "data/tbi2_admit_icd.xlsx"
ROOT_NAME = os.path.splitext(DF_PATH)[0]
df_init: DataFrame = pd.read_excel(DF_PATH)

#%% Merging of raw admission entries into single entries grouped by subject ID,

df_icds: DataFrame = df_init.sort_values(["SUBJECT_ID", "SEQ_NUM"]) \
    .groupby(["SUBJECT_ID", "GENDER", "DOB"])["ICD9_CODE"] \
    .apply(",".join) \
    .reset_index(name = "comorb") # Gather ICD9 codes for each subject and store them in list

#%%
df_admit: DataFrame = df_init.groupby(["SUBJECT_ID"])["ADMITTIME"].min() # Get first admission time
df_discharge: DataFrame = df_init.groupby(["SUBJECT_ID"])["DISCHTIME"].min() # Get first discharge time
# TBI_v2 = 2713 - 2629 extra admissions

#%%
df_death: DataFrame = df_init.groupby(["SUBJECT_ID"])["DOD"].max() # Get last DOD in case if there are extra

df_icds: DataFrame = df_icds.merge(df_admit, on="SUBJECT_ID", how="left") \
    .merge(df_discharge, on="SUBJECT_ID", how="left") \
    .merge(df_death, on="SUBJECT_ID", how="left")
    # Merge admission, discharge, death values (each should only have 1 per patient) on SUBJECT_ID

#%%
# Data format processing and adding of age
df_icds["DOB"] = pd.to_datetime(df_icds["DOB"]).dt.date # .dt.date conversion needed to prevent overflow during addition
df_icds["ADMITTIME"] = pd.to_datetime(df_icds["ADMITTIME"]).dt.date
df_icds["DISCHTIME"] = pd.to_datetime(df_icds["DISCHTIME"]).dt.date

# Age & LOS
df_icds["age"] = df_icds.apply(lambda row: (row["ADMITTIME"] - row["DOB"]).days/365, axis=1) # Need .apply() method by bypass overflow addition error as the dates are scrambled and some are 300+ years in the future
df_icds["age"] = df_icds.apply(lambda row: row["age"] if row["age"] < 90 else 90, axis=1) # Merge all ages above 90 to 90 since MIMIC-III shifts all ages above 89 to 300
max_age = max(df_icds["age"])
df_icds["age_normalized"] = df_icds.apply(lambda row: row["age"]/max_age, axis=1) # Generate normalized age by dividing by max age in group 


df_icds["los_days"] = df_icds.apply(lambda row: (row["DISCHTIME"] - row["ADMITTIME"]).days, axis=1) # Need .apply() method by bypass overflow addition error as the dates are scrambled and some are 300+ years in the future

#%%
# ICD

df_icds["comorb"] = [code_list.split(",") for code_list in df_icds["comorb"]] # Convert ICD code string into list 
df_comorb = getComorbGroups(df_icds, "SUBJECT_ID", ["comorb"]) # Retrieve elixhauser comorbidity groups

df_final = df_icds.merge(df_comorb, on="SUBJECT_ID", how="left")
df_final.to_excel(F"{ROOT_NAME}_age_elix.xlsx")


#%% Getting df for population info
col_origin = ["congestive_heart_failure", "cardiac_arrhythmia", "valvular_disease", "pulmonary_circulation_disorder", "peripheral_vascular_disorder", "hypertension_uncomplicated", "hypertension_complicated", "paralysis", "other_neurological_disorder", "chronic_pulmonary_disease", "diabetes_uncomplicated", "diabetes_complicated", "hypothyroidism", "renal_failure", "liver_disease", "peptic_ulcer_disease_excluding_bleeding", "aids_hiv", "lymphoma", "metastatic_cancer", "solid_tumor_wo_metastasis", "rheumatoid_arhritis", "coagulopathy", "obesity", "weight_loss", "fluid_and_electrolyte_disorders", "blood_loss_anemia", "deficiency_anemia", "alcohol_abuse", "drug_abuse", "psychoses", "depression"]
df_raw_info = pd.read_excel(F"{ROOT_NAME}_age_elix.xlsx")
num_subjs = len(df_raw_info) # Each row is a unique patient 

#%%
df1 = df_raw_info[["SUBJECT_ID", "age", "GENDER"]]
df1["alive_logical"] = df_raw_info["DOD"].isnull() # NaN counts as null, if Nan then is considered alive
df1["alive"] = df1["alive_logical"].apply(lambda x: "Alive" if x else "Expired") # Label for coding category (ggplot2 doesn't consider logical as categorical)
df1["num_comorbs"] = df_raw_info[col_origin].sum(axis=1)
survival = df1["alive_logical"].sum(axis=0)/num_subjs # 0.5843676618584368
df1.to_excel(F"{ROOT_NAME}_popinfo.xlsx")


#%%
df2 = DataFrame()
df2["comorb_count"] = df_raw_info[col_origin].sum(axis=0)
df2["comorb_prop"] = df2["comorb_count"]/num_subjs
df2.to_excel(F"{ROOT_NAME}_comorbdistr.xlsx")

#%%
sns.set_theme()
fig, ax = plt.subplots()
fig.set_size_inches(18, 10)
ax.bar(df2.index, df2["comorb_prop"])
ax.set_xticklabels(df2.index, rotation=45, ha="right") # df2.index re-writes previous labels

#%% Age analysis
df3 = pd.read_excel(F"data/tbi2_admit_icd_age_elix_annotated.xlsx")

#%%
num_clusts = len(df3["df_annot"].value_counts())
df3.groupby(["df_annot"])["age"].plot(kind="kde", xticks=list(range(100)[::5]), xlim=(0, 100),legend=True) # Preview
#%% GCS data 

df = pd.read_excel(F"data/tbi2_admit_chevents_gcs.xlsx")

#%%
df["CHARTTIME"] = pd.to_datetime(df["CHARTTIME"])
df.groupby(["SUBJECT_ID"])["LABEL"].value_counts() # Report of number of each type of GCS measurement
df.groupby(["SUBJECT_ID"])["CHARTTIME"].value_counts() # Report of number of time points



df_gcs_time = df.groupby(["SUBJECT_ID", "LABEL"])["CHARTTIME"].min().reset_index() # reset_index() to flatten multi-index from groupby() and get SUBJECT_ID and LABEL included
df_gcs_time.isna().any(axis=1).sum() # Check for any NaN values (should be none)
#%%
df_gcs = df_gcs_time.merge(df[["SUBJECT_ID", "LABEL", "CHARTTIME", "VALUENUM"]],
                      on=["SUBJECT_ID", "LABEL", "CHARTTIME"], how="left")

df_gcs = df3.merge(df_gcs, on="SUBJECT_ID", how="left")


df4 = df3.merge(df_gcs[["SUBJECT_ID", "LABEL", "CHARTTIME", "VALUENUM"]], on="SUBJECT_ID", how="left")

df4 = df4.drop_duplicates(subset=["SUBJECT_ID", "LABEL", "CHARTTIME"])
df4 = df4.dropna(subset=["SUBJECT_ID", "LABEL", "CHARTTIME"]) # For some reason certain values get copied with NA values, starts with first df_gcs merge
# Duplication of 42 patients without GCS due to left join, 5 more patients with VALUENUM == NaN

# 4785 rows × 1 columns - total counts, df_gcs_time
# 4827 rows × 46 columns
# Not every patient has GCS (2587 of 2629, 42 pt without GCS)
df4.groupby(["SUBJECT_ID"])["LABEL"].count().value_counts() # Number of patients with each type of GCS measurement 


#%%
matches_warnings = []
matches_errors = []
df_gcs_annot = DataFrame()
for subj_id in df4["SUBJECT_ID"].unique():
    df_match = df4[df4["SUBJECT_ID"] == subj_id]
    
    gcs_components = {"GCS - Motor Response",
                      "GCS - Verbal Response",
                      "GCS - Eye Opening"}
    yes_gcs_total = "GCS Total" in set(df_match["LABEL"])
    yes_gcs_comps = gcs_components.issubset(set(df_match["LABEL"])) # Subset check, not equivalence if there are more than 3 
    
    if yes_gcs_total and yes_gcs_comps: # Both
        total_match = df_match[df_match["LABEL"] == "GCS Total"]
        comps_match = df_match[df_match["LABEL"].isin(gcs_components)]
        if total_match["CHARTTIME"].min() < comps_match["CHARTTIME"].min():
            assert len(total_match) == 1 # Only single entry
            gcs_total = total_match["VALUENUM"].sum()
        elif len(comps_match["CHARTTIME"].unique()) > 1: # If components is earlier but staggered, use total
            assert len(total_match) == 1 # Only single entry
            gcs_total = total_match["VALUENUM"].sum()
        else: # Assume components is earlier and intact
            assert len(comps_match) == 3 # Only 3 components
            gcs_total = comps_match["VALUENUM"].sum()
            
        
    elif yes_gcs_total and not yes_gcs_comps: # Only total
        assert len(df_match) == 1 # Only single entry
        gcs_total = df_match["VALUENUM"].sum()
    
    elif not yes_gcs_total and yes_gcs_comps: # Only components
        assert len(df_match) == 3 # Only 3 components
        gcs_total = df_match["VALUENUM"].sum()
        
        if len(df_match["CHARTTIME"].unique()) > 1: # If more than 1 times recorded
            matches_warnings.append(df_match) # Flag

    else: # No GCS, no motor
        gcs_total = np.nan
        
    if not np.isnan(gcs_total) and gcs_total < 3: # If not nan but below 3, flag
        matches_errors.append(df_match) # Flag
        gcs_total = np.nan

    
    new_entry = DataFrame({"SUBJECT_ID": [subj_id], "gcs_init_total": [gcs_total]})
    df_gcs_annot = pd.concat([df_gcs_annot, new_entry])

#%% 
df_gcs_totals = df3.merge(df_gcs_annot, on="SUBJECT_ID", how="left")
df_gcs_totals["gcs_init_cat"] = pd.cut(df_gcs_totals["gcs_init_total"],
                                       right=True,
                                       bins=[2, 8, 13, 15], # Left bound is exclusive
                                       labels=["severe", "moderate", "mild"]
                                       )
df_gcs_totals["age_cat"] = pd.cut(df_gcs_totals["age"],
                                       right=True,
                                       bins=[0, 40, 70, 100], # Left bound is exclusive
                                       labels=["young", "mild-aged", "old"]
                                       )

#%% NSX procedures
df5 = pd.read_excel(F"data/tbi2_admit_proced.xlsx")
df5.drop_duplicates(subset=["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"])

#%%
df5["nsx_icp"] = df5["ICD9_CODE"].isin([110]) # 180 vs 945
df5["nsx_ventriculostomy"] = df5["ICD9_CODE"].isin([221, 222]) # Only 2 patients
df5["nsx_crani"] = df5["ICD9_CODE"].isin([123, 124, 125, 131, 139]) # 508 vs 617

df_icp = df5.groupby(["SUBJECT_ID"])["nsx_icp"].any()
df_ventr = df5.groupby(["SUBJECT_ID"])["nsx_ventriculostomy"].any()
df_crani = df5.groupby(["SUBJECT_ID"])["nsx_crani"].any()
df_nsx = pd.concat([df_icp, df_ventr, df_crani], axis=1)
df_nsx["nsx_any"] = True
#%%
df_final_labels = df_gcs_totals.merge(df_nsx, on="SUBJECT_ID", how="left")
df_final_labels["nsx_icp"].fillna(False, inplace=True)
df_final_labels["nsx_ventriculostomy"].fillna(False, inplace=True)
df_final_labels["nsx_crani"].fillna(False, inplace=True)
df_final_labels["nsx_any"].fillna(False, inplace=True)
#%%
df_final_labels["survival"] = df_final_labels["DOD"].isnull() # NaN counts as null, if Nan then is considered alive
df_final_labels["survival"] = df_final_labels["survival"].apply(lambda x: "Alive" if x else "Expired") # Label for coding category (ggplot2 doesn't consider logical as categorical)

#%%
df_final_labels.to_excel("data/tbi2_admit_templabels.xlsx")

#%% Visualization 

df_final_labels = pd.read_excel("data/tbi2_admit_templabels.xlsx")
def visStackProp(df: DataFrame, primary_ind: str, secondary_ind: str):
    df_grped = DataFrame(df.groupby([primary_ind])[secondary_ind].value_counts(normalize=True))
    df_grped.columns = ["Proportion"]
    df_grped = df_grped.reset_index()
    df_grped.columns = [primary_ind, secondary_ind, "Proportion"]
    df_grped = df_grped.set_index([primary_ind, secondary_ind]).Proportion

    df_grped.unstack().plot(kind="bar", stacked=True)

visStackProp(df_final_labels, "df_annot", "gcs_init_cat")
visStackProp(df_final_labels, "df_annot", "age_cat")

#%%
df_type_gcs_surv = DataFrame(df_final_labels.groupby(["df_annot","gcs_init_cat"])["survival"].value_counts(normalize=True))
df_type_gcs_surv.columns = ["Proportion"]
df_type_gcs_surv = df_type_gcs_surv.reset_index()

fig, axes = plt.subplots(ncols=5)
fig.set_size_inches(15, 7)

for i in df_type_gcs_surv["df_annot"].unique():
    df_clust = df_type_gcs_surv[df_type_gcs_surv["df_annot"] == i]
    df_clust = df_clust.set_index(["gcs_init_cat", "survival"]).Proportion

    df_clust.unstack().plot(kind="bar", stacked=True, ax=axes[i-1])
    
fig
#%%
df_type_gcs_surv = DataFrame(df_final_labels.groupby(["df_annot","age_cat"])["survival"].value_counts(normalize=True))
df_type_gcs_surv.columns = ["Proportion"]
df_type_gcs_surv = df_type_gcs_surv.reset_index()

fig, axes = plt.subplots(ncols=5)
fig.set_size_inches(15, 7)

for i in df_type_gcs_surv["df_annot"].unique():
    df_clust = df_type_gcs_surv[df_type_gcs_surv["df_annot"] == i]
    df_clust = df_clust.set_index(["age_cat", "survival"]).Proportion

    df_clust.unstack().plot(kind="bar", stacked=True, ax=axes[i-1])
    
fig

#%% Check frequency of GCS reports across patients
df_temp1 = DataFrame(df.groupby(["SUBJECT_ID"])["CHARTTIME"].value_counts())
df_temp1.columns = ["VAL"]
df_temp1 = df_temp1.reset_index()
df_temp2 = df_temp1.groupby(["SUBJECT_ID"])["VAL"].count()
df_temp3 = df_temp2.value_counts()
df_temp3[[1, 2, 3]] # Counts for those with only 1, 2, 3 measurements 
