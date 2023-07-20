#%% DF Processing
import os, json
import seaborn as sns
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


#%% Constants/Data
DF_PATH = "data/tbi2_admit_icd.xlsx"
ROOT_NAME = os.path.splitext(DF_PATH)[0]
# TBI_v2 = (2713 - 2629) extra admissions

df_init: DataFrame = pd.read_excel(DF_PATH)
#%%
df_nsx_proc = pd.read_excel(F"data/tbi2_admit_proced.xlsx")
df_gcs_events = pd.read_excel(F"data/tbi2_admit_chevents_gcs.xlsx")
#%%
COL_AGE = "Age (years)"
COL_LOS = "Length of stay (days)"
COL_AGE_NORM = "Age (normalized to max age in group)"
COL_AGE_CAT = "Age category"
AGE_LABELS = ["1. Young", "2. Middle-aged", "3. Old"]
COL_SURV = "Survival to discharge"
COL_GCS = "First recorded total GCS score"
COL_GCS_CAT = "TBI severity by initial GCS"
GCS_LABELS = ["3. Severe", "2. Moderate", "1. Mild"]
COL_GCS_CAT2 = "TBI severity by initial GCS (Sev + Mod vs Mild)"
GCS_LABELS2 = ["2. Severe and Moderate", "1. Mild"]
COL_ICPMON = "ICP monitoring"
COL_VENTRIC = "Ventriculostomy"
COL_CRANI = "Craniotomy or craniectomy"
COL_NSX_ANY = "Any neurosurgical intervention"
COL_COMORBS = "No. Comorbs"



LABELS = ["congestive_heart_failure", "cardiac_arrhythmia", "valvular_disease",
             "pulmonary_circulation_disorder", "peripheral_vascular_disorder",
             "hypertension_uncomplicated", "hypertension_complicated", "paralysis",
             "other_neurological_disorder", "chronic_pulmonary_disease", "diabetes_uncomplicated",
             "diabetes_complicated", "hypothyroidism", "renal_failure", "liver_disease",
             "peptic_ulcer_disease_excluding_bleeding", "aids_hiv", "lymphoma", "metastatic_cancer",
             "solid_tumor_wo_metastasis", "rheumatoid_arhritis", "coagulopathy", "obesity",
             "weight_loss", "fluid_and_electrolyte_disorders", "blood_loss_anemia",
             "deficiency_anemia", "alcohol_abuse", "drug_abuse", "psychoses", "depression"]

BETTER_LABELS = ["CHF", "Arrhythmia", "Valvular disease", "Pulmonary circulation disorder",
                "Peripheral vascular disorder", "Uncomplicated hypertension",
                "Complicated hypertension", "Paralysis", "Other neurological disorder",
                "COPD", "Uncomplicated diabetes", "Complicated diabetes", "Hypothyroidism",
                "Renal failure", "Liver disease", "Peptic ulcer disease", "AID/HIV",
                "Lymphoma", "Metastatic cancer", "Solid tumor (no metastasis)",
                "Rheumatoid arthritis", "Coagulopathy", "Obesity", "Weight loss",
                "Fluid and electrolyte disorders", "Blood loss anemia", "Deficiency anemia",
                "Alcohol abuse", "Drug abuse", "Psychoses", "Depression"]

#%% Functions
def deriveAdmitData(df_base: DataFrame,
                    col_id = "SUBJECT_ID",
                    col_sex = "GENDER",
                    col_seq = "SEQ_NUM",
                    col_icd = "ICD9_CODE",
                    col_admittime = "ADMITTIME",
                    col_dischtime = "DISCHTIME",
                    col_death = "DOD",
                    col_birth = "DOB",
                    ):
    
    df_icd: DataFrame = df_base.sort_values([col_id, col_seq]) \
        .groupby([col_id, col_sex])[col_icd] \
        .apply(lambda x: x.to_json(orient="records")) \
        .reset_index(name = "comorb") # Gather ICD9 codes for each subject and store them in list
        # to_json() method: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.to_json.html
        # .apply(json.loads) to de-serialize
        # .apply(",".join) \
    df_admit: DataFrame = df_base.groupby([col_id])[col_admittime].min() # Get first admission time
    df_discharge: DataFrame = df_base.groupby([col_id])[col_dischtime].min() # Get first discharge time
    df_death: DataFrame = df_base.groupby([col_id])[col_death].max() # Get last DOD in case if there are extra
    df_dob: DataFrame = df_base.groupby([col_id])[col_birth].max() # Get max DOB in case of multiple entries

    df_grped_data: DataFrame = df_icd.merge(df_admit, on=col_id, how="left") \
        .merge(df_discharge, on=col_id, how="left") \
        .merge(df_death, on=col_id, how="left") \
        .merge(df_dob, on=col_id, how="left")
        # Merge admission, discharge, death values (each should only have 1 per patient) on SUBJECT_ID

    
    # Data format processing and adding of age
    df_grped_data[col_birth] = pd.to_datetime(df_grped_data[col_birth]).dt.date # .dt.date conversion needed to prevent overflow during addition
    df_grped_data[col_admittime] = pd.to_datetime(df_grped_data[col_admittime]).dt.date
    df_grped_data[col_dischtime] = pd.to_datetime(df_grped_data[col_dischtime]).dt.date

    # Demographics
    df_grped_data[COL_LOS] = df_grped_data.apply(lambda row: (row[col_dischtime] - row[col_admittime]).days, axis=1) # Need .apply() method by bypass overflow addition error as the dates are scrambled and some are 300+ years in the future
    df_grped_data[COL_AGE] = df_grped_data.apply(lambda row: (row[col_admittime] - row[col_birth]).days/365, axis=1) # Need .apply() method by bypass overflow addition error as the dates are scrambled and some are 300+ years in the future
    df_grped_data[COL_AGE] = df_grped_data.apply(lambda row: row[COL_AGE] if row[COL_AGE] < 90 else 90, axis=1) # Merge all ages above 90 to 90 since MIMIC-III shifts all ages above 89 to 300
    df_grped_data[COL_AGE_NORM] = df_grped_data[COL_AGE]/df_grped_data[COL_AGE].max() # Normalize by max value 
    df_grped_data[COL_AGE_CAT] = pd.cut(df_grped_data[COL_AGE],
                                        right=True,
                                        bins=[0, 40, 70, 100], # Left bound is exclusive
                                        labels=AGE_LABELS
                                        )
    df_grped_data[COL_SURV] = df_grped_data[col_death].isnull() # NaN counts as null, if Nan then is considered alive
    df_grped_data[COL_SURV] = df_grped_data[COL_SURV].apply(lambda x: "Alive" if x else "Expired") # Label for coding category (ggplot2 doesn't consider logical as categorical)

    return df_grped_data


def deriveComorb(df_raw_icds: DataFrame,
               mapping="icd9",
               col_index = "SUBJECT_ID",
               list_col_icds = ["comorb"],
               ):
    """
    Returns:
        New Pandas DataFrame containing the ids specified with same name along with the comorbidites containing Bools True \
        or False if the icd code was found in the mapping
    # Adapated from package from https://pypi.org/project/icd/
    """
    for idx in list_col_icds: # If multiple columns with ICDs in JSON, deserialize all of them 
        df_raw_icds[idx] = df_raw_icds[idx].apply(json.loads) # De-serialize json into list of strings
    
    validMappings = ["quan_elixhauser10", "charlson10",
                     "icd9"]  # Add extra mappings to this list

    if isinstance(mapping, str) and mapping not in validMappings:
        raise Exception(
            "Didn't recognize comorbidity mapping. Please use a valid comorbidity or create your own.")
    if isinstance(mapping, str):
        script_dir = os.path.dirname(__file__)
        rel_path = "comorbidity_mappings/" + mapping + ".json"
        file_path = os.path.join(script_dir, rel_path)

        mapping = json.load(open(file_path))
    elif isinstance(mapping, dict):
        pass
    else:
        raise Exception("Bad mapping type")

    comorb_cols = list(mapping.keys())
    df_comorb_annot = pd.DataFrame(index=df_raw_icds[col_index], columns=comorb_cols)

    for comorb in comorb_cols: # Construct mapping for each comorbidity column 
        # reset truth list then
        # Create a list of lists, will end up with rows of df by len(idxs) long
        # Line below is modified to suit column containing list format (items in the column must be in list format)
        truth_list = [[any([code for code in code_list if any(
            condition == code for condition in mapping[comorb])]) for code_list in df_raw_icds[idx]] for idx in list_col_icds]
        # Take a code from code_list in column containing ICD codes
        # Only take code if the code is equivalent to a condition contained in the mapping of the respective comorbidity mapping
        # Cannot just search for the code in condition or vice versa using string functions, otherwise shorter codes will be incorrectly mapped to longer codes that contain a small snippit of them
        # E.g., "42" for HIV will match positive for "8242" which may be a TBI diagnosis
        # Code_list is from a df in memory with items in lists rather than contained within a string 

        # Swapping dimensions on list of lists so we can apply a listwise operation to the longer dimension (rows of df)
        truth_list = list(map(list, zip(*truth_list)))

        # Condense the icd truth dimensions on a column basis to get a bool yes or no if multiple or at least one icd code is true
        condensed_truth = [any(truth_list[i]) ==
                           True for i in range(0, len(truth_list))]

        # assign the column of comorb_df with the appropriate truth values
        df_comorb_annot[comorb] = condensed_truth

    # comorb_df[idx] = comorb_df.index.tolist()

    return df_comorb_annot

def deriveNsxProcedures(df_nsx_proc: DataFrame,
                        col_id = "SUBJECT_ID",
                        col_icd = "ICD9_CODE",
                        rem_dupl_subset = ["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"],
                        ):
    df_nsx_proc.drop_duplicates(subset=rem_dupl_subset, inplace=True)
    
    df_nsx_proc[COL_ICPMON] = df_nsx_proc[col_icd].isin([110]) # 180 vs 945
    df_nsx_proc[COL_VENTRIC] = df_nsx_proc[col_icd].isin([221, 222]) # Only 2 patients
    df_nsx_proc[COL_CRANI] = df_nsx_proc[col_icd].isin([123, 124, 125, 131, 139]) # 508 vs 617

    df_icp = df_nsx_proc.groupby([col_id])[COL_ICPMON].any()
    df_ventr = df_nsx_proc.groupby([col_id])[COL_VENTRIC].any()
    df_crani = df_nsx_proc.groupby([col_id])[COL_CRANI].any()
    df_nsx_annot = pd.concat([df_icp, df_ventr, df_crani], axis=1)
    df_nsx_annot[COL_NSX_ANY] = True
    
    return df_nsx_annot

def deriveGCS(df_gcs_events: DataFrame,
              col_id = "SUBJECT_ID",
              col_charttime = "CHARTTIME",
              col_label = "LABEL",
              col_value = "VALUENUM",
              ):
    
    df_gcs_events[col_charttime] = pd.to_datetime(df_gcs_events[col_charttime])
    df_gcs_events.groupby([col_id])[col_label].value_counts() # Report of number of each type of GCS measurement
    df_gcs_events.groupby([col_id])[col_charttime].value_counts() # Report of number of time points

    df_gcs_time = df_gcs_events.groupby([col_id, col_label])[col_charttime].min().reset_index() # reset_index() to flatten multi-index from groupby() and get SUBJECT_ID and LABEL included
    df_gcs_time.isna().any(axis=1).sum() # Check for any NaN values (should be none)

    df_gcs_values = df_gcs_time.merge(df_gcs_events[[col_id, col_label, col_charttime, col_value]],
                        on=[col_id, col_label, col_charttime], how="left") # Get values from original chartevents
    df_gcs_values = df_gcs_values.drop_duplicates(subset=[col_id, col_label, col_charttime]) # Merge results in some duplication, should remove these
    # Duplication of 42 patients without GCS due to left join, 5 more patients with VALUENUM == NaN
    # 4785 rows × 1 columns - total counts, df_gcs_time
    # 4827 rows × 46 columns

    
    matches_warnings = []
    matches_errors = []
    df_gcs_annot = DataFrame()
    for subj_id in df_gcs_values[col_id].unique():
        df_match = df_gcs_values[df_gcs_values[col_id] == subj_id]
        
        gcs_components = {"GCS - Motor Response",
                        "GCS - Verbal Response",
                        "GCS - Eye Opening"}
        yes_gcs_total = "GCS Total" in set(df_match[col_label])
        yes_gcs_comps = gcs_components.issubset(set(df_match[col_label])) # Subset check, not equivalence if there are more than 3 
        
        if yes_gcs_total and yes_gcs_comps: # Both
            total_match = df_match[df_match[col_label] == "GCS Total"]
            comps_match = df_match[df_match[col_label].isin(gcs_components)]
            if total_match[col_charttime].min() < comps_match[col_charttime].min():
                assert len(total_match) == 1 # Only single entry
                gcs_total = total_match[col_value].sum()
            elif len(comps_match[col_charttime].unique()) > 1: # If components is earlier but staggered, use total
                assert len(total_match) == 1 # Only single entry
                gcs_total = total_match[col_value].sum()
            else: # Assume components is earlier and intact
                assert len(comps_match) == 3 # Only 3 components
                gcs_total = comps_match[col_value].sum()
                
            
        elif yes_gcs_total and not yes_gcs_comps: # Only total
            assert len(df_match) == 1 # Only single entry
            gcs_total = df_match[col_value].sum()
        
        elif not yes_gcs_total and yes_gcs_comps: # Only components
            assert len(df_match) == 3 # Only 3 components
            gcs_total = df_match[col_value].sum()
            
            if len(df_match[col_charttime].unique()) > 1: # If more than 1 times recorded
                matches_warnings.append(df_match) # Flag

        else: # No GCS, no motor
            gcs_total = np.nan
            
        if not np.isnan(gcs_total) and gcs_total < 3: # If not nan but below 3, flag
            matches_errors.append(df_match) # Flag
            gcs_total = np.nan

        
        new_entry = DataFrame({col_id: [subj_id], COL_GCS: [gcs_total]})
        df_gcs_annot = pd.concat([df_gcs_annot, new_entry])
        # Not every patient has GCS (2587 of 2629, 42 pt without GCS)

    df_gcs_annot[COL_GCS_CAT] = pd.cut(df_gcs_annot[COL_GCS],
                                        right=True,
                                        bins=[2, 8, 13, 15], # Left bound is exclusive
                                        labels=GCS_LABELS
                                        )
    df_gcs_annot[COL_GCS_CAT2] = pd.cut(df_gcs_annot[COL_GCS],
                                        right=True,
                                        bins=[2, 13, 15], # Left bound is exclusive
                                        labels=GCS_LABELS2
                                        )
    return df_gcs_annot

#%%
df_grped_data = deriveAdmitData(df_init)
df_elix_annot = deriveComorb(df_grped_data)
df_nsx_annot = deriveNsxProcedures(df_nsx_proc)
df_gcs_annot = deriveGCS(df_gcs_events)
#%%
df_labelled = df_grped_data.merge(df_nsx_annot, on="SUBJECT_ID", how="left") \
    .merge(df_gcs_annot, on="SUBJECT_ID", how="left") \
    .merge(df_elix_annot, on="SUBJECT_ID", how="left")
    
df_labelled.fillna({COL_NSX_ANY: False, COL_VENTRIC: False, # Need dict to specify multiple columns (can't use slice since read-only)
                        COL_CRANI: False, COL_ICPMON: False}, inplace=True) # Fill NAs after merge
df_labelled[COL_COMORBS] = df_labelled[LABELS].sum(axis=1)
#%%
df_labelled.to_excel(F"{ROOT_NAME}_dates_nsx_gcs_elix.xlsx")


###### End of core data processing pipeline 

if 0: # Auxilliary processing pipelines 
    #%% Rename columns
    DF_ANNOTATED_PATH = F"data/tbi2_admit_icd_dates_nsx_gcs_elix_annotated_v4.xlsx"
    DF_ANNOTATED_ROOT = os.path.splitext(DF_ANNOTATED_PATH)[0]

    df_annotated = pd.read_excel(DF_ANNOTATED_PATH)
    col_name_dict = {pair[0]: pair[1] for pair in zip(LABELS, BETTER_LABELS)}
    df_annotated.rename(columns=col_name_dict, inplace=True)
    df_annotated.to_excel(F"{DF_ANNOTATED_ROOT}_BL.xlsx")
    
    #%% Check frequency of GCS reports across patients
    df_temp1 = DataFrame(df_gcs_events.groupby(["SUBJECT_ID"])["CHARTTIME"].value_counts())
    df_temp1.columns = ["VAL"]
    df_temp1 = df_temp1.reset_index()
    df_temp2 = df_temp1.groupby(["SUBJECT_ID"])["VAL"].count()
    df_temp3 = df_temp2.value_counts()
    df_temp3[[1, 2, 3]] # Counts for those with only 1, 2, 3 measurements 


    #%% Getting df for population info
    col_origin = ["congestive_heart_failure", "cardiac_arrhythmia", "valvular_disease", "pulmonary_circulation_disorder", "peripheral_vascular_disorder", "hypertension_uncomplicated", "hypertension_complicated", "paralysis", "other_neurological_disorder", "chronic_pulmonary_disease", "diabetes_uncomplicated", "diabetes_complicated", "hypothyroidism", "renal_failure", "liver_disease", "peptic_ulcer_disease_excluding_bleeding", "aids_hiv", "lymphoma", "metastatic_cancer", "solid_tumor_wo_metastasis", "rheumatoid_arhritis", "coagulopathy", "obesity", "weight_loss", "fluid_and_electrolyte_disorders", "blood_loss_anemia", "deficiency_anemia", "alcohol_abuse", "drug_abuse", "psychoses", "depression"]
    df_raw_info = pd.read_excel(F"{ROOT_NAME}_age_elix.xlsx")
    num_subjs = len(df_raw_info) # Each row is a unique patient 

    #%% Export of population info
    df1 = df_raw_info[["SUBJECT_ID", COL_AGE, "GENDER"]]
    df1["alive_logical"] = df_raw_info["DOD"].isnull() # NaN counts as null, if Nan then is considered alive
    df1["alive"] = df1["alive_logical"].apply(lambda x: "Alive" if x else "Expired") # Label for coding category (ggplot2 doesn't consider logical as categorical)
    df1["num_comorbs"] = df_raw_info[col_origin].sum(axis=1)
    survival = df1["alive_logical"].sum(axis=0)/num_subjs # 0.5843676618584368
    df1.to_excel(F"{ROOT_NAME}_popinfo.xlsx")


    #%% Export of comorbidity distributions
    df2 = DataFrame()
    df2["comorb_count"] = df_raw_info[col_origin].sum(axis=0)
    df2["comorb_prop"] = df2["comorb_count"]/num_subjs
    df2.to_excel(F"{ROOT_NAME}_comorbdistr.xlsx")

    #%% Jupyter notebook visulization of comorbidity disitributions
    sns.set_theme()
    fig, ax = plt.subplots()
    fig.set_size_inches(18, 10)
    ax.bar(df2.index, df2["comorb_prop"])
    ax.set_xticklabels(df2.index, rotation=45, ha="right") # df2.index re-writes previous labels
