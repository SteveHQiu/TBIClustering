#%%
import os, json
from itertools import combinations

import seaborn as sns
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy import stats

from statsmodels.stats.proportion import proportions_chisquare_allpairs
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.compat.python import lzip
from statsmodels.stats.multitest import multipletests

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#%% Copied from processing (avoid importing)
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


#%%
df_labels = pd.read_excel("data/tbi2_admit_icd_dates_nsx_gcs_elix_annotated_v4.xlsx")
df_labels[COL_LOS] = df_labels[COL_LOS].mask(df_labels[COL_LOS].sub(df_labels[COL_LOS].mean())
                                             .div(df_labels[COL_LOS].std())
                                             .abs()
                                             .gt(2))

# Remove outliers from LOS, 95 values removed at SD of 2, 33 at SD of 3
#%%
df_labels[COL_AGE].mean()
df_labels["GENDER"].value_counts()
df_labels[COL_SURV].value_counts()
#%% Functions
DF_CONT = []

# Visualization Functions

def _desatColors(colors, percent):
    # Colors should be in np format of RGBA with range of 0-1
    vector_diff = 1 - colors # Subtract color from pure white
    return colors + vector_diff * percent # Add difference vector scaled by percent


def visStackedProp(df_labels: DataFrame, primary_ind: str, secondary_ind: str):
    df_grped = DataFrame(df_labels.groupby([primary_ind])[secondary_ind].value_counts(normalize=True))
    df_grped.columns = ["Proportion"]
    df_grped = df_grped.reset_index()
    df_grped.columns = [primary_ind, secondary_ind, "Proportion"]
    
    df_grped = df_grped.set_index([primary_ind, secondary_ind]).Proportion
    
    ax = df_grped.unstack().plot(kind="bar", stacked=True)
    ax.set_ylabel("Proportion")
    
    df_grped2 = DataFrame(df_labels.groupby([primary_ind])[secondary_ind].value_counts())
    df_grped3 = DataFrame(df_labels.groupby([primary_ind])[secondary_ind].value_counts(normalize=True))
    df_grped3.columns = ["Proportion"]
    df_report = pd.concat([df_grped2, df_grped3], axis=1)
    
    print(df_report)
    DF_CONT.append(df_report)


def visContinuous(df_labels: DataFrame, primary_ind: str, secondary_ind: str):
    df_grped = DataFrame(df_labels.groupby([primary_ind])[secondary_ind].mean())
    df_grped = df_grped.reset_index()
    df_grped = df_grped.set_index(primary_ind)
    
    ax = df_grped.plot(kind="bar")
    ax.set_ylabel(secondary_ind)
    
    
    df_grped2 = DataFrame(df_labels.groupby([primary_ind])[secondary_ind].mean())
    df_grped3 = DataFrame(df_labels.groupby([primary_ind])[secondary_ind].std())
    df_grped3.columns = ["Standard deviation"]
    df_report = pd.concat([df_grped2, df_grped3], axis=1)
    
    print(df_report)
    DF_CONT.append(df_report)


def visGrpedDichotomous(df_labels: DataFrame, col_prim_grp: str, col_sec_grp: str,
                    col_outcome: str, ylab: str = "Proportion"):
    df_grped = DataFrame(df_labels.groupby([col_prim_grp, col_sec_grp])[col_outcome].value_counts(normalize=True))
    df_grped.columns = ["Proportion"]
    df_grped = df_grped.reset_index()
    clust_labels = df_grped[col_prim_grp].unique()
    
    df_grped2 = DataFrame(df_labels.groupby([col_prim_grp, col_sec_grp])[col_outcome].value_counts())
    df_grped3 = DataFrame(df_labels.groupby([col_prim_grp, col_sec_grp])[col_outcome].value_counts(normalize=True))
    df_grped3.columns = ["Proportion"]
    df_report = pd.concat([df_grped2, df_grped3], axis=1)
    
    print(df_report)
    DF_CONT.append(df_report)
    
    colors_norm = plt.colormaps['gist_rainbow'](np.linspace(0.08, 0.89, len(clust_labels)))
    colors_desat = _desatColors(colors_norm, 0.75)
    
    fig, axes = plt.subplots(ncols=len(clust_labels))
    fig.set_size_inches(15, 7)
    axes[0].set_ylabel(ylab) # Set y-label for first plot

    for ind, label in enumerate(clust_labels):
        clust_color = [colors_norm[ind], colors_desat[ind]]
        df_clust = df_grped[df_grped[col_prim_grp] == label]
        df_clust = df_clust.set_index([col_sec_grp, col_outcome]).Proportion

        df_clust.unstack().plot(kind="bar", stacked=True, ax=axes[label-1],
                                color=clust_color)
        



def visGrpedContinuous(df_labels: DataFrame, col_prim_grp: str, col_sec_grp: str,
                    col_outcome: str, ylab: str = ""):
    df_grped = DataFrame(df_labels.groupby([col_prim_grp, col_sec_grp])[col_outcome].mean())
    df_grped = df_grped.reset_index()
    clust_labels = df_grped[col_prim_grp].unique()
    
    colors_norm = plt.colormaps['gist_rainbow'](np.linspace(0.08, 0.89, len(clust_labels)))

    df_grped2 = DataFrame(df_labels.groupby([col_prim_grp, col_sec_grp])[col_outcome].mean())
    df_grped3 = DataFrame(df_labels.groupby([col_prim_grp, col_sec_grp])[col_outcome].std())
    df_grped3.columns = ["Standard deviation"]
    df_report = pd.concat([df_grped2, df_grped3], axis=1)
    
    print(df_report)
    DF_CONT.append(df_report)


    fig, axes = plt.subplots(ncols=5)
    fig.set_size_inches(15, 7)
    if ylab:
        axes[0].set_ylabel(ylab) # Set y-label for first plot
    else:
        axes[0].set_ylabel(col_outcome) # If not ylab, use series name
        

    for ind, label in enumerate(clust_labels):
        df_clust = df_grped[df_grped[col_prim_grp] == label]
        df_clust = df_clust[[col_sec_grp, col_outcome]] # Filter out only second grouping and outcome
        df_clust = df_clust.set_index([col_sec_grp]) # Set index to second grouping for labels to work

        df_clust.plot(kind="bar", ylim=(0, max(df_grped[col_outcome])), ax=axes[label-1], color=colors_norm[ind])



# Stats test
def compareDichot(df_labels: DataFrame, col_groups: str, col_cat: str):

    df_grped = DataFrame(df_labels.groupby([col_groups])[col_cat].value_counts())
    df_grped.columns = ["Count"]
    df_grped = df_grped.reset_index()
    df_grped

    comparisons: list[list] = []
    for a, b in combinations(df_grped[col_groups].unique(), 2):
        counts_a = []
        counts_b = []
        for cat in df_grped[col_cat].unique(): # Iterate through individual categories to ensure proper pairing and fill in missing values
            results_a = df_grped.loc[(df_grped[col_groups] == a) & (df_grped[col_cat] == cat)]["Count"]
            results_b = df_grped.loc[(df_grped[col_groups] == b) & (df_grped[col_cat] == cat)]["Count"]
            if results_a.values.size > 0:
                val_a = results_a.values
            else:
                val_a = 0 # Append 0 if there was no entries 
                
            if results_b.values.size > 0:
                val_b = results_b.values
            else:
                val_b = 0
        
            if val_a != 0 and val_b != 0: # Only append if neither values are 0, otherwise will end up with expected value of 0 which throws error 
                counts_a.append(val_a)
                counts_b.append(val_b)
        
        data = [counts_a, counts_b]
        chi2, pval, dof, expected = stats.chi2_contingency(data)
        comparisons.append([(a, b), pval])

    pval_raw = [comp[1] for comp in comparisons]
    multitest_results = multipletests(pval_raw, method="h")
    pval_corr = multitest_results[1]
    for i, comp in enumerate(comparisons):
        comp.append(pval_corr[i]) # Modify original list by adding to items

    print("Comparison, Raw p-value, Corrected p-value")
    print(*comparisons, sep="\n") # Print each element on separate line

def compareContinuous(df_labels: DataFrame, col_groups: str, col_cont: str):
    comp_results = MultiComparison(df_labels[col_cont], df_labels[col_groups])
    # tbl, a1, a2 = comp_results.allpairtest(stats.ttest_ind, method="h")
    tbl, a1, a2 = comp_results.allpairtest(stats.ranksums, method="h")
    # tbl, a1, a2 = comp_results.allpairtest(stats.wilcoxon, method="h")
    print(tbl)

def compareGrpedDichotomous(df_labels: DataFrame, col_groups: str, col_strata: str,
                    col_outcome: str):
    
    df_grped2 = DataFrame(df_labels.groupby([col_groups, col_strata])[col_outcome].value_counts())
    df_grped2.columns = ["Count"]
    df_grped2 = df_grped2.reset_index()
    df_grped2

    cat_ref = df_grped2[col_outcome].unique()[0]

    true_dict = dict()
    obs_dict = dict()

    for strata in df_grped2[col_strata].unique():
        ntrue = np.array([])
        nobs = np.array([])
        
        for i in df_grped2[col_groups].unique():
            df_clust = df_grped2[(df_grped2[col_groups] == i) & (df_grped2[col_strata] == strata)]
            df_clust_true = df_clust[df_clust[col_outcome] == cat_ref]
            
            trues = df_clust_true["Count"].sum()
            totals = df_clust["Count"].sum()

            ntrue = np.append(ntrue, trues)
            nobs = np.append(nobs, totals)
            
        true_dict[strata] = ntrue
        obs_dict[strata] = nobs

    for strata in df_grped2[col_strata].unique():
        array_trues = true_dict[strata]
        array_totals = obs_dict[strata]
        comp_results = proportions_chisquare_allpairs(array_trues, array_totals, "h") # Holm correction
        k = comp_results.n_levels
        print(strata)
        print(F"Correction method: {comp_results.multitest_method}")
        print("Raw pvals")
        pvals_raw = np.zeros((k, k))
        pvals_raw[lzip(*comp_results.all_pairs)] = comp_results.pvals_raw # Prints in matrix rather than list, easier to interpret
        print(DataFrame(pvals_raw))
        print("Corrected pvals")
        pvals_corrected = np.zeros((k, k))
        pvals_corrected[lzip(*comp_results.all_pairs)] = comp_results.pval_corrected() # Prints in matrix rather than list, easier to interpret
        print(DataFrame(pvals_corrected))
        

def compareGrpedContinuous(df_labels: DataFrame, col_groups: str, col_strata: str,
                    col_outcome: str):

    for strata in df_labels[col_strata].unique():
        if isinstance(strata, str): # Screen out NaN values 
            
            print(strata, type(strata))

            df_clust = df_labels[df_labels[col_strata] == strata]
            df_clust = df_clust.dropna(subset=[col_outcome, col_groups])
            
            # print(df_clust[measure])
            # print(df_clust[fst])
                    
            comp_results = MultiComparison(df_clust[col_outcome], df_clust[col_groups])
            # tbl, a1, a2 = comp_results.allpairtest(stats.ttest_ind, method="h")
            tbl, a1, a2 = comp_results.allpairtest(stats.ranksums, method="h")
            # tbl, a1, a2 = comp_results.allpairtest(stats.wilcoxon, method="h")
            
            print(tbl)
            
#%% Survival visualization
visGrpedDichotomous(df_labels, "Endotype", COL_GCS_CAT, COL_SURV)
visGrpedDichotomous(df_labels, "Endotype", COL_GCS_CAT2, COL_SURV)
visGrpedDichotomous(df_labels, "Endotype", COL_AGE_CAT, COL_SURV)
visGrpedDichotomous(df_labels, "Endotype", "GENDER", COL_SURV)

#%% Surg visualization
visGrpedDichotomous(df_labels, "Endotype", COL_GCS_CAT, COL_NSX_ANY)
visGrpedDichotomous(df_labels, "Endotype", COL_GCS_CAT2, COL_NSX_ANY)
visGrpedDichotomous(df_labels, "Endotype", COL_AGE_CAT, COL_NSX_ANY)
visGrpedDichotomous(df_labels, "Endotype", "GENDER", COL_NSX_ANY)


#%% LOS visualization
visGrpedContinuous(df_labels, "Endotype", COL_GCS_CAT, COL_LOS)
visGrpedContinuous(df_labels, "Endotype", COL_GCS_CAT2, COL_LOS)
visGrpedContinuous(df_labels, "Endotype", COL_AGE_CAT, COL_LOS)
visGrpedContinuous(df_labels, "Endotype", "GENDER", COL_LOS)


#%% Survival analysis
compareGrpedDichotomous(df_labels, "Endotype", COL_GCS_CAT, COL_SURV)
compareGrpedDichotomous(df_labels, "Endotype", COL_GCS_CAT2, COL_SURV)
compareGrpedDichotomous(df_labels, "Endotype", COL_AGE_CAT, COL_SURV)
compareGrpedDichotomous(df_labels, "Endotype", "GENDER", COL_SURV)

#%% NSX intervention analysis
compareGrpedDichotomous(df_labels, "Endotype", COL_GCS_CAT, COL_NSX_ANY)
compareGrpedDichotomous(df_labels, "Endotype", COL_GCS_CAT2, COL_NSX_ANY)
compareGrpedDichotomous(df_labels, "Endotype", COL_AGE_CAT, COL_NSX_ANY)
compareGrpedDichotomous(df_labels, "Endotype", "GENDER", COL_NSX_ANY)

#%% LOS analysis
compareGrpedContinuous(df_labels, "Endotype", COL_GCS_CAT, COL_LOS)
compareGrpedContinuous(df_labels, "Endotype", COL_GCS_CAT2, COL_LOS)
compareGrpedContinuous(df_labels, "Endotype", COL_AGE_CAT, COL_LOS)
compareGrpedContinuous(df_labels, "Endotype", "GENDER", COL_LOS)

#%% Age analysis with clusters
df3 = pd.read_excel(F"data/tbi2_admit_icd_age_elix_annotated.xlsx")
num_clusts = len(df3["Endotype"].value_counts())
df3.groupby(["Endotype"])[COL_AGE].plot(kind="kde", xticks=list(range(100)[::5]), xlim=(0, 100),legend=True) # Preview

#%%
from scipy.stats import shapiro # Normality test
df_labels.groupby(["Endotype", COL_GCS_CAT])["Length of stay (days)"].apply(shapiro)

######### Relevant analyses for paper 
#%% Demographic segmentation (dichot)
visStackedProp(df_labels, "Endotype", COL_GCS_CAT)
visStackedProp(df_labels, "Endotype", COL_AGE_CAT)
visStackedProp(df_labels, "Endotype", "GENDER")
#%% Demographic segmentation (continuous)
visContinuous(df_labels, "Endotype", COL_AGE)
visContinuous(df_labels, "Endotype", COL_COMORBS)

#%% Outcomes 
visStackedProp(df_labels, "Endotype", COL_SURV)
visStackedProp(df_labels, "Endotype", COL_NSX_ANY)
visContinuous(df_labels, "Endotype", COL_LOS)

#%% Segmentation stats
compareDichot(df_labels, "Endotype", COL_AGE_CAT)
compareDichot(df_labels, "Endotype", COL_GCS_CAT)
compareDichot(df_labels, "Endotype", "GENDER")

#%% Survival
visGrpedDichotomous(df_labels, "Endotype", COL_GCS_CAT, COL_SURV)
visGrpedDichotomous(df_labels, "Endotype", COL_AGE_CAT, COL_SURV)

compareGrpedDichotomous(df_labels, "Endotype", COL_GCS_CAT, COL_SURV)
compareGrpedDichotomous(df_labels, "Endotype", COL_AGE_CAT, COL_SURV)

#%% Surg
visGrpedDichotomous(df_labels, "Endotype", COL_GCS_CAT, COL_NSX_ANY)
visGrpedDichotomous(df_labels, "Endotype", COL_AGE_CAT, COL_NSX_ANY)

compareGrpedDichotomous(df_labels, "Endotype", COL_GCS_CAT, COL_NSX_ANY)
compareGrpedDichotomous(df_labels, "Endotype", COL_AGE_CAT, COL_NSX_ANY)

#%% LOS
visGrpedContinuous(df_labels, "Endotype", COL_GCS_CAT, COL_LOS)
visGrpedContinuous(df_labels, "Endotype", COL_AGE_CAT, COL_LOS)

compareGrpedContinuous(df_labels, "Endotype", COL_GCS_CAT, COL_LOS)
compareGrpedContinuous(df_labels, "Endotype", COL_AGE_CAT, COL_LOS)


#%% Differences in age groups between endotypes 
visGrpedContinuous(df_labels, "Endotype", COL_AGE_CAT, COL_AGE)
compareGrpedContinuous(df_labels, "Endotype", COL_AGE_CAT, COL_AGE)

#%% Stratification analysis 
visGrpedContinuous(df_labels, "Endotype", COL_AGE_CAT, COL_GCS)
visGrpedDichotomous(df_labels, "Endotype", COL_AGE_CAT, COL_GCS_CAT)
compareGrpedContinuous(df_labels, "Endotype", COL_AGE_CAT, COL_GCS)

visGrpedContinuous(df_labels, "Endotype", COL_GCS_CAT, COL_AGE)
visGrpedDichotomous(df_labels, "Endotype", COL_GCS_CAT, COL_AGE_CAT)
compareGrpedContinuous(df_labels, "Endotype", COL_GCS_CAT, COL_AGE)

#%% Other analyses 
visGrpedContinuous(df_labels, "Endotype", COL_AGE_CAT, COL_AGE)
visGrpedContinuous(df_labels, "Endotype", COL_GCS_CAT, COL_GCS)
visGrpedContinuous(df_labels, "Endotype", COL_GCS_CAT, COL_COMORBS)


visGrpedDichotomous(df_labels, "Endotype", COL_GCS_CAT, "GENDER")
visGrpedDichotomous(df_labels, "Endotype", COL_AGE_CAT, "GENDER")


#%% Young only
df_custom3 = df_labels[~((df_labels["Endotype"] == 2) & (df_labels[COL_AGE_CAT] != AGE_LABELS[0]))]
df_custom3 = df_custom3[~((df_custom3["Endotype"] == 4) & (df_custom3[COL_AGE_CAT] != AGE_LABELS[0]))]
df_custom3 = df_custom3[~((df_custom3["Endotype"] == 5) & (df_custom3[COL_AGE_CAT] != AGE_LABELS[0]))]

# df_custom3 = df_labels[df_labels[COL_AGE_CAT] == AGE_LABELS[1]]

visGrpedDichotomous(df_custom3, "Endotype", COL_GCS_CAT, COL_AGE_CAT)

visGrpedContinuous(df_custom3, "Endotype", COL_AGE_CAT, COL_GCS)
compareGrpedContinuous(df_custom3, "Endotype", COL_AGE_CAT, COL_GCS)
visGrpedContinuous(df_custom3, "Endotype", COL_GCS_CAT, COL_AGE)
compareGrpedContinuous(df_custom3, "Endotype", COL_GCS_CAT, COL_AGE)
#%%
visGrpedDichotomous(df_custom3, "Endotype", COL_GCS_CAT, COL_SURV)
compareGrpedDichotomous(df_custom3, "Endotype", COL_GCS_CAT, COL_SURV)
visGrpedDichotomous(df_custom3, "Endotype", COL_GCS_CAT, COL_NSX_ANY)
compareGrpedDichotomous(df_custom3, "Endotype", COL_GCS_CAT, COL_NSX_ANY)
visGrpedContinuous(df_custom3, "Endotype", COL_GCS_CAT, COL_LOS)
compareGrpedContinuous(df_custom3, "Endotype", COL_GCS_CAT, COL_LOS)
#%% Middle-aged only
df_custom1 = df_labels[~((df_labels["Endotype"] == 2) & (df_labels[COL_AGE_CAT] != AGE_LABELS[1]))]
df_custom1 = df_custom1[~((df_custom1["Endotype"] == 4) & (df_custom1[COL_AGE_CAT] != AGE_LABELS[1]))]
df_custom1 = df_custom1[~((df_custom1["Endotype"] == 5) & (df_custom1[COL_AGE_CAT] != AGE_LABELS[1]))]

# df_custom1 = df_labels[df_labels[COL_AGE_CAT] == AGE_LABELS[1]]

visGrpedDichotomous(df_custom1, "Endotype", COL_GCS_CAT, COL_AGE_CAT)

visGrpedContinuous(df_custom1, "Endotype", COL_AGE_CAT, COL_GCS)
compareGrpedContinuous(df_custom1, "Endotype", COL_AGE_CAT, COL_GCS)
visGrpedContinuous(df_custom1, "Endotype", COL_GCS_CAT, COL_AGE)
compareGrpedContinuous(df_custom1, "Endotype", COL_GCS_CAT, COL_AGE)
#%%
visGrpedDichotomous(df_custom1, "Endotype", COL_GCS_CAT, COL_SURV)
compareGrpedDichotomous(df_custom1, "Endotype", COL_GCS_CAT, COL_SURV)
visGrpedDichotomous(df_custom1, "Endotype", COL_GCS_CAT, COL_NSX_ANY)
compareGrpedDichotomous(df_custom1, "Endotype", COL_GCS_CAT, COL_NSX_ANY)
visGrpedContinuous(df_custom1, "Endotype", COL_GCS_CAT, COL_LOS)
compareGrpedContinuous(df_custom1, "Endotype", COL_GCS_CAT, COL_LOS)
#%% Old Aged only
df_custom2 = df_labels[~((df_labels["Endotype"] == 2) & (df_labels[COL_AGE_CAT] != AGE_LABELS[2]))]
df_custom2 = df_custom2[~((df_custom2["Endotype"] == 4) & (df_custom2[COL_AGE_CAT] != AGE_LABELS[2]))]
df_custom2 = df_custom2[~((df_custom2["Endotype"] == 5) & (df_custom2[COL_AGE_CAT] != AGE_LABELS[2]))]

# df_custom2 = df_labels[df_labels[COL_AGE_CAT] == AGE_LABELS[2]]

visGrpedDichotomous(df_custom2, "Endotype", COL_GCS_CAT, COL_AGE_CAT)

visGrpedContinuous(df_custom2, "Endotype", COL_AGE_CAT, COL_GCS)
compareGrpedContinuous(df_custom2, "Endotype", COL_AGE_CAT, COL_GCS)
visGrpedContinuous(df_custom2, "Endotype", COL_GCS_CAT, COL_AGE)
compareGrpedContinuous(df_custom2, "Endotype", COL_GCS_CAT, COL_AGE)

#%%
visGrpedDichotomous(df_custom2, "Endotype", COL_GCS_CAT, COL_SURV)
compareGrpedDichotomous(df_custom2, "Endotype", COL_GCS_CAT, COL_SURV)
visGrpedDichotomous(df_custom2, "Endotype", COL_GCS_CAT, COL_NSX_ANY)
compareGrpedDichotomous(df_custom2, "Endotype", COL_GCS_CAT, COL_NSX_ANY)
visGrpedContinuous(df_custom2, "Endotype", COL_GCS_CAT, COL_LOS)
compareGrpedContinuous(df_custom2, "Endotype", COL_GCS_CAT, COL_LOS)

# %%
visStackedProp(df_labels, "Endotype", COL_NSX_ANY)
visContinuous(df_labels, "Endotype", COL_LOS)

# %%
compareDichot(df_labels, "Endotype", COL_SURV)
compareDichot(df_labels, "Endotype", COL_NSX_ANY)
compareContinuous(df_labels, "Endotype", COL_LOS)
#%%
visGrpedContinuous(df_labels, "Endotype", COL_AGE_CAT, COL_GCS)
visGrpedContinuous(df_labels, "Endotype", COL_GCS_CAT, COL_AGE)

#%%
df_final = pd.concat(DF_CONT)
df_final.to_csv("Table outputs.csv")