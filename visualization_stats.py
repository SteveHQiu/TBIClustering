#%%
import os, json
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

#%%
df_labels = pd.read_excel("data/tbi2_admit_icd_dates_nsx_gcs_elix_annotated_v4.xlsx")
df_labels[COL_LOS] = df_labels[COL_LOS].mask(df_labels[COL_LOS].sub(df_labels[COL_LOS].mean()).div(df_labels[COL_LOS].std()).abs().gt(3))
# Remove outliers from LOS, 95 values removed at SD of 2, 33 at 3
#%%
def visStackedProp(df_labels: DataFrame, primary_ind: str, secondary_ind: str):
    df_grped = DataFrame(df_labels.groupby([primary_ind])[secondary_ind].value_counts(normalize=True))
    df_grped.columns = ["Proportion"]
    df_grped = df_grped.reset_index()
    df_grped.columns = [primary_ind, secondary_ind, "Proportion"]
    df_grped = df_grped.set_index([primary_ind, secondary_ind]).Proportion

    ax = df_grped.unstack().plot(kind="bar", stacked=True)
    ax.set_ylabel("Proportion")

visStackedProp(df_labels, "Endotype", COL_GCS_CAT)
visStackedProp(df_labels, "Endotype", COL_GCS_CAT2)
visStackedProp(df_labels, "Endotype", COL_AGE_CAT)

#%%
def _desatColors(colors, percent):
    # Colors should be in np format of RGBA with range of 0-1
    vector_diff = 1 - colors # Subtract color from pure white
    return colors + vector_diff * percent # Add difference vector scaled by percent
    

def visGrpedDichotomous(df_labels: DataFrame, col_prim_grp: str, col_sec_grp: str,
                    col_outcome: str, ylab: str = "Proportion"):
    df_grped = DataFrame(df_labels.groupby([col_prim_grp, col_sec_grp])[col_outcome].value_counts(normalize=True))
    df_grped.columns = ["Proportion"]
    df_grped = df_grped.reset_index()
    clust_labels = df_grped[col_prim_grp].unique()
    
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

#%% Survival
visGrpedDichotomous(df_labels, "Endotype", COL_GCS_CAT, COL_SURV)
visGrpedDichotomous(df_labels, "Endotype", COL_GCS_CAT2, COL_SURV)
visGrpedDichotomous(df_labels, "Endotype", COL_AGE_CAT, COL_SURV)
visGrpedDichotomous(df_labels, "Endotype", "GENDER", COL_SURV)

#%% Surg
visGrpedDichotomous(df_labels, "Endotype", COL_GCS_CAT, COL_NSX_ANY)
visGrpedDichotomous(df_labels, "Endotype", COL_GCS_CAT2, COL_NSX_ANY)
visGrpedDichotomous(df_labels, "Endotype", COL_AGE_CAT, COL_NSX_ANY)
visGrpedDichotomous(df_labels, "Endotype", "GENDER", COL_NSX_ANY)


#%% LOS
visGrpedContinuous(df_labels, "Endotype", COL_GCS_CAT, COL_LOS)
visGrpedContinuous(df_labels, "Endotype", COL_GCS_CAT2, COL_LOS)
visGrpedContinuous(df_labels, "Endotype", COL_AGE_CAT, COL_LOS)
visGrpedContinuous(df_labels, "Endotype", "GENDER", COL_LOS)

#%% Stats test

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
#%% Survival
compareGrpedDichotomous(df_labels, "Endotype", COL_GCS_CAT, COL_SURV)
compareGrpedDichotomous(df_labels, "Endotype", COL_GCS_CAT2, COL_SURV)
compareGrpedDichotomous(df_labels, "Endotype", COL_AGE_CAT, COL_SURV)
compareGrpedDichotomous(df_labels, "Endotype", "GENDER", COL_SURV)

#%% NSX intervention 
compareGrpedDichotomous(df_labels, "Endotype", COL_GCS_CAT, COL_NSX_ANY)
compareGrpedDichotomous(df_labels, "Endotype", COL_GCS_CAT2, COL_NSX_ANY)
compareGrpedDichotomous(df_labels, "Endotype", COL_AGE_CAT, COL_NSX_ANY)
compareGrpedDichotomous(df_labels, "Endotype", "GENDER", COL_NSX_ANY)

#%% LOS
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