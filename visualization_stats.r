#%% Imports
library(mclust) # Main clustering LPA algorithm 
library(poLCA) # LCA clustering algorithm
library(readxl)
library(writexl)
library(dplyr) # Df tools 
library(ggplot2)
library(patchwork) # For adding plots together
library(viridis) # Colormaps
library(colorspace) # Color manipulation (e.g., desaturation colors)
library(inflection) # For finding knee point
library(comprehenr) # For python comprehensions
library(multcomp)
library(pROC) # AUC calculation 
#%%
#%% Jitter plot

df_plot <- df_orig[df_orig$`Age (years)` >= 10, ]
ggplot(df_plot) +
  aes(
    x = `No. Comorbs`,
    y = `Age (years)`,
    colour = `Survival to discharge`
  ) +
  geom_jitter(size = 1.5) +
  scale_color_manual(
    values = c(Alive = "#1BA204",
               Expired = "#F33B3B")
  ) +
  theme_minimal()
ggsave("figures/survival_jitter.png", dpi=300, width=10, height=5)

#%%
#%% Regression for outcome prediction
col_labels <- c("congestive_heart_failure", "cardiac_arrhythmia", "valvular_disease",
                "pulmonary_circulation_disorder", "peripheral_vascular_disorder",
                "hypertension_uncomplicated", "hypertension_complicated", "paralysis",
                "other_neurological_disorder", "chronic_pulmonary_disease", "diabetes_uncomplicated",
                "diabetes_complicated", "hypothyroidism", "renal_failure", "liver_disease",
                "aids_hiv", "lymphoma", "metastatic_cancer",
                "solid_tumor_wo_metastasis", "rheumatoid_arhritis", "coagulopathy", "obesity",
                "weight_loss", "fluid_and_electrolyte_disorders", "blood_loss_anemia",
                "deficiency_anemia", "alcohol_abuse", "drug_abuse", "psychoses", "depression")
col_symbols <- base::lapply(col_labels, as.name)
DF_PATH <- "data/tbi2_admit_icd_dates_nsx_gcs_elix.xlsx"
df_orig <- read_excel(DF_PATH)
df_annot <- read_excel("data/tbi2_admit_icd_dates_nsx_gcs_elix_annotated_v4.xlsx")



df <- df_orig %>% dplyr::select(!!!col_symbols, `Age (years)`, `First recorded total GCS score`, `GENDER`) # Select only columns containing comorbs
df <- df_orig %>% dplyr::select(!!!col_symbols) # Select only columns containing comorbs
extra_var <- "Endotype"

# Change every "1" guard to "0" for all outcomes except those of interest
if (1) { # For survival
  outcome_raw <- "Survival to discharge"
  outcome <- sprintf("`%s`", outcome_raw) # Need to wrap around literal brackets
  df[[outcome_raw]] <- with(df_orig, ifelse(is.na(DOD), TRUE, FALSE)) # Append survival as column
  col_symbols <- append(col_symbols, as.name(outcome)) # Add new survival variable
  
}
if (0) { # For intervention
  outcome_raw <- "Any neurosurgical intervention"
  outcome <- sprintf("`%s`", outcome_raw) # Need to wrap around literal brackets
  df[[outcome_raw]] <- df_annot[[outcome_raw]]
  col_symbols <- append(col_symbols, as.name(outcome_raw)) # Add new survival variable
  
}
if (0) { # For LOS
  outcome_raw <- "Length of stay (days)"
  outcome <- sprintf("`%s`", outcome_raw) # Need to wrap around literal brackets
  df[[outcome_raw]] <- df_annot[[outcome_raw]]
  col_symbols <- append(col_symbols, as.name(outcome_raw)) # Add new survival variable
}

formula <- as.formula(sprintf("%s ~ .", outcome))
lg_model <- lm(formula, data=df) # lm() rather than glm() since lm() easier to pass on to ANOVA 
print("Regression using only comorbidities")
print(summary(lg_model))
predicted <- predict(lg_model, df, type="response")
auc(df[[outcome_raw]], predicted)

df[[extra_var]] <- as.factor(df_annot[[extra_var]])
col_symbols <- append(col_symbols, as.name(extra_var)) # Add new endotype variable
formula <- as.formula(sprintf("%s ~ .", outcome))
lg_model2 <- lm(formula, data=df)
print("Regression with endotypes")
print(summary(lg_model2))
predicted <- predict(lg_model2, df, type="response")
auc(df[[outcome_raw]], predicted)

formula <- as.formula(sprintf("%s ~ %s + `Age (years)`+ `First recorded total GCS score`+ `GENDER`", outcome, extra_var))
formula <- as.formula(sprintf("%s ~ %s", outcome, extra_var))
lg_model3 <- lm(formula, data=df) # Use only endotype
print("Regression with endotypes")
print(summary(lg_model3))
predicted <- predict(lg_model3, df, type="response")
auc(df[[outcome_raw]], predicted)

print("Significance of adding endotypes")
print(anova(lg_model, lg_model2))
print("Significance of between raw data and endotypes only")
print(anova(lg_model, lg_model3))

#%%
#%% Simple regressions and ANCOVA 
df_annot <- read_excel("data/tbi2_admit_icd_dates_nsx_gcs_elix_annotated_v4.xlsx")
df_annot[["Survival to discharge"]] <- with(df_annot, ifelse(is.na(DOD), TRUE, FALSE))
df_annot[["Endotype"]] <- as.factor(df_annot[["Endotype"]])

for (i in 1:5) { # Go through every endotype as reference
  df_annot[["Endotype"]] <- relevel(df_annot[["Endotype"]], ref=i) # Try every endotype as reference
  formula <- `Survival to discharge` ~ `Endotype` + `Age (years)` + `First recorded total GCS score` + `GENDER`
  surv_model <- glm(formula, data=df_annot)
  print(summary(surv_model))

  formula <- `Any neurosurgical intervention` ~ `Endotype` + `Age (years)` + `First recorded total GCS score` + `GENDER`
  nsx_model <- glm(formula, data=df_annot)
  print(summary(nsx_model))

  library(multcomp)
  formula <- `Length of stay (days)` ~ `Endotype` + `Age (years)` + `First recorded total GCS score` + `GENDER`
  los_model <- aov(formula, data=df_annot)
  post_hocs <- glht(los_model, linfct = mcp(`Endotype` = "Tukey"))
  print(summary(los_model))
  print(summary(post_hocs))

}


#%%
#%% Multitest for stratified survival analysis
path_df_annotated <- "data/tbi2_admit_icd_dates_nsx_gcs_elix_annotated_v4_BL.xlsx"

df_orig <- read_excel(path_df_annotated)
df_orig$`Survival to discharge` <- with(df_orig, ifelse(is.na(DOD), "Alive", "Expired"))
df <- df_orig %>% dplyr::select("Survival to discharge", "Endotype")
cross_tab <- table(df$Endotype, df$`Survival to discharge`)
pairwise.prop.test(cross_tab, p.adjust.method="holm")
#%%



