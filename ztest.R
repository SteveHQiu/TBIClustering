library(poLCA)
library(dplyr) # Df tools 
library(readxl) # To read excel files
library(writexl) 
library(comprehenr) # For python comprehensions
library(ggplot2)
library(patchwork) # For adding plots together

#%% LPA for LCA results 

df_orig <- read_excel("data/tbi_admit_icd_v2_age_elix.xlsx")
df <- df_orig %>% dplyr::select(congestive_heart_failure, cardiac_arrhythmia, valvular_disease,
                         pulmonary_circulation_disorder, peripheral_vascular_disorder,
                         hypertension_uncomplicated, hypertension_complicated, paralysis,
                         other_neurological_disorder, chronic_pulmonary_disease, 
                         diabetes_uncomplicated, diabetes_complicated, hypothyroidism,
                         renal_failure, liver_disease, peptic_ulcer_disease_excluding_bleeding, 
                         aids_hiv, lymphoma, metastatic_cancer, solid_tumor_wo_metastasis, 
                         rheumatoid_arhritis, coagulopathy, obesity, weight_loss, 
                         fluid_and_electrolyte_disorders, blood_loss_anemia, deficiency_anemia,
                         alcohol_abuse, drug_abuse, psychoses, depression
)
df <- df * 1 # Turns TRUE/FALSE into 1/0
# Manifest values needs to be an integer starting from 1 and not 0 https://stackoverflow.com/questions/52008147/polca-alert-values-that-are-not-positive-integers
df <- df + 1 # Adds 1 to every value

# Define LCA model
model <- cbind(congestive_heart_failure, cardiac_arrhythmia, valvular_disease,
               pulmonary_circulation_disorder, peripheral_vascular_disorder,
               hypertension_uncomplicated, hypertension_complicated, paralysis,
               other_neurological_disorder, chronic_pulmonary_disease, 
               diabetes_uncomplicated, diabetes_complicated, hypothyroidism,
               renal_failure, liver_disease, peptic_ulcer_disease_excluding_bleeding, 
               aids_hiv, lymphoma, metastatic_cancer, solid_tumor_wo_metastasis, 
               rheumatoid_arhritis, coagulopathy, obesity, weight_loss, 
               fluid_and_electrolyte_disorders, blood_loss_anemia, deficiency_anemia,
               alcohol_abuse, drug_abuse, psychoses, depression
) ~ 1


n_clust <- 7
bics <- c()
aics <- c()
clusts <- c()

cluster <- poLCA(model, data = df, nclass = n_clust, na.rm = TRUE, maxiter=10000)
df_results <- data.frame(cluster$probs) # Outputs probability of each input group for all clusters (1=False, 2=True)
df_means <- df_results[,seq(2, ncol(df_results), 2)] # Get every second column starting at col 2
# df_means <- df_means * 100 # Scale by 100
bics <- c(bics, cluster$bic)
aics <- c(aics, cluster$aic)
clusts <- c(clusts, cluster)


for (i in 1:29) {
  cluster <- poLCA(model, data = df, nclass = n_clust, na.rm = TRUE, maxiter=10000)
  df_results <- data.frame(cluster$probs) # Outputs probability of each input group for all clusters (1=False, 2=True)
  df_means_t <- df_results[,seq(2, ncol(df_results), 2)] # Get every second column starting at col 2
  # df_means_t <- df_means_t * 100 # Scale by 100
  df_means <- rbind(df_means, df_means_t)
  bics <- c(bics, cluster$bic)
  aics <- c(aics, cluster$aic)
  clusts <- c(clusts, cluster)
}

new_bics <- sort(bics, decreasing=TRUE)
ind <- match(new_bics[1], bics)
clusts[ind]




#%%
#%% Process survival/death
df <- read_excel("data/tbi_admit_icd_age_elix.xlsx")
df$survival <- with(df, ifelse(is.na(DOD), TRUE, FALSE))
write_xlsx(df, "data/tbi_admit_icd_age_elix_surv.xlsx")

#%%
#%% Regression
df <- read_excel("data/tbi_admit_icd_age_elix_surv.xlsx")
df <- df %>% select(congestive_heart_failure, cardiac_arrhythmia, valvular_disease,
                    pulmonary_circulation_disorder, peripheral_vascular_disorder,
                    hypertension_uncomplicated, hypertension_complicated, paralysis,
                    other_neurological_disorder, chronic_pulmonary_disease, 
                    diabetes_uncomplicated, diabetes_complicated, hypothyroidism,
                    renal_failure, liver_disease, peptic_ulcer_disease_excluding_bleeding, 
                    aids_hiv, lymphoma, metastatic_cancer, solid_tumor_wo_metastasis, 
                    rheumatoid_arhritis, coagulopathy, obesity, weight_loss, 
                    fluid_and_electrolyte_disorders, blood_loss_anemia, deficiency_anemia,
                    alcohol_abuse, drug_abuse, psychoses, depression,
                    survival)
lg_model <- glm(survival ~ ., data=df, family="binomial")
summary(lg_model)
# Calculate R2 https://youtu.be/C4N3_XJJ-jU?t=865
ll_null <- lg_model$null.deviance/-2
ll_proposed <- lg_model$deviance/-2
r2 <- (ll_null - ll_proposed)/ll_null
r2
p_val <- 1 - pchisq(2*(ll_proposed - ll_null), df=(length(lg_model$coefficients)-1))
p_val

#%%
#%% MIMIC III characterization 

df1 <- read_excel("data/tbi_admit_icd_popinfo.xlsx")
df2 <- read_excel("data/tbi_admit_icd_comorbdistr.xlsx")
df <- read_excel("data/tbi_admit_icd_age_elix.xlsx")


ggplot(df) +
 aes(x = age, fill = GENDER) +
 geom_histogram(bins = 40L) +
 scale_fill_hue(direction = 1) +
 labs(x = "Age", y = "Number of patients") +
 theme_minimal() + 
scale_x_continuous(breaks = round(seq(min(df$age), max(df$age), by = 10),1))


#%%
