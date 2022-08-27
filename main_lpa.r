#%% Installation
install.packages("tidyLPA")
install.packages("dplyr")

#%%
#%% Imports
library(tidyLPA)
library(dplyr)
library(readxl) # To read excel files
library(mclust)

#%%
#%% Execution
data <- read_excel("data/merged_elix_formatted_betterlabels.xlsx")
data <- read_excel("data/merged_output_elix.xlsx")
data <- read_excel("data/merged_elix_formatted.xlsx")
data <- read_excel("data/elix_truncated.xlsx")



# %>% operator comes from dplyr, works like a pipe operator in that the results of previous line is input as arg of next function
data_fmt <- data %>%
    select(AGE, congestive_heart_failure, cardiac_arrhythmia, valvular_disease,
               pulmonary_circulation_disorder, peripheral_vascular_disorder,
               hypertension_uncomplicated, hypertension_complicated, paralysis,
               other_neurological_disorder, chronic_pulmonary_disease, 
               diabetes_uncomplicated, diabetes_complicated, hypothyroidism,
               renal_failure, liver_disease, peptic_ulcer_disease_excluding_bleeding, 
               aids_hiv, lymphoma, metastatic_cancer, solid_tumor_wo_metastasis, 
               rheumatoid_arhritis, coagulopathy, obesity, weight_loss, 
               fluid_and_electrolyte_disorders, blood_loss_anemia, deficiency_anemia,
               alcohol_abuse, drug_abuse, psychoses, depression) %>% # Select cols that you need
    single_imputation() %>%



#%%
#%% Show entries with missing data
new_DF <- data[rowSums(is.na(data)) > 0,]

#%%
#%% Test
df1 <- data %>%
  select(AGE, congestive_heart_failure, cardiac_arrhythmia, valvular_disease,
         pulmonary_circulation_disorder, peripheral_vascular_disorder,
         hypertension_uncomplicated, hypertension_complicated, paralysis,
         other_neurological_disorder, chronic_pulmonary_disease, 
         diabetes_uncomplicated, diabetes_complicated, hypothyroidism,
         renal_failure, liver_disease, peptic_ulcer_disease_excluding_bleeding, 
         aids_hiv, lymphoma, metastatic_cancer, solid_tumor_wo_metastasis, 
         rheumatoid_arhritis, coagulopathy, obesity, weight_loss, 
         fluid_and_electrolyte_disorders, blood_loss_anemia, deficiency_anemia,
         alcohol_abuse, drug_abuse, psychoses, depression)
model = Mclust(df1, G = 5:15) # Default is G = 1:9, need to set this manually
