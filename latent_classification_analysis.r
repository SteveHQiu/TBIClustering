# Import from LCA_R.ipynb
# Imports
getwd()
library(poLCA)
library(readxl)
library(writexl)

# Import data
df <- read_excel("tbi_admission_comorbidities_age_elix_formatted.xlsx")
df <- read_excel("elix_formatted.xlsx")
# Manifest values needs to be an integer starting from 1 and not 0 https://stackoverflow.com/questions/52008147/polca-alert-values-that-are-not-positive-integers
df <- df+1 # Adds 1 to every value

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
cluster <- poLCA(model, data = df, nclass = 6, graphs = TRUE, na.rm = TRUE)

# Export data if needed
write_xlsx(df, "exported_output.xlsx")
