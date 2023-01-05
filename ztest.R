# Multitest development 
path_df_annotated <- "data/tbi2_admit_icd_dates_nsx_gcs_elix_annotated_v4_BL.xlsx"

df_orig <- read_excel(path_df_annotated)
df_orig$`Survival to discharge` <- with(df_orig, ifelse(is.na(DOD), "Alive", "Expired"))
df <- df_orig %>% dplyr::select("Survival to discharge", "Endotype")
cross_tab <- table(df$Endotype, df$`Survival to discharge`)
pairwise.prop.test(cross_tab, p.adjust.method="holm")
# pairwise.prop.test(cross_tab, p.adjust.method="bonferroni") # More conservative method




df_counts <- as.data.frame.matrix(t(cross_tab)) # Coerce into same shape (can't use data.frame())
df_percent <- apply(df_counts, 2, function(x){x*100/sum(x,na.rm=T)})
barplot(df_percent)

df_counts_series <- data.frame(cross_tab)
colnames(df_counts_series) <- c("Endotype", "Survival", "Count")
ggplot(df_counts_series, aes(fill=Survival, y=Count, x=Endotype))


## 
df_orig <- read_excel("data/tbi2_admit_icd_dates_nsx_gcs_elix.xlsx")
df_orig$`Survival to discharge` <- with(df_orig, ifelse(is.na(DOD), TRUE, FALSE))
df <- df_orig %>% dplyr::select("Survival to discharge", "Age (years)", "No. Comorbs")
lg_model <- lm(`Survival to discharge` ~ `Age (years)` * `No. Comorbs`, data=df) # "*" to indicate interaction term
print(summary(lg_model))
lg_model <- lm(`Survival to discharge` ~ `Age (years)` + `No. Comorbs`, data=df) # "*" to indicate interaction term
print(summary(lg_model))
lg_model <- lm(`Survival to discharge` ~  `No. Comorbs`, data=df) # "*" to indicate interaction term
print(summary(lg_model))
lg_model <- lm(`No. Comorbs` ~ `Age (years)`, data=df) # "*" to indicate interaction term
print(summary(lg_model))
