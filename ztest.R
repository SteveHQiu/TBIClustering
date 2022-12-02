path_df_annotated <- "data/tbi2_admit_icd_age_elix_annotated_v1.xlsx"

df_orig <- read_excel(path_df_annotated)
df_orig$survival <- with(df_orig, ifelse(is.na(DOD), "Alive", "Expired"))
df <- df_orig %>% dplyr::select("survival", "df_annot")
cross_tab <- table(df$df_annot, df$survival)
pairwise.prop.test(cross_tab, p.adjust.method="holm")
# pairwise.prop.test(cross_tab, p.adjust.method="bonferroni") # More conservative method



df_counts <- as.data.frame.matrix(t(cross_tab)) # Coerce into same shape (can't use data.frame())
df_percent <- apply(df_counts, 2, function(x){x*100/sum(x,na.rm=T)})
barplot(df_percent)

df_counts_series <- data.frame(cross_tab)
colnames(df_counts_series) <- c("Endotype", "Survival", "Count")
ggplot(df_counts_series, aes(fill=Survival, y=Count, x=Endotype))



aggregate(df_orig$age, list(df_orig$df_annot), FUN=mean)
