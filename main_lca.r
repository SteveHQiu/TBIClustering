# Installation 

# Imports

library(poLCA)
library(readxl)
library(writexl)
library(dplyr) # Df tools 
library(ggplot2)
library(patchwork) # For adding plots together
library(viridis) # Colormaps
library(colorspace) # Color manipulation (e.g., desaturation colors)
library(inflection) # For finding knee point
library(comprehenr) # For python comprehensions


# Import data
df_orig <- read_excel("data/tbi_admit_icd_age_elix.xlsx")
df <- df_orig %>% select(congestive_heart_failure, cardiac_arrhythmia, valvular_disease,
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
ind <- c(3:15)
bics <- c()
aics <- c()
for (i in ind) {
  cluster <-poLCA(model, data=df, nclass=i, maxiter=10000)
  bics <- c(bics, cluster$bic)
  aics <- c(aics, cluster$aic)
}
plot(cbind(ind, aics), type="l")
plot(cbind(ind, bics), type="l")

a <- aics
c <- to_vec(for(i in seq_along(a)) a[i] - a[i-1])
plot(seq_along(c), c, type = "l")
check_curve(seq_along(c), c)
# Different ways of finding inflection point, some require inputting the convexity (obtained via check_curve)
uik(seq_along(c), c)
d2uik(seq_along(c), c)
ese(seq_along(c), c, 0)
ede(seq_along(c), c, 0)


## Cluster visualization

cluster <- poLCA(model, data = df, nclass = 6, na.rm = TRUE, maxiter=10000)
df_results <- data.frame(cluster$probs) # Outputs probability of each input group for all clusters (1=False, 2=True)
df_annot <- cluster$predclass
df_means <- df_results[,seq(2, ncol(df_results), 2)] # Get every second column starting at col 2
model_means <- as.data.frame(t(df_means))
model_means <- model_means * 100 # Scale everything by 100
colnames(model_means) <- paste("Endotype_", seq(1, ncol(model_means)), sep="")
clust_names <- c(colnames(model_means))
clust_colors <- rainbow(length(colnames(model_means)))
clust_colors_desat <- lighten(clust_colors, 0.75)
clust_seq <- seq_len(length(clust_colors))
# image(clust_seq, 1, as.matrix(clust_seq), col=clust_colors) # Show colors
# image(clust_seq, 1, as.matrix(clust_seq), col=clust_colors_desat) # Show colors

df_means <- data.frame(model_means)
df_means$vars <- row.names(df_means) # Make separate variable of row names to be accessed later
df_means$id <- seq(1, nrow(df_means))
df_means$labels <- c("CHF", "Arrhythmia", "Valvular disease", "Pulmonary circulation disorder",
                     "Peripheral vascular disorder", "Uncomplicated hypertension",
                     "Complicated hypertension", "Paralysis", "Other neurological disorder",
                     "COPD", "Uncomplicated diabetes", "Complicated diabetes", "Hypothyroidism",
                     "Renal failure", "Liver disease", "AID/HIV",
                     "Lymphoma", "Metastatic cancer", "Solid tumor (no metastasis)",
                     "Rheumatoid arthritis", "Coagulopathy", "Obesity", "Weight loss",
                     "Fluid and electrolyte disorders", "Blood loss anemia", "Deficiency anemia",
                     "Alcohol abuse", "Drug abuse", "Psychoses", "Depression") # Age and peptic ulcer removed
num_rows = nrow(df_means)
angles <- 90 - 360 * (df_means$id - 0.5)/num_rows # Need to use sequence attached to df to maintain index, otherwise labels and angles are mismatched
# Subtract 0.5 because the letter must have the angle of the center of the bars. Not extreme right(1) or extreme left (0)
df_means$hjust<-ifelse(angles < -90, 1, 0) # Assign left or right alignment 
df_means$angles<-ifelse(angles < -90, angles+180, angles) # Rectify angles if needed

first_graph_col <- "Endotype_1"
clust_num = match(first_graph_col, clust_names)
plot_color = clust_colors[clust_num]
plot_color_desat = clust_colors_desat[clust_num]
start_graph <- ggplot(df_means, aes_string(x = "id", y = first_graph_col, fill = first_graph_col)) + 
  geom_bar(stat = "identity") + 
  ylim(-100, 150) +
  scale_fill_gradient(low = plot_color_desat, high = plot_color, limits = c(0, 100),
                      name = "Probability (%)") +
  theme_minimal() + # Remove grid and titles
  guides(fill = guide_colorbar(title.position = "top", direction = "vertical")) +
  theme(
    axis.text = element_blank(),
    axis.title = element_blank(),
    legend.position = "right",
    legend.box.just = "center",
    # legend.key.width = unit(2.5, "cm"), # To scale colorbar size
    # legend.title.align = 0.5,
    # legend.title = element_text(angle = 90), # Adjust angle of title
    # legend.direction = "horizontal",
    # panel.grid = element_blank()
  ) +
  coord_polar(start = 0, clip = "off") + # This addition makes graph use polar coordinates, clip = "off" to prevent label clipping
  geom_text(data = df_means, aes(label = labels, hjust = hjust), size = 3, angle = df_means$angles)

df_clust <- data.frame(model_means)
for (clust_name in colnames(df_clust)) {
  if (clust_name != first_graph_col) {
    clust_num = match(clust_name, clust_names)
    plot_color = clust_colors[clust_num]
    plot_color_desat = clust_colors_desat[clust_num]
    rgraph <- ggplot(df_means, aes_string(x = "id", y = clust_name, fill = clust_name)) + # Note that id is a factor. If x is numeric, there is some space between the first bar
      geom_bar(stat = "identity") + 
      ylim(-100, 150) +
      scale_fill_gradient(low = plot_color_desat, high = plot_color, limits = c(0, 100),
                          name = "Probability (%)") +
      theme_minimal() + # Remove grid and titles
      guides(fill = guide_colorbar(title.position = "top", direction = "vertical")) +
      theme(
        axis.text = element_blank(),
        axis.title = element_blank(),
        legend.position = "right",
        legend.box.just = "center",
        # legend.key.width = unit(2.5, "cm"), # To scale colorbar size
        # legend.title.align = 0.5,
        # legend.title = element_text(angle = 90), # Adjust angle of title
        # legend.direction = "horizontal",
        # panel.grid = element_blank()
      ) +
      coord_polar(start = 0, clip = "off") + # This addition makes graph use polar coordinates, clip = "off" to prevent label clipping
      geom_text(data = df_means, aes(label = labels, hjust = hjust), size = 3, angle = df_means$angles)
    start_graph <- start_graph + rgraph
  }
}

final_graph <- start_graph + plot_layout(guides = "auto") # guides = "collect" to collect duplicate legends
ggsave("figures/LCA clusters.png", final_graph, width = 17, height = 17)

#%% Regression

df_orig$survival <- with(df_orig, ifelse(is.na(DOD), TRUE, FALSE))
df_orig$endotype <- as.factor(df_annot) # Make sure 
df_regr <- df_orig %>% select(congestive_heart_failure, cardiac_arrhythmia, valvular_disease,
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
lg_model <- glm(survival ~ ., data=df_regr, family="binomial")
summary(lg_model)
# Calculate R2 https://youtu.be/C4N3_XJJ-jU?t=865
ll_null <- lg_model$null.deviance/-2
ll_proposed <- lg_model$deviance/-2
r2 <- (ll_null - ll_proposed)/ll_null
r2
p_val <- 1 - pchisq(2*(ll_proposed - ll_null), df=(length(lg_model$coefficients)-1))
p_val

df_regr <- df_orig %>% select(congestive_heart_failure, cardiac_arrhythmia, valvular_disease,
                              pulmonary_circulation_disorder, peripheral_vascular_disorder,
                              hypertension_uncomplicated, hypertension_complicated, paralysis,
                              other_neurological_disorder, chronic_pulmonary_disease, 
                              diabetes_uncomplicated, diabetes_complicated, hypothyroidism,
                              renal_failure, liver_disease, peptic_ulcer_disease_excluding_bleeding, 
                              aids_hiv, lymphoma, metastatic_cancer, solid_tumor_wo_metastasis, 
                              rheumatoid_arhritis, coagulopathy, obesity, weight_loss, 
                              fluid_and_electrolyte_disorders, blood_loss_anemia, deficiency_anemia,
                              alcohol_abuse, drug_abuse, psychoses, depression,
                              endotype,
                              survival)
lg_model2 <- glm(survival ~ ., data=df_regr, family="binomial")
summary(lg_model2)
anova(lg_model2, lg_model)
