#%% Installation
install.packages("tidyLPA")

install.packages("tidyverse")
install.packages("dplyr")
install.packages("comprehenr")
install.packages("inflection")
install.packages("patchwork")


#%%
#%% Imports
library(tidyLPA) # Wrapper around mclust for easier usage, but doesn't support mixed categorical and continuous data

library(dplyr) # Df tools 
library(readxl) # To read excel files
library(mclust) # Main clustering algorithm 
library(comprehenr) # For python comprehensions
library(inflection) # For finding knee point
library(ggplot2)
library(patchwork) # For adding plots together


#%%
#%% Load data
data <- read_excel("data/tbi_admit_icd_age_elix.xlsx")

# %>% operator comes from dplyr, works like a pipe operator in that the results of previous line is input as arg of next function
data_fmt <- data %>%
  # Need normalized age for clusters to make sense (otherwise age's distance affects it too much)
  select(age_normalized, congestive_heart_failure, cardiac_arrhythmia, valvular_disease,
               pulmonary_circulation_disorder, peripheral_vascular_disorder,
               hypertension_uncomplicated, hypertension_complicated, paralysis,
               other_neurological_disorder, chronic_pulmonary_disease, 
               diabetes_uncomplicated, diabetes_complicated, hypothyroidism,
               renal_failure, liver_disease, peptic_ulcer_disease_excluding_bleeding, 
               aids_hiv, lymphoma, metastatic_cancer, solid_tumor_wo_metastasis, 
               rheumatoid_arhritis, coagulopathy, obesity, weight_loss, 
               fluid_and_electrolyte_disorders, blood_loss_anemia, deficiency_anemia,
               alcohol_abuse, drug_abuse, psychoses, depression) %>% # Select cols that you need
    single_imputation()
    

#%%
#%% Model generation 

# Models:
# "VII": spherical, unequal volume - Tends not to work well with normalized range and some clusters will only be composed of age characteristics
# "EII": spherical, equal volume - Better with normalized age range

model = Mclust(data_fmt, G = 5:30, modelNames = c("EII")) # Default is G = 1:9, need to set this manually, can also specify certain type of model (E.g., EII, VII)
model = Mclust(data_fmt, G = 5:30, modelNames = c("VII")) # Default is G = 1:9, need to set this manually, can also specify certain type of model (E.g., EII, VII)
model = Mclust(data_fmt, G = 5:30) # General clustering to try every type of model
model = mclustICL(data_fmt, G = 5:30) # Based on ICL rather than BIC
model$BIC
plot(model$BIC)
model$parameters$mean
#%%
#%% Model selection
# Finding inflection point when giving a range of range values
a <- c(model$BIC)
c <- to_vec(for(i in seq_along(a)) a[i] - a[i-1])
plot(seq_along(c), c, type = "l")
check_curve(seq_along(c), c)
# Different ways of finding inflection point, some require inputting the convexity (obtained via check_curve)
uik(seq_along(c), c)
d2uik(seq_along(c), c)
ese(seq_along(c), c, 0)
ede(seq_along(c), c, 0)

# Use smoothing to get knee point 

lo <- loess(c~seq_along(c))
smoothed <- predict(lo)
plot(seq_along(c), c, type = "o", col = "blue", xlab = "No. of components", ylab = "BIC")
lines(seq_along(c), smoothed, type = "o", col = "red")

uik(seq_along(c), smoothed)
d2uik(seq_along(c), smoothed)
ese(seq_along(c), smoothed, 0)
ede(seq_along(c), smoothed, 0)

#%%
#%% Model visualization https://r-graph-gallery.com/295-basic-circular-barplot.html
model1 = Mclust(data_fmt, G = 10, modelNames = c("EII"))
model_means <- model1$parameters$mean
model_means["age",] <- model_means["age",]/100 # Divide by 100 to give age in decimal format to match probabilities
model_means <- model_means * 100 # Scale everything by 100
# model_means <- model_means + 50 # Add 50 to all values as base


df_means <- data.frame(model_means)
df_means$vars <- row.names(df_means) # Make separate variable of row names to be accessed later
df_means$id <- seq(1, nrow(df_means))
df_means$labels <- c("Age", "CHF", "Arrhythmia", "Valvular disease", "Pulmonary circulation disorder",
                     "Peripheral vascular disorder", "Uncomplicated hypertension",
                     "Complicated hypertension", "Paralysis", "Other neurological disorder",
                     "COPD", "Uncomplicated diabetes", "Complicated diabetes", "Hypothyroidism",
                     "Renal failure", "Liver disease", "Peptic ulcer disease", "AID/HIV",
                     "Lymphoma", "Metastatic cancer", "Solid tumor (no metastasis)",
                     "Rheumatoid arthritis", "Coagulopathy", "Obesity", "Weight loss",
                     "Fluid and electrolyte disorders", "Blood loss anemia", "Deficiency anemia",
                     "Alcohol abuse", "Drug abuse", "Psychoses", "Depression")
num_rows = nrow(df_means)
angles <- 90 - 360 * (df_means$id - 0.5)/num_rows # Need to use sequence attached to df to maintain index, otherwise labels and angles are mismatched
# Subtract 0.5 because the letter must have the angle of the center of the bars. Not extreme right(1) or extreme left (0)
df_means$hjust<-ifelse(angles < -90, 1, 0) # Assign left or right alignment 
df_means$angles<-ifelse(angles < -90, angles+180, angles) # Rectify angles if needed

first_graph_col <- "X1"
start_graph <- ggplot(df_means, aes_string(x = "id", y = first_graph_col, fill = first_graph_col)) + 
  geom_bar(stat = "identity") + 
  ylim(-100, 150) +
  scale_fill_gradient(low = "light blue", high = "blue", limits = c(0, 100),
                      name = "Probability (%) or Age (Years)") +
  theme_minimal() + # Remove grid and titles
  guides(fill = guide_colorbar(title.position = "top", direction = "vertical")) +
  theme(
    axis.text = element_blank(),
    axis.title = element_blank(),
    legend.position = "top",
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
for (i in colnames(df_clust)) {
  if (i != first_graph_col) {
    rgraph <- ggplot(df_means, aes_string(x = "id", y = i, fill = i)) + # Note that id is a factor. If x is numeric, there is some space between the first bar
      geom_bar(stat = "identity") + 
      ylim(-100, 150) +
      scale_fill_gradient(low = "light blue", high = "blue", limits = c(0, 100),
                          name = "Probability (%) or Age (Years)") +
      theme_minimal() + # Remove grid and titles
      guides(fill = guide_colorbar(title.position = "top", direction = "vertical")) +
      theme(
        axis.text = element_blank(),
        axis.title = element_blank(),
        legend.position = "top",
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

final_graph <- start_graph + plot_layout(guides = "collect")
ggsave("LPA clusters.png", final_graph, width = 17, height = 17)
#%%
#%% Vertical graph
vgraph <- ggplot(df_means, aes(x = vars, y = X1, fill = X1)) + # Generates vertical graph
  geom_bar(stat = "identity") + 
  scale_fill_gradient(low = "light blue", high = "blue", limits = c(0, 100)) + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) 
vgraph


#%%
#%% Show entries with missing data
new_DF <- data[rowSums(is.na(data)) > 0,]


#%%
#%% Test using diabetes dataset
data(diabetes)
X <- diabetes[,-1] # Remove class variable
testmodel <- Mclust(X)

#%%
