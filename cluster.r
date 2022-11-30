#%% Installation
# install.packages("mclust")
# install.packages("poLCA")
# install.packages("tidyLPA")
# install.packages("tidyverse")
# install.packages("dplyr")
# install.packages("comprehenr")
# install.packages("inflection")
# install.packages("patchwork")
# install.packages("viridis")
# install.packages("colorspace")

#%%
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
#%%
#%% Definitions 
df_orig <- read_excel("data/tbi_admit_icd_age_elix.xlsx")
col_labels <- c("congestive_heart_failure", "cardiac_arrhythmia", "valvular_disease",
             "pulmonary_circulation_disorder", "peripheral_vascular_disorder",
             "hypertension_uncomplicated", "hypertension_complicated", "paralysis",
             "other_neurological_disorder", "chronic_pulmonary_disease", "diabetes_uncomplicated",
             "diabetes_complicated", "hypothyroidism", "renal_failure", "liver_disease",
             "peptic_ulcer_disease_excluding_bleeding", "aids_hiv", "lymphoma", "metastatic_cancer",
             "solid_tumor_wo_metastasis", "rheumatoid_arhritis", "coagulopathy", "obesity",
             "weight_loss", "fluid_and_electrolyte_disorders", "blood_loss_anemia",
             "deficiency_anemia", "alcohol_abuse", "drug_abuse", "psychoses", "depression")
col_symbols <- base::lapply(col_labels, as.name)
fig_labels <- c("CHF", "Arrhythmia", "Valvular disease", "Pulmonary circulation disorder",
                "Peripheral vascular disorder", "Uncomplicated hypertension",
                "Complicated hypertension", "Paralysis", "Other neurological disorder",
                "COPD", "Uncomplicated diabetes", "Complicated diabetes", "Hypothyroidism",
                "Renal failure", "Liver disease", "Peptic ulcer disease", "AID/HIV",
                "Lymphoma", "Metastatic cancer", "Solid tumor (no metastasis)",
                "Rheumatoid arthritis", "Coagulopathy", "Obesity", "Weight loss",
                "Fluid and electrolyte disorders", "Blood loss anemia", "Deficiency anemia",
                "Alcohol abuse", "Drug abuse", "Psychoses", "Depression")

# Remove certain factors
col_symbols <- col_symbols[!col_symbols %in% c(as.name("peptic_ulcer_disease_excluding_bleeding"))] # Specific variables to remove from cluster
fig_labels <- fig_labels[!fig_labels %in% c("Peptic ulcer disease")] # Specific labels to remove from cluster figures

# Add age as needed
# col_symbols <- append(col_symbols, as.name("age_normalized"), 0) # Add age if needed
# fig_labels <- append(fig_labels, "Age", 0) # Add age if needed 


genClusts <- function(df_orig, col_symbols = NULL, n_clusts = 7, mode = "lca", lpa_mode = "EII", scale100 = TRUE) {

  if (!is.null(col_symbols)) {
    df <- df_orig %>% select(!!!col_symbols) # !!! is a spice operator to inject/unpack symbols for vector of symbols
  } else {
    df <- df_orig
  }

  
  
  if (mode == "lca") {
    df <- df * 1 # Turns TRUE/FALSE into 1/0
    # Manifest values needs to be an integer starting from 1 and not 0 https://stackoverflow.com/questions/52008147/polca-alert-values-that-are-not-positive-integers
    df <- df + 1 # Adds 1 to every value
    model <- rlang::inject(cbind(!!!col_symbols) ~ 1) # Define model, can probably just refer to col.names instead of re-using symbols since col.names will already be filtered
    cluster <- poLCA(model, data=df, nclass=n_clusts, na.rm=TRUE, maxiter=10000)
    df_results <- data.frame(cluster$probs) # Outputs probability of each input group for all clusters (1=False, 2=True)
    df_means <- df_results[,seq(2, ncol(df_results), 2)] # Get every second column starting at col 2
    model_means <- as.data.frame(t(df_means))

    df_annot <- cluster$predclass
  } else if (mode == "lpa") {
    cluster = Mclust(df, G=n_clusts, modelNames=c(lpa_mode))
    model_means <- cluster$parameters$mean

    df_annot <- data.frame() #FIXME

  } else {
    print("Invalid mode given")
    return() # Exit function early
  }
  if (scale100) {
    model_means <- model_means * 100 # Scale everything by 100
  }
  # model_means <- model_means + 50 # Add 50 to all values as base

  # Outputs df where every column is an endotype, each row is a factor used in endotyping 
  return(list(model_means, cluster, df_annot)) # Need to use list() to make array since c() coerces elements into same type
}


genMultClusts <- function(n_clusts, n_cycles, mode = "lca") {
  # Note: this has only been verified for LCA mode for first round of clustering
  meta_means <- data.frame()
  bics <- list()
  aics <- list()
  clusts <- list()
  
  for (i in 1:n_cycles) {
    results <- genClusts(df_orig=df_orig, col_symbols=col_symbols, n_clusts=n_clusts, mode=mode)
    model_means <- results[[1]] # Need double brackets for list indexing
    model_means_t <- data.frame(t(model_means)) # Transpose to have clusters as case listings and clusters as columns
    cluster <- results[[2]]
    
    meta_means <- rbind(meta_means, model_means_t)
    bics <- c(bics, cluster$bic)
    aics <- c(aics, cluster$aic)
    clusts <- c(clusts, cluster)    
  }
  
  return(list(meta_means, bics, aics, clusts))
  
}

analyzeClusts <- function(meta_means, n_mclusts = NULL, lpa_mode = "EEI") {
  if (is.null(n_mclusts)) { # If no number of meta-clusters given, do a preview instead
    meta_models = Mclust(meta_means, G=5:15) # Exploration
    plot(meta_models$BIC)
    
  } else {
    meta_results <- genClusts(df_orig=meta_means, n_clusts=n_mclusts, mode="lpa", lpa_mode=lpa_mode, scale100=FALSE) # No col_symbols, will take from dataframe, don't scale by 100 again
    
    return(meta_results)
    
  }
}

visRadialPlots <- function(model_means, fig_labels, save_path = "Clusters.png") {
  # Basically visualizing row values for every column, each column having own radial plot
  # model_means should be a df with factors in rows with columns representing groupings

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
  df_means$labels <- fig_labels # Better labels
  num_rows = nrow(df_means)
  angles <- 90 - 360 * (df_means$id - 0.5)/num_rows # Need to use sequence attached to df to maintain index, otherwise labels and angles are mismatched
  # Subtract 0.5 because the letter must have the angle of the center of the bars. Not extreme right(1) or extreme left (0)
  df_means$hjust<-ifelse(angles < -90, 1, 0) # Assign left or right alignment 
  df_means$angles<-ifelse(angles < -90, angles+180, angles) # Rectify angles if needed
  
  
  
  df_clust <- data.frame(model_means)
  for (clust_name in colnames(df_clust)) {
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
    if (clust_num == 1) {
      start_graph <- rgraph # If first cluster, initiate figure
    }
    else {
      start_graph <- start_graph + rgraph # Otherwise, add to main figure
    }
  }
  
  final_graph <- start_graph + plot_layout(guides = "auto") # guides = "collect" to collect duplicate legends
  ggsave(save_path, final_graph, width = 17, height = 17)
}

explModelsLCA <- function(df_orig, col_symbols) {
  df <- df_orig %>% select(!!!col_symbols) # !!! is a spice operator to inject/unpack symbols for vector of symbols
  df <- df * 1 # Turns TRUE/FALSE into 1/0
  # Manifest values needs to be an integer starting from 1 and not 0 https://stackoverflow.com/questions/52008147/polca-alert-values-that-are-not-positive-integers
  df <- df + 1 # Adds 1 to every value
  
  model <- rlang::inject(cbind(!!!col_symbols) ~ 1) # Define model
  
  ## Cluster visualization
  
  cluster <- poLCA(model, data = df, nclass = 7, na.rm = TRUE, maxiter=10000)
  
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
}

explModelsLPA <- function(df_orig, col_symbols) {
  #%% Model generation 
  df <- df_orig %>% select(!!!col_symbols) # !!! is a spice operator to inject/unpack symbols for vector of symbols
  # Models:
  # "VII": spherical, unequal volume - Tends not to work well with normalized range and some clusters will only be composed of age characteristics
  # "EII": spherical, equal volume - Better with normalized age range

  model = Mclust(df, G = 2:15, modelNames = c("EII"),) # Default is G = 1:9, need to set this manually, can also specify certain type of model (E.g., EII, VII)
  # model = Mclust(df, G = 5:30, modelNames = c("VII")) # Default is G = 1:9, need to set this manually, can also specify certain type of model (E.g., EII, VII)
  # model = Mclust(df, G = 5:30) # General clustering to try every type of model
  # model = mclustICL(df, G = 5:30) # Based on ICL rather than BIC
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

}

# Bar graph for 
anlzSurvival <- function(df_orig, df_annot, col_symbols) {

  genReport <- function(model) {
    print(summary(model))
    # ll_null <- model$null.deviance/-2
    # ll_proposed <- model$deviance/-2
    # r2 <- (ll_null - ll_proposed)/ll_null
    # print(sprintf("R2: %s", r2))
    # p_val <- 1 - pchisq(2*(ll_proposed - ll_null), df=(length(model$coefficients)-1))
    # print(sprintf("P-value: %s", p_val))
  }
  df <- df_orig %>% select(!!!col_symbols) # !!! is a spice operator to inject/unpack symbols for vector of symbols

  df$survival <- with(df_orig, ifelse(is.na(DOD), TRUE, FALSE))
  col_symbols <- append(col_symbols, as.name("survival")) # Add new survival variable

  lg_model <- lm(survival ~ ., data=df)
  print("Regression using only comorbidities")
  print(summary(lg_model))

  df$endotype <- as.factor(df_annot)
  col_symbols <- append(col_symbols, as.name("endotype")) # Add new endotype variable
  lg_model2 <- lm(survival ~ ., data=df)
  print("Regression with endotypes")
  print(summary(lg_model2))

  lg_model3 <- lm(survival ~ endotype, data=df) # Use only endotype
  print("Regression with endotypes")
  print(summary(lg_model3))

  print("Significance of adding endotypes")
  print(anova(lg_model, lg_model2))
  print("Significance of between raw data and endotypes only")
  print(anova(lg_model, lg_model3))

}
  


#%%
#%% Misc displays/analyses

# Exploration
explModelsLCA(df_orig=df_orig, col_symbols=col_symbols)
explModelsLPA(df_orig=df_orig, col_symbols=col_symbols)


# Regression analysis for survival
results <- genClusts(df_orig=df_orig, col_symbols=col_symbols, n_clusts=7, mode="lca")
model_means <- results[[1]] # Need double brackets for list
df_annot <- results[[3]]
anlzSurvival(df_orig=df_orig, df_annot=df_annot, col_symbols=col_symbols)



#%%
#%% Single cluster
results <- genClusts(df_orig=df_orig, col_symbols=col_symbols, n_clusts=7, mode="lca")
model_means <- results[[1]] # Need double brackets for list
visRadialPlots(model_means=model_means, fig_labels=fig_labels)

#%%
#%% LPA on LCA results 

results <- genMultClusts(n_clust=7, n_cycles=5)
model_means <- results[[1]] # Need double brackets for list
analyzeClusts(meta_means=model_means) # Preview
meta_results <- analyzeClusts(meta_means=model_means, n_mclusts=7, lpa_mode="VEI")
meta_means <- meta_results[[1]]
visRadialPlots(model_means=meta_means, fig_labels=fig_labels)

#%%
