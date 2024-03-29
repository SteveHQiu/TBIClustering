#%% Installation
install.packages("mclust")
install.packages("poLCA")
install.packages("tidyLPA")
install.packages("tidyverse")
install.packages("dplyr")
install.packages("comprehenr")
install.packages("inflection")
install.packages("patchwork")
install.packages("viridis")
install.packages("colorspace")
install.packages("multcomp")
install.packages("pROC")


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
library(multcomp)
library(pROC) # AUC calculation 
#%%
#%% Constants
DF_PATH <- "data/tbi2_admit_icd_dates_nsx_gcs_elix.xlsx"
df_orig <- read_excel(DF_PATH)
DF_ROOTNAME <- tools::file_path_sans_ext(DF_PATH)
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
fig_labels <- c("CHF", "Arrhythmia", "Valvular disease", "Pulmonary circulation disorder",
                "Peripheral vascular disorder", "Uncomplicated hypertension",
                "Complicated hypertension", "Paralysis", "Other neurological disorder",
                "COPD", "Uncomplicated diabetes", "Complicated diabetes", "Hypothyroidism",
                "Renal failure", "Liver disease", "AID/HIV",
                "Lymphoma", "Metastatic cancer", "Solid tumor (no metastasis)",
                "Rheumatoid arthritis", "Coagulopathy", "Obesity", "Weight loss",
                "Fluid and electrolyte disorders", "Blood loss anemia", "Deficiency anemia",
                "Alcohol abuse", "Drug abuse", "Psychoses", "Depression")


genClusts <- function(df_orig, col_symbols = NULL, n_clusts = 7, mode = "lca", lpa_mode = "EII", scale100 = TRUE, annot = FALSE) {

  if (!is.null(col_symbols)) {
    df <- df_orig %>% dplyr::select(!!!col_symbols) # !!! is a spice operator to inject/unpack symbols for vector of symbols
  } else {
    df <- df_orig
  }

  
  
  if (mode == "lca") {
    df <- df * 1 # Turns TRUE/FALSE into 1/0
    # Manifest values needs to be an integer starting from 1 and not 0 https://stackoverflow.com/questions/52008147/polca-alert-values-that-are-not-positive-integers
    df <- df + 1 # Adds 1 to every value
    new_col_symbols <- lapply(colnames(df), as.name)
    model <- rlang::inject(cbind(!!!new_col_symbols) ~ 1) # Define model, can probably just refer to col.names instead of re-using symbols since col.names will already be filtered
    cluster <- poLCA(model, data=df, nclass=n_clusts, na.rm=TRUE, maxiter=10000)
    df_results <- data.frame(cluster$probs) # Outputs probability of each input group for all clusters (1=False, 2=True)
    df_means <- df_results[,seq(2, ncol(df_results), 2)] # Get every second column starting at col 2
    model_means <- as.data.frame(t(df_means))

    df_annot <- cluster$predclass
  } else if (mode == "lpa") {
    cluster <- Mclust(df, G=n_clusts, modelNames=c(lpa_mode))
    model_means <- cluster$parameters$mean

    df_annot <- cluster$classification 
  } else {
    print("Invalid mode given")
    return() # Exit function early
  }
  if (scale100) {
    model_means <- model_b_means * 100 # Scale everything by 100
  }
  # model_means <- model_means + 50 # Add 50 to all values as base

  if (annot) {
    df_annot <- data.frame(df_annot)
    colnames(df_annot) <- "Endotype"
    df_annotated <- cbind(df_orig, df_annot)
    write_xlsx(df_annotated, sprintf("%s_annotated.xlsx", DF_ROOTNAME))
  }

  clust_counts <- table(df_annot) # Tabulate cluster counts

  # Outputs df where every column is an endotype, each row is a factor used in endotyping 
  return(list(model_means=model_means, cluster=cluster,
  clust_counts=clust_counts, df_annot=df_annot)) # Need to use list() to make array since c() coerces elements into same type
}


genMultClusts <- function(n_clusts, n_cycles, mode = "lca") {
  # Note: this has only been verified for LCA mode for first round of clustering
  meta_means <- data.frame()
  all_means <- list()
  bics <- c()
  aics <- c()
  clusts <- list()
  annotations <- list()
  clust_counts <- list()
  
  for (i in 1:n_cycles) {
    results <- genClusts(df_orig=df_orig, col_symbols=col_symbols, n_clusts=n_clusts, mode=mode)
    model_means <- results[[1]] # Need double brackets for list indexing
    model_means_t <- data.frame(t(model_means)) # Transpose to have clusters as case listings and clusters as columns
    cluster <- results[[2]]
    df_annot <- results[[4]]
    
    meta_means <- rbind(meta_means, model_means_t)
    all_means[[i]] <- model_means # Store via index to separate each df into its own
    bics <- c(bics, cluster$bic)
    aics <- c(aics, cluster$aic)
    clusts[[i]] <- cluster # Store via index to separate each cluster to avoid merging
    annotations[[i]] <- df_annot # Store via index to separate each df into its own
    clust_counts[[i]] <- table(df_annot) 
    
  }
  
  return(list(meta_means=meta_means, all_means=all_means,
   bics=bics, aics=aics, clusts=clusts, annotations=annotations,
   clust_counts=clust_counts))
  
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

visRadialPlots <- function(model_means, fig_labels, clust_counts = NULL, clust_model = NULL,
                           save_path = NULL) {
  # Basically visualizing row values for every column, each column having own radial plot
  # model_means should be a df with factors in rows with columns representing groupings

  col_labels <- paste("Endotype_", seq(1, ncol(model_means)), sep="")
  endotype_labels <- paste("Endotype ", seq(1, ncol(model_means)), sep="") # Separate list for labels without spaces 
  colnames(model_means) <- col_labels
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
  
  
  
  for (clust_name in col_labels) {
    clust_num <- match(clust_name, col_labels)
    clust_title <- endotype_labels[clust_num]
    plot_color <- clust_colors[clust_num]
    plot_color_desat <- clust_colors_desat[clust_num]

    rgraph <- ggplot(df_means, aes_string(x = "id", y = clust_name, fill = clust_name)) + # Note that id is a factor. If x is numeric, there is some space between the first bar
      ggtitle(clust_title) +
      geom_bar(stat = "identity") + 
      ylim(-100, 150) +
      scale_fill_gradient(low = plot_color_desat, high = plot_color, limits = c(0, 100),
                          name = "Probability (%)") +
      theme_minimal() + # Remove background
      guides(fill = guide_colorbar(title.position = "top", direction = "vertical")) +
      theme(
        axis.text.x = element_blank(), # Clear x-axis ticks
        axis.text.y = element_blank(), # Clear y-axis ticks
        axis.title = element_blank(), # Clear possible titles
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
    if (!is.null(clust_counts)) {
      clust_count <- clust_counts[[clust_num]]
      rgraph <- rgraph + ggtitle(sprintf("%s (%s patients)", clust_title, clust_count)) # Overwrite title if clust_counts available
    }

    if (clust_num == 1) {
      start_graph <- rgraph # If first cluster, initiate figure
    }
    else {
      start_graph <- start_graph + rgraph # Otherwise, add to main figure
    }

  }
  
  if (!is.null(clust_model)) {
    if (!is.null(clust_model$aic)) {
      clust_aic <- clust_model$aic # From poLCA model
    }
    else {
      clust_aic <- 2*clust_model$df - 2*clust_model$loglik # Assume LPA MClust model, manually calculate
    }
    fig_title <- sprintf("Endotype comorbidity profile (BIC: %s | AIC: %s)", clust_model$bic, clust_aic)
  }
  else {
    fig_title <- "Endotype comorbidity profile"
  }
  
  final_graph <- start_graph +
    plot_annotation(title=fig_title,
                    theme=theme(plot.title=element_text(size=18, hjust=0.5))) + # Add title, center it
    plot_layout(guides = "auto") # guides = "collect" to collect duplicate legends

  if (!is.null(save_path)) {
    ggsave(save_path, final_graph, width = 17, height = 17)
  }
  else {
    print(final_graph) # Explicitly render it
  }
}

visPerformance <- function(indices, measures, label, smoothing = FALSE) {
  fig_title <- sprintf("%s vs no. of clusters", label)
  

  plot(indices, measures, main=fig_title, ylab=label, xlab="No. of clusters", type="o")
  if (smoothing) {
    lo <- loess(measures~indices)
    smoothed <- predict(lo) # Smoothed output
    lines(indices, smoothed, type="o", col="blue")
  }
  print(sprintf("Inflection for %s series", label))
  calculateInflection(indices, measures)
  deltas <- analyzeDeltas(indices, measures, series_label=label)
  print(sprintf("Inflection for %s deltas", label))
  calculateInflection(indices[-1], deltas) # Remove first item from indices
}

visSurvival <- function(df_annotated, save_path=NULL) {
  # Note that df_annotated can be imported directly from xlsx containing exported annotations
  df <- df_annotated %>% dplyr::select("Survival to discharge", "Endotype")
  cross_tab <- table(df$Endotype, df$`Survival to discharge`)
  pairwise.prop.test(cross_tab, p.adjust.method="holm")
  # pairwise.prop.test(cross_tab, p.adjust.method="bonferroni") # More conservative method
  
  
  endotypes <- factor(df$Endotype)
  clust_colors <- rainbow(length(attributes(endotypes)$levels)) # Base colors
  bar_colors <- append(lighten(clust_colors, 0.2), lighten(clust_colors, 0.75))
  
  
  # ggplot visual
  df_counts_series <- data.frame(cross_tab)
  colnames(df_counts_series) <- c("Endotype", "Survival", "Count")
  df_percent <- df_counts_series %>% group_by(Endotype) %>%
    mutate(Percent=100*Count/sum(Count))
  surv_plot <- ggplot(df_percent, aes(fill=Survival, y=Percent, x=Endotype)) + 
    geom_bar(position="stack", stat="identity", fill=bar_colors)
  
  if (!is.null(save_path)) {
    ggsave(save_path, surv_plot)
  }
  else {
    print(surv_plot) # Explicitly render it
  }
  
}

analyzeDeltas <- function(indices, series, series_label) {
  deltas <- to_vec(for(i in seq_along(series)) series[i] - series[i-1]) # Calculate change in value per change in index
  fig_title <- sprintf("Change in %s with increase in number of clusters", series_label)
  y_label <- sprintf("Change in %s from previous number of clusters", series_label)
  indices <- indices[-1] # Remove first item
  plot(indices, deltas, main=fig_title, ylab=y_label, xlab="No. of clusters", type="o",)
  return(deltas)
}

calculateInflection <- function(indices, series, smoothing = FALSE) {
  # Different ways of finding inflection point, some require inputting the convexity (obtained via check_curve)
  if (smoothing) {
      lo <- loess(series~indices)
      series <- predict(lo) # Smoothed output
  }

  curve_report <- check_curve(indices, series)
  
  print(sprintf("UIK: %s", uik(indices, series)))
  print(sprintf("d2UIK: %s", d2uik(indices, series)))
  print(ese(indices, series, curve_report$index))
  print(ede(indices, series, curve_report$index))
}


explModelsLCA <- function(df_orig, col_symbols, upper_bound) {
  df <- df_orig %>% dplyr::select(!!!col_symbols) # !!! is a spice operator to inject/unpack symbols for vector of symbols
  df <- df * 1 # Turns TRUE/FALSE into 1/0
  # Manifest values needs to be an integer starting from 1 and not 0 https://stackoverflow.com/questions/52008147/polca-alert-values-that-are-not-positive-integers
  df <- df + 1 # Adds 1 to every value
  
  model <- rlang::inject(cbind(!!!col_symbols) ~ 1) # Define model
  
  ## Cluster visualization
  
  indices <- c(2:upper_bound)

  bics <- c()
  aics <- c()
  clusters <- list()

  for (i in indices) {
    cluster <-poLCA(model, data=df, nclass=i, maxiter=10000)
    bics <- c(bics, cluster$bic)
    aics <- c(aics, cluster$aic)
    clusters <- append(clusters, cluster)
  }
  
  return(list(indices=indices, bics=bics, aics=aics, clusters=clusters))
}

explModelsLPA <- function(df_orig, col_symbols, upper_bound) {
  #%% Model generation 
  df <- df_orig %>% dplyr::select(!!!col_symbols) # !!! is a spice operator to inject/unpack symbols for vector of symbols
  # Models:
  # "VII": spherical, unequal volume - Tends not to work well with normalized range and some clusters will only be composed of age characteristics
  # "EII": spherical, equal volume - Better with normalized age range

  model = Mclust(df, G = 2:upper_bound, modelNames = c("EII")) # Default is G = 1:9, need to set this manually, can also specify certain type of model (E.g., EII, VII)
  # model = Mclust(df, G = 5:30, modelNames = c("VII")) # Default is G = 1:9, need to set this manually, can also specify certain type of model (E.g., EII, VII)
  # model = Mclust(df, G = 5:30) # General clustering to try every type of model
  # model = mclustICL(df, G = 5:30) # Based on ICL rather than BIC

  indices <- c(2:upper_bound)
  bics <- model$BIC
  aics <- NULL
  clusters <- model

  return(list(indices, bics, aics, clusters))


}


regressionSurvival <- function(df_orig, df_annot, col_symbols) {

  df <- df_orig %>% dplyr::select(!!!col_symbols) # !!! is a spice operator to inject/unpack symbols for vector of symbols

  df$`Survival to discharge` <- with(df_orig, ifelse(is.na(DOD), TRUE, FALSE))
  col_symbols <- append(col_symbols, as.name("Survival to discharge")) # Add new survival variable

  lg_model <- lm(`Survival to discharge` ~ ., data=df) # lm() rather than glm() since lm() easier to pass on to ANOVA 
  print("Regression using only comorbidities")
  print(summary(lg_model))

  df$endotype <- as.factor(df_annot)
  col_symbols <- append(col_symbols, as.name("endotype")) # Add new endotype variable
  lg_model2 <- lm(`Survival to discharge` ~ ., data=df)
  print("Regression with endotypes")
  print(summary(lg_model2))

  lg_model3 <- lm(`Survival to discharge` ~ endotype, data=df) # Use only endotype
  print("Regression with endotypes")
  print(summary(lg_model3))

  print("Significance of adding endotypes")
  print(anova(lg_model, lg_model2))
  print("Significance of between raw data and endotypes only")
  print(anova(lg_model, lg_model3))

}

  
chisqSurvival <- function(df_annotated, col_outcome = "Survival to discharge") {
  df <- df_annotated %>% dplyr::select(col_outcome, "Endotype")
  cross_tab <- table(df$Endotype, df[[col_outcome]]) # Need to use brackets to extract using string variables
  pairwise.prop.test(cross_tab, p.adjust.method="holm")

}



#%%
#%% Misc displays/analyses

# Exploration
if (1) {
  for (i in 1:7) {
    expl_results <- explModelsLCA(df_orig=df_orig, col_symbols=col_symbols, upper_bound=15)
    visPerformance(expl_results[[1]], expl_results[[2]], "BIC")
    visPerformance(expl_results[[1]], expl_results[[3]], "AIC")

  }
}

if (1) {
expl_results <- explModelsLPA(df_orig=df_orig, col_symbols=col_symbols, upper_bound=15)
visPerformance(expl_results[[1]], expl_results[[2]], "BIC")
}


#%%
#%% Visualizing lowest AIC/BIC clusters (i.e., finding best performing clusters)
if (1) {
  results <- genMultClusts(n_clust=5, n_cycles=30, mode="lca")
  
  if (0) { # Activate for rounding
  bics <- round(results[["bics"]], digits=3) # Round to 3 decimals since some clusters are very similar 
  aics <- round(results[["aics"]], digits=3)
  }
  else { # Activate for rounding
  bics <- results[["bics"]]
  aics <- results[["aics"]]
  }
  r_bics <- round(results[["bics"]], digits=3)
  r_aics <- round(results[["aics"]], digits=3)
  
  
  
  lowest_aics <- which(bics == min(bics)) # Match all results with same rounded AIC if rounding active, otherwise will only visualize the best cluster 
  for (i in lowest_aics) {
    model_means <- results[["all_means"]][[i]] # Need double brackets as all of these are lists
    clust_model <- results[["clusts"]][[i]]
    clust_counts <- results[["clust_counts"]][[i]]

    visRadialPlots(model_means=model_means, fig_labels=fig_labels,
                   clust_counts=clust_counts, clust_model=clust_model, save_path="figures/LCA_5v5.png")

    df_annot <- results[["annotations"]][[i]]
    df_annot <- data.frame(df_annot)
    colnames(df_annot) <- "Endotype"
    df_annotated <- cbind(df_orig, df_annot)
    write_xlsx(df_annotated, sprintf("%s_annotated.xlsx", DF_ROOTNAME))
    visSurvival(df_annotated, save_path="figures/LCA_5v5_survival.png")
    print(chisqSurvival(df_annotated))
  }
  
}
#%%
#%% LPA on LCA results 
if (1) {
results <- genMultClusts(n_clust=5, n_cycles=30, mode="lca")
model_means <- results[["model_means"]] # Need double brackets for list
analyzeClusts(meta_means=model_means) # Preview


}

if (1) { # Visualize after getting n_mclusts from preview
meta_results <- analyzeClusts(meta_means=model_means, n_mclusts=13, lpa_mode="VEI")
meta_means <- meta_results[["model_means"]]
meta_counts <- meta_results[["clust_counts"]]
visRadialPlots(model_means=meta_means, fig_labels=fig_labels, clust_counts=meta_counts)

}


#%%
#%% Focusing on Single cluster (main analysis)
if (1) {
results <- genClusts(df_orig=df_orig, col_symbols=col_symbols, n_clusts=5, mode="lca", annot=TRUE)
model_means <- results[["model_means"]] # Need double brackets for list
clust_model <- results[["cluster"]]
clust_counts <- results[["clust_counts"]]
visRadialPlots(model_means=model_means, fig_labels=fig_labels, clust_counts=clust_counts,
               clust_model=clust_model, save_path="figures/LCA_5.png")
}

#%%
