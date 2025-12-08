##------------------Subgroup analysis: Age, Gender, Urban/rural residence, Hypertension, Lipid-lowering drugs------------------
#Split by Age
Age_list <- split(test_standardized, test_standardized$Age)

#Split by Gender
Gender_list <- split(test_standardized, test_standardized$Gender)

#Split by Current_residence_location
crl_list <- split(test_standardized, test_standardized$Current_residence_location)

#Split by Hypertension
Hypertension_list <- split(test_standardized, test_standardized$Hypertension)

#Split by Lipid_lowering_drugs
lld_list <- split(test_standardized, test_standardized$Lipid_lowering_drugs)

#Function to compute performance for any classification task
evaluate_task_metrics <- function(data, dataset_name = "dataset_name") {
  #Extract final predictors and outcome
  subgroup_boruta <- data[, c(confirmed_features, "RDKF")]
  #Create task
  task <- as_task_classif(subgroup_boruta, target = "RDKF")
  
  #Predict using the already-trained learners
  lr_pred       <- lr_learner$predict(task)
  rf_pred       <- rf_learner$predict(task)
  svm_pred      <- svm_learner$predict(task)
  xgb_pred      <- xgb_learner$predict(task)
  catb_pred     <- catb_learner$predict(task)
  lightgbm_pred <- lightgbm_learner$predict(task)
  
  preds_list <- list(
    LR       = lr_pred,
    RF       = rf_pred,
    SVM      = svm_pred,
    XGBoost  = xgb_pred,
    CatBoost = catb_pred,
    LightGBM = lightgbm_pred
  )
  
  metrics_df <- data.frame()
  
  for (model_name in names(preds_list)) {
    pred_obj   <- preds_list[[model_name]]
    pred_prob  <- pred_obj$prob[, "2"]
    true_class <- pred_obj$truth
    true_class <- ifelse(true_class == "2", 1, 0)
    
    # If a subgroup has only one outcome class, ROC/AUC cannot be computed; skip it
    if (length(unique(true_class)) < 2) {
      next
    }
    
    roc_obj <- roc(true_class, pred_prob)
    auc_val <- roc_obj$auc
    ci_auc  <- ci(roc_obj)
    
    # Combine AUC and 95% CI
    auc_with_ci <- sprintf("%.3f\n(%.3f-%.3f)", auc_val, ci_auc[1], ci_auc[3]) 
    
    # Retrieve this model's optimal threshold recorded from the training cohort
    best_threshold_train <- train_metrics[train_metrics$Model == model_name, "Threshold"]
    
    # Compute performance on this task using the training cohort threshold 
    
    best_metrics <- calculate_metrics(pred_prob, true_class, best_threshold_train)
    
    # Brier score
    brier_score <- mean((pred_prob - true_class)^2)
    
    # bootstrap CI
    set.seed(123)
    ci_boot <- bootstrap_ci(pred_prob, true_class, best_threshold_train, B = 1000)
    
    Sensitivity_CI <- format_ci(best_metrics$Sensitivity, ci_boot[, "Sensitivity"])
    Specificity_CI <- format_ci(best_metrics$Specificity, ci_boot[, "Specificity"])
    PPV_CI         <- format_ci(best_metrics$PPV,        ci_boot[, "PPV"])
    NPV_CI         <- format_ci(best_metrics$NPV,        ci_boot[, "NPV"])
    CCR_CI         <- format_ci(best_metrics$CCR,        ci_boot[, "CCR"])
    F1_score_CI    <- format_ci(best_metrics$F1_score,   ci_boot[, "F1_score"])
    Brier_score_CI <- format_ci(brier_score,             ci_boot[, "Brier_score"])
    
    one_row <- data.frame(
      Model          = model_name,
      Dataset        = dataset_name,
      AUC_CI         = auc_with_ci, 
      Sensitivity_CI = Sensitivity_CI,
      Specificity_CI = Specificity_CI,
      PPV_CI         = PPV_CI,
      NPV_CI         = NPV_CI,
      CCR_CI         = CCR_CI,
      F1_score_CI    = F1_score_CI,
      Brier_score_CI = Brier_score_CI, 
      stringsAsFactors = FALSE
    )
    
    metrics_df <- rbind(metrics_df, one_row)
  }
  
  return(metrics_df)
}

#Performance for each subgroup
Age1_metrics <- evaluate_task_metrics(Age_list[["1"]], dataset_name = "Age (45-54 years)")
Age2_metrics <- evaluate_task_metrics(Age_list[["2"]], dataset_name = "Age (55-64 years)")
Age3_metrics <- evaluate_task_metrics(Age_list[["3"]], dataset_name = "Age (¡Ý 65 years)")
Gender1_metrics <- evaluate_task_metrics(Gender_list[["1"]], dataset_name = "Gender (Man)")
Gender2_metrics <- evaluate_task_metrics(Gender_list[["2"]], dataset_name = "Gender (Woman)")
crl1_metrics <- evaluate_task_metrics(crl_list[["1"]], dataset_name = "Current residence location (Urban)")
crl2_metrics <- evaluate_task_metrics(crl_list[["2"]], dataset_name = "Current residence location (Rural)")
Hypertension1_metrics <- evaluate_task_metrics(Hypertension_list[["1"]], dataset_name = "Hypertension (No)")
Hypertension2_metrics <- evaluate_task_metrics(Hypertension_list[["2"]], dataset_name = "Hypertension (Yes)")
lld1_metrics <- evaluate_task_metrics(lld_list[["1"]], dataset_name = "Lipid-lowering drugs (No)")
lld2_metrics <- evaluate_task_metrics(lld_list[["2"]], dataset_name = "Lipid-lowering drugs (Yes)")

#Put all subgroup results into a named list
subgroup_metrics_list <- list(
  "Age (45-54 years)"                    = Age1_metrics,
  "Age (55-64 years)"                    = Age2_metrics,
  "Age (¡Ý65 years)"                      = Age3_metrics,
  "Gender (Man)"                         = Gender1_metrics,
  "Gender (Woman)"                       = Gender2_metrics,
  "Current residence location (Urban)"   = crl1_metrics,
  "Current residence location (Rural)"   = crl2_metrics,
  "Hypertension (No)"                    = Hypertension1_metrics,
  "Hypertension (Yes)"                   = Hypertension2_metrics,
  "Lipid-lowering drugs (No)"            = lld1_metrics,
  "Lipid-lowering drugs (Yes)"           = lld2_metrics
)

#Transpose function
transpose_metrics <- function(df) {
  # Modify column names
  names(df)[c(3:10)] <- c("AUC\n(95% CI)", 
                          "Sensitivity\n(95% CI)", 
                          "Specificity\n(95% CI)", 
                          "PPV\n(95% CI)", 
                          "NPV\n(95% CI)", 
                          "CCR\n(95% CI)", 
                          "F1 score\n(95% CI)", 
                          "Brier score\n(95% CI)")
  
  # Transpose + convert row names into the first column Metric
  df_t <- tibble::rownames_to_column(as.data.frame(t(df)))
  
  # The first row is 'Model'; use it to name the columns
  colnames(df_t) <- c("Metric", df_t[1, -1])
  
  # Remove the 'Model' row
  df_t <- df_t[-1, ]
  
  return(df_t)
}

#Transpose all subgroup results
subgroup_t_list <- lapply(subgroup_metrics_list, transpose_metrics)
#Combine all subgroup results (Table S10)
subgroup_t_all <- bind_rows(subgroup_t_list)

##---------Subgroup SHAP summary plots
#Subgroup SHAP function
subgroup_shap <- function(data) {
  #Extract final predictors and outcome
  subgroup_boruta <- data[, c(confirmed_features, "RDKF")]
  #Create task
  task <- as_task_classif(subgroup_boruta, target = "RDKF")
  #Extract predictors for each subgroup
  subgroup_x <- task$data(cols = task$feature_names)
  #Compute SHAP values: svm_learner is the trained model; X = test_x is the dataset to be explained; bg_X = train_x_200 is the background dataset used as the baseline for SHAP; predict_type = "prob" specifies probability predictions; verbose = F disables verbose output.
  shap_value <- kernelshap(svm_learner, X = subgroup_x, bg_X = train_x_200, predict_type = "prob", verbose = F)
  #Build visualization object
  sv_svm <- shapviz(shap_value, which_class = 2) 
  #Map variable names via data dictionary
  shap_values_df <- as.data.frame(sv_svm$S)
  colnames(shap_values_df) <- c("Anti-diabetic drugs", "Anti-hypertensive drugs", "Dyslipidemia", 
                                "Gender", "Fasting glucose", "HDL-C", "HbA1c", "Hematocrit", "Hemoglobin", 
                                "Hypertension", "LDL-C", "Lipid-lowering drugs", "TC", "TG", 
                                "UA", "Waist circumference")
  sv_svm$S <- as.matrix(shap_values_df)
  colnames(sv_svm$X) <- data_dict$long[match(names(sv_svm$X), data_dict$short)]
  
  #SHAP beeswarm plot
  shap_beeswarm <- sv_importance(sv_svm, 
                                 kind = "beeswarm", #beeswarm plot
                                 size = 0.5, # point size (beeswarm plot)
                                 bee_width = 0.5, #horizontal spread width of points
                                 max_display = Inf) + #display all features
    scale_color_gradient(low = "#6601F7", high = "#48EDFE") + 
    scale_x_continuous(limits = c(-0.45, 0.6), #limit x-axis range
                       breaks = seq(-0.4, 0.6, by = 0.2)) + #set x-axis breaks
    theme(
      panel.background = element_rect(fill = "white"), #panel background is white
      plot.background = element_rect(fill = "white"), #entire plotting area background is white
      axis.title = element_text(size = 40), #axis title font size
      axis.text = element_text(size = 30), #axis tick label font size
      legend.title = element_text(size = 30), #legend title font size
      legend.text = element_text(size = 25), #legend text font size
      legend.key.height = unit(1.25, "cm"), #legend bar height
      legend.key.width = unit(0.075, "cm"), #legend bar width
      plot.tag = element_text(size = 50, face = "bold"), #set tag text size
      plot.margin = margin(0, 0, 0, 0, "cm"), #adjust margins
      axis.ticks = element_line(size = 0.25), #tick line width
      axis.ticks.length = unit(0.05, "cm"), #tick marks outward length
      axis.ticks.y = element_blank() #remove y-axis ticks
    )
  
  return(list(
    sv_svm   = sv_svm,
    shap_beeswarm = shap_beeswarm
  ))
}

#Call function to plot SHAP for each subgroup
Age1_shap <- subgroup_shap(Age_list[["1"]])  # 45¨C54 years
Age2_shap <- subgroup_shap(Age_list[["2"]])  # 55¨C64 years
Age3_shap <- subgroup_shap(Age_list[["3"]])  # ¡Ý65 years
Gender1_shap <- subgroup_shap(Gender_list[["1"]])  # Man
Gender2_shap <- subgroup_shap(Gender_list[["2"]])  # Woman
crl1_shap <- subgroup_shap(crl_list[["1"]])  # Urban
crl2_shap <- subgroup_shap(crl_list[["2"]])  # Rural
Hypertension1_shap <- subgroup_shap(Hypertension_list[["1"]])  # Hypertension = No
Hypertension2_shap <- subgroup_shap(Hypertension_list[["2"]])  # Hypertension = Yes
lld1_shap <- subgroup_shap(lld_list[["1"]])  # Lipid-lowering drugs = No
lld2_shap <- subgroup_shap(lld_list[["2"]])  # Lipid-lowering drugs = Yes

##Summary plot--age
Age1_shap_beeswarm <- sv_importance(Age1_shap[["sv_svm"]], 
                                    kind = "beeswarm", #beeswarm plot
                                    size = 0.3, # point size (beeswarm plot)
                                    bee_width = 0.3, #horizontal spread width of points
                                    max_display = Inf) + #display all features
  scale_color_gradient(low = "#6601F7", high = "#48EDFE") + 
  scale_x_continuous(limits = c(-0.45, 0.6), #limit x-axis range
                     breaks = seq(-0.4, 0.6, by = 0.2)) + #set x-axis breaks
  theme(
    panel.background = element_rect(fill = "white"), #panel background is white
    plot.background = element_rect(fill = "white"), #entire plotting area background is white
    axis.title = element_text(size = 30), #axis title font size
    axis.text = element_text(size = 20), #axis tick label font size
    legend.title = element_text(size = 20), #legend title font size
    legend.text = element_text(size = 15), #legend text font size
    legend.key.height = unit(0.9, "cm"), #legend bar height
    legend.key.width = unit(0.075, "cm"), #legend bar width
    plot.tag = element_text(size = 50, face = "bold"), #set tag text size
    plot.margin = margin(0, 0, 0, 0, "cm"), #adjust margins
    axis.ticks = element_line(size = 0.2), #tick line width
    axis.ticks.length = unit(0.04, "cm"), #tick marks outward length
    axis.ticks.y = element_blank() #remove y-axis ticks
  )

Age2_shap_beeswarm <- sv_importance(Age2_shap[["sv_svm"]], 
                                    kind = "beeswarm", #beeswarm plot
                                    size = 0.3, # point size (beeswarm plot)
                                    bee_width = 0.3, #horizontal spread width of points
                                    max_display = Inf) + #display all features
  scale_color_gradient(low = "#6601F7", high = "#48EDFE") + 
  scale_x_continuous(limits = c(-0.45, 0.6), #limit x-axis range
                     breaks = seq(-0.4, 0.6, by = 0.2)) + #set x-axis breaks
  theme(
    panel.background = element_rect(fill = "white"), #panel background is white
    plot.background = element_rect(fill = "white"), #entire plotting area background is white
    axis.title = element_text(size = 30), #axis title font size
    axis.text = element_text(size = 20), #axis tick label font size
    legend.title = element_text(size = 20), #legend title font size
    legend.text = element_text(size = 15), #legend text font size
    legend.key.height = unit(0.9, "cm"), #legend bar height
    legend.key.width = unit(0.075, "cm"), #legend bar width
    plot.tag = element_text(size = 50, face = "bold"), #set tag text size
    plot.margin = margin(0, 0, 0, 0, "cm"), #adjust margins
    axis.ticks = element_line(size = 0.2), #tick line width
    axis.ticks.length = unit(0.04, "cm"), #tick marks outward length
    axis.ticks.y = element_blank() #remove y-axis ticks
  )

Age3_shap_beeswarm <- sv_importance(Age3_shap[["sv_svm"]], 
                                    kind = "beeswarm", #beeswarm plot
                                    size = 0.3, # point size (beeswarm plot)
                                    bee_width = 0.3, #horizontal spread width of points
                                    max_display = Inf) + #display all features
  scale_color_gradient(low = "#6601F7", high = "#48EDFE") + 
  scale_x_continuous(limits = c(-0.45, 0.6), #limit x-axis range
                     breaks = seq(-0.4, 0.6, by = 0.2)) + #set x-axis breaks
  theme(
    panel.background = element_rect(fill = "white"), #panel background is white
    plot.background = element_rect(fill = "white"), #entire plotting area background is white
    axis.title = element_text(size = 30), #axis title font size
    axis.text = element_text(size = 20), #axis tick label font size
    legend.title = element_text(size = 20), #legend title font size
    legend.text = element_text(size = 15), #legend text font size
    legend.key.height = unit(0.9, "cm"), #legend bar height
    legend.key.width = unit(0.075, "cm"), #legend bar width
    plot.tag = element_text(size = 50, face = "bold"), #set tag text size
    plot.margin = margin(0, 0, 0, 0, "cm"), #adjust margins
    axis.ticks = element_line(size = 0.2), #tick line width
    axis.ticks.length = unit(0.04, "cm"), #tick marks outward length
    axis.ticks.y = element_blank() #remove y-axis ticks
  )

#Merge summary plot--age
shap_plot_Age <- Age1_shap_beeswarm + Age2_shap_beeswarm + Age3_shap_beeswarm + 
  plot_annotation(tag_levels = "A") #Automatically add A-C labels

#Export summary plot--age
ggsave(plot = shap_plot_Age, 
       filename = "shap_plot_Age.png", 
       width = 17, 
       height = 6,
       units = "cm", 
       dpi = 600)

#Merge summary plot--Gender
shap_plot_Gender <- Gender1_shap[["shap_beeswarm"]] + Gender2_shap[["shap_beeswarm"]] + 
  plot_annotation(tag_levels = "A") #Automatically add A and B labels

#Export summary plot--Gender
ggsave(plot = shap_plot_Gender, 
       filename = "shap_plot_Gender.png", 
       width = 17, 
       height = 8,
       units = "cm", 
       dpi = 600)

#Merge summary plot--Current residence location
shap_plot_crl <- crl1_shap[["shap_beeswarm"]] + crl2_shap[["shap_beeswarm"]] + 
  plot_annotation(tag_levels = "A") #Automatically add A and B labels

#Export summary plot--Current residence location
ggsave(plot = shap_plot_crl, 
       filename = "shap_plot_crl.png", 
       width = 17, 
       height = 8,
       units = "cm", 
       dpi = 600)

#Merge summary plot--Hypertension
shap_plot_Hypertension <- Hypertensio1_shap[["shap_beeswarm"]] + Hypertensio2_shap[["shap_beeswarm"]] + 
  plot_annotation(tag_levels = "A") #Automatically add A and B labels

#Export summary plot--Hypertension
ggsave(plot = shap_plot_Hypertension, 
       filename = "shap_plot_Hypertension.png", 
       width = 17, 
       height = 8,
       units = "cm", 
       dpi = 600)

#Merge summary plot--Lipid-lowering drugs 
shap_plot_lld <- lld1_shap[["shap_beeswarm"]] + lld2_shap[["shap_beeswarm"]] + 
  plot_annotation(tag_levels = "A") #Automatically add A and B labels

#Export summary plot--Lipid-lowering drugs
ggsave(plot = shap_plot_lld, 
       filename = "shap_plot_lld.png", 
       width = 17, 
       height = 8,
       units = "cm", 
       dpi = 600)
