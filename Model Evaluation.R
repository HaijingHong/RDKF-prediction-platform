## -------- Model performance evaluation --------
# Create lists of model predictions for training and testing cohorts
train_preds <- list(
  LR = train_lr_pred, 
  RF = train_rf_pred, 
  SVM = train_svm_pred, 
  XGBoost = train_xgb_pred, 
  CatBoost = train_catb_pred, 
  LightGBM = train_lightgbm_pred
)
test_preds <- list(
  LR = test_lr_pred, 
  RF = test_rf_pred, 
  SVM = test_svm_pred, 
  XGBoost = test_xgb_pred, 
  CatBoost = test_catb_pred, 
  LightGBM = test_lightgbm_pred
)

# Create empty data frames to store performance metrics for training and testing cohorts
train_metrics <- data.frame()
test_metrics <- data.frame()

# Create empty data frames to store ROC data for training and testing cohorts
train_roc_data <- data.frame()
test_roc_data <- data.frame()

# Create lists to store ROC() results for training and testing cohorts for DeLong's AUC test
train_roc_auc <- list()
test_roc_auc <- list()

# Create empty data frames to store DCA data for training and testing cohorts
train_dca_data <- data.frame()
test_dca_data <- data.frame()

# Create lists to store predicted probabilities for training and testing cohorts
train_prob_data <- list()
test_prob_data <- list()

#Bootstrap confidence interval function
bootstrap_ci <- function(pred_prob, true_class, threshold, B = 1000) {
  n <- length(true_class)
  
  boot_metrics <- data.frame(
    Sensitivity = numeric(B),
    Specificity = numeric(B),
    PPV = numeric(B),
    NPV = numeric(B),
    CCR = numeric(B),
    F1_score = numeric(B),
    Brier_score = numeric(B)
  )
  
  for (b in 1:B) {
    idx <- sample(1:n, n, replace = TRUE)
    
    boot_prob <- pred_prob[idx]
    boot_true <- true_class[idx]
    
    metrics <- calculate_metrics(boot_prob, boot_true, threshold)
    
    boot_metrics[b, "Sensitivity"] <- metrics$Sensitivity
    boot_metrics[b, "Specificity"] <- metrics$Specificity
    boot_metrics[b, "PPV"] <- metrics$PPV
    boot_metrics[b, "NPV"] <- metrics$NPV
    boot_metrics[b, "CCR"] <- metrics$CCR
    boot_metrics[b, "F1_score"] <- metrics$F1_score
    boot_metrics[b, "Brier_score"] <- mean((boot_prob - boot_true)^2)
  }
  
  apply(boot_metrics, 2, quantile, probs = c(0.025, 0.975), na.rm = TRUE)
}

format_ci <- function(estimate, ci_vec) {
  sprintf("%.3f\n(%.3f-%.3f)", estimate, ci_vec[1], ci_vec[2])
}

# Calculate performance metrics for the training cohort
for (model_name in names(train_preds)) {
  # Get training cohort model predictions
  pred_train <- train_preds[[model_name]]
  # Get predicted probabilities for the training cohort
  pred_prob_train <- pred_train$prob[, "2"]
  
  # Store predicted probabilities for the training cohort in a list
  train_prob_data[[model_name]] <- pred_prob_train
  
  # Get true class labels for the training cohort
  true_class_train <- pred_train$truth
  true_class_train <- ifelse(true_class_train == "2", 1, 0)
  
  roc_train <- roc(true_class_train, pred_prob_train)
  auc_train <- roc_train$auc
  ci_train <- ci(roc_train)
  # Combine AUC and 95% CI (3 decimals)
  auc_with_ci_train <- sprintf("%.3f\n(%.3f-%.3f)", auc_train, ci_train[1], ci_train[3]) 
  
  # Store ROC() results in a list
  train_roc_auc[[model_name]] <- roc_train
  
  # Extract data for plotting ROC curve and append to data frame
  roc_data_train <- data.frame(
    FPR = 1-roc_train$specificities, # False positive rate (FPR): 1 - specificities
    TPR = roc_train$sensitivities, # True positive rate (TPR): sensitivities
    model = model_name # model name
  )
  # Order model names
  roc_data_train$model <- factor(roc_data_train$model, levels = c("LR", "RF", "SVM", "XGBoost", "CatBoost", "LightGBM"))
  # Combine ROC data for all models in the training cohort
  train_roc_data <- rbind(train_roc_data, roc_data_train)
  
  ## Define performance metrics
  calculate_metrics <- function(pred_prob, true_class, threshold) {
    # Generate predicted class based on threshold
    pred_class <- ifelse(pred_prob >= threshold, "1", "0") 
    # Build confusion matrix ensuring all classes are present
    cm <- table(
      factor(pred_class, levels = c("0", "1")), # Ensure predicted classes include 0 and 1
      factor(true_class, levels = c("0", "1")) # Ensure true classes include 0 and 1
    )
    # Extract elements of confusion matrix to avoid errors when some classes are missing
    TP <- cm["1", "1"] # True positives
    TN <- cm["0", "0"] # True negatives
    FP <- cm["1", "0"] # False positives
    FN <- cm["0", "1"] # False negatives
    # Compute metrics
    sensitivity <- TP / (TP + FN) # Sensitivity (Recall)
    specificity <- TN / (TN + FP) # Specificity
    PPV <- TP / (TP + FP) # Positive predictive value / Precision
    NPV <- TN / (TN + FN) # Negative predictive value
    CCR <- (TP + TN) / sum(cm) # Accuracy
    f1_score <- 2 * (PPV * sensitivity) / (PPV + sensitivity) # F1 score
    # Return results
    list(Sensitivity = sensitivity, Specificity = specificity, PPV = PPV, NPV = NPV, CCR = CCR, F1_score = f1_score)
  }
  
  ## Compute the optimal threshold for the training cohort
  # Define threshold range
  threshold <- seq(0, 1, by = 0.001)
  # Compute performance metrics across thresholds
  metrics_list_train <- sapply(threshold, function(t) {
    calculate_metrics(pred_prob_train, true_class_train, t)
  }, simplify = F)
  
  distances <- sapply(metrics_list_train, function(metrics) {
    sqrt((1-metrics$Sensitivity)^2 + (1-metrics$Specificity)^2)
  })
  # Find the optimal threshold for the training cohort
  best_threshold_train <- threshold[which.min(distances)]
  
  # Using the optimal threshold, compute performance metrics for the training cohort
  best_metrics_train <- calculate_metrics(pred_prob_train, true_class_train, best_threshold_train)
  
  # Compute Brier score
  brier_score_train <- mean((pred_prob_train - true_class_train)^2)
  
  #Bootstrap CI
  set.seed(123)
  ci_train_metrics <- bootstrap_ci(pred_prob_train, true_class_train, best_threshold_train, B = 1000)
  
  Sensitivity_CI  <- format_ci(best_metrics_train$Sensitivity,  ci_train_metrics[, "Sensitivity"])
  Specificity_CI  <- format_ci(best_metrics_train$Specificity,  ci_train_metrics[, "Specificity"])
  PPV_CI          <- format_ci(best_metrics_train$PPV,          ci_train_metrics[, "PPV"])
  NPV_CI          <- format_ci(best_metrics_train$NPV,          ci_train_metrics[, "NPV"])
  CCR_CI          <- format_ci(best_metrics_train$CCR,          ci_train_metrics[, "CCR"])
  F1_score_CI     <- format_ci(best_metrics_train$F1_score,     ci_train_metrics[, "F1_score"])
  Brier_score_CI  <- format_ci(brier_score_train,               ci_train_metrics[, "Brier_score"])
  
  # Summarize model results for the training cohort
  train_metrics_result <- data.frame(
    Model = model_name,
    Dataset = "Training cohort",
    AUC = auc_train,
    AUC_CI = auc_with_ci_train, 
    Threshold = round(best_threshold_train, 3),
    Sensitivity_CI = Sensitivity_CI,
    Specificity_CI = Specificity_CI,
    PPV_CI = PPV_CI,
    NPV_CI = NPV_CI,
    CCR_CI = CCR_CI,
    F1_score_CI = F1_score_CI,
    Brier_score_CI = Brier_score_CI,
    Sensitivity = round(best_metrics_train$Sensitivity, 3),
    Specificity = round(best_metrics_train$Specificity, 3),
    PPV = round(best_metrics_train$PPV, 3),
    NPV = round(best_metrics_train$NPV, 3),
    CCR = round(best_metrics_train$CCR, 3),
    F1_score = round(best_metrics_train$F1_score, 3),
    Brier_score = round(brier_score_train, 3)
  )
  # Append each model's training-cohort results to the data frame
  train_metrics <- rbind(train_metrics, train_metrics_result)
  
  ## Extract DCA data for the training cohort
  # Construct data frame: truth and predicted probabilities
  train_truth_p <- data.frame(
    truth = true_class_train,  # Convert truth to 0/1
    p = pred_prob_train  # Extract predicted probabilities
  )
  
  # Compute decision curve
  train_dca_result <- decision_curve(truth ~ p, data = train_truth_p, family = binomial, thresholds = seq(0, 1, by = 0.01))
  # Extract data and add to data frame
  train_thresholds <- train_dca_result$derived.data$thresholds
  train_net_benefit <- train_dca_result$derived.data$sNB
  train_model <- train_dca_result$derived.data$model
  train_model <- ifelse(train_model == "truth ~ p", model_name, train_model)
  # Combine decision-curve data
  train_dca_data <- rbind(train_dca_data, data.frame(
    threshold = train_thresholds,
    net_benefit = train_net_benefit,
    model = train_model
  ))
  # Set factor order for model names
  train_dca_data$model <- factor(train_dca_data$model, levels = c("LR", "RF", "SVM", "XGBoost", "CatBoost", "LightGBM", "All", "None"))
}

# Compute performance metrics for the test cohort
for (model_name in names(test_preds)) {
  # Get model predictions for the test cohort
  pred_test <- test_preds[[model_name]]
  # Get predicted probabilities for the test cohort
  pred_prob_test <- pred_test$prob[, "2"]
  
  # Store test-cohort predicted probabilities in list
  test_prob_data[[model_name]] <- pred_prob_test
  
  # Get true classes for the test cohort
  true_class_test <- pred_test$truth
  true_class_test <- ifelse(true_class_test == "2", 1, 0)
  
  # Compute AUC and 95% CI for the test cohort
  roc_test <- roc(true_class_test, pred_prob_test)
  auc_test <- roc_test$auc
  ci_test <- ci(roc_test)
  # Combine AUC and 95% CI, keep three decimals
  auc_with_ci_test <- sprintf("%.3f\n(%.3f-%.3f)", auc_test, ci_test[1], ci_test[3]) 
  
  # Store ROC() result in list
  test_roc_auc[[model_name]] <- roc_test
  
  # Extract data for ROC plotting and append to data frame
  roc_data <- data.frame(
    FPR = 1-roc_test$specificities, # False positive rate FPR: 1 - specificities
    TPR = roc_test$sensitivities, # True positive rate TPR: sensitivities
    model = model_name # Model name
  )
  # Set factor order for model names
  roc_data$model <- factor(roc_data$model, levels = c("LR", "RF", "SVM", "XGBoost", "CatBoost", "LightGBM"))
  # Combine ROC data for all models in the test cohort
  test_roc_data <- rbind(test_roc_data, roc_data)
  
  # Retrieve the optimal threshold from the training cohort
  best_threshold_train <- train_metrics[train_metrics$Model == model_name, "Threshold"]
  # Evaluate test-cohort performance using the training optimal threshold
  best_metrics_test <- calculate_metrics(pred_prob_test, true_class_test, best_threshold_train)
  
  # Compute Brier score for the test cohort
  brier_score_test <- mean((pred_prob_test - true_class_test)^2)
  
  #Bootstrap CI
  set.seed(123)
  ci_test_metrics <- bootstrap_ci(pred_prob_test, true_class_test, best_threshold_train, B = 1000)
  
  Sensitivity_CI  <- format_ci(best_metrics_test$Sensitivity,  ci_test_metrics[, "Sensitivity"])
  Specificity_CI  <- format_ci(best_metrics_test$Specificity,  ci_test_metrics[, "Specificity"])
  PPV_CI          <- format_ci(best_metrics_test$PPV,          ci_test_metrics[, "PPV"])
  NPV_CI          <- format_ci(best_metrics_test$NPV,          ci_test_metrics[, "NPV"])
  CCR_CI          <- format_ci(best_metrics_test$CCR,          ci_test_metrics[, "CCR"])
  F1_score_CI     <- format_ci(best_metrics_test$F1_score,     ci_test_metrics[, "F1_score"])
  Brier_score_CI  <- format_ci(brier_score_test,               ci_test_metrics[, "Brier_score"])
  
  # Summarize model results of the test cohort into a data frame
  test_metrics_result <- data.frame(
    Model = model_name,
    Dataset = "Testing cohort",
    AUC = auc_test,
    AUC_CI = auc_with_ci_test, 
    Threshold = round(best_threshold_train, 3),  # use optimal threshold from the training cohort
    Sensitivity_CI = Sensitivity_CI,
    Specificity_CI = Specificity_CI,
    PPV_CI = PPV_CI,
    NPV_CI = NPV_CI,
    CCR_CI = CCR_CI,
    F1_score_CI = F1_score_CI,
    Brier_score_CI = Brier_score_CI,
    Sensitivity = round(best_metrics_test$Sensitivity, 3),
    Specificity = round(best_metrics_test$Specificity, 3),
    PPV = round(best_metrics_test$PPV, 3),
    NPV = round(best_metrics_test$NPV, 3),
    CCR = round(best_metrics_test$CCR, 3),
    F1_score = round(best_metrics_test$F1_score, 3),
    Brier_score = round(brier_score_test, 3)
  )
  # Append results from each model in the test cohort
  test_metrics <- rbind(test_metrics, test_metrics_result)
  
  ## Extract DCA data for the test cohort
  # Construct data frame: truth and predicted probability
  test_truth_p <- data.frame(
    truth = true_class_test,  # convert truth to 0/1
    p = pred_prob_test  # extract predicted probability
  )
  
  # Compute decision curve
  test_dca_result <- decision_curve(truth ~ p, data = test_truth_p, family = binomial, thresholds = seq(0, 1, by = 0.01))
  # Extract data and add to data frame
  test_thresholds <- test_dca_result$derived.data$thresholds
  test_net_benefit <- test_dca_result$derived.data$sNB
  test_model <- test_dca_result$derived.data$model
  test_model <- ifelse(test_model == "truth ~ p", model_name, test_model)
  # Combine decision-curve data
  test_dca_data <- rbind(test_dca_data, data.frame(
    threshold = test_thresholds,
    net_benefit = test_net_benefit,
    model = test_model
  ))
  # Set factor order for model names
  test_dca_data$model <- factor(test_dca_data$model, levels = c("LR", "RF", "SVM", "XGBoost", "CatBoost", "LightGBM", "All", "None"))
}

## Combine model performance results of training and test cohorts
# Transpose model results of the training cohort
train_metrics_t <- train_metrics[, -3] # remove unnecessary variable
names(train_metrics_t)[c(3, 5:11)] <- c("AUC\n(95% CI)", "Sensitivity\n(95% CI)", 
                                        "Specificity\n(95% CI)", "PPV\n(95% CI)", 
                                        "NPV\n(95% CI)", "CCR\n(95% CI)", 
                                        "F1 score\n(95% CI)", "Brier score\n(95% CI)")
train_metrics_t <- tibble::rownames_to_column(as.data.frame(t(train_metrics_t)))
colnames(train_metrics_t) <- c("Metric", train_metrics_t[1, -1])
train_metrics_t <- train_metrics_t[-1, ]
# Transpose model results of the test cohort
test_metrics_t <- test_metrics[, -3] # remove unnecessary variable
names(test_metrics_t)[c(3, 5:11)] <- c("AUC\n(95% CI)", "Sensitivity\n(95% CI)", 
                                       "Specificity\n(95% CI)", "PPV\n(95% CI)", 
                                       "NPV\n(95% CI)", "CCR\n(95% CI)", 
                                       "F1 score\n(95% CI)", "Brier score\n(95% CI)")
test_metrics_t <- tibble::rownames_to_column(as.data.frame(t(test_metrics_t)))
colnames(test_metrics_t) <- c("Metric", test_metrics_t[1, -1])
test_metrics_t <- test_metrics_t[-1, ]
# Combine results (Table 2)
train_test_metrics <- rbind(train_metrics_t, test_metrics_t)

## DeLong test
# Initialize an empty data frame to store DeLong test results for the training cohort
train_auc_comparisons <- data.frame()
# Get pairwise combinations of model names in the training cohort
model_names <- names(train_roc_auc)
# Compare ROC curves pairwise and directly extract Z-scores and p-values for the training cohort
for (i in 1:(length(model_names) - 1)) {
  for (j in (i + 1):length(model_names)) {
    # Get ROC objects for the two models
    train_roc1 <- train_roc_auc[[model_names[i]]]
    train_roc2 <- train_roc_auc[[model_names[j]]]
    # DeLong test
    train_delong_test <- roc.test(train_roc1, train_roc2, method = "delong")
    # Extract Z-score and p-value, keep three decimals
    train_Z_score <- round(train_delong_test$statistic, 3)
    train_P_value <- round(train_delong_test$p.value, 3)
    # If p-value < 0.001, label as "<0.001"
    if (train_P_value < 0.001) {
      train_P_value <- "<0.001"
    } else {
      train_P_value <- train_P_value
    }
    # Add results to the data frame and combine
    train_auc_comparisons <- rbind(train_auc_comparisons, data.frame(
      Model_comparison = paste(model_names[i], model_names[j], sep = " vs "),
      train_Z_score = train_Z_score,
      train_P_value = train_P_value
    ))
  }
}

### Radar chart
## Draw radar charts for the training cohort
# Convert data to long format, suitable for radar plots
train_radar_data <- train_metrics[, c(3, 13:19)] # Select required variables
train_radar_data <- t(train_radar_data)  # Transpose data frame
train_radar_data <- as.data.frame(train_radar_data)  # Convert back to data frame
colnames(train_radar_data) <- c("LR", "RF", "SVM", "XGBoost", "CatBoost", "LightGBM") # Add column names
train_radar_data$metrics <- c("AUC", "Sensitivity", "Specificity", "PPV", "NPV", "CCR", "F1 score", "Brier score") # Add metric column
train_radar_data <- train_radar_data[, c("metrics", "LR", "RF", "SVM", "XGBoost", "CatBoost", "LightGBM")] # Reorder
train_radar_data$metrics <- factor(train_radar_data$metrics, levels=c("AUC", "Sensitivity", "Specificity", "PPV", "NPV", "CCR", "F1 score", "Brier score")) # Convert to factor
# Plot radar chart - AUC
train_AUC <- ggradar(train_radar_data[1,], 
                     grid.line.width = 0.4, # Grid line width
                     grid.label.size = 11, # Grid label font size
                     axis.label.size = 9, # Axis label font size
                     grid.min = 0, # Grid minimum
                     grid.mid = 0.4, # Grid midpoint
                     grid.max = 0.8, # Grid maximum
                     values.radar = c(0, 0.4, 0.8), # Axis label ticks
                     background.circle.colour = "white", # Background color
                     group.line.width = 0.5, # Line width
                     group.point.size = 1, # Point size
                     fill = T, # Fill
                     fill.alpha = 0.3, # Fill alpha
                     group.colours = "#95B0E0") +
  ggtitle("AUC") + # Add title
  theme(plot.title = element_text(hjust = 0.5, size = 40), # hjust=0.5 centers title; set title font size
        plot.margin = margin(0, 0, 0, 0, "cm"), # Adjust margins
        plot.tag = element_text(size = 50, face = "bold") # Top-right tag font size 
  ) +
  coord_fixed(ratio = 1) # Set square aspect ratio

# Plot radar chart - Specificity
train_Sensitivity <- ggradar(train_radar_data[2,], 
                             grid.line.width = 0.4, # Grid line width
                             grid.label.size = 11, # Grid label font size
                             axis.label.size = 9, # Axis label font size
                             grid.min = 0, # Grid minimum
                             grid.mid = 0.4, # Grid midpoint
                             grid.max = 0.8, # Grid maximum
                             values.radar = c(0, 0.4, 0.8), # Axis label ticks
                             background.circle.colour = "white", # Background color
                             group.line.width = 0.5, # Line width
                             group.point.size = 1, # Point size
                             fill = T, # Fill
                             fill.alpha = 0.3, # Fill alpha
                             group.colours = "#56AEDE") +
  ggtitle("Sensitivity") +# Add title
  theme(plot.title = element_text(hjust = 0.5, size = 40), # hjust=0.5 centers title; set title font size
        plot.margin = margin(0, 0, 0, 0, "cm"), # Adjust margins
        plot.tag = element_text(size = 50, face = "bold") # Top-right tag font size 
  ) +
  coord_fixed(ratio = 1) # Set square aspect ratio (translate Chinese comments into English only; do not change other formatting; do not make other modifications)

# Plot radar chart - Specificity
train_Specificity <- ggradar(train_radar_data[3,], 
                             grid.line.width = 0.4, # Grid line width
                             grid.label.size = 11, # Grid label font size
                             axis.label.size = 9, # Axis label font size
                             grid.min = 0, # Grid minimum
                             grid.mid = 0.4, # Grid midpoint
                             grid.max = 0.8, # Grid maximum
                             values.radar = c(0, 0.4, 0.8), # Axis label ticks
                             background.circle.colour = "white", # Background color
                             group.line.width = 0.5, # Line width
                             group.point.size = 1, # Point size
                             fill = T, # Fill
                             fill.alpha = 0.2, # Fill alpha
                             group.colours = "#EE7A5F") +
  ggtitle("Specificity") + # Add title
  theme(plot.title = element_text(hjust = 0.5, size = 40), # hjust=0.5 centers title; set title font size
        plot.margin = margin(0, 0, 0, 0, "cm"), # Adjust margins
        plot.tag = element_text(size = 50, face = "bold") # Top-right tag font size 
  ) +
  coord_fixed(ratio = 1) # Set square aspect ratio

# Plot radar chart - PPV
train_PPV <- ggradar(train_radar_data[4,], 
                     grid.line.width = 0.4, # Grid line width
                     grid.label.size = 11, # Grid label font size
                     axis.label.size = 9, # Axis label font size
                     grid.min = 0, # Grid minimum
                     grid.mid = 0.1, # Grid midpoint
                     grid.max = 0.2, # Grid maximum
                     values.radar = c(0, 0.1, 0.2), # Axis label ticks
                     background.circle.colour = "white", # Background color
                     group.line.width = 0.5, # Line width
                     group.point.size = 1, # Point size
                     fill = T, # Fill
                     fill.alpha = 0.2, # Fill alpha
                     group.colours = "#FEAF8A") +
  ggtitle("PPV") + # Add title
  theme(plot.title = element_text(hjust = 0.5, size = 40), # hjust=0.5 centers title; set title font size
        plot.margin = margin(0, 0, 0, 0, "cm"), # Adjust margins
        plot.tag = element_text(size = 50, face = "bold") # Top-right tag font size 
  ) +
  coord_fixed(ratio = 1) # Set square aspect ratio

# Plot radar chart - NPV
train_NPV <- ggradar(train_radar_data[5,], 
                     grid.line.width = 0.4, # Grid line width
                     grid.label.size = 11, # Grid label font size
                     axis.label.size = 9, # Axis label font size
                     grid.min = 0, # Grid minimum
                     grid.mid = 0.5, # Grid midpoint
                     grid.max = 1, # Grid maximum
                     values.radar = c(0, 0.5, 1), # Axis label ticks
                     background.circle.colour = "white", # Background color
                     group.line.width = 0.5, # Line width
                     group.point.size = 1, # Point size
                     fill = T, # Fill
                     fill.alpha = 0.3, # Fill alpha
                     group.colours = "#F6B7C6") +
  ggtitle("NPV") + # Add title
  theme(plot.title = element_text(hjust = 0.5, size = 40), # hjust=0.5 centers title; set title font size
        plot.margin = margin(0, 0, 0, 0, "cm"), # Adjust margins
        plot.tag = element_text(size = 50, face = "bold") # Top-right tag font size 
  ) +
  coord_fixed(ratio = 1) # Set square aspect ratio

# Plot radar chart - CCR
train_CCR <- ggradar(train_radar_data[6,], 
                     grid.line.width = 0.4, # Grid line width
                     grid.label.size = 11, # Grid label font size
                     axis.label.size = 9, # Axis label font size
                     grid.min = 0, # Grid minimum
                     grid.mid = 0.4, # Grid midpoint
                     grid.max = 0.8, # Grid maximum
                     values.radar = c(0, 0.4, 0.8), # Axis label ticks
                     background.circle.colour = "white", # Background color
                     group.line.width = 0.5, # Line width
                     group.point.size = 1, # Point size
                     fill = T, # Fill
                     fill.alpha = 0.3, # Fill alpha
                     group.colours = "#D8CBF0") +
  ggtitle("CCR") + # Add title
  theme(plot.title = element_text(hjust = 0.5, size = 40), # hjust=0.5 centers title; set title font size
        plot.margin = margin(0, 0, 0, 0, "cm"), # Adjust margins
        plot.tag = element_text(size = 50, face = "bold") # Top-right tag font size 
  ) +
  coord_fixed(ratio = 1) # Set square aspect ratio

# Plot radar chart - F1 score
train_F1_score <- ggradar(train_radar_data[7,], 
                          grid.line.width = 0.4, # Grid line width
                          grid.label.size = 11, # Grid label font size
                          axis.label.size = 9, # Axis label font size
                          grid.min = 0, # Grid minimum
                          grid.mid = 0.15, # Grid midpoint
                          grid.max = 0.3, # Grid maximum
                          values.radar = c(0, 0.15, 0.3), # Axis label ticks
                          background.circle.colour = "white", # Background color
                          group.line.width = 0.5, # Line width
                          group.point.size = 1, # Point size
                          fill = T, # Fill
                          fill.alpha = 0.3, # Fill alpha
                          group.colours = "#9FCDC9") +
  ggtitle("F1 score") + # Add title
  theme(plot.title = element_text(hjust = 0.5, size = 40), # hjust=0.5 centers title; set title font size
        plot.margin = margin(0, 0, 0, 0, "cm"), # Adjust margins
        plot.tag = element_text(size = 50, face = "bold") # Top-right tag font size 
  ) +
  coord_fixed(ratio = 1) # Set square aspect ratio

# Plot radar chart - Brier score
train_Brier_score <- ggradar(train_radar_data[8,], 
                             grid.line.width = 0.4, # Grid line width
                             grid.label.size = 11, # Grid label font size
                             axis.label.size = 9, # Axis label font size
                             grid.min = 0, # Grid minimum
                             grid.mid = 0.15, # Grid midpoint
                             grid.max = 0.3, # Grid maximum
                             values.radar = c(0, 0.15, 0.3), # Axis label ticks
                             background.circle.colour = "white", # Background color
                             group.line.width = 0.5, # Line width
                             group.point.size = 1, # Point size
                             fill = T, # Fill
                             fill.alpha = 0.2, # Fill alpha
                             group.colours = "#7EB87B") +
  ggtitle("Brier score") + # Add title
  theme(plot.title = element_text(hjust = 0.5, size = 40), # hjust=0.5 centers title; set title font size
        plot.margin = margin(0, 0, 0, 0, "cm"), # Adjust margins
        plot.tag = element_text(size = 50, face = "bold") # Top-right tag font size 
  ) +
  coord_fixed(ratio = 1) # Set square aspect ratio

# Combine radar charts of training cohort metrics
train_radar <- (train_AUC + train_Sensitivity + train_Specificity + train_PPV + 
                  train_NPV + train_CCR + train_F1_score + train_Brier_score) + 
  plot_annotation(tag_levels = "A") + # Automatically add A and B tags
  plot_layout(ncol = 4, nrow = 2) # Figure 2*4

# Export training cohort radar charts
ggsave(plot = train_radar, 
       filename = "Figure S4 train_radar.png", 
       width = 17, 
       height = 8.5,
       units = "cm", 
       dpi = 600)

## Plot radar charts for the testing cohort
# Convert data for radar plotting
test_radar_data <- test_metrics[, c(3, 13:19)] # Select required variables
test_radar_data <- t(test_radar_data)  # Transpose data frame
test_radar_data <- as.data.frame(test_radar_data)  # Convert back to data frame
colnames(test_radar_data) <- c("LR", "RF", "SVM", "XGBoost", "CatBoost", "LightGBM") # Add column names
test_radar_data$metrics <- c("AUC", "Sensitivity", "Specificity", "PPV", "NPV", "CCR", "F1 score", "Brier score") # Add metric column
test_radar_data <- test_radar_data[, c("metrics", "LR", "RF", "SVM", "XGBoost", "CatBoost", "LightGBM")] # Reorder
test_radar_data$metrics <- factor(test_radar_data$metrics, levels=c("AUC", "Sensitivity", "Specificity", "PPV", "NPV", "CCR", "F1 score", "Brier score")) # Convert to factor
# Plot radar chart - AUC
test_AUC <- ggradar(test_radar_data[1,], 
                    grid.line.width = 0.4, # Grid line width
                    grid.label.size = 11, # Grid label font size
                    axis.label.size = 9, # Axis label font size
                    grid.min = 0, # Grid minimum
                    grid.mid = 0.4, # Grid midpoint
                    grid.max = 0.8, # Grid maximum
                    values.radar = c(0, 0.4, 0.8), # Axis label ticks
                    background.circle.colour = "white", # Background color
                    group.line.width = 0.5, # Line width
                    group.point.size = 1, # Point size
                    fill = T, # Fill
                    fill.alpha = 0.3, # Fill alpha
                    group.colours = "#95B0E0") +
  ggtitle("AUC") + # Add title
  theme(plot.title = element_text(hjust = 0.5, size = 40), # hjust=0.5 centers title; set title font size
        plot.margin = margin(0, 0, 0, 0, "cm"), # Adjust margins
        plot.tag = element_text(size = 50, face = "bold") # Top-right tag font size 
  ) +
  coord_fixed(ratio = 1) # Set square aspect ratio

# Plot radar chart - Specificity
test_Sensitivity <- ggradar(test_radar_data[2,], 
                            grid.line.width = 0.4, # Grid line width
                            grid.label.size = 11, # Grid label font size
                            axis.label.size = 9, # Axis label font size
                            grid.min = 0, # Grid minimum
                            grid.mid = 0.4, # Grid midpoint
                            grid.max = 0.8, # Grid maximum
                            values.radar = c(0, 0.4, 0.8), # Axis label ticks
                            background.circle.colour = "white", # Background color
                            group.line.width = 0.5, # Line width
                            group.point.size = 1, # Point size
                            fill = T, # Fill
                            fill.alpha = 0.3, # Fill alpha
                            group.colours = "#56AEDE") +
  ggtitle("Sensitivity") +# Add title
  theme(plot.title = element_text(hjust = 0.5, size = 40), # hjust=0.5 centers title; set title font size
        plot.margin = margin(0, 0, 0, 0, "cm"), # Adjust margins
        plot.tag = element_text(size = 50, face = "bold") # Top-right tag font size 
  ) +
  coord_fixed(ratio = 1) # Set square aspect ratio

# Plot radar chart - Specificity
test_Specificity <- ggradar(test_radar_data[3,], 
                            grid.line.width = 0.4, # Grid line width
                            grid.label.size = 11, # Grid label font size
                            axis.label.size = 9, # Axis label font size
                            grid.min = 0, # Grid minimum
                            grid.mid = 0.4, # Grid midpoint
                            grid.max = 0.8, # Grid maximum
                            values.radar = c(0, 0.4, 0.8), # Axis label ticks
                            background.circle.colour = "white", # Background color
                            group.line.width = 0.5, # Line width
                            group.point.size = 1, # Point size
                            fill = T, # Fill
                            fill.alpha = 0.2, # Fill alpha
                            group.colours = "#EE7A5F") +
  ggtitle("Specificity") + # Add title
  theme(plot.title = element_text(hjust = 0.5, size = 40), # hjust=0.5 centers title; set title font size
        plot.margin = margin(0, 0, 0, 0, "cm"), # Adjust margins
        plot.tag = element_text(size = 50, face = "bold") # Top-right tag font size 
  ) +
  coord_fixed(ratio = 1) # Set square aspect ratio

# Plot radar chart - PPV
test_PPV <- ggradar(test_radar_data[4,], 
                    grid.line.width = 0.4, # Grid line width
                    grid.label.size = 11, # Grid label font size
                    axis.label.size = 9, # Axis label font size
                    grid.min = 0, # Grid minimum
                    grid.mid = 0.1, # Grid midpoint
                    grid.max = 0.2, # Grid maximum
                    values.radar = c(0, 0.1, 0.2), # Axis label ticks
                    background.circle.colour = "white", # Background color
                    group.line.width = 0.5, # Line width
                    group.point.size = 1, # Point size
                    fill = T, # Fill
                    fill.alpha = 0.2, # Fill alpha
                    group.colours = "#FEAF8A") +
  ggtitle("PPV") + # Add title
  theme(plot.title = element_text(hjust = 0.5, size = 40), # hjust=0.5 centers title; set title font size
        plot.margin = margin(0, 0, 0, 0, "cm"), # Adjust margins
        plot.tag = element_text(size = 50, face = "bold") # Top-right tag font size 
  ) +
  coord_fixed(ratio = 1) # Set square aspect ratio

# Plot radar chart - NPV
test_NPV <- ggradar(test_radar_data[5,], 
                    grid.line.width = 0.4, # Grid line width
                    grid.label.size = 11, # Grid label font size
                    axis.label.size = 9, # Axis label font size
                    grid.min = 0, # Grid minimum
                    grid.mid = 0.5, # Grid midpoint
                    grid.max = 1, # Grid maximum
                    values.radar = c(0, 0.5, 1), # Axis label ticks
                    background.circle.colour = "white", # Background color
                    group.line.width = 0.5, # Line width
                    group.point.size = 1, # Point size
                    fill = T, # Fill
                    fill.alpha = 0.3, # Fill alpha
                    group.colours = "#F6B7C6") +
  ggtitle("NPV") + # Add title
  theme(plot.title = element_text(hjust = 0.5, size = 40), # hjust=0.5 centers title; set title font size
        plot.margin = margin(0, 0, 0, 0, "cm"), # Adjust margins
        plot.tag = element_text(size = 50, face = "bold") # Top-right tag font size 
  ) +
  coord_fixed(ratio = 1) # Set square aspect ratio

# Plot radar chart - CCR
test_CCR <- ggradar(test_radar_data[6,], 
                    grid.line.width = 0.4, # Grid line width
                    grid.label.size = 11, # Grid label font size
                    axis.label.size = 9, # Axis label font size
                    grid.min = 0, # Grid minimum
                    grid.mid = 0.4, # Grid midpoint
                    grid.max = 0.8, # Grid maximum
                    values.radar = c(0, 0.4, 0.8), # Axis label ticks
                    background.circle.colour = "white", # Background color
                    group.line.width = 0.5, # Line width
                    group.point.size = 1, # Point size
                    fill = T, # Fill
                    fill.alpha = 0.3, # Fill alpha
                    group.colours = "#D8CBF0") +
  ggtitle("CCR") + # Add title
  theme(plot.title = element_text(hjust = 0.5, size = 40), # hjust=0.5 centers title; set title font size
        plot.margin = margin(0, 0, 0, 0, "cm"), # Adjust margins
        plot.tag = element_text(size = 50, face = "bold") # Top-right tag font size 
  ) +
  coord_fixed(ratio = 1) # Set square aspect ratio

# Plot radar chart - F1 score
test_F1_score <- ggradar(test_radar_data[7,], 
                         grid.line.width = 0.4, # Grid line width
                         grid.label.size = 11, # Grid label font size
                         axis.label.size = 9, # Axis label font size
                         grid.min = 0, # Grid minimum
                         grid.mid = 0.15, # Grid midpoint
                         grid.max = 0.3, # Grid maximum
                         values.radar = c(0, 0.15, 0.3), # Axis label ticks
                         background.circle.colour = "white", # Background color
                         group.line.width = 0.5, # Line width
                         group.point.size = 1, # Point size
                         fill = T, # Fill
                         fill.alpha = 0.3, # Fill alpha
                         group.colours = "#9FCDC9") +
  ggtitle("F1 score") + # Add title
  theme(plot.title = element_text(hjust = 0.5, size = 40), # hjust=0.5 centers title; set title font size
        plot.margin = margin(0, 0, 0, 0, "cm"), # Adjust margins
        plot.tag = element_text(size = 50, face = "bold") # Top-right tag font size 
  ) +
  coord_fixed(ratio = 1) # Set square aspect ratio

# Plot radar chart - Brier score
test_Brier_score <- ggradar(test_radar_data[8,], 
                            grid.line.width = 0.4, # Grid line width
                            grid.label.size = 11, # Grid label font size
                            axis.label.size = 9, # Axis label font size
                            grid.min = 0, # Grid minimum
                            grid.mid = 0.15, # Grid midpoint
                            grid.max = 0.3, # Grid maximum
                            values.radar = c(0, 0.15, 0.3), # Axis label ticks
                            background.circle.colour = "white", # Background color
                            group.line.width = 0.5, # Line width
                            group.point.size = 1, # Point size
                            fill = T, # Fill
                            fill.alpha = 0.2, # Fill alpha
                            group.colours = "#7EB87B") +
  ggtitle("Brier score") + # Add title
  theme(plot.title = element_text(hjust = 0.5, size = 40), # hjust=0.5 centers title; set title font size
        plot.margin = margin(0, 0, 0, 0, "cm"), # Adjust margins
        plot.tag = element_text(size = 50, face = "bold") # Top-right tag font size 
  ) +
  coord_fixed(ratio = 1) # Set square aspect ratio

# Combine radar charts for the test cohort
test_radar <- (test_AUC + test_Sensitivity + test_Specificity + test_PPV + 
                 test_NPV + test_CCR + test_F1_score + test_Brier_score) + 
  plot_annotation(tag_levels = "A") + # automatically add A and B tags
  plot_layout(ncol = 4, nrow = 2) # 2x4 grid

# Export radar charts for the test cohort
ggsave(plot = test_radar, 
       filename = "Figure S7 test_radar.png", 
       width = 17, 
       height = 8.5,
       units = "cm", 
       dpi = 600)

# Initialize empty data frames to store DeLong test results for train and test cohorts
train_auc_comparisons <- data.frame()
test_auc_comparisons <- data.frame()
# Get combinations of model names for the training cohort
model_names <- names(train_roc_auc)

# P value format
pvalue_format_vec <- Vectorize(function(p) {
  if (is.na(p)) return("")
  
  if (p < 0.001) {
    return("<.001") 
  } else if (p < 0.01) {
    return(sub("^0", "", sprintf("%.3f", p)))
  } else {
    return(sub("^0", "", sprintf("%.2f", p)))
  }
})

# Pairwise ROC comparisons and extract Z and p-value for the training cohort
for (i in 1:(length(model_names) - 1)) {
  for (j in (i + 1):length(model_names)) {
    # Get ROC objects of the two models
    train_roc1 <- train_roc_auc[[model_names[i]]]
    train_roc2 <- train_roc_auc[[model_names[j]]]
    # DeLong test
    train_delong_test <- roc.test(train_roc1, train_roc2, method = "delong")
    # Extract Z and p-value (Z rounded to 3 decimals; p formatted with custom function)
    train_Z_score <- round(train_delong_test$statistic, 3)
    train_P_value <- pvalue_format_vec(train_delong_test$p.value)
    
    # Append results to data frame
    train_auc_comparisons <- rbind(train_auc_comparisons, data.frame(
      Model_comparison = paste(model_names[i], model_names[j], sep = " vs "),
      train_Z_score = train_Z_score,
      train_P_value = train_P_value
    ))
  }
}

# Pairwise ROC comparisons and extract Z and p-value for the test cohort
for (i in 1:(length(model_names) - 1)) {
  for (j in (i + 1):length(model_names)) {
    # Get ROC objects of the two models
    test_roc1 <- test_roc_auc[[model_names[i]]]
    test_roc2 <- test_roc_auc[[model_names[j]]]
    # DeLong test
    test_delong_test <- roc.test(test_roc1, test_roc2, method = "delong")
    # Extract Z and p-value (Z rounded to 3 decimals; p formatted with custom function)
    test_Z_score <- round(test_delong_test$statistic, 3)
    test_P_value <- pvalue_format_vec(test_delong_test$p.value)
    
    # Append results to data frame
    test_auc_comparisons <- rbind(test_auc_comparisons, data.frame(
      Model_comparison = paste(model_names[i], model_names[j], sep = " vs "),
      test_Z_score = test_Z_score,
      test_P_value = test_P_value
    ))
  }
}

# Merge DeLong test results of train and test cohorts (Table S6)
train_test_roc_comparisons <- merge(train_auc_comparisons, test_auc_comparisons, by = "Model_comparison")

### Plot calibration curves
## Plot calibration curves for the training cohort
# Extract training task data
train_task_data <- train_task$data()
train_task_data$RDKF <- factor(ifelse(train_task_data$RDKF == "2", 1, 0))

# Score models
train_score <- Score(list(LR = train_prob_data[["LR"]],
                          RF = train_prob_data[["RF"]],
                          SVM = train_prob_data[["SVM"]],
                          XGBoost = train_prob_data[["XGBoost"]],
                          CatBoost = train_prob_data[["CatBoost"]],
                          LightGBM = train_prob_data[["LightGBM"]]),
                     formula = RDKF ~ 1, # Model evaluation formula
                     null.model = F, # Do not compare with a null model
                     plots = "calibration", # Draw calibration curves
                     data = train_task_data)

# Extract calibration curve data
train_calibration_plot <- plotCalibration(train_score, plot = FALSE)
train_calibration_data <- imap_dfr(train_calibration_plot$plotFrames, ~ {
  .x %>% 
    as_tibble() %>% 
    mutate(model = .y)  # .y is the list element name (lr/rf/svm/etc.)
})
train_calibration_data$model <- factor(train_calibration_data$model, levels = c("LR", "RF", "SVM", "XGBoost", "CatBoost", "LightGBM"))

# Create a custom segment for the ideal calibration line
segment_data <- data.frame(
  x = 0, y = 0, xend = 1,  yend = 1, 
  segment_type = "Ideal"  # used to label the line in legend
)

# Plot calibration curves for the training cohort
train_calibration <- ggplot(train_calibration_data, aes(x = Pred, y = Obs, color = model)) +
  geom_line(linewidth = 0.5) + # set line width
  # draw the ideal calibration line with geom_segment()
  geom_segment(data = segment_data, aes(x = x, y = y, xend = xend, yend = yend, color = segment_type), linewidth = 0.5, linetype = "dotted") +
  scale_x_continuous(limits = c(0, 1),  # limit x-axis range
                     breaks = seq(0, 1, by = 0.2), name = "Predicted Probability") + # set x-axis breaks and label
  scale_y_continuous(limits = c(0, 1),  # limit y-axis range
                     breaks = seq(0, 1, by = 0.2), name = "Actual Probability") + # set y-axis breaks and label
  # customize colors and legend labels
  scale_color_manual(
    values = c("#8DD3C7", "#BEBADA", "#FB8072", "#80B1D3", "#FDB462", "#B3DE69", "black"),
    labels = c("LR", "RF", "SVM", "XGBoost", "CatBoost", "LightGBM", "Ideal")
  ) +
  labs(color = "") + # empty legend title
  theme_minimal() + # set plot theme
  theme(
    plot.background = element_rect(fill = "white", color = NA), # overall background; no border color
    panel.grid = element_blank(), # remove grid lines
    panel.border = element_rect(color = "black", fill = NA), # add border
    axis.ticks.x = element_line(color = "black"), # x-axis tick marks
    axis.ticks.y = element_line(color = "black"), # y-axis tick marks
    axis.ticks.length = unit(0.1, "cm"), # tick marks outside and their length
    legend.position = c(0.17, 0.82), # place legend at top-left corner
    legend.key.height = unit(0.45, "cm"), # control each legend key height
    axis.title = element_text(size = 80), # axis title font size
    axis.text = element_text(size = 70), # axis tick label font size
    legend.text = element_text(size = 50), # legend text size
    plot.tag = element_text(size = 90, face = "bold") # set tag text size
  ) +
  coord_fixed(ratio = 1) # make plot square

## Plot calibration curves for the test cohort
# Extract test cohort task data
test_task_data <- test_task$data()
test_task_data$RDKF <- factor(ifelse(test_task_data$RDKF == "2", 1, 0))

# Score models
test_score <- Score(list(LR = test_prob_data[["LR"]],
                         RF = test_prob_data[["RF"]],
                         SVM = test_prob_data[["SVM"]],
                         XGBoost = test_prob_data[["XGBoost"]],
                         CatBoost = test_prob_data[["CatBoost"]],
                         LightGBM = test_prob_data[["LightGBM"]]),
                    formula = RDKF ~ 1, # model evaluation formula
                    null.model = F, # do not compare with a null model
                    plots = "calibration", # draw calibration curves
                    data = test_task_data)

# Extract calibration curve data
test_calibration_plot <- plotCalibration(test_score, plot = FALSE)
test_calibration_data <- imap_dfr(test_calibration_plot$plotFrames, ~ {
  .x %>% 
    as_tibble() %>% 
    mutate(model = .y)  # .y is the list element name (lr/rf/svm etc.)
})
test_calibration_data$model <- factor(test_calibration_data$model, levels = c("LR", "RF", "SVM", "XGBoost", "CatBoost", "LightGBM"))

# Plot calibration curves for the test cohort
test_calibration <- ggplot(test_calibration_data, aes(x = Pred, y = Obs, color = model)) +
  geom_line(linewidth = 0.5) + # set line width
  # draw the ideal calibration line with geom_segment()
  geom_segment(data = segment_data, aes(x = x, y = y, xend = xend, yend = yend, color = segment_type), linewidth = 0.5, linetype = "dotted") +
  scale_x_continuous(limits = c(0, 1),  # limit x-axis range
                     breaks = seq(0, 1, by = 0.2), name = "Predicted Probability") + # set x-axis breaks and label
  scale_y_continuous(limits = c(0, 1),  # limit y-axis range
                     breaks = seq(0, 1, by = 0.2), name = "Actual Probability") + # set y-axis breaks and label
  # customize colors and legend labels
  scale_color_manual(
    values = c("#8DD3C7", "#BEBADA", "#FB8072", "#80B1D3", "#FDB462", "#B3DE69", "black"),
    labels = c("LR", "RF", "SVM", "XGBoost", "CatBoost", "LightGBM", "Ideal")
  ) +
  labs(color = "") + # empty legend title
  theme_minimal() + # set plot theme
  theme(
    plot.background = element_rect(fill = "white", color = NA), # overall background; no border color
    panel.grid = element_blank(), # remove grid lines
    panel.border = element_rect(color = "black", fill = NA), # add border
    axis.ticks.x = element_line(color = "black"), # x-axis tick marks
    axis.ticks.y = element_line(color = "black"), # y-axis tick marks
    axis.ticks.length = unit(0.1, "cm"), # tick marks outside and their length
    legend.position = c(0.17, 0.82), # place legend at top-left corner
    legend.key.height = unit(0.45, "cm"), # control each legend key height
    axis.title = element_text(size = 80), # axis title font size
    axis.text = element_text(size = 70), # axis tick label font size
    legend.text = element_text(size = 50), # legend text size
    plot.tag = element_text(size = 90, face = "bold") # set tag text size
  ) +
  coord_fixed(ratio = 1) # make plot square

# Combine calibration plots for training and test cohorts
train_test_calibration <- (train_calibration + test_calibration) + 
  plot_annotation(tag_levels = "A") # automatically add A and B tags 

# Export calibration plots for training and test cohorts
ggsave(plot = train_test_calibration, 
       filename = "Figure S5 train_test_calibration.png", 
       width = 23, 
       height = 11.5,
       units = "cm", 
       dpi = 600)

## Plot DCA
# Plot DCA for the training cohort
train_dca <- ggplot(train_dca_data, aes(x = threshold, y = net_benefit, color = model)) +
  geom_line(linewidth = 0.5) + # set line width
  scale_x_continuous(limits = c(0, 0.6), # limit x-axis range
                     breaks = seq(0, 0.6, by = 0.2), # set x-axis breaks
                     name = "High Risk Threshold") + # set x-axis label
  scale_y_continuous(limits = c(-0.2, 0.8), # limit y-axis range
                     breaks = seq(-0.2, 0.8, by = 0.2), # set y-axis breaks
                     name = "Standardized Net Benefit") + # set y-axis label
  scale_color_manual(values = c("#8DD3C7", "#BEBADA", "#FB8072", "#80B1D3", 
                                "#FDB462", "#B3DE69", "gray", "black")) + # custom colors
  theme_minimal() + # set plot theme
  theme(
    panel.grid = element_blank(), # remove grid lines
    panel.border = element_rect(color = "black", fill = NA), # add border
    axis.ticks.x = element_line(color = "black"), # x-axis tick marks
    axis.ticks.y = element_line(color = "black"), # y-axis tick marks
    axis.ticks.length = unit(0.1, "cm"), # tick marks outside and length
    legend.title = element_blank(), # remove legend title
    legend.position = c(0.85, 0.75), # place legend at top-right
    legend.key.height = unit(0.45, "cm"), # control each legend key height
    axis.title = element_text(size = 80), # axis title font size
    axis.text = element_text(size = 70), # axis tick label font size
    legend.text = element_text(size = 50), # legend text size
    plot.tag = element_text(size = 90, face = "bold") # set tag text size
  ) +
  coord_fixed(ratio = 2/4) # make plot square

# Plot DCA for the test cohort
test_dca <- ggplot(test_dca_data, aes(x = threshold, y = net_benefit, color = model)) +
  geom_line(linewidth = 0.5) + # set line width
  scale_x_continuous(limits = c(0, 0.6), # limit x-axis range
                     breaks = seq(0, 0.6, by = 0.2), # set x-axis breaks
                     name = "High Risk Threshold") + # set x-axis label
  scale_y_continuous(limits = c(-0.2, 0.8), # limit y-axis range
                     breaks = seq(-0.2, 0.8, by = 0.2), # set y-axis breaks
                     name = "Standardized Net Benefit") + # set y-axis label
  scale_color_manual(values = c("#8DD3C7", "#BEBADA", "#FB8072", "#80B1D3", 
                                "#FDB462", "#B3DE69", "gray", "black")) + # custom colors
  theme_minimal() + # set plot theme
  theme(
    panel.grid = element_blank(), # remove grid lines
    panel.border = element_rect(color = "black", fill = NA), # add border
    axis.ticks.x = element_line(color = "black"), # x-axis tick marks
    axis.ticks.y = element_line(color = "black"), # y-axis tick marks
    axis.ticks.length = unit(0.1, "cm"), # tick marks outside and length
    legend.title = element_blank(), # remove legend title
    legend.position = c(0.85, 0.75), # place legend at top-right
    legend.key.height = unit(0.45, "cm"), # control each legend key height
    axis.title = element_text(size = 80), # axis title font size
    axis.text = element_text(size = 70), # axis tick label font size
    legend.text = element_text(size = 50), # legend text size
    plot.tag = element_text(size = 90, face = "bold") # set tag text size
  ) +
  coord_fixed(ratio = 2/4) # make plot square

# Combine DCA plots for training and test cohorts
train_test_dca <- (train_dca + test_dca) + 
  plot_annotation(tag_levels = "A") # automatically add A and B tags 

# Export DCA plots for training and test cohorts
ggsave(plot = train_test_dca, 
       filename = "Figure S6 train_test_dca.png", 
       width = 23, 
       height = 10,
       units = "cm", 
       dpi = 600)
