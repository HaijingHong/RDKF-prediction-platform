##------------------Sensitivity analysis------------------
#---------(1) Sensitivity dataset 1 (complete data)
sensitivity1 <- na.omit(rckd3)
#Standardize sensitivity dataset 1 using the training cohort's standardization parameters
sensitivity1_standardized <- predict(standardized_para, newdata = sensitivity1)
#Convert all factor variables to numeric
sensitivity1_standardized <- sensitivity1_standardized %>% 
  mutate(across(where(is.factor), ~ as.numeric(.)))
#Convert outcome variable RDKF to factor
sensitivity1_standardized$RDKF <- factor(sensitivity1_standardized$RDKF)
#Extract selected features and outcome
sensitivity1_boruta <- sensitivity1_standardized[, c(confirmed_features, "RDKF")]
##Create task for sensitivity dataset 1
sensitivity1_task <- as_task_classif(sensitivity1_boruta, target = "RDKF")
#Predict on sensitivity dataset 1
sensitivity1_lr_pred <- lr_learner$predict(sensitivity1_task)
sensitivity1_rf_pred <- rf_learner$predict(sensitivity1_task)
sensitivity1_svm_pred <- svm_learner$predict(sensitivity1_task)
sensitivity1_xgb_pred <- xgb_learner$predict(sensitivity1_task)
sensitivity1_catb_pred <- catb_learner$predict(sensitivity1_task)
sensitivity1_lightgbm_pred <- lightgbm_learner$predict(sensitivity1_task)
#Create a list of model predictions for sensitivity dataset 1
sensitivity1_preds <- list(
  LR = sensitivity1_lr_pred, 
  RF = sensitivity1_rf_pred, 
  SVM = sensitivity1_svm_pred, 
  XGBoost = sensitivity1_xgb_pred, 
  CatBoost = sensitivity1_catb_pred, 
  LightGBM = sensitivity1_lightgbm_pred
)
#Create an empty data frame to store performance metrics for sensitivity dataset 1
sensitivity1_metrics <- data.frame()
#Compute performance metrics for sensitivity dataset 1
for (model_name in names(sensitivity1_preds)) {
  pred_sensitivity1 <- sensitivity1_preds[[model_name]]
  pred_prob_sensitivity1 <- pred_sensitivity1$prob[, "2"]
  true_class_sensitivity1 <- pred_sensitivity1$truth
  true_class_sensitivity1 <- ifelse(true_class_sensitivity1 == "2", 1, 0)
  roc_sensitivity1 <- roc(true_class_sensitivity1, pred_prob_sensitivity1)
  auc_sensitivity1 <- roc_sensitivity1$auc
  ci_sensitivity1 <- ci(roc_sensitivity1)
  #Combine AUC and 95% CI, keep three decimals
  auc_with_ci_sensitivity1 <- sprintf("%.3f\n(%.3f-%.3f)", auc_sensitivity1, ci_sensitivity1[1], ci_sensitivity1[3]) 
  
  #Get the best threshold from the training cohort
  best_threshold_train <- train_metrics[train_metrics$Model == model_name, "Threshold"]
  #Use the best threshold from the training cohort to compute the performance of the test cohort
  best_metrics_sensitivity1 <- calculate_metrics(pred_prob_sensitivity1, true_class_sensitivity1, best_threshold_train)
  
  #Compute Brier score
  brier_score_sensitivity1 <- mean((pred_prob_sensitivity1 - true_class_sensitivity1)^2)
  
  #Bootstrap CI
  set.seed(123)
  ci_sensitivity1_metrics <- bootstrap_ci(pred_prob_sensitivity1, true_class_sensitivity1, best_threshold_train, B = 1000)
  
  Sensitivity_CI  <- format_ci(best_metrics_sensitivity1$Sensitivity,  ci_sensitivity1_metrics[, "Sensitivity"])
  Specificity_CI  <- format_ci(best_metrics_sensitivity1$Specificity,  ci_sensitivity1_metrics[, "Specificity"])
  PPV_CI          <- format_ci(best_metrics_sensitivity1$PPV,          ci_sensitivity1_metrics[, "PPV"])
  NPV_CI          <- format_ci(best_metrics_sensitivity1$NPV,          ci_sensitivity1_metrics[, "NPV"])
  CCR_CI          <- format_ci(best_metrics_sensitivity1$CCR,          ci_sensitivity1_metrics[, "CCR"])
  F1_CI           <- format_ci(best_metrics_sensitivity1$F1_score,     ci_sensitivity1_metrics[, "F1_score"])
  Brier_CI        <- format_ci(brier_score_sensitivity1,               ci_sensitivity1_metrics[, "Brier_score"])
  
  sensitivity1_metrics_result <- data.frame(
    Model = model_name,
    Dataset = "sensitivity1",
    AUC_CI = auc_with_ci_sensitivity1, 
    Sensitivity_CI = Sensitivity_CI,
    Specificity_CI = Specificity_CI,
    PPV_CI = PPV_CI,
    NPV_CI = NPV_CI,
    CCR_CI = CCR_CI,
    F1_CI = F1_CI,
    Brier_CI = Brier_CI
  )
  sensitivity1_metrics <- rbind(sensitivity1_metrics, sensitivity1_metrics_result)
}

#---------(2) Sensitivity dataset 2 (RDKF imputed as No)
sensitivity2 <- rckd2 %>%
  mutate(RDKF = ifelse(is.na(RDKF), 1, RDKF))
sensitivity2$RDKF <- factor(sensitivity2$RDKF, levels=c(1, 2), labels=c("No", "Yes"))
#Random forest imputation for missing values
#Set seed to ensure reproducibility of imputation
set.seed(10)
#Use missForest() to impute missing values for rckd3
data_missresult2 <- missForest(sensitivity2)
#Extract imputed data and convert to data frame
imputed_data2 <- as.data.frame(data_missresult2$ximp)
#Standardize sensitivity dataset 2 using the training cohort's standardization parameters
sensitivity2_standardized <- predict(standardized_para, newdata = imputed_data2)
#Convert all factor variables to numeric
sensitivity2_standardized <- sensitivity2_standardized %>% 
  mutate(across(where(is.factor), ~ as.numeric(.)))
#Convert outcome variable RDKF to factor
sensitivity2_standardized$RDKF <- factor(sensitivity2_standardized$RDKF)
#Extract selected features and outcome
sensitivity2_boruta <- sensitivity2_standardized[, c(confirmed_features, "RDKF")]
##Create task for sensitivity dataset 2
sensitivity2_task <- as_task_classif(sensitivity2_boruta, target = "RDKF")
#Predict on sensitivity dataset 1
sensitivity2_lr_pred <- lr_learner$predict(sensitivity2_task)
sensitivity2_rf_pred <- rf_learner$predict(sensitivity2_task)
sensitivity2_svm_pred <- svm_learner$predict(sensitivity2_task)
sensitivity2_xgb_pred <- xgb_learner$predict(sensitivity2_task)
sensitivity2_catb_pred <- catb_learner$predict(sensitivity2_task)
sensitivity2_lightgbm_pred <- lightgbm_learner$predict(sensitivity2_task)
#Create a list of model predictions for sensitivity dataset 2
sensitivity2_preds <- list(
  LR = sensitivity2_lr_pred, 
  RF = sensitivity2_rf_pred, 
  SVM = sensitivity2_svm_pred, 
  XGBoost = sensitivity2_xgb_pred, 
  CatBoost = sensitivity2_catb_pred, 
  LightGBM = sensitivity2_lightgbm_pred
)
#Create an empty data frame to store performance metrics for sensitivity dataset 2
sensitivity2_metrics <- data.frame()
#Compute performance metrics for sensitivity dataset 2
for (model_name in names(sensitivity2_preds)) {
  pred_sensitivity2 <- sensitivity2_preds[[model_name]]
  pred_prob_sensitivity2 <- pred_sensitivity2$prob[, "2"]
  true_class_sensitivity2 <- pred_sensitivity2$truth
  true_class_sensitivity2 <- ifelse(true_class_sensitivity2 == "2", 1, 0)
  roc_sensitivity2 <- roc(true_class_sensitivity2, pred_prob_sensitivity2)
  auc_sensitivity2 <- roc_sensitivity2$auc
  ci_sensitivity2 <- ci(roc_sensitivity2)
  #Combine AUC and 95% CI, keep three decimals
  auc_with_ci_sensitivity2 <- sprintf("%.3f\n(%.3f-%.3f)", auc_sensitivity2, ci_sensitivity2[1], ci_sensitivity2[3]) 
  
  #Get the best threshold from the training cohort
  best_threshold_train <- train_metrics[train_metrics$Model == model_name, "Threshold"]
  #Use the best threshold from the training cohort to compute the performance of the test cohort
  best_metrics_sensitivity2 <- calculate_metrics(pred_prob_sensitivity2, true_class_sensitivity2, best_threshold_train)
  
  #Compute Brier score
  brier_score_sensitivity2 <- mean((pred_prob_sensitivity2 - true_class_sensitivity2)^2)
  
  #Bootstrap CI
  set.seed(123)
  ci_sensitivity2_metrics <- bootstrap_ci(pred_prob_sensitivity2, true_class_sensitivity2, best_threshold_train, B = 1000)
  
  Sensitivity_CI  <- format_ci(best_metrics_sensitivity2$Sensitivity,  ci_sensitivity2_metrics[, "Sensitivity"])
  Specificity_CI  <- format_ci(best_metrics_sensitivity2$Specificity,  ci_sensitivity2_metrics[, "Specificity"])
  PPV_CI          <- format_ci(best_metrics_sensitivity2$PPV,          ci_sensitivity2_metrics[, "PPV"])
  NPV_CI          <- format_ci(best_metrics_sensitivity2$NPV,          ci_sensitivity2_metrics[, "NPV"])
  CCR_CI          <- format_ci(best_metrics_sensitivity2$CCR,          ci_sensitivity2_metrics[, "CCR"])
  F1_CI           <- format_ci(best_metrics_sensitivity2$F1_score,     ci_sensitivity2_metrics[, "F1_score"])
  Brier_CI        <- format_ci(brier_score_sensitivity2,               ci_sensitivity2_metrics[, "Brier_score"])
  
  sensitivity2_metrics_result <- data.frame(
    Model = model_name,
    Dataset = "sensitivity2",
    AUC_CI = auc_with_ci_sensitivity2, 
    Sensitivity_CI = Sensitivity_CI,
    Specificity_CI = Specificity_CI,
    PPV_CI = PPV_CI,
    NPV_CI = NPV_CI,
    CCR_CI = CCR_CI,
    F1_CI = F1_CI,
    Brier_CI = Brier_CI
  )
  sensitivity2_metrics <- rbind(sensitivity2_metrics, sensitivity2_metrics_result)
}

#---------(3) Sensitivity dataset 3 (RDKF imputed as Yes)
sensitivity3 <- rckd2 %>%
  mutate(RDKF = ifelse(is.na(RDKF), 2, RDKF))
sensitivity3$RDKF <- factor(sensitivity3$RDKF, levels=c(1, 2), labels=c("No", "Yes"))
#Random forest imputation for missing values
#Set seed to ensure reproducibility of imputation
set.seed(10)
#Use missForest() to impute missing values for sensitivity3
data_missresult3 <- missForest(sensitivity3)
#Extract imputed data and convert to data frame
imputed_data3 <- as.data.frame(data_missresult3$ximp)
#Standardize sensitivity dataset 3 using the training cohort's standardization parameters
sensitivity3_standardized <- predict(standardized_para, newdata = imputed_data3)
#Convert all factor variables to numeric
sensitivity3_standardized <- sensitivity3_standardized %>% 
  mutate(across(where(is.factor), ~ as.numeric(.)))
#Convert outcome variable RDKF to factor
sensitivity3_standardized$RDKF <- factor(sensitivity3_standardized$RDKF)
#Extract selected features and outcome
sensitivity3_boruta <- sensitivity3_standardized[, c(confirmed_features, "RDKF")]
##Create task for sensitivity dataset 3
sensitivity3_task <- as_task_classif(sensitivity3_boruta, target = "RDKF")
#Predict on sensitivity dataset 3
sensitivity3_lr_pred <- lr_learner$predict(sensitivity3_task)
sensitivity3_rf_pred <- rf_learner$predict(sensitivity3_task)
sensitivity3_svm_pred <- svm_learner$predict(sensitivity3_task)
sensitivity3_xgb_pred <- xgb_learner$predict(sensitivity3_task)
sensitivity3_catb_pred <- catb_learner$predict(sensitivity3_task)
sensitivity3_lightgbm_pred <- lightgbm_learner$predict(sensitivity3_task)
#Create a list of model predictions for sensitivity dataset 3
sensitivity3_preds <- list(
  LR = sensitivity3_lr_pred, 
  RF = sensitivity3_rf_pred, 
  SVM = sensitivity3_svm_pred, 
  XGBoost = sensitivity3_xgb_pred, 
  CatBoost = sensitivity3_catb_pred, 
  LightGBM = sensitivity3_lightgbm_pred
)
#Create an empty data frame to store performance metrics for sensitivity dataset 3
sensitivity3_metrics <- data.frame()
#Compute performance metrics for sensitivity dataset 3
for (model_name in names(sensitivity3_preds)) {
  pred_sensitivity3 <- sensitivity3_preds[[model_name]]
  pred_prob_sensitivity3 <- pred_sensitivity3$prob[, "2"]
  true_class_sensitivity3 <- pred_sensitivity3$truth
  true_class_sensitivity3 <- ifelse(true_class_sensitivity3 == "2", 1, 0)
  roc_sensitivity3 <- roc(true_class_sensitivity3, pred_prob_sensitivity3)
  auc_sensitivity3 <- roc_sensitivity3$auc
  ci_sensitivity3 <- ci(roc_sensitivity3)
  #Combine AUC and 95% CI, keep three decimals
  auc_with_ci_sensitivity3 <- sprintf("%.3f\n(%.3f-%.3f)", auc_sensitivity3, ci_sensitivity3[1], ci_sensitivity3[3]) 
  
  #Get the best threshold from the training cohort
  best_threshold_train <- train_metrics[train_metrics$Model == model_name, "Threshold"]
  #Use the best threshold from the training cohort to compute the performance of the test cohort
  best_metrics_sensitivity3 <- calculate_metrics(pred_prob_sensitivity3, true_class_sensitivity3, best_threshold_train)
  
  #Compute Brier score
  brier_score_sensitivity3 <- mean((pred_prob_sensitivity3 - true_class_sensitivity3)^2)
  
  #Bootstrap CI
  set.seed(123)
  ci_sensitivity3_metrics <- bootstrap_ci(pred_prob_sensitivity3, true_class_sensitivity3, best_threshold_train, B = 1000)
  
  Sensitivity_CI  <- format_ci(best_metrics_sensitivity3$Sensitivity,  ci_sensitivity3_metrics[, "Sensitivity"])
  Specificity_CI  <- format_ci(best_metrics_sensitivity3$Specificity,  ci_sensitivity3_metrics[, "Specificity"])
  PPV_CI          <- format_ci(best_metrics_sensitivity3$PPV,          ci_sensitivity3_metrics[, "PPV"])
  NPV_CI          <- format_ci(best_metrics_sensitivity3$NPV,          ci_sensitivity3_metrics[, "NPV"])
  CCR_CI          <- format_ci(best_metrics_sensitivity3$CCR,          ci_sensitivity3_metrics[, "CCR"])
  F1_CI           <- format_ci(best_metrics_sensitivity3$F1_score,     ci_sensitivity3_metrics[, "F1_score"])
  Brier_CI        <- format_ci(brier_score_sensitivity3,               ci_sensitivity3_metrics[, "Brier_score"])
  
  sensitivity3_metrics_result <- data.frame(
    Model = model_name,
    Dataset = "sensitivity3",
    AUC_CI = auc_with_ci_sensitivity3, 
    Sensitivity_CI = Sensitivity_CI,
    Specificity_CI = Specificity_CI,
    PPV_CI = PPV_CI,
    NPV_CI = NPV_CI,
    CCR_CI = CCR_CI,
    F1_CI = F1_CI,
    Brier_CI = Brier_CI
  )
  sensitivity3_metrics <- rbind(sensitivity3_metrics, sensitivity3_metrics_result)
}

##---------(4) Sensitivity dataset 4 (RDKF: multiple imputation)
#Generate default method template
meth <- make.method(rckd2)
#Select continuous variables (numeric)
num_vars <- c("PEF", "CRP", "HbA1c", "TC", "HDL_C", "LDL_C", "TG", "BUN", 
              "Glucose", "UA", "WBC", "Hemoglobin", "MCV", "Platelets", "Hematocrit")
#Specify imputation method for continuous variables: pmm
meth[num_vars] <- "pmm"

#Select multi-category categorical variables
polyreg_vars <- c("Age", "Education", "Material_wealth", "Medical_insurance",
                  "Self_rated_health", "Nighttime_sleep", "Afternoon_napping", 
                  "Social_activities","Intellectual_activities")
#Specify polyreg imputation method
meth[polyreg_vars] <- "polyreg"

#Select binary variables
logreg_vars <- c("Gender", "Occupation", "Marital_status", "Living_status", 
                 "Housing_tenure", "Current_residence_location", "Hypertension", 
                 "Diabetes", "Chronic_lung_diseases", "Heart_diseases", "Stroke", 
                 "Psychiatric_problems", "Arthritis_or_rheumatism", "Dyslipidemia", 
                 "Liver_diseases", "Digestive_diseases", "Asthma", 
                 "Memory_related_diseases", "Depression", "Vision_impairment", 
                 "Hearing_impairment", "ADLs_disability", "Pain", "Complete_tooth_loss", 
                 "Eating_habit", "Smoking", "Drinking", "Waist_circumference", 
                 "SBP", "DBP", "Low_HGS", "Anti_hypertensive_drugs", 
                 "Anti_diabetic_drugs", "Lipid_lowering_drugs", "RDKF")
#Specify logreg method
meth[logreg_vars] <- "logreg"

#Perform multiple imputation
set.seed(123)
mi <- mice( rckd2, m = 5, method = meth, maxit = 20)

#Extract the first imputed complete dataset
rckd2_mi1 <- complete(mi, action = 1)

#Standardize sensitivity dataset 4 (multiple imputation 1) using the training cohort's standardization parameters
sensitivity4_standardized <- predict(standardized_para, newdata = rckd2_mi1)
#Convert all factor variables to numeric
sensitivity4_standardized <- sensitivity4_standardized %>% 
  mutate(across(where(is.factor), ~ as.numeric(.)))
#Convert outcome variable RDKF to factor
sensitivity4_standardized$RDKF <- factor(sensitivity4_standardized$RDKF)
#Extract selected features and outcome
sensitivity4_boruta <- sensitivity4_standardized[, c(confirmed_features, "RDKF")]
##Create task for sensitivity dataset 4
sensitivity4_task <- as_task_classif(sensitivity4_boruta, target = "RDKF")
#Predict on sensitivity dataset 4
sensitivity4_lr_pred <- lr_learner$predict(sensitivity4_task)
sensitivity4_rf_pred <- rf_learner$predict(sensitivity4_task)
sensitivity4_svm_pred <- svm_learner$predict(sensitivity4_task)
sensitivity4_xgb_pred <- xgb_learner$predict(sensitivity4_task)
sensitivity4_catb_pred <- catb_learner$predict(sensitivity4_task)
sensitivity4_lightgbm_pred <- lightgbm_learner$predict(sensitivity4_task)
#Create a list of model predictions for sensitivity dataset 4
sensitivity4_preds <- list(
  LR = sensitivity4_lr_pred, 
  RF = sensitivity4_rf_pred, 
  SVM = sensitivity4_svm_pred, 
  XGBoost = sensitivity4_xgb_pred, 
  CatBoost = sensitivity4_catb_pred, 
  LightGBM = sensitivity4_lightgbm_pred
)
#Create an empty data frame to store performance metrics for sensitivity dataset 4
sensitivity4_metrics <- data.frame()
#Compute performance metrics for sensitivity dataset 4
for (model_name in names(sensitivity4_preds)) {
  pred_sensitivity4 <- sensitivity4_preds[[model_name]]
  pred_prob_sensitivity4 <- pred_sensitivity4$prob[, "2"]
  true_class_sensitivity4 <- pred_sensitivity4$truth
  true_class_sensitivity4 <- ifelse(true_class_sensitivity4 == "2", 1, 0)
  roc_sensitivity4 <- roc(true_class_sensitivity4, pred_prob_sensitivity4)
  auc_sensitivity4 <- roc_sensitivity4$auc
  ci_sensitivity4 <- ci(roc_sensitivity4)
  #Combine AUC and 95% CI, keep three decimals
  auc_with_ci_sensitivity4 <- sprintf("%.3f\n(%.3f-%.3f)", auc_sensitivity4, ci_sensitivity4[1], ci_sensitivity4[3]) 
  
  #Get the best threshold from the training cohort
  best_threshold_train <- train_metrics[train_metrics$Model == model_name, "Threshold"]
  #Use the best threshold from the training cohort to compute the performance of the test cohort
  best_metrics_sensitivity4 <- calculate_metrics(pred_prob_sensitivity4, true_class_sensitivity4, best_threshold_train)
  
  #Compute Brier score
  brier_score_sensitivity4 <- mean((pred_prob_sensitivity4 - true_class_sensitivity4)^2)
  
  #Bootstrap CI
  set.seed(123)
  ci_sensitivity4_metrics <- bootstrap_ci(pred_prob_sensitivity4, true_class_sensitivity4, best_threshold_train, B = 1000)
  
  Sensitivity_CI  <- format_ci(best_metrics_sensitivity4$Sensitivity,  ci_sensitivity4_metrics[, "Sensitivity"])
  Specificity_CI  <- format_ci(best_metrics_sensitivity4$Specificity,  ci_sensitivity4_metrics[, "Specificity"])
  PPV_CI          <- format_ci(best_metrics_sensitivity4$PPV,          ci_sensitivity4_metrics[, "PPV"])
  NPV_CI          <- format_ci(best_metrics_sensitivity4$NPV,          ci_sensitivity4_metrics[, "NPV"])
  CCR_CI          <- format_ci(best_metrics_sensitivity4$CCR,          ci_sensitivity4_metrics[, "CCR"])
  F1_CI           <- format_ci(best_metrics_sensitivity4$F1_score,     ci_sensitivity4_metrics[, "F1_score"])
  Brier_CI        <- format_ci(brier_score_sensitivity4,               ci_sensitivity4_metrics[, "Brier_score"])
  
  sensitivity4_metrics_result <- data.frame(
    Model = model_name,
    Dataset = "sensitivity4",
    AUC_CI = auc_with_ci_sensitivity4, 
    Sensitivity_CI = Sensitivity_CI,
    Specificity_CI = Specificity_CI,
    PPV_CI = PPV_CI,
    NPV_CI = NPV_CI,
    CCR_CI = CCR_CI,
    F1_CI = F1_CI,
    Brier_CI = Brier_CI
  )
  sensitivity4_metrics <- rbind(sensitivity4_metrics, sensitivity4_metrics_result)
}

sensitivity_metrics_list <- list(
  sensitivity1_metrics = sensitivity1_metrics,
  sensitivity2_metrics = sensitivity2_metrics,
  sensitivity3_metrics = sensitivity3_metrics,
  sensitivity4_metrics = sensitivity4_metrics
)

#Transpose sensitivity results 1-4
sensitivity_t_list <- lapply(sensitivity_metrics_list, transpose_metrics)
#Combine sensitivity results 1-4 (Table S7)
sensitivity_t_all <- bind_rows(sensitivity_t_list)



#---------(5) Sensitivity analysis: Youden index

#Create empty data frames to store performance evaluation results for the training and test cohorts separately
train_metrics_youden <- data.frame()
test_metrics_youden <- data.frame()

#Compute performance metrics for the training cohort
for (model_name in names(train_preds)) {
  #Get the training cohort model predictions
  pred_train <- train_preds[[model_name]]
  #Get the training cohort predicted probabilities
  pred_prob_train <- pred_train$prob[, "2"]
  
  #Get the actual classes for the training cohort
  true_class_train <- pred_train$truth
  true_class_train <- ifelse(true_class_train == "2", 1, 0)
  
  roc_train <- roc(true_class_train, pred_prob_train)
  auc_train <- roc_train$auc
  ci_train <- ci(roc_train)
  #Combine AUC and 95% CI, keep three decimals
  auc_with_ci_train <- sprintf("%.3f\n(%.3f-%.3f)", auc_train, ci_train[1], ci_train[3]) 
  
  ##Define performance metrics
  calculate_metrics <- function(pred_prob, true_class, threshold) {
    #Generate predicted classes based on the threshold
    pred_class <- ifelse(pred_prob >= threshold, "1", "0") 
    #Construct the confusion matrix ensuring all classes are present
    cm <- table(
      factor(pred_class, levels = c("0", "1")), #Ensure predicted classes include No and Yes
      factor(true_class, levels = c("0", "1")) #Ensure actual classes include No and Yes
    )
    #Extract confusion matrix elements to avoid errors when some classes are missing
    TP <- cm["1", "1"] #True Positive
    TN <- cm["0", "0"] #True Negative
    FP <- cm["1", "0"] #False Positive
    FN <- cm["0", "1"] #False Negative
    #Compute metrics
    sensitivity <- TP / (TP + FN) #Sensitivity (Recall)
    specificity <- TN / (TN + FP) #Specificity
    PPV <- TP / (TP + FP) #Positive Predictive Value/Precision
    NPV <- TN / (TN + FN) #Negative Predictive Value
    CCR <- (TP + TN) / sum(cm) #Accuracy
    f1_score <- 2 * (PPV * sensitivity) / (PPV + sensitivity) #F1 score
    #Return results
    list(Sensitivity = sensitivity, Specificity = specificity, PPV = PPV, NPV = NPV, CCR = CCR, F1_score = f1_score)
  }
  
  ##Compute the best threshold for the training cohort
  #Define threshold range
  threshold <- seq(0, 1, by = 0.001)
  #Compute performance metrics for different thresholds
  metrics_list_train <- sapply(threshold, function(t) {
    calculate_metrics(pred_prob_train, true_class_train, t)
  }, simplify = F)
  
  #Compute Youden index
  youden_indices <- sapply(metrics_list_train, function(metrics) {
    metrics$Sensitivity + metrics$Specificity - 1
  })
  #Obtain the best threshold
  best_threshold_train <- threshold[which.max(youden_indices)]
  
  #Use the best threshold from the training cohort to compute its performance metrics
  best_metrics_train <- calculate_metrics(pred_prob_train, true_class_train, best_threshold_train)
  
  #Compute Brier score
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
  
  #Summarize model results for the training cohort
  train_metrics_result <- data.frame(
    Model = model_name,
    Dataset = "Training cohort",
    Threshold = round(best_threshold_train, 3),
    Sensitivity_CI = Sensitivity_CI,
    Specificity_CI = Specificity_CI,
    PPV_CI = PPV_CI,
    NPV_CI = NPV_CI,
    CCR_CI = CCR_CI,
    F1_score_CI = F1_score_CI
  )
  #Append each model's results for the training cohort to the data frame
  train_metrics_youden <- rbind(train_metrics_youden, train_metrics_result)
}

#Compute performance metrics for the test cohort
for (model_name in names(test_preds)) {
  #Get test cohort model predictions
  pred_test <- test_preds[[model_name]]
  #Get test cohort predicted probabilities
  pred_prob_test <- pred_test$prob[, "2"]
  
  #Get actual classes for the test cohort
  true_class_test <- pred_test$truth
  true_class_test <- ifelse(true_class_test == "2", 1, 0)
  
  #Compute AUC and 95% CI for the test cohort
  roc_test <- roc(true_class_test, pred_prob_test)
  auc_test <- roc_test$auc
  ci_test <- ci(roc_test)
  #Combine AUC and 95% CI, keep three decimals
  auc_with_ci_test <- sprintf("%.3f\n(%.3f-%.3f)", auc_test, ci_test[1], ci_test[3]) 
  
  #Get the best threshold from the training cohort
  best_threshold_train <- train_metrics_youden[train_metrics_youden$Model == model_name, "Threshold"]
  #Use the training cohort's best threshold to compute test performance
  best_metrics_test <- calculate_metrics(pred_prob_test, true_class_test, best_threshold_train)
  
  #Compute Brier score for the test cohort
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
  
  #Summarize model results for the test cohort into a data frame
  test_metrics_result <- data.frame(
    Model = model_name,
    Dataset = "Testing cohort",
    Threshold = round(best_threshold_train, 3),  #Use the best threshold from the training cohort
    Sensitivity_CI = Sensitivity_CI,
    Specificity_CI = Specificity_CI,
    PPV_CI = PPV_CI,
    NPV_CI = NPV_CI,
    CCR_CI = CCR_CI,
    F1_score_CI = F1_score_CI
  )
  #Combine each model's results for the test cohort
  test_metrics_youden <- rbind(test_metrics_youden, test_metrics_result)
}

##Combine model performance results of the training and test cohorts
#Transpose the training cohort model results
train_metrics_youden_t <- train_metrics_youden
names(train_metrics_youden_t)[c(4:9)] <- c("Sensitivity\n(95% CI)", 
                                           "Specificity\n(95% CI)", "PPV\n(95% CI)", 
                                           "NPV\n(95% CI)", "CCR\n(95% CI)", 
                                           "F1 score\n(95% CI)")
train_metrics_youden_t <- tibble::rownames_to_column(as.data.frame(t(train_metrics_youden_t)))
colnames(train_metrics_youden_t) <- c("Metric", train_metrics_youden_t[1, -1])
train_metrics_youden_t <- train_metrics_youden_t[-1, ]
#Transpose the test cohort model results
test_metrics_youden_t <- test_metrics_youden
names(test_metrics_youden_t)[c(4:9)] <- c("Sensitivity\n(95% CI)", 
                                          "Specificity\n(95% CI)", "PPV\n(95% CI)", 
                                          "NPV\n(95% CI)", "CCR\n(95% CI)", 
                                          "F1 score\n(95% CI)")
test_metrics_youden_t <- tibble::rownames_to_column(as.data.frame(t(test_metrics_youden_t)))
colnames(test_metrics_youden_t) <- c("Metric", test_metrics_youden_t[1, -1])
test_metrics_youden_t <- test_metrics_youden_t[-1, ]
#Combine results (Table S9)
train_test_metrics_youden <- rbind(train_metrics_youden_t, test_metrics_youden_t)

#---------(6) Sensitivity analysis: SMOTENC
## Handle class imbalance in the dataset train_boruta using SMOTENC
# Define preprocessing recipe
rec <- recipe(RDKF ~ ., data = train_boruta)

# Set random seed to ensure reproducibility of SMOTENC results
set.seed(131)

# Apply SMOTENC to address class imbalance
train_smotenc <- rec %>% 
  step_smotenc(RDKF, over_ratio = 1) %>% # Balance classes to a 1:1 ratio
  prep() %>% # Apply the recipe to the training data and complete the augmented preprocessing steps
  bake(new_data = NULL) # Output the processed dataset

train_smotenc_task <- as_task_classif(train_smotenc, target = "RDKF")

## Logistic regression (LR)
# Select algorithm with probability prediction output
lr_learner_smotenc <- lrn("classif.log_reg", predict_type = "prob")

# Train the model on the training dataset
lr_learner_smotenc$train(train_smotenc_task)

# Predict on the training dataset
train_lr_pred_smotenc <- lr_learner_smotenc$predict(train_smotenc_task)

# Predict on the testing dataset
test_lr_pred_smotenc <- lr_learner_smotenc$predict(test_task)

## Random forest (RF)
# Select algorithm with probability prediction output
rf_learner_smotenc <- lrn("classif.ranger", predict_type = "prob")   

# Define tuned hyperparameters and their ranges
rf_learner_smotenc$param_set$values <- list(
  num.trees = to_tune(10, 100), # Number of trees: controls the number of trees in the forest, affecting stability and computation
  max.depth = to_tune(1, 5), # Maximum tree depth: limits tree depth to prevent overfitting
  min.node.size = to_tune(p_int(1, 5)), # Minimum samples per internal node: controls the splitting of trees
  mtry = to_tune(1, 5), # Number of random features at each split: balances diversity and accuracy
  sample.fraction = to_tune(0.3, 1) # Fraction of samples randomly selected
)

# Set random seed to ensure reproducibility of tuning
set.seed(123)

# Perform hyperparameter tuning
rf_smotenc <- tune(
  tuner = tnr("grid_search", resolution = 5), # Grid search; resolution = 5 divides each parameter range into 5 values
  task = train_smotenc_task, # Specify tuning task
  learner = rf_learner_smotenc, # Specify learner
  resampling = rsmp("cv", folds = 5), # 5-fold cross-validation
  measure = msr("classif.auc") # Model evaluation metric: AUC
)

# Apply the best hyperparameters to the model
rf_learner_smotenc$param_set$values <- rf_smotenc$result_learner_param_vals

# Train the model on the training dataset
rf_learner_smotenc$train(train_smotenc_task)

# Predict on the training dataset
train_rf_pred_smotenc <- rf_learner_smotenc$predict(train_smotenc_task)

# Predict on the testing dataset
test_rf_pred_smotenc <- rf_learner_smotenc$predict(test_task)

## Support vector machine (SVM)
svm_learner_smotenc <- lrn("classif.svm", predict_type = "prob")

svm_learner_smotenc$param_set$values <- list(
  type = "C-classification", # Classification task
  kernel = "radial", # Radial Basis Function (RBF) kernel for nonlinear classification
  cost = to_tune(0.001, 1), # Penalty parameter C: controls the penalty for misclassification
  gamma = to_tune(0.001, 1) # Gamma parameter: defines the width of the RBF kernel
)

set.seed(123)

svm_smotenc <- tune(
  tuner = tnr("grid_search", resolution = 5), 
  task = train_smotenc_task, 
  learner = svm_learner_smotenc, 
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.auc")
)

svm_learner_smotenc$param_set$values <- svm_smotenc$result_learner_param_vals

svm_learner_smotenc$train(train_smotenc_task)

train_svm_pred_smotenc <- svm_learner_smotenc$predict(train_smotenc_task)
test_svm_pred_smotenc <- svm_learner_smotenc$predict(test_task)

## XGBoost
xgb_learner_smotenc <- lrn("classif.xgboost", predict_type = "prob")

xgb_learner_smotenc$param_set$values <- list(
  nrounds = to_tune(500, 1000), # Number of trees: controls the number of boosting iterations
  max_depth = to_tune(1, 10), # Maximum tree depth: controls model complexity
  eta = to_tune(0.001, 0.05), # Learning rate: controls learning speed and convergence stability
  min_child_weight = to_tune(1, 10), # Minimum instance weight per leaf node
  subsample = to_tune(0.1, 0.5) # Subsample ratio: introduces randomness and reduces overfitting
)

set.seed(123)

xgb_smotenc <- tune(
  tuner = tnr("grid_search", resolution = 5),
  task = train_smotenc_task,
  learner = xgb_learner_smotenc,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.auc")
)

xgb_learner_smotenc$param_set$values <- xgb_smotenc$result_learner_param_vals

xgb_learner_smotenc$train(train_smotenc_task)

train_xgb_pred_smotenc <- xgb_learner_smotenc$predict(train_smotenc_task)
test_xgb_pred_smotenc <- xgb_learner_smotenc$predict(test_task)

test_svm_pred_smotenc$score(msr("classif.auc"))

## CatBoost
catb_learner_smotenc <- lrn("classif.catboost", predict_type = "prob")

catb_learner_smotenc$param_set$values <- list(
  learning_rate = to_tune(0.1, 0.5), # Learning rate for gradient steps
  depth = to_tune(1, 5), # Tree depth
  rsm = to_tune(0.001, 0.01), # Feature subsampling ratio for each split
  loss_function_twoclass = "Logloss" # Loss function for binary classification
)

set.seed(123)

catb_smotenc <- tune(
  tuner = tnr("grid_search", resolution = 5),
  task = train_smotenc_task,
  learner = catb_learner_smotenc,
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.auc")
)

catb_learner_smotenc$param_set$values <- catb_smotenc$result_learner_param_vals

catb_learner_smotenc$train(train_smotenc_task)

train_catb_pred_smotenc <- catb_learner_smotenc$predict(train_smotenc_task)
test_catb_pred_smotenc <- catb_learner_smotenc$predict(test_task)

## LightGBM
lightgbm_learner_smotenc <- lrn("classif.lightgbm", predict_type = "prob")

lightgbm_learner_smotenc$param_set$values <- list(
  learning_rate = to_tune(0.001, 0.2), # Learning rate
  bagging_fraction = to_tune(0.8, 1), # Sample fraction
  max_depth = to_tune(1, 5), # Maximum tree depth
  num_iterations = to_tune(500, 1000), # Number of iterations
  objective = "binary" # Objective function for binary classification
)

set.seed(123)

lightgbm_smotenc <- tune(
  tuner = tnr("grid_search", resolution = 5), 
  task = train_smotenc_task, 
  learner = lightgbm_learner_smotenc, 
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.auc")
)

lightgbm_learner_smotenc$param_set$values <- lightgbm_smotenc$result_learner_param_vals

lightgbm_learner_smotenc$train(train_smotenc_task)

train_lightgbm_pred_smotenc <- lightgbm_learner_smotenc$predict(train_smotenc_task)
test_lightgbm_pred_smotenc <- lightgbm_learner_smotenc$predict(test_task)

## -------- Model performance evaluation --------
# Create lists of prediction results for the training and testing datasets
train_preds_smotenc <- list(
  LR = train_lr_pred_smotenc, 
  RF = train_rf_pred_smotenc, 
  SVM = train_svm_pred_smotenc, 
  XGBoost = train_xgb_pred_smotenc, 
  CatBoost = train_catb_pred_smotenc, 
  LightGBM = train_lightgbm_pred_smotenc
)
test_preds_smotenc <- list(
  LR = test_lr_pred_smotenc, 
  RF = test_rf_pred_smotenc, 
  SVM = test_svm_pred_smotenc, 
  XGBoost = test_xgb_pred_smotenc, 
  CatBoost = test_catb_pred_smotenc, 
  LightGBM = test_lightgbm_pred_smotenc
)

# Create empty data frames to store performance evaluation results for the training and testing datasets
train_metrics_smotenc <- data.frame()
test_metrics_smotenc <- data.frame()

# Calculate performance metrics for the training dataset
for (model_name in names(train_preds_smotenc)) {
  
  # Obtain prediction results from the training dataset
  pred_train <- train_preds_smotenc[[model_name]]
  
  # Obtain predicted probabilities from the training dataset
  pred_prob_train <- pred_train$prob[, "2"]
  
  # Obtain true class labels for the training dataset
  true_class_train <- pred_train$truth
  true_class_train <- ifelse(true_class_train == "2", 1, 0)
  
  roc_train <- roc(true_class_train, pred_prob_train)
  auc_train <- roc_train$auc
  ci_train <- ci(roc_train)
  
  # Combine AUC and 95% CI, retaining three decimal places
  auc_with_ci_train <- sprintf("%.3f\n(%.3f-%.3f)", auc_train, ci_train[1], ci_train[3]) 
  
  ## Define performance metrics
  calculate_metrics <- function(pred_prob, true_class, threshold) {
    
    # Generate predicted class based on the threshold
    pred_class <- ifelse(pred_prob >= threshold, "1", "0") 
    
    # Construct confusion matrix ensuring both classes exist
    cm <- table(
      factor(pred_class, levels = c("0", "1")), # Ensure predicted classes include both categories
      factor(true_class, levels = c("0", "1")) # Ensure true classes include both categories
    )
    
    # Extract confusion matrix elements
    TP <- cm["1", "1"] # True positive
    TN <- cm["0", "0"] # True negative
    FP <- cm["1", "0"] # False positive
    FN <- cm["0", "1"] # False negative
    
    # Calculate performance metrics
    sensitivity <- TP / (TP + FN) # Sensitivity (Recall)
    specificity <- TN / (TN + FP) # Specificity
    PPV <- TP / (TP + FP) # Positive predictive value (Precision)
    NPV <- TN / (TN + FN) # Negative predictive value
    CCR <- (TP + TN) / sum(cm) # Overall accuracy
    f1_score <- 2 * (PPV * sensitivity) / (PPV + sensitivity) # F1 score
    
    # Return results
    list(Sensitivity = sensitivity, Specificity = specificity, PPV = PPV, NPV = NPV, CCR = CCR, F1_score = f1_score)
  }
  
  ## Calculate the optimal threshold for the training dataset
  # Define threshold range
  threshold <- seq(0, 1, by = 0.001)
  
  # Calculate performance metrics across thresholds
  metrics_list_train <- sapply(threshold, function(t) {
    calculate_metrics(pred_prob_train, true_class_train, t)
  }, simplify = F)
  
  distances <- sapply(metrics_list_train, function(metrics) {
    sqrt((1-metrics$Sensitivity)^2 + (1-metrics$Specificity)^2)
  })
  
  # Identify the optimal threshold for the training dataset
  best_threshold_train <- threshold[which.min(distances)]
  
  # Calculate performance metrics using the optimal threshold
  best_metrics_train <- calculate_metrics(pred_prob_train, true_class_train, best_threshold_train)
  
  # Calculate Brier score
  brier_score_train <- mean((pred_prob_train - true_class_train)^2)
  
  # Bootstrap CI
  set.seed(123)
  ci_train_metrics <- bootstrap_ci(pred_prob_train, true_class_train, best_threshold_train, B = 1000)
  
  Sensitivity_CI  <- format_ci(best_metrics_train$Sensitivity,  ci_train_metrics[, "Sensitivity"])
  Specificity_CI  <- format_ci(best_metrics_train$Specificity,  ci_train_metrics[, "Specificity"])
  PPV_CI          <- format_ci(best_metrics_train$PPV,          ci_train_metrics[, "PPV"])
  NPV_CI          <- format_ci(best_metrics_train$NPV,          ci_train_metrics[, "NPV"])
  CCR_CI          <- format_ci(best_metrics_train$CCR,          ci_train_metrics[, "CCR"])
  F1_score_CI     <- format_ci(best_metrics_train$F1_score,     ci_train_metrics[, "F1_score"])
  Brier_score_CI  <- format_ci(brier_score_train,               ci_train_metrics[, "Brier_score"])
  
  # Summarize training dataset model results
  train_metrics_result <- data.frame(
    Model = model_name,
    Dataset = "Training cohort",
    AUC_CI = auc_with_ci_train, 
    Threshold = round(best_threshold_train, 3),
    Sensitivity_CI = Sensitivity_CI,
    Specificity_CI = Specificity_CI,
    PPV_CI = PPV_CI,
    NPV_CI = NPV_CI,
    CCR_CI = CCR_CI,
    F1_score_CI = F1_score_CI,
    Brier_score_CI = Brier_score_CI
  )
  
  # Append training results to the data frame
  train_metrics_smotenc <- rbind(train_metrics_smotenc, train_metrics_result)
}

# Calculate performance metrics for the testing dataset
for (model_name in names(test_preds_smotenc)) {
  
  # Obtain prediction results from the testing dataset
  pred_test <- test_preds_smotenc[[model_name]]
  
  # Obtain predicted probabilities from the testing dataset
  pred_prob_test <- pred_test$prob[, "2"]
  
  # Obtain true class labels for the testing dataset
  true_class_test <- pred_test$truth
  true_class_test <- ifelse(true_class_test == "2", 1, 0)
  
  # Calculate AUC and 95% CI for the testing dataset
  roc_test <- roc(true_class_test, pred_prob_test)
  auc_test <- roc_test$auc
  ci_test <- ci(roc_test)
  
  # Combine AUC and 95% CI, retaining three decimal places
  auc_with_ci_test <- sprintf("%.3f\n(%.3f-%.3f)", auc_test, ci_test[1], ci_test[3]) 
  
  # Retrieve the optimal threshold from the training dataset
  best_threshold_train <- train_metrics_smotenc[train_metrics_smotenc$Model == model_name, "Threshold"]
  
  # Calculate testing performance using the training optimal threshold
  best_metrics_test <- calculate_metrics(pred_prob_test, true_class_test, best_threshold_train)
  
  # Calculate Brier score for the testing dataset
  brier_score_test <- mean((pred_prob_test - true_class_test)^2)
  
  # Bootstrap CI
  set.seed(123)
  ci_test_metrics <- bootstrap_ci(pred_prob_test, true_class_test, best_threshold_train, B = 1000)
  
  Sensitivity_CI  <- format_ci(best_metrics_test$Sensitivity,  ci_test_metrics[, "Sensitivity"])
  Specificity_CI  <- format_ci(best_metrics_test$Specificity,  ci_test_metrics[, "Specificity"])
  PPV_CI          <- format_ci(best_metrics_test$PPV,          ci_test_metrics[, "PPV"])
  NPV_CI          <- format_ci(best_metrics_test$NPV,          ci_test_metrics[, "NPV"])
  CCR_CI          <- format_ci(best_metrics_test$CCR,          ci_test_metrics[, "CCR"])
  F1_score_CI     <- format_ci(best_metrics_test$F1_score,     ci_test_metrics[, "F1_score"])
  Brier_score_CI  <- format_ci(brier_score_test,               ci_test_metrics[, "Brier_score"])
  
  # Summarize testing dataset model results
  test_metrics_result <- data.frame(
    Model = model_name,
    Dataset = "Testing cohort",
    AUC_CI = auc_with_ci_test, 
    Threshold = round(best_threshold_train, 3),  # Use the optimal threshold from the training dataset
    Sensitivity_CI = Sensitivity_CI,
    Specificity_CI = Specificity_CI,
    PPV_CI = PPV_CI,
    NPV_CI = NPV_CI,
    CCR_CI = CCR_CI,
    F1_score_CI = F1_score_CI,
    Brier_score_CI = Brier_score_CI
  )
  
  # Append testing results
  test_metrics_smotenc <- rbind(test_metrics_smotenc, test_metrics_result)
}

## Combine training and testing model performance results
# Transpose training dataset model results
train_metrics_smotenc_t <- train_metrics_smotenc
names(train_metrics_smotenc_t)[c(3, 5:11)] <- c("AUC\n(95% CI)", "Sensitivity\n(95% CI)", 
                                                "Specificity\n(95% CI)", "PPV\n(95% CI)", 
                                                "NPV\n(95% CI)", "CCR\n(95% CI)", 
                                                "F1 score\n(95% CI)", "Brier score\n(95% CI)")

train_metrics_smotenc_t <- tibble::rownames_to_column(as.data.frame(t(train_metrics_smotenc_t)))
colnames(train_metrics_smotenc_t) <- c("Metric", train_metrics_smotenc_t[1, -1])
train_metrics_smotenc_t <- train_metrics_smotenc_t[-1, ]

# Transpose testing dataset model results
test_metrics_smotenc_t <- test_metrics_smotenc
names(test_metrics_smotenc_t)[c(3, 5:11)] <- c("AUC\n(95% CI)", "Sensitivity\n(95% CI)", 
                                               "Specificity\n(95% CI)", "PPV\n(95% CI)", 
                                               "NPV\n(95% CI)", "CCR\n(95% CI)", 
                                               "F1 score\n(95% CI)", "Brier score\n(95% CI)")

test_metrics_smotenc_t <- tibble::rownames_to_column(as.data.frame(t(test_metrics_smotenc_t)))
colnames(test_metrics_smotenc_t) <- c("Metric", test_metrics_smotenc_t[1, -1])
test_metrics_smotenc_t <- test_metrics_smotenc_t[-1, ]

# Merge results
train_test_metrics_smotenc <- rbind(train_metrics_smotenc_t, test_metrics_smotenc_t)
