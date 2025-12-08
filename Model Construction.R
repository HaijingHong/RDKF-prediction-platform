##-------- Build predictive models with mlr3 --------
# Suppress console output during training; 'warn' outputs only warnings and errors
lgr::get_logger("mlr3")$set_threshold("warn")
lgr::get_logger("bbotk")$set_threshold("warn")

# Create training task
train_task <- as_task_classif(train_boruta, target = "RDKF")
# Create test task
test_task <- as_task_classif(test_boruta, target = "RDKF")

## Logistic Regression (LR)
# Choose learner; output predicted probabilities
lr_learner <- lrn("classif.log_reg", predict_type = "prob")
# Train on training cohort
lr_learner$train(train_task)
# Predict on training cohort
train_lr_pred <- lr_learner$predict(train_task)
# Predict on testing cohort
test_lr_pred <- lr_learner$predict(test_task)

# Compute class-imbalance weights
ratio <- sum(train_boruta$RDKF == "1") / sum(train_boruta$RDKF == "2")

## Random Forest (RF)
# Choose learner; output predicted probabilities
rf_learner <- lrn("classif.ranger", predict_type = "prob")   
# Set hyperparameters and ranges
rf_learner$param_set$values <- list(
  num.trees = to_tune(10, 100), # Number of trees: controls ensemble size; affects stability and efficiency.
  max.depth = to_tune(1, 5), # Max tree depth: limits depth to prevent overfitting.
  min.node.size = to_tune(p_int(1, 5)), # Min samples at a node: controls split granularity. p_int() defines an integer range (mlr3tuning).
  mtry = to_tune(1, 5), # Number of features at split: balances diversity vs. accuracy.
  sample.fraction = to_tune(0.3, 1), # Subsample fraction
  class.weights = c("2" = ratio, "1" = 1) # Set class weights
)
# Set seed for reproducible tuning
set.seed(123)
# Run hyperparameter tuning
rf <- tune(tuner = tnr("grid_search", resolution = 5), # Grid search; resolution = 5 means 5 evenly spaced values per parameter
           task = train_task, # Tuning task
           learner = rf_learner, # Learner to tune
           resampling = rsmp("cv", folds = 5), # 5-fold cross-validation
           measure = msr("classif.auc") # Evaluation metric: AUC
)
# Apply best hyperparameters to learner
rf_learner$param_set$values <- rf$result_learner_param_vals
# Train on training cohort
rf_learner$train(train_task)
# Predict on training cohort
train_rf_pred <- rf_learner$predict(train_task)
# Predict on testing cohort
test_rf_pred <- rf_learner$predict(test_task)

## Support Vector Machine (SVM)
svm_learner <- lrn("classif.svm", predict_type = "prob")
svm_learner$param_set$values <- list(
  type = "C-classification", # classification task
  kernel = "radial", # Radial Basis Function (RBF) kernel: handles complex nonlinear classification
  cost = to_tune(1, 100), # C (penalty) parameter: degree of penalty for misclassification
  gamma = to_tune(0.001, 1), # gamma parameter: width of the RBF kernel; affects the mapping to high-dimensional space
  class.weights = c("2" = ratio, "1" = 1) # set class weights
)
set.seed(123)
svm <- tune(tuner = tnr("grid_search", resolution = 5), 
            task = train_task, 
            learner = svm_learner, 
            resampling = rsmp("cv", folds = 5),
            measure = msr("classif.auc")
)
svm_learner$param_set$values <- svm$result_learner_param_vals
svm_learner$train(train_task)
train_svm_pred <- svm_learner$predict(train_task)
test_svm_pred <- svm_learner$predict(test_task)

## XGBoost
xgb_learner <- lrn("classif.xgboost", predict_type = "prob")
xgb_learner$param_set$values <- list(
  nrounds = to_tune(500, 1000), # number of boosting rounds: controls iterations and learning extent
  max_depth = to_tune(1, 10), # maximum tree depth: controls complexity and prevents overfitting
  eta = to_tune(0.001, 0.05), # learning rate: step size; affects convergence and stability
  min_child_weight = to_tune(1, 10), # minimum sum of instance weight per leaf: restricts splits, prevents overfitting
  subsample = to_tune(0.1, 0.5), # subsample ratio: introduces randomness, reduces overfitting, increases diversity
  scale_pos_weight = ratio # set class weights
)
set.seed(123)
xgb <- tune(tuner = tnr("grid_search", resolution = 5),
            task = train_task,
            learner = xgb_learner,
            resampling = rsmp("cv", folds = 5),
            measure = msr("classif.auc")
)
xgb_learner$param_set$values <- xgb$result_learner_param_vals
xgb_learner$train(train_task)
train_xgb_pred <- xgb_learner$predict(train_task)
test_xgb_pred <- xgb_learner$predict(test_task)

## CatBoost
catb_learner <- lrn("classif.catboost", predict_type = "prob")
catb_learner$param_set$values <- list(
  learning_rate = to_tune(0.1, 0.5), # learning rate: reduces gradient step size
  depth = to_tune(1, 5), # tree depth
  rsm = to_tune(0.001, 0.01), # size of the feature subspace considered at each split (Random Subspace Method)
  loss_function_twoclass = "Logloss", # used to assign weights for binary classification
  class_weights = c("2" = ratio, "1" = 1) # set class weights
)
set.seed(123)
catb <- tune(tuner = tnr("grid_search", resolution = 5),
             task = train_task,
             learner = catb_learner,
             resampling = rsmp("cv", folds = 5),
             measure = msr("classif.auc")
)
catb_learner$param_set$values <- catb$result_learner_param_vals
catb_learner$train(train_task)
train_catb_pred <- catb_learner$predict(train_task)
test_catb_pred <- catb_learner$predict(test_task)

# LightGBM
lightgbm_learner <- lrn("classif.lightgbm", predict_type = "prob")
lightgbm_learner$param_set$values <- list(
  learning_rate = to_tune(0.001, 0.2), # learning rate
  bagging_fraction = to_tune(0.8, 1), # sample fraction
  max_depth = to_tune(1, 5), # maximum tree depth
  num_iterations = to_tune(500, 1000), # number of iterations
  objective = "binary", # objective for binary classification
  scale_pos_weight = ratio # set class weights
)
set.seed(123)
lightgbm <- tune(tuner = tnr("grid_search", resolution = 5), 
                 task = train_task, 
                 learner = lightgbm_learner, 
                 resampling = rsmp("cv", folds = 5),
                 measure = msr("classif.auc")
)
lightgbm_learner$param_set$values <- lightgbm$result_learner_param_vals
lightgbm_learner$train(train_task)
train_lightgbm_pred <- lightgbm_learner$predict(train_task)
test_lightgbm_pred <- lightgbm_learner$predict(test_task)
