library(shiny)
library(DT)
library(shinydashboard)
library(ggplot2)
library(caret)
library(dplyr)
library(mlr3)
library(mlr3verse)
library(mlr3learners)
library(e1071)
library(pROC)
library(ResourceSelection)
library(rmda)
library(riskRegression)
library(tidyverse)
library(kernelshap) 
library(shapviz)

ui <- dashboardPage(
  dashboardHeader(title = "RDKF prediction platform", titleWidth = 260),
  dashboardSidebar(
    width = 150, 
    sidebarMenu(
      menuItem("Introduction", tabName = "Introduction"), 
      menuItem("External validation", tabName = "External_Validation"), 
      menuItem("Batch prediction", tabName = "Batch_prediction"), 
      menuItem("Case prediction", tabName = "Case_prediction")
    )
  ),
  dashboardBody(
    tabItems(
      tabItem(tabName = "Introduction",
              fluidRow(
                box(
                  title = "Introduction", 
                  width = 6, 
                  height = "820px",
                  status = "primary",
                  collapsible = FALSE,
                  solidHeader = TRUE,
                  p(span("Upon initial access, the model requires approximately 5 minutes for deployment before the platform becomes fully operational.", style = "color: red; font-weight: bold")),
                  p(strong("Purpose:")),
                  p("This platform aims to predict the 4-year risk of rapid kidney function decline (RDKF), facilitating early identification of individuals at risk and offering timely opportunities for targeted interventions in primary care and community settings."),
                  p(strong("Scopes:")),
                  p("Clinicians, public health personnel, researchers, and individuals."),
                  p(strong("Functional modules:")),
                  p("The platform mainly consists of four functional modules: introduction, external validation, batch prediction, and case prediction."),
                  p("The introduction module displays basic information about the platform, detailed information on model variables, the receiver operating characteristic (ROC) curve of the optimal model, along with the calibration curve and decision curve analysis (DCA)."),
                  p("The external validation module allows users to upload external datasets for validating the performance of the model. Upon data upload, the platform automatically generates a comprehensive set of performance evaluation results, including AUC, threshold, sensitivity, specificity, positive predictive value (PPV), negative predictive value (NPV), correctly classified rate (CCR), F1 score, Brier score, ROC curve, calibration curve, and DCA. Additionally, a downloadable data template is provided, ensuring users can format their data correctly before uploading, facilitating a smooth validation process."),
                  p("Similarly, the batch prediction module enables users to upload datasets for bulk prediction of RDKF incidence rates across multiple cases. After uploading, users obtain aggregated predictions for the entire dataset, and the platform provides a downloadable template to help users structure their data appropriately for batch processing."),
                  p("In the case prediction module, users can input the individual values for each predictor, and by clicking the 'Predict' button, they will obtain a predicted probability for RDKF risk along with a visual SHAP force plot, which displays the contribution of each predictor to the model's prediction.")
                ),
                box(
                  title = "Variables in the model",
                  width = 6, 
                  height = "820px",
                  status = "primary",
                  collapsible = FALSE,
                  solidHeader = TRUE,
                  DTOutput("variables_info"), 
                  p("Notes: The normal waist circumference is men < 85 cm or women < 80 cm, while the abnormal range is men >= 85 cm or women >= 80 cm. HbA1c, glycosylated hemoglobin; TC, total cholesterol; HDL-C, high-density lipoprotein cholesterol; LDL-C, low-density lipoprotein cholesterol; TG, triglyceride; BUN, blood urea nitrogen; UA, uric acid; WBC, white blood cell; RDKF, rapid decline in kidney function.")
                )
              ),
              fluidRow(
                box(
                  width = 4, 
                  height = "380px",
                  status = "primary",
                  collapsible = FALSE,
                  solidHeader = FALSE,
                  imageOutput("roc_curve", height = "320px"),
                  p("Fig. 1 Receiver operating characteristic curve of support vector machine model (AUC = 0.727)")
                ),
                box(
                  width = 4, 
                  height = "380px",
                  status = "primary",
                  collapsible = FALSE,
                  solidHeader = FALSE,
                  imageOutput("calibration_curve", height = "320px"),
                  p("Fig. 2 Calibration curve of support vector machine model")
                ),
                box(
                  width = 4, 
                  height = "380px",
                  status = "primary",
                  collapsible = FALSE,
                  solidHeader = FALSE,
                  imageOutput("dca_curve", height = "320px"),
                  p("Fig. 3 Decision curve analysis of support vector machine model")
                )
              )
      ),
      tabItem(tabName = "External_Validation",
              fluidRow(
                box(
                  title = "External validation",
                  width = 12,
                  status = "primary",
                  collapsible = FALSE,
                  solidHeader = TRUE,
                  downloadButton("downloadTemplate1", "Download data template"), 
                  fileInput("external_data", "Please upload a .csv file.", accept = ".csv"),
                  fluidRow(
                    box(
                      title = "External validation performance results",
                      width = 12, 
                      height = "150px",
                      status = "primary",
                      collapsible = FALSE,
                      solidHeader = TRUE,
                      DTOutput("external_performance")
                    )
                  ),
                  fluidRow(
                    box(
                      title = "Receiver operating characteristic curve",
                      width = 4, 
                      height = "370px",
                      status = "primary",
                      collapsible = FALSE,
                      solidHeader = TRUE,
                      plotOutput("roc_curve2", height = "320px")
                    ),
                    box(
                      title = "Calibration curve",
                      width = 4, 
                      height = "370px",
                      status = "primary",
                      collapsible = FALSE,
                      solidHeader = TRUE,
                      plotOutput("calibration_curve2", height = "320px")
                    ),
                    box(
                      title = "Decision curve analysis",
                      width = 4, 
                      height = "370px",
                      status = "primary",
                      collapsible = FALSE,
                      solidHeader = T,
                      plotOutput("dca_curve2", height = "320px")
                    )
                  )
                )
              )
      ),
      tabItem(tabName = "Batch_prediction",
              box(
                title = "Batch prediction",
                width = 13,
                status = "primary",
                collapsible = TRUE,
                solidHeader = TRUE,
                downloadButton("downloadTemplate2", "Download data template"), 
                fileInput("population_data", "Please upload a .csv file.", accept = ".csv"),
                fluidRow(
                  box(
                    title = "Prediction results",
                    width = 12, 
                    height = "580px",
                    status = "primary",
                    collapsible = FALSE,
                    solidHeader = T,
                    DTOutput("population_predict_results"),
                    downloadButton("download_results", "Download results")
                  )
                )
                
              )
      ),
      tabItem(tabName = "Case_prediction",
              fluidRow(
                box(
                  title = "Input variables", 
                  width = 12,
                  height = "410px",
                  status = "primary",
                  collapsible = TRUE,
                  solidHeader = TRUE,
                  fluidRow(
                    column(width = 3, 
                           selectInput("Gender", label = "Gender (1: Man, 2: Woman)", choices = c("Man", "Woman"), selected = "Man")),
                    column(width = 3, 
                           selectInput("Hypertension", label = "Hypertension (1: No, 2: Yes)", choices = c("No", "Yes"), selected = "No")),
                    column(width = 3, 
                           selectInput("Dyslipidemia", label = "Dyslipidemia (1: No, 2: Yes)", choices = c("No", "Yes"), selected = "No")),
                    column(width = 3, 
                           numericInput("Waist_circumference", label = "Waist Circumference (cm; 1: Normal, 2: Abnormal)", value = 80))
                  ),
                  fluidRow(
                    column(width = 3, 
                           selectInput("Anti_hypertensive_drugs", label = "Anti-hypertensive drugs (1: No, 2: Yes)", choices = c("No", "Yes"), selected = "No")), 
                    column(width = 3, 
                           selectInput("Anti_diabetic_drugs", label = "Anti-diabetic drugs (1: No, 2: Yes)", choices = c("No", "Yes"), selected = "No")), 
                    column(width = 3, 
                           selectInput("Lipid_lowering_drugs", label = "Lipid-lowering drugs (1: No, 2: Yes)", choices = c("No", "Yes"), selected = "No")),
                    column(width = 3, 
                           numericInput("HbA1c", label = "HbA1c (%)", value = 5.3))
                  ),
                  fluidRow(
                    column(width = 3, 
                           numericInput("TC", label = "TC (mg/dL)", value = 190)),
                    column(width = 3, 
                           numericInput("HDL_C", label = "HDL-C (mg/dL)", value = 50)), 
                    column(width = 3, 
                           numericInput("LDL_C", label = "LDL-C (mg/dL)", value = 115)),
                    column(width = 3, 
                           numericInput("TG", label = "TG (mg/dL)", value = 135))
                  ),
                  fluidRow(
                    column(width = 3, 
                           numericInput("Glucose", label = "Fasting glucose (g/dL)", value = 110)),
                    column(width = 3, 
                           numericInput("UA", label = "UA (g/dL)", value = 4.6)), 
                    column(width = 3, 
                           numericInput("Hemoglobin", label = "Hemoglobin (g/dL)", value = 15)),
                    column(width = 3, 
                           numericInput("Hematocrit", label = "Hematocrit (%)", value = 40))
                  ),
                  fluidRow(
                    column(width = 3, 
                           actionButton("Predict", label = "Predict", icon = icon("play")))
                  )
                )
              ),
              fluidRow(
                box(
                  title = "Prediction result", 
                  width = 12, 
                  height = "100px", 
                  status = "primary", 
                  solidHeader = TRUE, 
                  tags$div(style = "font-size: 20px;", textOutput("prediction_result"))
                )
              ),
              fluidRow(
                box(
                  title = "SHAP force plot", 
                  width = 12, 
                  height = "380px", 
                  status = "primary", 
                  solidHeader = TRUE, 
                  plotOutput("shap_force_plot") 
                )
              ),
      )
    )
  )
)

server <- function(input, output) {
  data_dict <- data.frame(rbind(
    c("Age", "Age"), 
    c("Gender", "Gender"), 
    c("Education", "Education"), 
    c("Occupation", "Occupation"), 
    c("Marital_status", "Marital status"), 
    c("Living_status", "Living status"), 
    c("Housing_tenure", "Housing tenure"), 
    c("Current_residence_location", "Current residence location"), 
    c("Material_wealth", "Material wealth"), 
    c("Medical_insurance", "Medical insurance"), 
    c("Pension_insurance", "Pension insurance"), 
    c("Hypertension", "Hypertension"), 
    c("Diabetes", "Diabetes"), 
    c("Chronic_lung_diseases", "Chronic lung diseases"), 
    c("Heart_diseases", "Heart diseases"), 
    c("Stroke", "Stroke"), 
    c("Psychiatric_problems", "Psychiatric problems"), 
    c("Arthritis_or_rheumatism", "Arthritis or rheumatism"), 
    c("Dyslipidemia", "Dyslipidemia"), 
    c("Liver_diseases", "Liver diseases"), 
    c("Digestive_diseases", "Digestive diseases"), 
    c("Asthma", "Asthma"), 
    c("Memory_related_diseases", "Memory-related diseases"), 
    c("Depression", "Depression"), 
    c("Cognitive_impairment", "Cognitive impairment"), 
    c("Vision_impairment", "Vision impairment"), 
    c("Hearing_impairment", "Hearing impairment"), 
    c("ADLs_disability", "ADLs disability"), 
    c("Pain", "Pain"), 
    c("Complete_tooth_loss", "Complete tooth loss"), 
    c("Self_rated_health", "Self-rated health"), 
    c("Eating_habit", "Eating habit"), 
    c("Nighttime_sleep", "Nighttime sleep"), 
    c("Afternoon_napping", "Afternoon napping"), 
    c("Smoking", "Smoking"), 
    c("Drinking", "Drinking"), 
    c("Social_activities", "Social activities"), 
    c("Intellectual_activities", "Intellectual activities"), 
    c("Waist_circumference", "Waist circumference"), 
    c("PEF", "PEF"), 
    c("SBP", "SBP"), 
    c("DBP", "DBP"), 
    c("Low_HGS", "Low HGS"), 
    c("Anti_hypertensive_drugs", "Anti-hypertensive drugs"), 
    c("Anti_diabetic_drugs", "Anti-diabetic drugs"), 
    c("Lipid_lowering_drugs", "Lipid-lowering drugs"), 
    c("CRP", "CRP"), 
    c("HbA1c", "HbA1c"), 
    c("TC", "TC"), 
    c("HDL_C", "HDL-C"), 
    c("LDL_C", "LDL-C"), 
    c("TG", "TG"), 
    c("BUN", "BUN"), 
    c("Glucose", "Fasting glucose"), 
    c("UA", "UA"), 
    c("WBC", "WBC"), 
    c("Hemoglobin", "Hemoglobin"), 
    c("MCV", "MCV"), 
    c("Platelets", "Platelets"), 
    c("Hematocrit", "Hematocrit"), 
    c("RDKF", "RDKF"), 
    c("group", "group")
  ))
  names(data_dict) <- c("short", "long")
  
  output$variables_info <- renderDT({
    description <- read.csv("data/description.csv", header = T)
    description <- description[, -1]
    datatable(
      description,
      rownames = FALSE,
      options = list(
        dom = "t",
        paging = FALSE
      )
    )  
  })
  
  output$roc_curve <- renderImage({
    train_roc_data <- read.csv("data/train_roc_data.csv", header = T)
    train_roc_data_svm <- train_roc_data[train_roc_data$model == "SVM", ]
    train_roc_svm <- ggplot(train_roc_data_svm, aes(x = FPR, y = TPR, color = model)) + 
      geom_line(linewidth = 0.5) + 
      geom_abline(slope = 1, intercept = 0, linetype = "dashed") + 
      scale_x_continuous(breaks = seq(0, 1, by = 0.2), name = "1-Specificity") + 
      scale_y_continuous(breaks = seq(0, 1, by = 0.2), name = "Sensitivity") + 
      scale_color_manual(values = "#377EB8") + 
      theme_minimal() + 
      theme(
        panel.grid = element_blank(), 
        panel.border = element_rect(color = "black", fill = NA), 
        axis.ticks.x = element_line(color = "black"), 
        axis.ticks.y = element_line(color = "black"), 
        axis.ticks.length = unit(0.1, "cm"), 
        legend.title = element_blank(), 
        legend.position = c(0.85, 0.1)
      ) +
      coord_fixed(ratio = 1) 
    
    ggsave(plot = train_roc_svm, 
           filename = "train_roc_svm.png", 
           width = 10, 
           height = 10,
           units = "cm", 
           dpi = 600)
    
    list(
      src = "train_roc_svm.png", 
      height = "98%"
    )
  }, deleteFile = T) 
  
  segment_data <- data.frame(
    x = 0, y = 0, xend = 1,  yend = 1, 
    segment_type = "Ideal"
  )
  
  output$calibration_curve <- renderImage({
    train_calibration_data <- read.csv("data/train_calibration_data.csv", header = T)
    train_calibration_data_svm <- train_calibration_data[train_calibration_data$model == "SVM", ]
    train_calibration_data_svm$model <- factor(train_calibration_data_svm$model)
    train_calibration_svm <- ggplot(train_calibration_data_svm, aes(x = Pred, y = Obs, color = model)) +
      geom_line(linewidth = 0.5) + 
      geom_segment(data = segment_data, aes(x = x, y = y, xend = xend, yend = yend, color = segment_type), linewidth = 0.5, linetype = "dotted") +
      scale_x_continuous(limits = c(0, 1), 
                         breaks = seq(0, 1, by = 0.2), name = "Predicted Probability") + 
      scale_y_continuous(limits = c(0, 1), 
                         breaks = seq(0, 1, by = 0.2), name = "Actual Probability") + 
      scale_color_manual(
        values = c("#377EB8", "black"),
        labels = c("SVM", "Ideal")
      ) +
      labs(color = "") + 
      theme_minimal() + 
      theme(
        plot.background = element_rect(fill = "white", color = NA), 
        panel.grid = element_blank(), 
        panel.border = element_rect(color = "black", fill = NA), 
        axis.ticks.x = element_line(color = "black"), 
        axis.ticks.y = element_line(color = "black"), 
        axis.ticks.length = unit(0.1, "cm"), 
        legend.position = c(0.85, 0.15)
      ) +
      coord_fixed(ratio = 1) 
    
    
    ggsave(plot = train_calibration_svm, 
           filename = "train_calibration_svm.png", 
           width = 10, 
           height = 10,
           units = "cm", 
           dpi = 600)
    
    list(
      src = "train_calibration_svm.png", 
      height = "98%"
    )
  }, deleteFile = T)
  
  output$dca_curve <- renderImage({
    train_dca_data <- read.csv("data/train_dca_data.csv", header = T)
    train_dca_data_svm <- train_dca_data[train_dca_data$model == "SVM" | 
                                           train_dca_data$model == "All" | 
                                           train_dca_data$model == "None", ]
    train_dca_data_svm$model <- factor(train_dca_data_svm$model, levels=c("SVM", "All", "None"))
    train_dca_svm <- ggplot(train_dca_data_svm, aes(x = threshold, y = net_benefit, color = model)) +
      geom_line(linewidth = 0.5) + 
      scale_x_continuous(limits = c(0, 0.6), 
                         breaks = seq(0, 0.6, by = 0.2), 
                         name = "High Risk Threshold") + 
      scale_y_continuous(limits = c(-0.2, 0.8), 
                         breaks = seq(-0.2, 0.8, by = 0.2), 
                         name = "Standardized Net Benefit") + 
      scale_color_manual(values = c("#377EB8", "gray", "black")) +
      theme_minimal() + 
      theme(
        panel.grid = element_blank(), 
        panel.border = element_rect(color = "black", fill = NA), 
        axis.ticks.x = element_line(color = "black"), 
        axis.ticks.y = element_line(color = "black"), 
        axis.ticks.length = unit(0.1, "cm"), 
        legend.title = element_blank(), 
        legend.position = c(0.85, 0.8) 
      ) +
      coord_fixed(ratio = 2/4) 
    
    ggsave(plot = train_dca_svm, 
           filename = "train_dca_svm.png", 
           width = 10, 
           height = 10,
           units = "cm", 
           dpi = 600)
    
    list(
      src = "train_dca_svm.png",
      height = "98%"
    )
  }, deleteFile = T)
  
  data_template1 <- data.frame(
    ID = c("001", "002"),
    Gender = c("Man", "Woman"),
    Hypertension = c("No", "Yes"),
    Dyslipidemia = c("No", "Yes"),
    Waist_circumference = c(80, 100),
    Anti_hypertensive_drugs = c("No", "Yes"),
    Anti_diabetic_drugs = c("No", "Yes"),
    Lipid_lowering_drugs = c("No", "Yes"),
    HbA1c = c(5.3, 5.3),
    TC = c(190, 151),
    HDL_C = c(124, 82),
    LDL_C = c(372, 172),
    TG = c(42, 70),
    Fasting_glucose = c(163, 218),
    UA = c(4.6, 4.6),
    Hemoglobin = c(15, 20),
    Hematocrit = c(40, 50),
    RDKF = c("No", "Yes")
  )
  data_template2 <- data_template1[, -18]
  
  output$downloadTemplate1 <- downloadHandler(
    filename = function() {
      paste("data template", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      write.csv(data_template1, file, row.names = FALSE)
    }
  )
  
  train_data <- read.csv("data/train_data.csv", header = T)
  train_data <- train_data %>%
    mutate_if(is.character, as.factor)
  train_boruta <- read.csv("data/train_boruta.csv", header = T)
  train_boruta <- train_boruta[, -1]
  train_boruta$RDKF <- factor(train_boruta$RDKF, levels = c(1, 2))
  ratio <- sum(train_boruta$RDKF == "1") / sum(train_boruta$RDKF == "2")
  lgr::get_logger("mlr3")$set_threshold("warn")
  lgr::get_logger("bbotk")$set_threshold("warn")
  train_task <- as_task_classif(train_boruta, target = "RDKF")
  svm_learner <- lrn("classif.svm", predict_type = "prob")
  svm_learner$param_set$values <- list(
    type = "C-classification",
    kernel = "radial",
    cost = to_tune(1, 100),
    gamma = to_tune(0.001, 1),
    class.weights = c("2" = ratio, "1" = 1)
  )
  set.seed(123)
  svm <- mlr3verse::tune(tuner = tnr("grid_search", resolution = 5), 
                         task = train_task, 
                         learner = svm_learner, 
                         resampling = rsmp("cv", folds = 5),
                         measure = msr("classif.auc")
  )
  svm_learner$param_set$values <- svm$result_learner_param_vals
  svm_learner$train(train_task)
  
  external_result <- reactive({
    req(input$external_data)
    ext1 <- tools::file_ext(input$external_data$name)
    file1 <- switch(ext1,
                    csv = vroom::vroom(input$external_data$datapath, delim = ","),
                    validate("Invalid file; Please upload a .csv file.")
    )
    file1 <- file1[, -1]
    file1$Gender <- factor(file1$Gender, levels=c("Man", "Woman"))
    file1$Dyslipidemia <- factor(file1$Dyslipidemia, levels=c("No", "Yes"))
    file1$Hypertension <- factor(file1$Hypertension, levels=c("No", "Yes"))
    file1$Anti_hypertensive_drugs <- factor(file1$Anti_hypertensive_drugs, levels=c("No", "Yes"))
    file1$Anti_diabetic_drugs <- factor(file1$Anti_diabetic_drugs, levels=c("No", "Yes"))
    file1$Lipid_lowering_drugs <- factor(file1$Lipid_lowering_drugs, levels=c("No", "Yes"))
    file1$RDKF <- factor(file1$RDKF, levels=c("No", "Yes"))
    file1$Waist_circumference <- ifelse((file1$Gender == "Man" & file1$Waist_circumference >= 85) | 
                                          (file1$Gender == "Woman" & file1$Waist_circumference >= 80), "Abnormal", "Normal")
    file1$Waist_circumference <- factor(file1$Waist_circumference, levels=c("Normal", "Abnormal"))
    file1[, 8:16] <- lapply(file1[, 8:16], as.numeric)
    
    train_data17 <- train_data[, c("Gender", "Hypertension", "Dyslipidemia", 
                                   "Waist_circumference", "Anti_hypertensive_drugs", 
                                   "Anti_diabetic_drugs", "Lipid_lowering_drugs", 
                                   "HbA1c", "TC", "HDL_C", "LDL_C", "TG", "Glucose", 
                                   "UA", "Hemoglobin", "Hematocrit", "RDKF")]
    standardized_para_17 <- preProcess(train_data17, method = c("center", "scale"))
    file1_standardized <- predict(standardized_para_17, newdata = file1)
    file1_standardized <- file1_standardized %>% 
      mutate(across(where(is.factor), ~ as.numeric(.)))
    file1_standardized$RDKF <- factor(file1_standardized$RDKF)
    file1_task <- as_task_classif(file1_standardized, target = "RDKF")
    file1_pred <- svm_learner$predict(file1_task)
    file1_truth <- ifelse(file1_pred$truth == "2", 1, 0)
    file1_prob <- file1_pred$prob[, "2"]
    file1_auc <- file1_pred$score(msr("classif.auc"))
    file1_roc <- roc(file1_truth, file1_prob)
    file1_ci <- ci(file1_roc)
    file1_auc_with_ci <- sprintf("%.3f (%.3f-%.3f)", file1_auc, file1_ci[1], file1_ci[3]) 
    calculate_metrics <- function(pred_prob, true_class, threshold) {
      pred_class <- ifelse(pred_prob >= threshold, "1", "0")  
      cm <- table(
        factor(pred_class, levels = c("0", "1")),
        factor(true_class, levels = c("0", "1"))
      )
      TP <- cm["1", "1"]
      TN <- cm["0", "0"]
      FP <- cm["1", "0"]
      FN <- cm["0", "1"]
      sensitivity <- TP / (TP + FN)
      specificity <- TN / (TN + FP)
      PPV <- TP / (TP + FP)
      NPV <- TN / (TN + FN)
      CCR <- (TP + TN) / sum(cm)
      f1_score <- 2 * (PPV * sensitivity) / (PPV + sensitivity)
      list(Sensitivity = sensitivity, Specificity = specificity, PPV = PPV, NPV = NPV, CCR = CCR, F1_score =f1_score)
    }
    best_metrics_svm <- calculate_metrics(file1_prob, file1_truth, 0.071)
    brier_score_svm <- mean((file1_prob - file1_truth)^2)
    
    svm_metrics_result <- data.frame(
      AUC_CI = file1_auc_with_ci, 
      Threshold = 0.071, 
      Sensitivity = round(best_metrics_svm$Sensitivity, 3),
      Specificity = round(best_metrics_svm$Specificity, 3),
      PPV = round(best_metrics_svm$PPV, 3),
      NPV = round(best_metrics_svm$NPV, 3),
      CCR = round(best_metrics_svm$CCR, 3),
      F1_score = round(best_metrics_svm$F1_score, 3),
      Brier_score = round(brier_score_svm, 3)
    )
    
    file1_roc_data <- data.frame(
      FPR = 1-file1_roc$specificities, 
      TPR = file1_roc$sensitivities, 
      model = "SVM" 
    )
    
    roc_validation <- ggplot(file1_roc_data, aes(x = FPR, y = TPR, color = model)) + 
      geom_line(linewidth = 0.5) + 
      geom_abline(slope = 1, intercept = 0, linetype = "dashed") + 
      scale_x_continuous(breaks = seq(0, 1, by = 0.2), name = "1-Specificity") + 
      scale_y_continuous(breaks = seq(0, 1, by = 0.2), name = "Sensitivity") + 
      scale_color_manual(values = "#377EB8") + 
      theme_minimal() + 
      theme(
        panel.grid = element_blank(), 
        panel.border = element_rect(color = "black", fill = NA), 
        axis.ticks.x = element_line(color = "black"),
        axis.ticks.y = element_line(color = "black"),
        axis.ticks.length = unit(0.1, "cm"),
        legend.title = element_blank(),
        legend.position = c(0.85, 0.1)
      ) +
      coord_fixed(ratio = 1)
    
    validation_roc <- ggsave(plot = roc_validation, 
                             filename = "roc_validation.png", 
                             width = 10, 
                             height = 10,
                             units = "cm", 
                             dpi = 600)
    
    file1_task_data <- file1_task$data()
    file1_task_data$RDKF <- factor(ifelse(file1_task_data$RDKF == "2", 1, 0))
    
    file1_score <- Score(list(SVM = file1_pred$prob[, "2"]),
                         formula = RDKF ~ 1,
                         null.model = F,
                         plots = "calibration",
                         data = file1_task_data)
    
    file1_calibration_plot <- plotCalibration(file1_score, plot = FALSE)
    file1_calibration_data <- data.frame(
      Pred = file1_calibration_plot[["plotFrames"]][["SVM"]]$Pred, 
      Obs = file1_calibration_plot[["plotFrames"]][["SVM"]]$Obs,
      model = "SVM" 
    )
    file1_calibration_data$model <- factor(file1_calibration_data$model)
    
    file1_calibration <- ggplot(file1_calibration_data, aes(x = Pred, y = Obs, color = model)) +
      geom_line(linewidth = 0.5) +
      geom_segment(data = segment_data, aes(x = x, y = y, xend = xend, yend = yend, color = segment_type), linewidth = 0.5, linetype = "dotted") +
      scale_x_continuous(limits = c(0, 1),
                         breaks = seq(0, 1, by = 0.2), name = "Predicted Probability") + 
      scale_y_continuous(limits = c(0, 1),
                         breaks = seq(0, 1, by = 0.2), name = "Actual Probability") + 
      scale_color_manual(
        values = c("#377EB8", "black"),
        labels = c("SVM", "Ideal")
      ) + 
      labs(color = "") + 
      theme_minimal() + 
      theme(
        plot.background = element_rect(fill = "white", color = NA), 
        panel.grid = element_blank(), 
        panel.border = element_rect(color = "black", fill = NA), 
        axis.ticks.x = element_line(color = "black"), 
        axis.ticks.y = element_line(color = "black"),
        axis.ticks.length = unit(0.1, "cm"),
        legend.position = c(0.85, 0.15)
      ) +
      coord_fixed(ratio = 1)
    
    validation_calibration <- ggsave(plot = file1_calibration, 
                                     filename = "file1_calibration.png", 
                                     width = 10, 
                                     height = 10,
                                     units = "cm", 
                                     dpi = 600)
    
    file1_truth_p <- data.frame(
      truth = file1_truth,
      p = file1_prob
    )
    file1_dca_result <- decision_curve(truth ~ p, data = file1_truth_p, family = binomial, thresholds = seq(0, 1, by = 0.01))
    file1_dca_result$derived.data$model <- ifelse(file1_dca_result$derived.data$model == "truth ~ p", "SVM", file1_dca_result$derived.data$model)
    file1_dca_result$derived.data$model <- factor(file1_dca_result$derived.data$model, levels = c("SVM", "All", "None"))
    file1_dca <- ggplot(file1_dca_result$derived.data, aes(x = thresholds, y = sNB, color = model)) +
      geom_line(linewidth = 0.5) + 
      scale_x_continuous(limits = c(0, 0.6),
                         breaks = seq(0, 0.6, by = 0.2),
                         name = "High Risk Threshold") +
      scale_y_continuous(limits = c(-0.2, 0.8),
                         breaks = seq(-0.2, 0.8, by = 0.2),
                         name = "Standardized Net Benefit") +
      scale_color_manual(values = c("#377EB8", "gray", "black")) +
      theme_minimal() +
      theme(
        panel.grid = element_blank(),
        panel.border = element_rect(color = "black", fill = NA),
        axis.ticks.x = element_line(color = "black"),
        axis.ticks.y = element_line(color = "black"),
        axis.ticks.length = unit(0.1, "cm"),
        legend.title = element_blank(),
        legend.position = c(0.85, 0.8)
      ) +
      coord_fixed(ratio = 2/4)
    
    validation_dca <- ggsave(plot = file1_dca, 
                             filename = "file1_dca.png", 
                             width = 10, 
                             height = 10,
                             units = "cm", 
                             dpi = 600)
    
    return(list(svm_metrics_result = svm_metrics_result, 
                validation_roc = validation_roc, 
                validation_calibration = validation_calibration, 
                validation_dca = validation_dca))
  })
  
  output$external_performance <- renderDT({
    datatable(
      external_result()$svm_metrics_result,
      rownames = FALSE,
      colnames = c("AUC (95% CI)", "Threshold", "Sensitivity", "Specificity", "PPV", "NPV", "CCR", "F1 score", "Brier score"),
      options = list(
        dom = "t",
        paging = FALSE
      )
    )
  })
  
  output$roc_curve2 <- renderImage({
    list(
      src = external_result()$validation_roc,
      height = "96%"
    )
  }, deleteFile = T)
  
  output$calibration_curve2 <- renderImage({
    list(
      src = external_result()$validation_calibration,
      height = "96%"
    )
  }, deleteFile = T)
  
  output$dca_curve2 <- renderImage({
    list(
      src = external_result()$validation_dca,
      height = "96%"
    )
  }, deleteFile = T)
  
  output$downloadTemplate2 <- downloadHandler(
    filename = function() {
      paste("data template", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      write.csv(data_template2, file, row.names = FALSE)
    }
  )
  
  population_result <- reactive({
    req(input$population_data)
    ext2 <- tools::file_ext(input$population_data$name)
    file2 <- switch(ext2,
                    csv = vroom::vroom(input$population_data$datapath, delim = ","),
                    validate("Invalid file; Please upload a .csv file.")
    )
    ID <- file2$ID
    file2 <- file2[, -1]
    file2$Gender <- factor(file2$Gender, levels=c("Man", "Woman"))
    file2$Hypertension <- factor(file2$Hypertension, levels=c("No", "Yes"))
    file2$Dyslipidemia <- factor(file2$Dyslipidemia, levels=c("No", "Yes"))
    file2$Anti_hypertensive_drugs <- factor(file2$Anti_hypertensive_drugs, levels=c("No", "Yes"))
    file2$Anti_diabetic_drugs <- factor(file2$Anti_diabetic_drugs, levels=c("No", "Yes"))
    file2$Lipid_lowering_drugs <- factor(file2$Lipid_lowering_drugs, levels=c("No", "Yes"))
    
    file2$Waist_circumference <- ifelse((file2$Gender == "Man" & file2$Waist_circumference >= 85) | 
                                          (file2$Gender == "Woman" & file2$Waist_circumference >= 80), "Abnormal", "Normal")
    file2$Waist_circumference <- factor(file2$Waist_circumference, levels=c("Normal", "Abnormal"))
    file2[, 8:16] <- lapply(file2[, 8:16], as.numeric)
    
    train_data16 <- train_data[, c("Gender", "Hypertension", "Dyslipidemia", 
                                   "Waist_circumference", "Anti_hypertensive_drugs", 
                                   "Anti_diabetic_drugs", "Lipid_lowering_drugs", 
                                   "HbA1c", "TC", "HDL_C", "LDL_C", "TG", "Glucose", 
                                   "UA", "Hemoglobin", "Hematocrit")]
    standardized_para_16 <- preProcess(train_data16, method = c("center", "scale"))
    file2_standardized <- predict(standardized_para_16, newdata = file2)
    file2_standardized <- file2_standardized %>% 
      mutate(across(where(is.factor), ~ as.numeric(.)))
    
    file2_pred <- svm_learner$predict_newdata(file2_standardized)
    file2_prob <- data.frame(
      ID = ID,
      Prob1 = file2_pred$prob[, "1"],
      Prob2 = file2_pred$prob[, "2"]
    )
    file2_prob$response <- ifelse(file2_prob$Prob2 >= 0.071, "Yes", "No")
    file2_prob
  })
  
  output$population_predict_results <- renderDT({
    datatable(
      population_result(),
      rownames = FALSE,
      colnames = c("ID", "prob_No", "prob_Yes", "prediction class"),
    )
  })
  
  output$download_results <- downloadHandler(
    filename = function() {
      paste("Batch prediction results", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      write.csv(population_result(), file, row.names = FALSE)
    }
  )
  
  results <- eventReactive(input$Predict, {
    input_data <- data.frame(
      Gender = input$Gender,
      Hypertension = input$Hypertension,
      Dyslipidemia = input$Dyslipidemia,
      Waist_circumference = input$Waist_circumference, 
      Anti_hypertensive_drugs = input$Anti_hypertensive_drugs,
      Anti_diabetic_drugs = input$Anti_diabetic_drugs,
      Lipid_lowering_drugs = input$Lipid_lowering_drugs,
      HbA1c = input$HbA1c,
      TC = input$TC,
      HDL_C = input$HDL_C,
      LDL_C = input$LDL_C,
      TG = input$TG,
      Glucose = input$Glucose,
      UA = input$UA,
      Hemoglobin = input$Hemoglobin,
      Hematocrit = input$Hematocrit
    )
    input_data$Gender <- factor(input_data$Gender, levels=c("Man", "Woman"))
    input_data$Hypertension <- factor(input_data$Hypertension, levels=c("No", "Yes"))
    input_data$Dyslipidemia <- factor(input_data$Dyslipidemia, levels=c("No", "Yes"))
    input_data$Anti_hypertensive_drugs <- factor(input_data$Anti_hypertensive_drugs, levels=c("No", "Yes"))
    input_data$Anti_diabetic_drugs <- factor(input_data$Anti_diabetic_drugs, levels=c("No", "Yes"))
    input_data$Lipid_lowering_drugs <- factor(input_data$Lipid_lowering_drugs, levels=c("No", "Yes"))
    
    input_data$Waist_circumference <- ifelse((input_data$Gender == "Man" & input_data$Waist_circumference >= 85) | 
                                               (input_data$Gender == "Woman" & input_data$Waist_circumference >= 80), "Abnormal", "Normal")
    input_data$Waist_circumference <- factor(input_data$Waist_circumference, levels=c("Normal", "Abnormal"))
    train_data16 <- train_data[, c("Gender", "Hypertension", "Dyslipidemia", 
                                   "Waist_circumference", "Anti_hypertensive_drugs", 
                                   "Anti_diabetic_drugs", "Lipid_lowering_drugs", 
                                   "HbA1c", "TC", "HDL_C", "LDL_C", "TG", "Glucose", 
                                   "UA", "Hemoglobin", "Hematocrit")]
    standardized_para_16 <- preProcess(train_data16, method = c("center", "scale"))
    input_data_standardized <- predict(standardized_para_16, newdata = input_data)
    input_data_standardized <- input_data_standardized %>% 
      mutate(across(where(is.factor), ~ as.numeric(.)))
    pred <- svm_learner$predict_newdata(input_data_standardized)
    prob <- pred$prob[, "2"]
    train_x <- train_task$data(cols = train_task$feature_names)
    set.seed(123)
    train_x_200 <- train_x %>% 
      slice_sample(n = 200)
    
    input_SHAP_value <- kernelshap(svm_learner, X = input_data_standardized, bg_X = train_x_200, predict_type = "prob", verbose = F)
    input_sv_svm <- shapviz(input_SHAP_value, which_class = 2)
    input_shap_values_df <- as.data.frame(input_sv_svm$S)
    colnames(input_shap_values_df) <- c("Anti-diabetic drugs", "Anti-hypertensive drugs", "Dyslipidemia", 
                                        "Gender", "Fasting glucose", "HDL-C", "HbA1c", "Hematocrit", "Hemoglobin", 
                                        "Hypertension", "LDL-C", "Lipid-lowering drugs", "TC", "TG", 
                                        "UA", "Waist circumference")
    input_sv_svm$S <- as.matrix(input_shap_values_df)
    colnames(input_sv_svm$X) <- data_dict$long[match(names(input_sv_svm$X), data_dict$short)]
    
    input_force_plot <- sv_force(input_sv_svm, row_id = 1, max_display = Inf, 
                                 fill_colors = c("#ca0020", "#377EB8")) +
      labs(x = "SHAP value") +
      theme(panel.grid = element_blank())
    
    force_plot <- ggsave(plot = input_force_plot, 
                         filename = "input_force_plot.png", 
                         width = 20, 
                         height = 6,
                         units = "cm", 
                         dpi = 600)
    
    return(list(prob = prob, force_plot = force_plot))
  })
  
  output$prediction_result <- renderText({
    paste0(round(results()$prob * 100, 2), "%")
  })
  
  output$shap_force_plot <- renderImage({
    list(
      src = results()$force_plot,
      height = "77.5%"
    )
  }, deleteFile = T)
}

shinyApp(ui, server)

