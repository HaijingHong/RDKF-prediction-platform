## Install packages (run only if needed)
install.packages("haven")                 # read .dta (Stata) files
install.packages("dplyr")                 # data manipulation: anti_join(), mutate(), etc.
install.packages(c("dlookr", "ggplot2"))  # missing-data visualization, plots
install.packages("itertools")             # ensure missForest installation/operation
install.packages("missForest")            # missing-value imputation: missForest()
install.packages("caret")                 # data split: createDataPartition(); scaling: preProcess()
install.packages("Boruta")                # feature selection: Boruta()
install.packages("mlr3")                  # core ML framework
install.packages("mlr3verse")             # mlr3 ecosystem extensions
install.packages("devtools")              # needed for installing extra learners from GitHub
devtools::install_github("https://github.com/mlr-org/mlr3extralearners")  # extra algorithms for mlr3
install.packages("ranger")                # random forest (mlr3 does not bundle ranger)
install.packages("e1071")                 # SVM (mlr3 does not bundle e1071)
install.packages("xgboost")               # XGBoost (mlr3 does not bundle xgboost)
install.packages("remotes")               # required for installing catboost from URL
# install catboost (Windows prebuilt binary)
remotes::install_url('https://github.com/catboost/catboost/releases/download/v1.2.7/catboost-R-windows-x86_64-1.2.7.tgz',INSTALL_opts = c("--no-multiarch", "--no-test-load"))
install.packages("lightgbm")              # LightGBM (not bundled in mlr3)
install.packages("pROC")                  # ROC curves with 95% CI
install.packages("ResourceSelection")     # Hosmer¨CLemeshow test
install.packages("patchwork")             # combine ggplot figures
install.packages("rmda")                  # decision curve analysis
install.packages("kernelshap")            # SHAP value computation
install.packages("shapviz")               # SHAP visualizations
install.packages(c("officer", "flextable")) # export to Word
install.packages(c("gtsummary", "cardx")) # baseline characteristics tables
install.packages("rvest")                 # tidyverse-adjacent scraping dependency
install.packages(c("riskRegression", "tidyverse")) # calibration plots & data extraction
devtools::install_github("ricardo-bion/ggradar")   # radar charts
install.packages("Hmisc")                 # dependency for rms
install.packages("rms")                   # nomogram/clinical modeling
install.packages("psych")                 # correlation analysis
install.packages("corrplot")              # correlation heatmaps
install.packages("mice") #multiple interpolation
install.packages("tidymodels")            #recipe()
install.packages("themis")                #step_smotenc()
# shiny app dependencies
install.packages(c("shiny", "DT", "plotly", "iml", "shinydashboard", "shinyjs", "progressr"))
install.packages("rsconnect")             # cloud deployment

# Load packages
library(haven)
library(dplyr)
library(dlookr)
library(ggplot2)
library(missForest)
library(caret)
library(Boruta)
library(mlr3)
library(mlr3verse)
library(mlr3learners) #ML Algorithm packages
library(mlr3extralearners)
library(mlr3tuning)
library(pROC)
library(ResourceSelection)
library(patchwork)
library(rmda)
library(kernelshap) 
library(shapviz)
library(officer)
library(flextable)
library(gtsummary)
library(cardx)
library(riskRegression)
library(tidyverse)
library(ggradar)
library(rms)
library(psych)
library(corrplot)
library(mice)
library(tidymodels)
library(themis)

#Read .dta data from CHARLS
data_harmonized <- read_dta("D:/data/H_CHARLS_D_Data.dta")
data2011_household_income <- read_dta("D:/data/household_income.dta")
data2011_work_retirement_and_pension <- read_dta("D:/data/work_retirement_and_pension.dta")
data2011_health_care_and_insurance <- read_dta("D:/data/health_care_and_insurance.dta")
data2011_health_status_and_functioning <- read_dta("D:/data/health_status_and_functioning.dta")
data2011_blood <- read_dta("D:/data/Blood_20140429.dta")
data2015_blood <- read_dta("D:/data/Blood.dta")

#Data merging
ckd <- merge(data_harmonized, data2011_household_income[, c(1, 602, 983:1000)], by.x = "householdID_w1", by.y="householdID", all.x = T)
ckd <- merge(ckd, data2011_work_retirement_and_pension[, c(1:2, 4)], by.x = "ID_w1", by.y="ID", all.x = T)
ckd <- merge(ckd, data2011_health_care_and_insurance[, c(1, 4:13)], by = "ID_w1", by.y="ID", all.x = T)
ckd <- merge(ckd, data2011_health_status_and_functioning[, c(1, 29, 191, 197:199, 234:235, 251:264, 266:268, 270:272, 274, 289, 293, 518)], by = "ID_w1", by.y="ID", all.x = T)
ckd <- merge(ckd, data2011_blood[, c(1, 3:19)], by = "ID_w1", by.y="ID", all.x = T)
ckd <- merge(ckd, data2015_blood[, c(1, 5:22)], by = "ID", by.y="ID", all.x = T)

#Select required variables
rckd <- ckd[, c("ID_w1", "r1agey", "ragender", "ha065s1", "ha065s2", "ha065s3", 
                "ha065s4", "ha065s5", "ha065s6", "ha065s7", "ha065s8", "ha065s9", 
                "ha065s10", "ha065s11", "ha065s12", "ha065s13", "ha065s14", 
                "ha065s15", "ha065s16", "ha065s17", "ha065s18", "ha007", "raeduc_c", 
                "fa001", "h1rural", "r1mstat", "ea001s1", "ea001s2", "ea001s3", 
                "ea001s4", "ea001s5", "ea001s6", "ea001s7", "ea001s8", "ea001s9", 
                "ea001s10", "r1jcpen", "da033", "da039", "r1hibpe", "r1diabe", 
                "r1cancre", "r1lunge", "r1hearte", "r1stroke", "r1psyche", "r1arthre", 
                "r1dyslipe", "r1livere", "r1kidneye", "r1digeste", "r1asthmae", 
                "da007_12_", "da041", "da040", "r1shlta", "r1mwaist", "da058", 
                "da049", "da050", "r1smoken", "da067", "da069", "da056s1", "da056s2", 
                "da056s3", "da056s4", "da056s5", "da056s6", "da056s7", "da056s8", 
                "da056s9", "da056s10", "da056s11", "da056s12", "da057_1_", "da057_4_", 
                "da057_5_", "da057_6_", "da057_2_", "da057_8_", "da057_9_", 
                "da057_10_", "r1cesd10", "r1imrc", "r1dlrc", "r1ser7", "r1mo", 
                "r1dy", "r1yr", "r1dw", "dc003", "r1draw", "r1puff", "r1systo1", 
                "r1systo2", "r1systo3", "r1diasto1", "r1diasto2", "r1diasto3", 
                "r1adlab_c", "newcrp", "newhba1c", "newcho", "newhdl", "newldl", 
                "newtg", "newbun", "newcrea", "newglu", "newua", "cystatinc", 
                "qc1_vb002", "qc1_vb004", "bl_crea", "bl_cysc", "r1rxhibp_c", 
                "r1rxdiab_c", "r1rxdyslip_c", "r1lgrip", "r1rgrip", "r1iwy", 
                "r1iwm", "r3iwstat", "r3iwy", "r3iwm", "r3agey", "qc1_vb005", 
                "qc1_vb006", "qc1_vb009", "h1hhres", "qc1_va003")]

#Variable renaming
names(rckd)[c(2:3, 25, 37, 40:52, 57, 61, 94, 102:119, 122:130)] <- 
  c("Age_w1", "Gender", "Current_residence_location", "Pension_insurance", 
    "Hypertension", "Diabetes", "Cancer", "Chronic_lung_diseases", "Heart_diseases", 
    "Stroke", "Psychiatric_problems", "Arthritis_or_rheumatism", 
    "Dyslipidemia", "Liver_diseases", "Kidney_diseases", "Digestive_diseases", 
    "Asthma", "Waist_circumference", "Smoking", "PEF", "CRP", "HbA1c", "TC", 
    "HDL_C", "LDL_C", "TG", "BUN", "Creatinine_w1", "Glucose", "UA", "CystatinC_w1", 
    "WBC", "Hemoglobin", "Creatinine_w3", "CystatinC_w3", "Anti_hypertensive_drugs", 
    "Anti_diabetic_drugs", "Lipid_lowering_drugs", "Year_w1", "Month_w1", 
    "Alive_w3", "Year_w3", "Month_w3", "Age_w3", "Hematocrit", "MCV", "Platelets")

#Convert 0¨C1 variables to 1¨C2 by adding 1
rckd[, c(25, 37, 40:52, 61, 117:119)] <- rckd[, c(25, 37, 40:52, 61, 117:119)] + 1

### Variable coding
## Age: 1= 45-54 years, 2= 55-64 years, 3= ¡Ý65 years
rckd$Age <- ifelse(rckd$Age_w1 < 55, 1, 
                   ifelse(rckd$Age_w1 < 65, 2, 3))

## Waist_circumference: 1=Normal (men < 85 cm or women < 80 cm), 2=Abnormal (men ¡Ý 85 cm or women ¡Ý 80 cm) 
rckd$Waist_circumference <- ifelse((rckd$Gender == 1 & rckd$Waist_circumference >= 85) | 
                                     (rckd$Gender == 2 & rckd$Waist_circumference >= 80), 2, 1)

## Depression: 1=No (r1cesd10 < 10), 2=Yes (r1cesd10 ¡Ý 10)
rckd$Depression <- ifelse(rckd$r1cesd10 >= 10, 2, 1) 

#...

### Calculate rapid decline in kidney function (RDKF)
## Custom 2021 CKD-EPI formula using creatinine and cystatin C to calculate eGFR
eGFR <- function(x, y, age, gender) {
  A <- ifelse(gender == 2, 0.7, 0.9)
  B <- ifelse((gender == 2 & x <= 0.7), -0.219, 
              ifelse(gender == 1 & x <= 0.9, -0.144, -0.544))
  C <-  ifelse(y <= 0.8, -0.323, -0.778)
  D <- ifelse(gender == 2, 0.963, 1)
  135 * (x/A)^B * (y/0.8)^C * 0.9961^age * D
}
# Calculate eGFR in 2011: eGFR_w1
rckd$eGFR_w1 <- eGFR(x = rckd$Creatinine_w1, y = rckd$CystatinC_w1, age = rckd$Age_w1, gender = rckd$Gender)
# Calculate eGFR in 2015: eGFR_w3
rckd$eGFR_w3 <- eGFR(x = rckd$Creatinine_w3, y = rckd$CystatinC_w3, age = rckd$Age_w3, gender = rckd$Gender)
# Calculate the difference in eGFR between 2011 and 2015: diff_eGFR
rckd$diff_eGFR <- rckd$eGFR_w1 - rckd$eGFR_w3

## Calculate the number of years between two visit dates
# Extract 2011 visit date, set day to 01
rckd$Date_w1 <- as.Date(paste(rckd$Year_w1, rckd$Month_w1, "01", sep = "-"), format = "%Y-%m-%d")
# Extract 2015 visit date, set day to 01
rckd$Date_w3 <- as.Date(paste(rckd$Year_w3, rckd$Month_w3, "01", sep = "-"), format = "%Y-%m-%d")
# Compute year difference between two visit dates
rckd$diff_year <- as.numeric(difftime(rckd$Date_w3, rckd$Date_w1, units = "days"))/365
# Compute annual eGFR decline
rckd$eGFR_decline <- rckd$diff_eGFR/rckd$diff_year
# Rapid decline in kidney function (RDKF): 1 = eGFR_decline < 4, 2 = eGFR_decline ¡Ý 4
rckd$RDKF <- ifelse(rckd$eGFR_decline >= 4, 2, 1)

# Copy dataset as rckd1
rckd1 <- rckd

## Convert categorical variables to labeled factors
rckd1$Age <- factor(rckd1$Age, levels = c(1, 2, 3), labels = c("45-54 years", "55-64 years", "¡Ý65 years"))
rckd1$Gender <- factor(rckd1$Gender, levels = c(1, 2), labels = c("Man", "Woman"))
rckd1$Material_wealth <- factor(rckd1$Material_wealth, levels = c(1, 2, 3), labels = c("Low", "Medium", "High"))
rckd1$Housing_tenure <- factor(rckd1$Housing_tenure, levels = c(1, 2), labels = c("Non-ownership", "Partially/entirely ownership"))
rckd1$Education <- factor(rckd1$Education, levels = c(1, 2, 3), labels = c("Did not finish primary school", "Sishu/elementary school", "Middle school and above"))
rckd1$Occupation <- factor(rckd1$Occupation, levels = c(1, 2), labels = c("Non-agricultural work", "Agricultural work"))
rckd1$Current_residence_location <- factor(rckd1$Current_residence_location, levels = c(1, 2), labels = c("Urban", "Rural"))
rckd1$Marital_status <- factor(rckd1$Marital_status, levels = c(1, 2), labels = c("Other", "Married"))
rckd1$Living_status <- factor(rckd1$Living_status, levels = c(1, 2), labels = c("Not alone", "Alone"))
rckd1$Medical_insurance <- factor(rckd1$Medical_insurance, levels = c(1, 2, 3), labels = c("Low", "Medium", "High"))
rckd1$Pension_insurance <- factor(rckd1$Pension_insurance, levels = c(1, 2), labels = c("Others", "Government/institution/firm-provided social pension"))
rckd1$Vision_impairment <- factor(rckd1$Vision_impairment, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$Hearing_impairment <- factor(rckd1$Hearing_impairment, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$Hypertension <- factor(rckd1$Hypertension, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$Diabetes <- factor(rckd1$Diabetes, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$Cancer <- factor(rckd1$Cancer, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$Chronic_lung_diseases <- factor(rckd1$Chronic_lung_diseases, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$Heart_diseases <- factor(rckd1$Heart_diseases, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$Stroke <- factor(rckd1$Stroke, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$Psychiatric_problems <- factor(rckd1$Psychiatric_problems, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$Arthritis_or_rheumatism <- factor(rckd1$Arthritis_or_rheumatism, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$Dyslipidemia <- factor(rckd1$Dyslipidemia, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$Liver_diseases <- factor(rckd1$Liver_diseases, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$Kidney_diseases <- factor(rckd1$Kidney_diseases, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$Digestive_diseases <- factor(rckd1$Digestive_diseases, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$Asthma <- factor(rckd1$Asthma, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$Memory_related_diseases <- factor(rckd1$Memory_related_diseases, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$Pain <- factor(rckd1$Pain, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$Complete_tooth_loss <- factor(rckd1$Complete_tooth_loss, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$Self_rated_health <- factor(rckd1$Self_rated_health, levels = c(1, 2, 3), labels = c("Poor", "Fair", "Good"))
rckd1$Waist_circumference <- factor(rckd1$Waist_circumference, levels = c(1, 2), labels = c("Normal", "Abnormal"))
rckd1$Eating_habit <- factor(rckd1$Eating_habit, levels = c(1, 2), labels = c("¡Ý3 times/day", "<3 times/day"))
rckd1$Nighttime_sleep <- factor(rckd1$Nighttime_sleep, levels = c(1, 2, 3), labels = c("<6 h", "6-8 h", ">8 h"))
rckd1$Afternoon_napping <- factor(rckd1$Afternoon_napping, levels = c(1, 2, 3), labels = c("No napping", "<120 min", "¡Ý120 min"))
rckd1$Smoking <- factor(rckd1$Smoking, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$Drinking <- factor(rckd1$Drinking, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$Social_activities <- factor(rckd1$Social_activities, levels = c(1, 2, 3), labels = c("Sum score =0", "Sum score =1-2", "Sum score ¡Ý3"))
rckd1$Intellectual_activities <- factor(rckd1$Intellectual_activities, levels = c(1, 2, 3), labels = c("Sum score =0", "Sum score =1-2", "Sum score ¡Ý3"))
rckd1$Depression <- factor(rckd1$Depression, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$SBP <- factor(rckd1$SBP, levels = c(1, 2), labels = c("<140 mmHg", "¡Ý140 mmHg"))
rckd1$DBP <- factor(rckd1$DBP, levels = c(1, 2), labels = c("<90 mmHg", "¡Ý90 mmHg"))
rckd1$Anti_hypertensive_drugs <- factor(rckd1$Anti_hypertensive_drugs, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$Anti_diabetic_drugs <- factor(rckd1$Anti_diabetic_drugs, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$Lipid_lowering_drugs <- factor(rckd1$Lipid_lowering_drugs, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$ADLs_disability <- factor(rckd1$ADLs_disability, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$Low_HGS <- factor(rckd1$Low_HGS, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$Cognitive_impairment <- factor(rckd1$Cognitive_impairment, levels = c(1, 2), labels = c("No", "Yes"))
rckd1$RDKF <- factor(rckd1$RDKF, levels = c(1, 2), labels = c("No", "Yes"))

##-------- Inclusion/Exclusion Criteria --------
# Keep only participants surveyed in 2011  n=17708
rckd2 <- rckd1[nchar(rckd1$ID_w1) == 11 & !is.na(rckd1$ID_w1),]
# Exclude participants aged <45 years in 2011 or with missing age (777)  n=16931
rckd2 <- rckd2[rckd2$Age_w1 >= 45 & !is.na(rckd2$Age_w1), ]
# Exclude participants with kidney diseases in 2011 (1194)  n=15737
rckd2 <- rckd2[rckd2$Kidney_diseases == "No" & !is.na(rckd2$Kidney_diseases), ]
# Exclude participants with cancer in 2011 (198)  n=15539
rckd2 <- rckd2[rckd2$Cancer == "No" & !is.na(rckd2$Cancer), ]
# Exclude participants with missing eGFR in 2011 (7677)  n=7862
rckd2 <- rckd2[!is.na(rckd2$eGFR_w1), ] # final rckd2 is the dataset before applying 2015-related exclusions
# Exclude participants who withdrew/died by 2015 (999)  n=6863
rckd3 <- rckd2[rckd2$Alive_w3 == 1 & !is.na(rckd2$Alive_w3), ]
# Exclude participants with missing eGFR in 2015 (1851)  n=5012
rckd3 <- rckd3[!is.na(rckd3$eGFR_w3), ] # final rckd3 after applying 2015-related exclusions; dataset for analysis

# Dataset for baseline comparison between followed and lost-to-follow-up
# Use anti_join to find rows in rckd2 that are not in rckd3
not_in_rckd3 <- anti_join(rckd2, rckd3, by = "ID_w1")
# Add a 'group' column in rckd2_3: rckd2_3 is the dataset for baseline comparison of followed vs. lost-to-follow-up
rckd2_3 <- rckd2 %>%
  mutate(group = ifelse(ID_w1 %in% not_in_rckd3$ID_w1, "Lost to Follow-up", "Follow-up"))

#Keep only required predictors and outcome variables
variables <- c("Age", "Gender", "Education", "Occupation", "Marital_status", 
               "Living_status", "Housing_tenure", "Current_residence_location", 
               "Material_wealth", "Medical_insurance", "Pension_insurance", 
               "Hypertension", "Diabetes", "Chronic_lung_diseases", 
               "Heart_diseases", "Stroke", "Psychiatric_problems", 
               "Arthritis_or_rheumatism", "Dyslipidemia", "Liver_diseases", 
               "Digestive_diseases", "Asthma", "Memory_related_diseases", 
               "Depression", "Cognitive_impairment", "Vision_impairment", 
               "Hearing_impairment", "ADLs_disability", "Pain", "Complete_tooth_loss", 
               "Self_rated_health", "Eating_habit", "Nighttime_sleep", 
               "Afternoon_napping", "Smoking", "Drinking", "Social_activities", 
               "Intellectual_activities", "Waist_circumference", "PEF", "SBP", 
               "DBP", "Low_HGS", "Anti_hypertensive_drugs", "Anti_diabetic_drugs", 
               "Lipid_lowering_drugs", "CRP", "HbA1c", "TC", "HDL_C", "LDL_C", 
               "TG", "BUN", "Glucose", "UA", "WBC", "Hemoglobin", "MCV", 
               "Platelets", "Hematocrit")
rckd2 <- rckd2[, c(variables, "RDKF")]
rckd3 <- rckd3[, c(variables, "RDKF")]
rckd2_3 <- rckd2_3[, c(variables, "group")]

#Data dictionary
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

##--------------Multicollinearity diagnosis--------------
# Select numeric variables
numeric_vars <- rckd3[, sapply(rckd3, is.numeric)]
colnames(numeric_vars) <- data_dict$long[match(names(numeric_vars), data_dict$short)]
cor_matrix <- corr.test(numeric_vars, method = "pearson")
r <- cor_matrix$r # Extract correlation coefficients
png("Figure S1 corrplot.png", width = 3400, height = 3000, res = 600)
corrplot.mixed(r,
               lower = "number", # Lower triangle type is numbers
               number.cex = 3,
               upper = "color", # Upper triangle is color tiles
               lower.col = "black", # Lower triangle color
               tl.pos = "lt", # Show axis labels at left/top
               tl.col = "black", # Axis label color is black
               tl.srt = 30, # Top label rotation angle
               tl.cex = 4, # Axis label text size
               diag = "l", # 'l' means show lower panel content on diagonal
               cl.cex = 3, # Color legend text size
               cl.ratio = 0.1 # Color legend width ratio
) 
dev.off()

##--------------Missing value overview (Table S2)--------------
# Data dictionary mapping for rckd2 and rckd3 (run to export missingness tables and visualization)
# colnames(rckd2) <- data_dict$long[match(names(rckd2), data_dict$short)]
# colnames(rckd3) <- data_dict$long[match(names(rckd3), data_dict$short)]
# Calculate counts and proportions of missing values
missing <- data.frame(
  # Add variable names
  Variable = names(rckd3), 
  # Missing count in rckd2
  Missing_Count_rckd2 = sapply(rckd2, function(x) sum(is.na(x))), 
  # Missing percentage in rckd2, rounded to two decimals
  Missing_Percentage_rckd2 = sapply(rckd2, function(x) round(sum(is.na(x)) / nrow(rckd2) * 100, 2)),  
  # Missing count in rckd3
  Missing_Count_rckd3 = sapply(rckd3, function(x) sum(is.na(x))), 
  # Missing percentage in rckd3, rounded to two decimals
  Missing_Percentage_rckd3 = sapply(rckd3, function(x) round(sum(is.na(x)) / nrow(rckd3) * 100, 2))   
)
# Extract variable names in rckd3 with >20% missingness
missing_variable_rckd3 <- missing$Variable[missing$Missing_Percentage_rckd3 > 20]

# Determine variable type and add a Type column
missing$Type <- sapply(rckd3, function(x) if (is.factor(x)) "Categorical" else "Continuous")
# For categorical variables, output descriptions in the 'code - label' format
missing$Description <- sapply(names(rckd3), function(x) {
  if (is.factor(rckd3[[x]])) {
    levels_x <- levels(rckd3[[x]]) # Get all levels of variable x and store in levels_x
    paste(paste(1:length(levels_x), levels_x, sep = ": "), collapse = ", ") # Variable code: variable label
  } 
  # Unit descriptions for numeric variables
  else if (x == "PEF") {"L/min"} 
  else if (x == "CRP") {"mg/L"} 
  else if (x == "HbA1c" | x == "Hematocrit") {"%"} 
  else if (x == "TC" | x == "HDL-C" | x == "LDL-C" | x == "TG" | x == "BUN" | x == "Glucose" | x == "UA") {"mg/dL"} 
  else if (x == "WBC") {"in thousands"} 
  else if (x == "Hemoglobin") {"g/dL"} 
  else if (x == "MCV") {"fL"} 
  else if (x == "Platelets") {"10^9/L"} 
  else { NA } 
})
# Combine missing counts and percentages
missing$"Missing Count(Missing Percentage) (n=7862) [n(%)]" <- paste(missing$Missing_Count_rckd2, " (", missing$Missing_Percentage_rckd2, ")", sep = "")
missing$"Missing Count(Missing Percentage) (n=5012) [n(%)]" <- paste(missing$Missing_Count_rckd3, " (", missing$Missing_Percentage_rckd3, ")", sep = "")
# Select final columns needed
missing_description <- missing[, -c(2:5)]

## Plot missingness visualization: Pareto chart (share of missingness and cumulative percentage)
# only_na = T plots only the missing part; main = "": sets the main title to empty; grade = list(Low = 0.1, Middle = 0.2, High = 1): defines levels of missingness (low, middle, high).
missing_plot <- plot_na_pareto(rckd3, only_na = T, main = "", grade = list(Low = 0.1, Middle = 0.2, High = 1)) + 
  scale_fill_manual(
    name = "Missing Grade", # legend name is empty
    values = c("#00A088", "#A1D99C", "#FDD39F"), # legend colors
    labels = c("Low (¡Ü10%)", "Middle (¡Ü20%)", "High (¡Ü100%)")  # legend labels
  ) + 
  theme(plot.background = element_rect(fill = "white", color = NA), # overall plot background, no border color
        axis.title.x = element_blank(), # set x-axis title to empty
        axis.title.y = element_text(size = 60), # left y-axis title font size
        axis.title.y.right = element_text(size = 60), # right y-axis title font size
        axis.text.x = element_text(size = 50), # x-axis tick font size
        axis.text.y = element_text(size = 50), # y-axis tick font size
        legend.title = element_text(size = 50), # legend title font size
        legend.text = element_text(size = 50), # legend text font size
        legend.key.height = unit(0.5, "cm"), # legend key height
        legend.key.width = unit(0.5, "cm"), # legend key width
        plot.margin = margin(0, 0.5, 0.5, 0.5, "cm") # adjust margins
  )
# Adjust font size of numbers above bars
missing_plot$layers <- lapply(missing_plot$layers, function(layer) {
  if ("GeomText" %in% class(layer$geom)) {
    layer$aes_params <- modifyList(layer$aes_params, list(size = 10))
  }
  layer
})
# Export missingness visualization plot
ggsave(plot = missing_plot, 
       filename = "Figure S2 missing_plot.png", 
       width = 17, 
       height = 10,
       units = "cm", 
       dpi = 600)

# Remove variables with more than 20% missing: Pension_insurance, Cognitive_impairment
# Find column indices of these variable names in rckd3
missing_columns <- which(names(rckd3) %in% missing_variable_rckd3)
# Delete these columns
rckd2 <- rckd2[, -missing_columns]
rckd3 <- rckd3[, -missing_columns]
rckd2_3 <- rckd2_3[, -missing_columns]

##-------- Random forest imputation for missing values: missForest package --------
# Set random seed to ensure reproducibility of imputation
set.seed(10)
# Use missForest() to impute missing values in rckd3
data_missresult <- missForest(rckd3)
# Extract imputed data and convert to data frame
imputed_data <- as.data.frame(data_missresult$ximp)

##-------- Split dataset --------
# Set random seed to ensure reproducibility of the split
set.seed(10)
# createDataPartition() automatically samples equal proportions from each level of y; p specifies the train/test split ratio; list = F returns a vector; times = 1 means the split is executed once
trainindex <- createDataPartition(imputed_data$RDKF, p = 0.7, list = F, times = 1)
# Create training cohort
train_data <- imputed_data[trainindex, ]
# Create testing cohort
test_data <- imputed_data[-trainindex, ]

##-------- Standardization --------
# Use preProcess() to standardize the training cohort (centering and scaling) so each column has mean 0 and SD 1
standardized_para <- preProcess(train_data, method = c("center", "scale"))
# Standardize the training cohort
train_standardized <- predict(standardized_para, newdata = train_data)
# Standardize the testing cohort using the parameters learned from the training cohort
test_standardized <- predict(standardized_para, newdata = test_data)

# Convert all factor variables to numeric
train_standardized <- train_standardized %>% 
  mutate(across(where(is.factor), ~ as.numeric(.)))
test_standardized <- test_standardized %>% 
  mutate(across(where(is.factor), ~ as.numeric(.)))
# Convert the outcome variable RDKF to factor
train_standardized$RDKF <- factor(train_standardized$RDKF)
test_standardized$RDKF <- factor(test_standardized$RDKF)

#------------------Baseline characteristics comparison------------------
#Data dictionary 2
data_dict2 <- data.frame(rbind(
  c("Age", "Age, n (%)"), 
  c("Gender", "Gender, n (%)"), 
  c("Education", "Education, n (%)"), 
  c("Occupation", "Occupation, n (%)"), 
  c("Marital_status", "Marital status, n (%)"), 
  c("Living_status", "Living status, n (%)"), 
  c("Housing_tenure", "Housing tenure, n (%)"), 
  c("Current_residence_location", "Current residence location, n (%)"), 
  c("Material_wealth", "Material wealth, n (%)"), 
  c("Medical_insurance", "Medical insurance, n (%)"), 
  c("Pension_insurance", "Pension insurance, n (%)"), 
  c("Hypertension", "Hypertension, n (%)"), 
  c("Diabetes", "Diabetes, n (%)"), 
  c("Chronic_lung_diseases", "Chronic lung diseases, n (%)"), 
  c("Heart_diseases", "Heart diseases, n (%)"), 
  c("Stroke", "Stroke, n (%)"), 
  c("Psychiatric_problems", "Psychiatric problems, n (%)"), 
  c("Arthritis_or_rheumatism", "Arthritis or rheumatism, n (%)"), 
  c("Dyslipidemia", "Dyslipidemia, n (%)"), 
  c("Liver_diseases", "Liver diseases, n (%)"), 
  c("Digestive_diseases", "Digestive diseases, n (%)"), 
  c("Asthma", "Asthma, n (%)"), 
  c("Memory_related_diseases", "Memory-related diseases, n (%)"), 
  c("Depression", "Depression, n (%)"), 
  c("Cognitive_impairment", "Cognitive impairment, n (%)"), 
  c("Vision_impairment", "Vision impairment, n (%)"), 
  c("Hearing_impairment", "Hearing impairment, n (%)"), 
  c("ADLs_disability", "ADLs disability, n (%)"), 
  c("Pain", "Pain, n (%)"), 
  c("Complete_tooth_loss", "Complete tooth loss, n (%)"), 
  c("Self_rated_health", "Self-rated health, n (%)"), 
  c("Eating_habit", "Eating habit, n (%)"), 
  c("Nighttime_sleep", "Nighttime sleep, n (%)"), 
  c("Afternoon_napping", "Afternoon napping, n (%)"), 
  c("Smoking", "Smoking, n (%)"), 
  c("Drinking", "Drinking, n (%)"), 
  c("Social_activities", "Social activities, n (%)"), 
  c("Intellectual_activities", "Intellectual activities, n (%)"), 
  c("Waist_circumference", "Waist circumference, n (%)"), 
  c("PEF", "PEF (L/min), mean (SD)"), 
  c("SBP", "SBP, n (%)"), 
  c("DBP", "DBP, n (%)"), 
  c("Low_HGS", "Low HGS, n (%)"), 
  c("Anti_hypertensive_drugs", "Anti-hypertensive drugs, n (%)"), 
  c("Anti_diabetic_drugs", "Anti-diabetic drugs, n (%)"), 
  c("Lipid_lowering_drugs", "Lipid-lowering drugs, n (%)"), 
  c("CRP", "CRP (mg/L), mean (SD)"), 
  c("HbA1c", "HbA1c (%), mean (SD)"), 
  c("TC", "TC (mg/dL), mean (SD)"), 
  c("HDL_C", "HDL-C (mg/dL), mean (SD)"), 
  c("LDL_C", "LDL-C (mg/dL), mean (SD)"), 
  c("TG", "TG (mg/dL), mean (SD)"), 
  c("BUN", "BUN (mg/dL), mean (SD)"), 
  c("Glucose", "Fasting glucose (mg/dL), mean (SD)"), 
  c("UA", "UA (mg/dL), mean (SD)"), 
  c("WBC", "WBC (in thousands), mean (SD)"), 
  c("Hemoglobin", "Hemoglobin (g/dL), mean (SD)"), 
  c("MCV", "MCV (fL), mean (SD)"), 
  c("Platelets", "Platelets (10^9/L), mean (SD)"), 
  c("Hematocrit", "Hematocrit (%), mean (SD)"), 
  c("RDKF", "RDKF, n (%)"), 
  c("group", "group")
))
names(data_dict2) <- c("short", "long")

#---------(1) Baseline characteristics of non-lost and lost to follow-up
#rckd2_3 data dictionary mapping
colnames(rckd2_3) <- data_dict2$long[match(names(rckd2_3), data_dict2$short)]
baseline1 <- rckd2_3 %>% 
  tbl_summary(by = group, #grouped by group
              statistic = list(all_continuous() ~ "{mean} ({sd})", #continuous variables: mean (SD)
                               all_categorical() ~ "{n} ({p})"), #categorical variables: count (percentage)
              digits = list(all_continuous() ~ 2, #continuous variables keep 2 decimals for mean and SD
                            all_categorical() ~ c(0, 2)), #categorical variables' percentages keep 2 decimals
              type = list(all_continuous() ~ "continuous", #continuous variables display: mean (SD)
                          all_categorical() ~ "categorical"), #categorical variables display all levels: count (percentage)
              missing = "no" #do not show rows for missing values
  ) %>% 
  add_p(
    all_continuous() ~ "t.test", #t-test for continuous variables
    pvalue_fun = pvalue_format_vec #add P values and keep 3 decimals
  ) %>% 
  modify_header(statistic ~ "t test/¦Ö^2 (df)", p.value = "P value") %>% 
  modify_table_styling(
    columns = statistic,
    fmt_fun = function(x) style_number(x, digits = 3)
  ) %>%
  modify_table_styling(
    columns = parameter,
    fmt_fun = function(x) style_number(x, digits = 0)
  ) %>%
  modify_column_merge(
    pattern = "{statistic} ({parameter})",  # format: statistic (degrees of freedom)
    rows = !is.na(statistic)
  )

#export to Word
baseline1 %>%
  as_flex_table() %>%
  flextable::save_as_docx(baseline1, path = "Table S4.docx")

#---------(2) Baseline characteristics by RDKF status
colnames(rckd3) <- data_dict2$long[match(names(rckd3), data_dict2$short)]
baseline2 <- rckd3 %>% 
  tbl_summary(by = "RDKF, n (%)", #grouped by RDKF
              statistic = list(all_continuous() ~ "{mean} ({sd})", #continuous variables: mean (SD)
                               all_categorical() ~ "{n} ({p})"), #categorical variables: count (percentage)
              digits = list(all_continuous() ~ 2, #continuous variables keep 2 decimals for mean and SD
                            all_categorical() ~ c(0, 2)), #categorical variables' percentages keep 2 decimals
              type = list(all_continuous() ~ "continuous", #continuous variables display: mean (SD)
                          all_categorical() ~ "categorical"), #categorical variables display all levels: count (percentage)
              missing = "no" #do not show rows for missing values
  ) %>% 
  add_p(
    all_continuous() ~ "t.test",# categorical variables automatically choose chi-square or Fisher's exact test    
    pvalue_fun = pvalue_format_vec) %>% #add P values and keep 3 decimals
  modify_header(statistic ~ "t test/¦Ö^2 (df)", p.value = "P value") %>% 
  modify_table_styling(
    columns = statistic,
    fmt_fun = function(x) style_number(x, digits = 3)
  ) %>%
  modify_table_styling(
    columns = parameter,
    fmt_fun = function(x) style_number(x, digits = 0)
  ) %>%
  modify_column_merge(
    pattern = "{statistic} ({parameter})",  # format: statistic (degrees of freedom)
    rows = !is.na(statistic)
  ) %>% 
  add_overall() #add an overall summary column

#export to Word
baseline2 %>%
  as_flex_table() %>%
  flextable::save_as_docx(baseline2, path = "Table 1.docx")

#---------(3) Baseline characteristics comparison between training and testing cohorts
#add grouping for training and testing cohorts
train_data$group <- "Training cohort"
test_data$group <- "Testing cohort"
#merge training and testing cohorts
train_test <- rbind(train_data, test_data)
#convert group to factor
train_test$group <- factor(train_test$group, levels=c("Training cohort", "Testing cohort"))
#train_test data dictionary mapping
colnames(train_test) <- data_dict2$long[match(names(train_test), data_dict2$short)]
#perform statistics
baseline3 <- train_test %>% 
  tbl_summary(by = group, #grouped by group
              statistic = list(all_continuous() ~ "{mean} ({sd})", #continuous variables: mean (SD)
                               all_categorical() ~ "{n} ({p})"), #categorical variables: count (percentage)
              digits = list(all_continuous() ~ 2, #continuous variables keep 2 decimals for mean and SD
                            all_categorical() ~ c(0, 2)), #categorical variables' percentages keep 2 decimals
              type = list(all_continuous() ~ "continuous", #continuous variables display: mean (SD)
                          all_categorical() ~ "categorical"), #categorical variables display all levels: count (percentage)
              missing = "no" #do not show rows for missing values
  ) %>% 
  add_p(
    all_continuous() ~ "t.test", #t-test for continuous variables
    pvalue_fun = pvalue_format_vec #add P values and keep 3 decimals
  ) %>% 
  modify_header(statistic ~ "t test/¦Ö^2 (df)", p.value = "P value") %>% 
  modify_table_styling(
    columns = statistic,
    fmt_fun = function(x) style_number(x, digits = 3)
  ) %>%
  modify_table_styling(
    columns = parameter,
    fmt_fun = function(x) style_number(x, digits = 0)
  ) %>%
  modify_column_merge(
    pattern = "{statistic} ({parameter})",  # format: statistic (degrees of freedom)
    rows = !is.na(statistic)
  ) %>% 
  add_overall() #add an overall summary column

#export to Word
baseline3 %>%
  as_flex_table() %>%
  flextable::save_as_docx(baseline3, path = "Table S5.docx")
