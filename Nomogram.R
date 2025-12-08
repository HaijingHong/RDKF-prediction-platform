##------------------Nomogram------------------
train_boruta_nom <- train_boruta
colnames(train_boruta_nom) <- data_dict$long[match(names(train_boruta_nom), data_dict$short)]
#Generate data distribution object (convert data to the format required by the rms package)
dd <- datadist(train_boruta_nom)
#Set data distribution object dd as a global option
options(datadist = "dd")
#Fit logistic regression model using the training cohort (16 variables)
lr_model <- lrm(RDKF ~ ., data = train_boruta_nom)
nomogram <- nomogram(lr_model,
                     lp = F, #Whether to display the coefficient axis
                     fun.at = seq(0, 1, by = 0.2), #Set ticks for the risk axis
                     fun = plogis,
                     funlabel = "Predicted Probability")

png("nomogram.png", width = 8000, height = 5000, res = 600)
par(mar = c(1, 0.5, 1, 0.5)) #Bottom, left, top, right margins
plot(nomogram,
     cex.axis = 5, # Font size of tick labels
     cex.var = 7, # Font size of variable names
     cex.sub = 7) # Subtitle font size (e.g., 'Points')
dev.off()

#Fit logistic regression model using the training cohort (the top 10 predictors ranked by the SHAP bar plot)
lr_model_10 <- lrm(
  RDKF ~ TC + `LDL-C` + `HDL-C` + Hypertension + `Anti-hypertensive drugs` + TG + 
    Hemoglobin + `Waist circumference` + `Lipid-lowering drugs` + Gender,
  data = train_boruta_nom)

# Generate predicted probabilities from the fitted logistic regression model (lr_model_10)
pred_prob <- predict(lr_model_10, type = "fitted")

# Create an ROC curve object using the true labels and predicted probabilities
roc_lr10 <- roc(train_boruta_nom$RDKF, pred_prob)

# Compute the confidence interval for the AUC of the ROC curve (default: 95% CI)
ci_lr10 <- ci(roc_lr10)

nomogram_10 <- nomogram(lr_model_10, 
                        lp = F, #Whether to display the coefficient axis
                        fun.at = seq(0, 1, by = 0.2), #Set ticks for the risk axis
                        fun = plogis, 
                        funlabel = "Predicted Probability")

png("nomogram_10.png", width = 8000, height = 3600, res = 600)
par(mar = c(1, 0.5, 1, 0.5)) #Bottom, left, top, right margins
plot(nomogram_10,
     cex.axis = 5, # Font size of tick labels
     cex.var = 7, # Font size of variable names
     cex.sub = 7) # Subtitle font size (e.g., 'Points')
dev.off()
