## SHAP for the test cohort
# Extract predictor variables for training and test cohorts
train_x <- train_task$data(cols = train_task$feature_names)
test_x <- test_task$data(cols = test_task$feature_names)
# Randomly sample 200 rows from the training cohort
set.seed(123)
train_x_200 <- train_x %>% 
  slice_sample(n = 200)
# Compute SHAP values: svm_learner is the trained model; X = test_x is the test dataset to explain; bg_X = train_x_200 is the background dataset used as baseline for SHAP; predict_type = "prob" specifies probability predictions; verbose = F turns off verbose output.
shap_value <- kernelshap(svm_learner, X = test_x, bg_X = train_x_200, predict_type = "prob", verbose = F)
# Build visualization object
sv_svm <- shapviz(shap_value, which_class = 2) 
# Map variable names using the data dictionary
shap_values_df <- as.data.frame(sv_svm$S)
colnames(shap_values_df) <- c("Anti-diabetic drugs", "Anti-hypertensive drugs", "Dyslipidemia", 
                              "Gender", "Fasting glucose", "HDL-C", "HbA1c", "Hematocrit", "Hemoglobin", 
                              "Hypertension", "LDL-C", "Lipid-lowering drugs", "TC", "TG", 
                              "UA", "Waist circumference")
sv_svm$S <- as.matrix(shap_values_df)
colnames(sv_svm$X) <- data_dict$long[match(names(sv_svm$X), data_dict$short)]

#SHAP bar plot
shap_bar <- sv_importance(sv_svm, 
                          kind = "bar", #bar chart
                          max_display = Inf, #show all features
                          show_numbers = T, #show numbers (i.e., feature importance scores)
                          fill = "#48EDFE") + #bar fill color
  scale_x_continuous(limits = c(0, 0.05), #limit x-axis range
                     breaks = seq(0, 0.05, by = 0.01)) + #set x-axis breaks
  theme(
    axis.text.y = element_text(vjust = 0.5, hjust = 1, margin = margin(r = 1)),
    panel.background = element_rect(fill = "white"), #panel background white
    plot.background = element_rect(fill = "white"), #whole plotting background white
    panel.grid.major.y = element_blank(), #remove horizontal grid lines
    panel.grid.minor = element_blank(), #remove minor grid lines
    axis.title = element_text(size = 40), #axis title font size
    axis.text = element_text(size = 30), #axis tick label font size
    plot.tag = element_text(size = 50, face = "bold"), #set tag text size
    plot.margin = margin(0, 0, 0, 0, "cm"), #adjust margins
    axis.ticks = element_line(size = 0.25), #y-axis tick line width
    axis.ticks.length = unit(0.05, "cm"), #tick marks outside and length
    axis.ticks.y = element_blank() #remove y-axis tick marks
  )

#font size of numbers above SHAP bar plot
shap_bar$layers <- lapply(shap_bar$layers, function(layer) {
  if ("GeomText" %in% class(layer$geom)) {
    layer$aes_params <- modifyList(layer$aes_params, list(size = 10))
  }
  layer
})

#SHAP summary plot
shap_beeswarm <- sv_importance(sv_svm, 
                               kind = "beeswarm", #beeswarm plot
                               size = 0.5, #point size (beeswarm)
                               bee_width = 0.5, #horizontal spread width of points
                               max_display = Inf) + #show all features
  scale_color_gradient(low = "#6601F7", high = "#48EDFE") + 
  scale_x_continuous(limits = c(-0.45, 0.6), #limit x-axis range
                     breaks = seq(-0.4, 0.6, by = 0.2)) + #set x-axis breaks
  theme(
    panel.background = element_rect(fill = "white"), #panel background white
    plot.background = element_rect(fill = "white"), #whole plotting background white
    axis.title = element_text(size = 40), #axis title font size
    axis.text = element_text(size = 30), #axis tick label font size
    legend.title = element_text(size = 30), #legend title font size
    legend.text = element_text(size = 25), #legend text size
    legend.key.height = unit(0.73, "cm"), #legend bar height
    legend.key.width = unit(0.075, "cm"), #legend bar width
    plot.tag = element_text(size = 50, face = "bold"), #set tag text size
    plot.margin = margin(0, 0, 0, 0, "cm"), #adjust margins
    axis.ticks = element_line(size = 0.25), #y-axis tick line width
    axis.ticks.length = unit(0.05, "cm"), #tick marks outside and length
    axis.ticks.y = element_blank() #remove y-axis tick marks
  )

#SHAP force plot (single sample)
shap_force <- sv_force(sv_svm, row_id = 1, max_display = Inf, size = 30, 
                       fill_colors = c("#48EDFE", "#6601F7"), #colors
                       bar_label_size = 10, #bar label font size
                       annotation_size = 10) + #annotation text size
  labs(x = "SHAP value") + #change x-axis title to "SHAP value"
  theme(
    axis.title = element_text(size = 40), #axis title font size
    axis.text = element_text(size = 30), #axis tick label font size
    axis.ticks = element_line(size = 0.25), #y-axis tick line width
    axis.ticks.length = unit(0.05, "cm"), #tick marks outside and length
    axis.line.x = element_line(size = 0.25),  #set x-axis main line width
    panel.grid = element_blank(), #remove grid lines
    plot.margin = margin(0, 0, 0, 0, "cm"), #adjust margins
    plot.tag = element_text(size = 50, face = "bold") #set tag text size
  )

#draw waterfall plot for the first observation
shap_waterfall <- sv_waterfall(sv_svm, row_id = 1, size = 10000, 
                               fill_colors = c("#48EDFE", "#6601F7"), #colors
                               annotation_size = 10) + #annotation text size
  labs(x = "SHAP value") + #change x-axis title to "SHAP value"
  scale_x_continuous(breaks = seq(0.08, 0.16, by = 0.02)) + #set x-axis breaks
  theme(
    axis.title = element_text(size = 40), #axis title font size
    axis.text = element_text(size = 30), #axis tick label font size
    axis.ticks = element_line(size = 0.25), #y-axis tick line width
    axis.ticks.length = unit(0.05, "cm"), #tick marks outside and length
    axis.line.x = element_line(size = 0.25),  #set x-axis main line width
    panel.grid = element_blank(), #remove grid lines
    plot.margin = margin(0, 0, 0, 0, "cm"), #adjust margins
    plot.tag = element_text(size = 50, face = "bold") #set tag text size
  )

#Variable dependence plot
x <- c(
  "TC",
  "LDL-C",
  "HDL-C",
  "Hypertension",
  "Anti-hypertensive drugs",
  "TG",
  "Hemoglobin",
  "Waist circumference",
  "Lipid-lowering drugs",
  "Gender",
  "Dyslipidemia",
  "UA",
  "Hematocrit",
  "Fasting glucose",
  "HbA1c",
  "Anti-diabetic drugs"
)
#, color_var = NULL, color = "#377EB8"
shap_dependence <- sv_dependence(sv_svm, v = x)


#Combine SHAP bar plot, summary plot, waterfall plot, and force plot
shap_plot <- (shap_bar + shap_beeswarm) / shap_waterfall / shap_force + 
  plot_annotation(tag_levels = "A") + #automatically add A and B labels
  plot_layout(widths = c(3, 2, 2), heights = c(3, 2, 2))

#Export SHAP bar plot, summary plot, waterfall plot, and force plot
ggsave(plot = shap_plot, 
       filename = "Figure 2 shap_plot.png", 
       width = 17, 
       height = 15,
       units = "cm", 
       dpi = 600)

#Export SHAP dependence plot
ggsave(plot = shap_dependence, 
       filename = "Figure S8 shap_dependence.png", 
       width = 17, 
       height = 15,
       units = "cm", 
       dpi = 600)
