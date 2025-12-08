##-------- Feature selection --------
# If drawing the Boruta importance plot, run this code first
# colnames(train_standardized) <- data_dict$long[match(names(train_standardized), data_dict$short)]
# Set random seed to ensure reproducibility
set.seed(10)
# Run Boruta(): doTrace = 0 to suppress progress, maxRuns = 50 for up to 50 iterations, getImp = getImpRfZ to compute feature importance
boruta_result <- Boruta(RDKF ~ ., data = train_standardized, doTrace = 0, maxRuns = 50, getImp = getImpRfZ)
# Get all confirmed features; withTentative = FALSE selects only features confirmed as important by Boruta
confirmed_features <- getSelectedAttributes(boruta_result, withTentative = F)
# Subset selected features and outcome
train_boruta <- train_standardized[, c(confirmed_features, "RDKF")]
test_boruta <- test_standardized[, c(confirmed_features, "RDKF")]

## Boruta importance plot
# Extract importance scores
importance_score <- boruta_result$ImpHistory %>%
  as.data.frame() %>%
  tidyr::gather(key = "Feature", value = "Importance")
# Add decision labels (Confirmed, Tentative, Rejected)
importance_score$Decision <- ifelse(
  importance_score$Feature %in% names(boruta_result$finalDecision[boruta_result$finalDecision == "Confirmed"]),
  "Confirmed",
  ifelse(
    importance_score$Feature %in% names(boruta_result$finalDecision[boruta_result$finalDecision == "Tentative"]),
    "Tentative",
    "Rejected"
  )
)
# Plot importance figure
boruta_plot <- ggplot(importance_score, aes(x = reorder(Feature, Importance, median), 
                                            y = Importance, 
                                            fill = Decision)) +
  geom_boxplot() + # draw boxplots
  scale_fill_manual(values = c("Confirmed" = "green", "Tentative" = "yellow", "Rejected" = "red")) + # set colors
  labs(x = "", # x-axis title
       y = "Importance: Z-Score" # y-axis title
  ) +
  theme_minimal() + # set plot theme
  theme(
    plot.background = element_rect(fill = "white", color = NA), # overall plot background; no border color
    axis.text.x = element_text(angle = 45, hjust = 1), # rotate X labels 45бу, right-justify
    panel.grid = element_blank(), # remove grid lines
    panel.border = element_rect(color = "black", fill = NA), # add border
    axis.ticks.x = element_line(color = "black"), # x-axis tick marks
    axis.ticks.y = element_line(color = "black"), # y-axis tick marks
    axis.ticks.length = unit(0.1, "cm"), # outer tick marks and length
    legend.title = element_blank(), # remove legend title
    legend.position = c(0.05, 0.9), # place legend at top-left
    plot.margin = margin(0.5, 0.5, 0, 0.5, "cm"), # adjust margins
    text = element_text(size = 60),  # global font size
    axis.text = element_text(size = 50)  # axis tick label size
  )
# Export Boruta importance plot
ggsave(plot = boruta_plot, 
       filename = "Figure S3 boruta_plot.png" , 
       width = 17, 
       height = 8.5,
       units = "cm", 
       dpi = 600)
