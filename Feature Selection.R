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

# Keep only the original features and shadowMax
all_features <- c(names(boruta_result$finalDecision), "shadowMax")
importance_score <- importance_score[importance_score$Feature %in% all_features, ]
importance_score$Feature[importance_score$Feature == "shadowMax"] <- "Shadow Max"

# Add decision labels (Confirmed, Tentative, Rejected)
importance_score$Decision <- ifelse(
  importance_score$Feature %in% names(boruta_result$finalDecision[boruta_result$finalDecision == "Confirmed"]),
  "Confirmed",
  ifelse(
    importance_score$Feature %in% names(boruta_result$finalDecision[boruta_result$finalDecision == "Tentative"]),
    "Tentative",
    ifelse(
      importance_score$Feature == "Shadow Max",
      "Shadow Max",
      "Rejected"
    )
  )
)

# Plot importance figure
boruta_plot <- ggplot(importance_score, aes(x = reorder(Feature, Importance, median), 
                                            y = Importance, 
                                            fill = Decision)) +
  geom_boxplot(
    linewidth = 0.2,      # Boxplot line width
    outlier.size = 0.5,   # Outlier point size
    outlier.stroke = 0.1  # Outlier border width
    ) + # Draw boxplot
  scale_fill_manual(values = c("Confirmed" = "green", "Tentative" = "yellow", 
                               "Rejected" = "red", "Shadow Max" = "blue")) + # Set colors
  labs(x = "", # X-axis title
       y = expression("Importance: " * italic(z) * " score") # Y-axis title
  ) +
  theme_minimal() + # Set plot theme
  theme(
    plot.background = element_rect(fill = "white", color = NA), # Overall plot background, no border color
    axis.text.x = element_text(angle = 45, hjust = 1), # Rotate X-axis labels to 45° and right-align
    panel.grid = element_blank(), # Remove grid lines
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.2), # Add panel border
    axis.ticks.x = element_line(color = "black", linewidth = 0.2), # X-axis tick marks
    axis.ticks.y = element_line(color = "black", linewidth = 0.2), # Y-axis tick marks
    axis.ticks.length = unit(0.07, "cm"), # Tick length (outside)
    legend.title = element_blank(), # Remove legend title
    legend.position = c(0.07, 0.85), # Place legend in the upper-left corner
    legend.key.size = unit(0.3, "cm"), # Legend key size
    legend.text = element_text(size = 30), # Legend text size
    plot.margin = margin(0.2, 0.2, -0.1, 0.2, "cm"), # Adjust plot margins
    text = element_text(size = 40),  # Global font size
    axis.text = element_text(size = 30)  # Axis tick label font size
  )

# Export Boruta importance plot
ggsave(plot = boruta_plot, 
       filename = "Figure S3 boruta_plot.png" , 
       width = 17, 
       height = 8.5,
       units = "cm", 
       dpi = 600)

