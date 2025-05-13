#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
json_input <- args[1]

library(xgboost)
library(jsonlite)

# Optional: define log file path
log_file <- "predict_log.txt"
log <- function(msg) {
  cat(sprintf("[%s] %s\n", Sys.time(), msg), file=log_file, append=TRUE)
}

log("Starting prediction script")

# Log raw input JSON
log(paste("Raw JSON input:", json_input))

# Load model and features
model <- xgb.load("xgboost_model.model")
features <- read.csv("important_features.csv", header=TRUE, as.is=TRUE)
input <- fromJSON(json_input)

# Ensure input is a data frame
if (!is.data.frame(input)) {
  input <- as.data.frame(input)
}

# Add missing columns with default value 0
missing_cols <- setdiff(features$x, names(input))
for (col in missing_cols) {
  input[[col]] <- 0
}

# Reorder columns
input <- input[features$x]

# Log final input passed to model
log(paste("Input to model:", toJSON(input)))

# Predict
dtest <- xgb.DMatrix(data = as.matrix(input))
score <- predict(model, dtest)

# Log prediction score
log(paste("Prediction score:", score))

# Decision
decision <- ifelse(score > 0.76, "Accept", "Reject")

# Log decision
log(paste("Decision:", decision))

# Output as JSON
output <- toJSON(list(
  probability = score,
  decision = decision
), auto_unbox = TRUE)

log(paste("Output JSON:", output))
log("Prediction script completed")
cat(output)
