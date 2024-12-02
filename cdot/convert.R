##################################################
# Script for converting .Rdata files to .txt files
##################################################
set.seed(123)

source("./cdot/Requirements.R")
library(ggplot2)
library(dplyr)
library(tidyr)

load(file = "./cdot/DecorrelatedData3b4b.RData")
# RData file data frames:
# Background.Classifier: Training data for the classifier
# Signal.Classifier: Training signal data for the classifier
# Background.CDOT: Training data for the decorrelation - VALIDATION DATA
# Background.Test.3b: Test 3b data
# Background.Test.4b: Test 4b data
# Signal.Test: Test signal data

export <- function(file, data) {
  write.table(x = data,
              file = file,
              append = FALSE,
              sep = "",
              col.names = FALSE,
              row.names = FALSE)

}

# Function to calculate Garwood confidence intervals
garwood_ci <- function(count, conf_level = 0.95) {
  if (count == 0) {
    # Special case for zero counts
    lower <- 0
    upper <- qchisq(conf_level, df = 2 * (count + 1)) / 2
  } else {
    alpha <- 1 - conf_level
    lower <- qchisq(alpha / 2, df = 2 * count) / 2
    upper <- qchisq(1 - alpha / 2, df = 2 * (count + 1)) / 2
  }
  c(lower, upper)
}

# Plot dataset
plot_dataset <- function(data, name) {
  # Create a data frame
  df <- data.frame(obs = 1 - exp(-0.003 * (data$m4j - min(data$m4j))),
                   weights = data$weight)

  # Subsampling using weights
  # sampled_indices <- sample(
  #   x = seq_len(nrow(df)),        # Indices of the dataframe
  #   size = 20000,                     # Desired sample size
  #   prob = df$weight,             # Weights for sampling
  #   replace = TRUE               # With replacement
  # )
  # df <- df[sampled_indices,]
  # df$weights <- 1

  n_bins <- 500
  bin_breaks <- seq(0, 1, length.out = n_bins)
  bin_mid <- (bin_breaks[-1] + bin_breaks[-n_bins]) / 2
  bins <- cut(df$obs, breaks = bin_breaks, include.lowest = TRUE)

  # Bin the data and compute confidence intervals
  binned_data <- df %>%
    mutate(bin = cut(obs, breaks = bin_breaks, include.lowest = TRUE)) %>%
    group_by(bin) %>%
    summarize(
      count = sum(weights),
      .groups = "drop"
    ) %>%
    rowwise() %>%
    mutate(
      ci = list(garwood_ci(count)),
      lower_ci = ci[[1]],
      upper_ci = ci[[2]]
    ) %>%
    ungroup() %>%
    complete(bin = bins,
             fill = list(count = 0,
                         lower_ci = 0,
                         upper_ci = qchisq(0.95, df = 2) / 2))

  binned_data$bin <- bin_mid

  # Plot histogram with confidence intervals
  p <- ggplot(binned_data, aes(x = bin, y = count)) +
    # geom_line(color = "blue") +               # Line plot
    geom_point(color = "red", size = 0.5, alpha = 1) +    # Points
    geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci),
                  width = 0.0001,
                  color = "black",
                  alpha = 0.6) +
    labs(title = paste(name,
                       " Histogram with Garwood CI (n=", dim(df)[1], ")",
                       sep = ''),
         x = "Invariant mass", y = "Counts") +
    # ylim(0, 800) +
    theme_minimal()

  ggsave(paste('./data/', name, '/distribution.png', sep = ''),
         plot = p,
         width = 8,
         height = 6, dpi = 300)
}

exports <- function(name, data) {
  export(paste('./data/', name, '/mass.txt', sep = ''),
         data$m4j)
  export(paste('./data/', name, '/tclass.txt', sep = ''),
         data$Trans_h)
  export(paste('./data/', name, '/class.txt', sep = ''),
         data$h)
  export(paste('./data/', name, '/weight.txt', sep = ''),
         data$weight)
  plot_dataset(data, name)
}

exports('3b/test/background', Background.Test.3b)
exports('3b/test/signal', Signal.Test)

exports('3b/val/background', Background.CDOT)
exports('3b/val/signal', Signal.Classifier)

exports('4b/test/background', Background.Test.4b)
exports('4b/test/signal', Signal.Test)
exports('4b/val/background', Background.CDOT)
exports('4b/val/signal', Signal.Classifier)
#####################################################################
# W-Tagging Datasets
#####################################################################

load(file = "./cdot/WTaggingDecorrelated.Rdata")
# RData file data frames:
# Train: Training data for the classifier (label: 0 - Signal, 1 - Background)
# Val: Training data for the decorrelation - VALIDATION DATA (label same as above)
# Test: Test data (label same as above)

exports <- function(name, data) {
  export(paste('./data/', name, '/mass.txt', sep = ''),
         data$mass)
  export(paste('./data/', name, '/tclass.txt', sep = ''),
         data$Trans_h)
  export(paste('./data/', name, '/class.txt', sep = ''),
         data$h)
}

# NOTE: label is 1 for background and 0 for signal
exports('WTagging/test/background', Test[Test$label == 1,])
exports('WTagging/test/signal', Test[Test$label == 0,])
exports('WTagging/val/background', Val[Val$label == 1,])
exports('WTagging/val/signal', Val[Val$label == 0,])






