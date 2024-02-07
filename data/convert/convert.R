load(file = "./data/convert/DecorrelatedData.RData")
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
              sep = "",
              col.names = FALSE,
              row.names = FALSE)

}

exports <- function(name, data) {
  export(paste('./data/', name, '/mass.txt', sep = ''),
         data$m4j)
  export(paste('./data/', name, '/tclass.txt', sep = ''),
         data$Trans_h)
  export(paste('./data/', name, '/class.txt', sep = ''),
         data$h)
}

exports('3b/background', Background.Test.3b)
exports('3b/signal', Signal.Test)
exports('3b/val/background', Background.CDOT)

exports('4b/background', Background.Test.4b)
exports('4b/signal', Signal.Test)
exports('4b/val/background', Background.CDOT)
#####################################################################
# W-Tagging Datasets
#####################################################################

load(file = "./data/convert/DataValWorkedDataSets.RData")
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
exports('WTagging/background', Test[Test$label == 1,])
exports('WTagging/signal', Test[Test$label == 0,])
exports('WTagging/val/background', Val[Val$label == 1,])






