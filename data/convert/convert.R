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
  export(paste('./data/convert/', name, '/mass.txt', sep = ''),
         data$m4j)
  export(paste('./data/convert/', name, '/tclass.txt', sep = ''),
         data$Trans_h)
  export(paste('./data/convert/', name, '/class.txt', sep = ''),
         data$h)
}

exports('3b/background', Background.Test.3b)
exports('4b/background', Background.Test.4b)
exports('val/background', Background.CDOT)
exports('signal', Signal.Test)




