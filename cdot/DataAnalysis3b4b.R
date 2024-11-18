source("./cdot/Requirements.R")
library("rhdf5")

# Load the data sets from .h5 files
# Loading 4b background
background_4b = h5read(paste0("bbbb", ".h5"), "/df/block0_values")
background_4b = as.data.frame(t(background_4b))
cols_4b = h5read(paste0("bbbb", ".h5"), "/df/block0_items")
colnames(background_4b) = cols_4b

# Loading 3b background
background_3b = h5read(paste0("./cdot/bbbj", ".h5"), "/df/block0_values")
background_3b = as.data.frame(t(background_3b))
cols_3b = h5read(paste0("./cdot/bbbj", ".h5"), "/df/block0_items")
colnames(background_3b) = cols_3b

# Loading signal
load("./cdot/Signal400.RData")
signal = Signal400
rm("Signal400")

h5closeAll()

# Explore the data sets
#dim(signal)
#dim(background_3b)
#dim(background_4b)
#colnames(signal)
#colnames(background_3b)
#colnames(background_4b)
#sum(colnames(background_3b) == colnames(background_4b))

#Choosing e, eta, phi, pt of the 4 jets, m4j and weight as variables of interest
col.names = c("e1", "e2", "e3", "e4", "eta1", "eta2", "eta3", "eta4",
              "phi1", "phi2", "phi3", "phi4", "pt1", "pt2", "pt3", "pt4",
              "m4j", "weight")
sig.cols = c(14:17, 6:9, 10:13, 2:5, 21, 27)
back.cols = c(2, 7, 12, 17, 3, 8, 13, 18, 5, 10, 15, 20, 6, 11, 16, 21, 39, 50)

# Signal and Background Data:
data_s = signal[, sig.cols]
colnames(data_s) = col.names
data_4b = background_4b[, back.cols]
colnames(data_4b) = col.names
data_3b = background_3b[, back.cols]
colnames(data_3b) = col.names
rm(list = c("background_4b", "background_3b", "signal"))

# Taking log transformation on the pt and e variables 

logind = c(1:4, 13:16)

logdata_s = data_s
logdata_s[, logind] = log(data_s[, logind])

logdata_4b = data_4b
logdata_4b[, logind] = log(data_4b[, logind])

logdata_3b = data_3b
logdata_3b[, logind] = log(data_3b[, logind])

# Transforming the 4 phi variables
# New transformed variables (3) give the angle between 
# the leading jet and the other 3 jets. 

phis = c(9:12)

logdata_s[, 10:12] = ((logdata_s[, 10:12] + pi -
  logdata_s[, 9]) %% (2 * pi)) - pi
logdata_s = logdata_s[, -9]

logdata_4b[, 10:12] = ((logdata_4b[, 10:12] + pi -
  logdata_4b[, 9]) %% (2 * pi)) - pi
logdata_4b = logdata_4b[, -9]

logdata_3b[, 10:12] = ((logdata_3b[, 10:12] + pi -
  logdata_3b[, 9]) %% (2 * pi)) - pi
logdata_3b = logdata_3b[, -9]
col.names = col.names[-9]

# Saving the data for future analysis
combdata = rbind(logdata_4b, logdata_3b, logdata_s)
combdata$Class = as.factor(c(rep("4b", nrow(logdata_4b)),
                             rep("3b", nrow(logdata_3b)),
                             rep("Signal", nrow(logdata_s))))
save(combdata, file = "./cdot/Data3b4b.RData")
