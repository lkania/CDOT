setwd("~/Dropbox/Anomaly Detection/Code/WTaggingData")
load("DataValWorkedDataSets.RData")
set.seed(7)
lambda = 0 # true overall signal strength.
n = 50000
nsim = 1000
ppermute = rep(0, nsim)
# NOTE: label is 1 for background and 0 for signal
Signal.Test = Test[Test$label == 0,]
Test = Test[Test$label == 1,] # From the RData file. 
threshold = 0.5


for(j in 1:nsim){
    n.sig = rbinom(1, size = n, prob = lambda)
    back.ind = sample(1:nrow(Test), size = n - n.sig)
    sig.ind = sample(1:nrow(Signal.Test), size = n.sig)
    Experimental = rbind(Test[back.ind,],
                         Signal.Test[sig.ind,]) # Mixture data (sub-sample)
    
    cut.data.index = which(Experimental$Trans_h > threshold)
    
    ppermute[j] = test(Experimental[cut.data.index]$mass)
}


# RData file data frames:

# Train: Training data for the classifier (label: 0 - Signal, 1 - Background)
# Val: Training data for the decorrelation - VALIDATION DATA (label same as above)
# Test: Test data (label same as above)


  

