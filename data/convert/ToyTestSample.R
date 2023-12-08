set.seed(7)
lambda = 0 # true overall signal strength.
n = 10000
nsim = 1000
ppermute = rep(0, nsim)
Test = Background.Test.3b # From the RData file.
threshold = 0.5


for(j in 1:nsim){
    n.sig = rbinom(1, size = n, prob = lambda)
    back.ind = sample(1:nrow(Test), size = n - n.sig)
    sig.ind = sample(1:nrow(Signal.Test), size = n.sig)
    Experimental = rbind(Test[back.ind,],
                         Signal.Test[sig.ind,]) # Mixture data (sub-sample)
    
    cut.data.index = which(Experimental$Trans_h > threshold)
    
    ppermute[j] = test(Experimental[cut.data.index]$m4j)
}


# RData file data frames:

# Background.Classifier: Training data for the classifier
# Signal.Classifier: Training signal data for the classifier
# Background.CDOT: Training data for the decorrelation - VALIDATION DATA
# Background.Test.3b: Test 3b data
# Background.Test.4b: Test 4b data
# Signal.Test: Test signal data

  

