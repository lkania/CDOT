source("./cdot/Requirements.R")
# Load necessary packages
library(ggplot2)
library(ranger)
library(np)
library(gridExtra)

# Load the file with the CDOT functions
source("./cdot/CDOTFunctions.R")

set.seed(7) # Setting a random seed

# Load the 3b, 4b background and 400 GeV signal datasets
load("./cdot/Data3b4b.RData")
summary(combdata)
dim(combdata)

sum(combdata$Class == "Signal")
sum(combdata$Class == "3b")
sum(combdata$Class == "4b")
data.3b = combdata[which(combdata$Class == "3b"),]
data.sig = combdata[which(combdata$Class == "Signal"),]
data.4b = combdata[which(combdata$Class == "4b"),]

# Set sample sizes for training, validation and test data sets
mb = 50000 # Sample size of background training data to train classifier
nCDOT = 2 * 60000 # Sample size of background validation data to train CDOT
ms = 40000 # Sample size of signal training data to train classifier

# Sample sizes for creating a mixture data of signal strength lambda
lambda = 0.01
n = nrow(data.4b)
ns = rbinom(1, size = n, prob = lambda)
nb = n - ns

#Generating Background Train and Test, Signal and Experimental Data:

# 3b Datasets

permute.3b = sample(1:nrow(data.3b), size = nrow(data.3b)) # permuting the rows

# Background training data to train classifier
index1 = permute.3b[1:floor(nrow(data.3b) / 2)]
background.ind = sample(index1, size = mb,
                        prob = data.3b$weight[index1] / sum(data.3b$weight[index1]))

# Background validation data to train CDOT
index2 = permute.3b[(floor(nrow(data.3b) / 2) + 1):nrow(data.3b)]
background.ind.2 = sample(index2, size = nCDOT,
                          prob = data.3b$weight[index2] / sum(data.3b$weight[index2]))

# Background 3b test data
index3 = permute.3b[-c(background.ind, background.ind.2)]
background.ind.3 = index3

Background.Classifier = data.3b[background.ind,]
Background.CDOT = data.3b[background.ind.2,]
Background.Test.3b = data.3b[background.ind.3,]

# Signal Datasets

set.seed(3)

permute.sig = sample(1:nrow(data.sig), size = nrow(data.sig))

# Signal training data to train classifier
signal.ind = permute.sig[1:ms]
Signal.Classifier = data.sig[signal.ind,]

# Signal test data
Signal.Test = data.sig[-signal.ind,]

# 4b Dataset - test data

Background.Test.4b = data.4b


# Fitting the classifier h on training 3b background and signal Data

combdata = rbind(Background.Classifier[, -c(16:17)],
                 Signal.Classifier[, -c(16:17)])
combdata$Class = factor(combdata$Class, levels = c("3b", "Signal"))


randomF = ranger(Class ~ ., data = combdata,
                 num.trees = 1000,
                 mtry = floor(sqrt(10)),
                 min.node.size = 100,
                 probability = T)

# Predicting the classifier on the different data sets

Background.Classifier$h = predict(randomF,
                                  Background.Classifier[, -c(16:17)])$predictions[, 2]
Background.CDOT$h = predict(randomF,
                            Background.CDOT[, -c(16:17)])$predictions[, 2]
Background.Test.3b$h = predict(randomF,
                               Background.Test.3b[, -c(16:17)])$predictions[, 2]
Signal.Classifier$h = predict(randomF,
                              Signal.Classifier[, -c(16:17)])$predictions[, 2]
Signal.Test$h = predict(randomF,
                        Signal.Test[, -c(16:17)])$predictions[, 2]
Background.Test.4b$h = predict(randomF,
                               Background.Test.4b[, -c(16:17)])$predictions[, 2]


# Fitting the Decorrelation Algorithm CDOT on Background 3b validation data

h.back = Background.CDOT$h
m.back = Background.CDOT$m4j

# Setting the parameters of the algorithm

splits = c(500, 1000, 1500)
scaled = FALSE
logit.y.scale = FALSE
log.x.scale = FALSE


# Training CDOT
CDOT.fit = fit.CDOT(x = m.back, y = h.back,
                    scaled = scaled,
                    logit.y.scale = logit.y.scale,
                    log.x.scale = log.x.scale,
                    splits = splits)

# Predicting CDOT on all the datasets

Background.CDOT.fit = predict.CDOT(CDOT.model = CDOT.fit,
                                   x = Background.CDOT$m4j,
                                   y = Background.CDOT$h,
                                   scaled = scaled,
                                   logit.y.scale = logit.y.scale,
                                   log.x.scale = log.x.scale,
                                   splits = splits)

Background.Classifier.fit = predict.CDOT(CDOT.model = CDOT.fit,
                                         x = Background.Classifier$m4j,
                                         y = Background.Classifier$h,
                                         scaled = scaled,
                                         logit.y.scale = logit.y.scale,
                                         log.x.scale = log.x.scale,
                                         splits = splits)


Background.Test.3b.fit = predict.CDOT(CDOT.model = CDOT.fit,
                                      x = Background.Test.3b$m4j,
                                      y = Background.Test.3b$h,
                                      scaled = scaled,
                                      logit.y.scale = logit.y.scale,
                                      log.x.scale = log.x.scale,
                                      splits = splits)

Background.Test.4b.fit = predict.CDOT(CDOT.model = CDOT.fit,
                                      x = Background.Test.4b$m4j,
                                      y = Background.Test.4b$h,
                                      scaled = scaled,
                                      logit.y.scale = logit.y.scale,
                                      log.x.scale = log.x.scale,
                                      splits = splits)


Signal.Test.fit = predict.CDOT(CDOT.model = CDOT.fit,
                               x = Signal.Test$m4j,
                               y = Signal.Test$h,
                               scaled = scaled,
                               logit.y.scale = logit.y.scale,
                               log.x.scale = log.x.scale,
                               splits = splits)

Signal.Classifier.fit = predict.CDOT(CDOT.model = CDOT.fit,
                                     x = Signal.Classifier$m4j,
                                     y = Signal.Classifier$h,
                                     scaled = scaled,
                                     logit.y.scale = logit.y.scale,
                                     log.x.scale = log.x.scale,
                                     splits = splits)


# Adding the transformed classifier scores to the data sets
Background.CDOT$Trans_h = Background.CDOT.fit$Trans_h
Background.Classifier$Trans_h = Background.Classifier.fit$Trans_h
Background.Test.3b$Trans_h = Background.Test.3b.fit$Trans_h
Background.Test.4b$Trans_h = Background.Test.4b.fit$Trans_h
Signal.Test$Trans_h = Signal.Test.fit$Trans_h
Signal.Classifier$Trans_h = Signal.Classifier.fit$Trans_h

# Saving the decorrelated datasets
save(Background.CDOT, Background.Classifier,
     Background.Test.3b, Background.Test.4b,
     Signal.Classifier, Signal.Test, file = "./cdot/DecorrelatedData3b4b.Rdata")
