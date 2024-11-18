source("./cdot/Requirements.R")
# Load required packages
library(ggplot2)
library(ranger)
library(np)
library(gridExtra)

# Load the file with the CDOT functions
source("./cdot/CDOTFunctions.R")

set.seed(7) # Setting a random seed

# Load the WTagging datasets
Train = read.csv(file = './cdot/WTaggingTrain.csv')
Test = read.csv(file = './cdot/WTaggingTest.csv')
Val = read.csv(file = './cdot/WTaggingVal.csv')

# Deleting the ID column from all the datasets
Train = Train[, -1]
Test = Test[, -1]
Val = Val[, -1]

# Storing the no. of rows
n.train = nrow(Train)
n.test = nrow(Test)
n.val = nrow(Val)

# Setting label as a categorical variable
# 0 is signal and 1 is background
Train$label = as.factor(Train$label)
Test$label = as.factor(Test$label)
Val$label = as.factor(Val$label)


# Training the Random Forest Supervised Classifier on training data
randomF = ranger(label ~ ., data = Train[, -1],
                 num.trees = 1000,
                 mtry = floor(sqrt(10)),
                 min.node.size = 800,
                 probability = T)

# Fitting the Classifier on the different data sets (h = P(Signal|x))

Train$h = predict(randomF, Train)$predictions[, 1]
Test$h = predict(randomF, Test)$predictions[, 1]
Val$h = predict(randomF, Val)$predictions[, 1]

# Setting the parameters of the algorithm

scaled = TRUE # Apply the scaling mentioned in Approach 2 of CDOT
logit.y.scale = TRUE # Take logit transformation of h
log.x.scale = TRUE # Take log transformation of mass

# The CDOT algorithm is trained on the background Validation data
train.h = Val$h[Val$label == 1]
train.m = Val$mass[Val$label == 1]

# Training the CDOT algorithm
CDOT.fit = fit.CDOT(x = train.m, y = train.h,
                    scaled = scaled,
                    logit.y.scale = logit.y.scale,
                    log.x.scale = log.x.scale)

# Predicting it on data
# Test data
Test.fit = predict.CDOT(CDOT.model = CDOT.fit,
                        x = Test$mass,
                        y = Test$h,
                        scaled = scaled,
                        logit.y.scale = logit.y.scale,
                        log.x.scale = log.x.scale)


# Validation data
Val.fit = predict.CDOT(CDOT.model = CDOT.fit,
                       x = Val$mass,
                       y = Val$h,
                       scaled = scaled,
                       logit.y.scale = logit.y.scale,
                       log.x.scale = log.x.scale)

# Training data
Train.fit = predict.CDOT(CDOT.model = CDOT.fit,
                         x = Train$mass,
                         y = Train$h,
                         scaled = scaled,
                         logit.y.scale = logit.y.scale,
                         log.x.scale = log.x.scale)


# Adding the transformed classifier output to the data frames

Train$Trans_h = Train.fit$Trans_h
Test$Trans_h = Test.fit$Trans_h
Val$Trans_h = Val.fit$Trans_h

# If in the above code predict.CDOT takes too long to run can use the below

#for(i in 1:(nrow(Test)/5000)){
#  print(i)
#  Test.fit.iter = predict.CDOT(CDOT.model = CDOT.fit,
#                               x = Test$mass[((i - 1)*5000+1):(i*5000)],
#                               y = Test$h[((i - 1)*5000+1):(i*5000)],
#                               scaled = scaled,
#                               logit.y.scale = logit.y.scale, 
#                               log.x.scale = log.x.scale)
#  Test$Trans_h[((i - 1)*5000+1):(i*5000)] = Test.fit.iter$Trans_h
#}

#for(i in 1:(nrow(Val)/5000)){
#  print(i)
#  Val.fit.iter = predict.CDOT(CDOT.model = CDOT.fit,
#                              x = Val$mass[((i - 1)*5000+1):(i*5000)],
#                              y = Val$h[((i - 1)*5000+1):(i*5000)],
#                              scaled = scaled,
#                              logit.y.scale = logit.y.scale, 
#                              log.x.scale = log.x.scale)
#  Val$Trans_h[((i - 1)*5000+1):(i*5000)] = Val.fit.iter$Trans_h
#}

#for(i in 1:(nrow(Train)/5000)){
#  print(i)
#  Train.fit.iter = predict.CDOT(CDOT.model = CDOT.fit,
#                                x = Train$mass[((i - 1)*5000+1):(i*5000)],
#                                y = Train$h[((i - 1)*5000+1):(i*5000)],
#                                scaled = scaled,
#                                logit.y.scale = logit.y.scale, 
#                                log.x.scale = log.x.scale)
#  Train$Trans_h[((i - 1)*5000+1):(i*5000)] = Train.fit.iter$Trans_h
#}

# Saving the data frame

save(Train, Test, Val, file = "./cdot/WTaggingDecorrelated.RData")

# Geodesic Path was computed on Python - Code in R50JSDPlot.ipynb
############## Data for Python Code ##############
Test.df = Test
# List of lambdas to compute the range of classifiers given by the geodesic 
mixprop = c(2^seq(-24, -4, by = 1), seq(0.065, 0.15, by = 0.01),
            seq(0.15, 1, by = 0.05))
save(Test.df, mixprop, file = "./cdot/DataforR50JSDPlot.Rdata")






