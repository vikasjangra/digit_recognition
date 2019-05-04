#---Checking and installing required packages -------#

packages = c("kernlab","dplyr","plyr","stringr","readr","caret","ggplot2","gridExtra", "foreach", "doParallel")

package.check <- lapply(packages, FUN = function(x) {
  if (!require(x, character.only = TRUE)) {
    install.packages(x, dependencies = TRUE)
    library(x, character.only = TRUE)
  }
})

# ---- Data understanding and cleaning ----- #

#Loading Data

train_data <- read.csv("mnist_train.csv", header = FALSE)
test_data <- read.csv("mnist_test.csv", header = FALSE)

#train_data - 60000 obs of 785 variables
#test_data - 10000 obs of 785 variables


#data cleaning
# checking for NA

sum(is.na(train_data))
sum(is.na(test_data))

summary(train_data)

#After looking at the data , it feels like some columns have all the same value for all the labels.
#lets check that.

x<- which(sapply(train_data, function(x) length(unique(x))) == 1)
y<- which(sapply(test_data, function(x) length(unique(x))) == 1)


length(x)
length(y)

z <- intersect(x,y)

train_data <- train_data[,-z]
test_data <- test_data[,-z]


#both the data does not have any Na values.

#Making our target class to factor

train_data$V1<-factor(train_data$V1)
test_data$V1 <- factor(test_data$V1)

# Split the data into train and test set

set.seed(1)
train.indices = sample(1:nrow(train_data), 0.15*nrow(train_data))
train_1 = train_data[train.indices, ]


set.seed(100)
test.indices = sample(1:nrow(test_data), 0.15*nrow(test_data))
test_1 = test_data[test.indices, ]

#Constructing Model

#Using Linear Kernel
Model_linear <- ksvm(V1~ ., data = train, scale = FALSE, kernel = "vanilladot")
Eval_linear<- predict(Model_linear, test)

#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear,test$V1)

#accuracy is 90.47%


#Using RBF Kernel to check for accuracy improvement
Model_RBF <- ksvm(V1~ ., data = train, scale = FALSE, kernel = "rbfdot")
Eval_RBF<- predict(Model_RBF, test)

#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF,test$V1)

print(Model_RBF)

#accuracy for rbf model is 95.53%
# The final values used for the model were sigma = 1.62e-07 and C = 1.

# ---it looks like rbf model is better. We will run 
############   Hyperparameter tuning and Cross Validation #####################

# Performing Cross validation with 5 folds

trainControl <- trainControl(method="cv", number=5)

# metric is accuracy

metric <- "Accuracy"

#Expand.grid to build the combination of C.

# We will use only single value for sigma and change the value for C to see the change as we want to limit the computatoinal time.

set.seed(7)
grid <- expand.grid(.sigma = 1.62e-07,.C=seq(1,5, by= 1))


fit.svm <- train(V1~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

print(fit.svm)

plot(fit.svm)
#Resampling results across tuning parameters:
  
#C  Accuracy   Kappa    
#1  0.9521105  0.9467614
#2  0.9565544  0.9517012
#3  0.9572209  0.9524417
#4  0.9587764  0.9541717
#5  0.9597768  0.9552839

#Tuning parameter 'sigma' was held constant at a value of 1.62e-07
#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were sigma = 1.62e-07 and C = 5.

#accuracy of the model is on cross validation is 95.98%

###----------Lets take logarithmic increment of sigma and cross validate model on sigma 1.62e-06 and other value for C---#

set.seed(7)
grid_1 <- expand.grid(.sigma = 1.62e-06,.C=seq(1,5, by= 1))


fit.svm.1 <- train(V1~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid_1, trControl=trainControl)

print(fit.svm.1)

plot(fit.svm.1)

###---------------------####
# Just to be sure, we can also check with cross validation of linear model if that works better than rbf or not.

grid_2 <- expand.grid(.C=seq(1,5, by= 1))

fit.svm.2 <- train(V1~., data=train, method="svmLinear", metric=metric, 
                 tuneGrid=grid_2, trControl=trainControl)

#Resampling results across tuning parameters:
  
#C  Accuracy   Kappa    
#1  0.9103332  0.9003105
#2  0.9103332  0.9003105
#3  0.9103332  0.9003105
#4  0.9103332  0.9003105
#5  0.9103332  0.9003105

# Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was C = 1.



# It clearly shows that accuracy is well below the accuracy which came for rbf.
###----------------------#####

# we will evaluate rbf model on test data 


Eval_final<- predict(fit.svm, test)
#confusion matrix - RBF Kernel
confusionMatrix(Eval_final,test$V1)

#----------------------------------------------------------------------#
#So, our final model is rbf kernel with the final values used for the model as sigma = 1.64e-07 and C = 5.
# accuracy for the model is 96.13% which is even better than tested on cross validation.
#----------------------------------------------------------------------#