##################################### problem1 ######################################
################################### Bagging #######################################
#load the dataset
diabetes <- read.csv("C:\\Users\\DELL\\Downloads\\Ensemble Model\\Diabeted_Ensemble.csv",stringsAsFactors = TRUE)

##Exploring and preparing the data ----
str(diabetes)

#splitting train and test data
library(caTools)
set.seed(0)
split <- sample.split(diabetes$Class.variable, SplitRatio = 0.8)
diabetes_train <- subset(diabetes, split == TRUE)
diabetes_test <- subset(diabetes, split == FALSE)

# install.packages("randomForest")
library(randomForest)

bagging <- randomForest(diabetes_train$Class.variable ~ .,mtree = 30,maxnodes = 4,data = diabetes_train, mtry = 17)
# bagging will take all the columns ---> mtry = all the attributes

test_pred <- predict(bagging, diabetes_test)

rmse_bagging <- sqrt(mean(diabetes_test$Class.variable == test_pred)^2)
rmse_bagging

# Prediction for trained data result
train_pred <- predict(bagging, diabetes_train)

# RMSE on Train Data
train_rmse <- sqrt(mean(diabetes_train$Class.variable == train_pred)^2)
train_rmse

help(randomForest)

#train and test accuracy is almost similar hence it is right fit model

################################# Boosting ######################
############################## Adaboosting #############################
install.packages("adabag")
library(adabag)

diabetes_train$Class.variable <- as.factor(diabetes_train$Class.variable)

adaboost <- boosting(Class.variable ~ .,data = diabetes_train, boos = TRUE)

# Test data
adaboost_test <- predict(adaboost, diabetes_test)

table(adaboost_test$class, diabetes_test$Class.variable)
mean(adaboost_test$class == diabetes_test$Class.variable)


# Train data
adaboost_train <- predict(adaboost, diabetes_train)

table(adaboost_train$class, diabetes_train$Class.variable)
mean(adaboost_train$class == diabetes_train$Class.variable)

help(boosting)

###################### XGBoosting #########################################
install.packages("xgboost")
library(xgboost)

train_y <- diabetes_train$Class.variable == "1"

str(diabetes_train)

# create dummy variables on attributes
train_x <- model.matrix(diabetes_train$Class.variable ~  .-1, data = diabetes_train)

train_x <- train_x[, -9]
# 'n-1' dummy variables are required, hence deleting the additional variables

test_y <- diabetes_test$Class.variable == "1"

# create dummy variables on attributes
test_x <- model.matrix(diabetes_test$Class.variable ~ .-1, data = diabetes_test)
test_x <- test_x[, -9]

# DMatrix on train
Xmatrix_train <- xgb.DMatrix(data = train_x, label = train_y)
# DMatrix on test 
Xmatrix_test <- xgb.DMatrix(data = test_x, label = test_y)


# Max number of boosting iterations - nround
xg_boosting <- xgboost(data = Xmatrix_train, nround = 50,
                       objective = "multi:softmax", eta = 0.3, 
                       num_class = 2, max_depth = 100)

# Prediction for test data
xgbpred_test <- predict(xg_boosting, Xmatrix_test)
table(test_y, xgbpred_test)
mean(test_y == xgbpred_test)

# Prediction for train data
xgbpred_train <- predict(xg_boosting, Xmatrix_train)
table(train_y, xgbpred_train)
mean(train_y == xgbpred_train)

################################ voting ###############################################
# load dataset with factors as strings
diabetes <- read.csv("C:\\Users\\DELL\\Downloads\\Ensemble Model\\Diabeted_Ensemble.csv", stringsAsFactors = TRUE)
str(diabetes)
set.seed(12345)
Train_Test <- sample(c("Train", "Test"), nrow(diabetes), replace = TRUE, prob = c(0.7, 0.3))
diabetes_Train <- diabetes[Train_Test == "Train",]
diabetes_TestX <- within(diabetes[Train_Test == "Test", ], rm(Class.variable))
diabetes_TestY <- diabetes[Train_Test == "Test", "Class.variable"]

library(randomForest)
# Random Forest Analysis
diabetes_RF <- randomForest(Class.variable ~ ., data = diabetes_Train, keep.inbag = TRUE, ntree = 500)
?randomForest

# Overall class prediction (hard voting)
diabetes_RF_Test_Margin <- predict(diabetes_RF, newdata = diabetes_TestX, type = "class")

# Prediction
diabetes_RF_Test_Predict <- predict(diabetes_RF, newdata = diabetes_TestX, type = "class", predict.all = TRUE)

sum(diabetes_RF_Test_Margin == diabetes_RF_Test_Predict$aggregate)
head(diabetes_RF_Test_Margin == diabetes_RF_Test_Predict$aggregate)

# Majority Voting
dim(diabetes_RF_Test_Predict$individual)

# View(cc_RF_Test_Predict$individual) # Prediction at each tree

Row_Count_Max <- function(x) names(which.max(table(x)))

Voting_Predict <- apply(diabetes_RF_Test_Predict$individual, 1, Row_Count_Max)

head(Voting_Predict)
tail(Voting_Predict)

all(Voting_Predict == diabetes_RF_Test_Predict$aggregate)
all(Voting_Predict == diabetes_RF_Test_Margin)

mean(Voting_Predict == diabetes_TestY)

############################ problem2 #######################################################
################################### Bagging #######################################
#load the dataset
tumor_ensemble <- read.csv("C:\\Users\\DELL\\Downloads\\Ensemble Model\\Tumor_Ensemble.csv",stringsAsFactors = TRUE)

##Exploring and preparing the data ----
str(tumor_ensemble)

#splitting train and test data
library(caTools)
set.seed(0)
split <- sample.split(tumor_ensemble$diagnosis, SplitRatio = 0.8)
tumor_ensemble_train <- subset(tumor_ensemble, split == TRUE)
tumor_ensemble_test <- subset(tumor_ensemble, split == FALSE)

# install.packages("randomForest")
library(randomForest)

bagging <- randomForest(tumor_ensemble_train$diagnosis ~ .,mtree = 30,maxnodes = 4,data = tumor_ensemble_train, mtry = 17)
# bagging will take all the columns ---> mtry = all the attributes

test_pred <- predict(bagging, tumor_ensemble_test)

rmse_bagging <- sqrt(mean(tumor_ensemble_test$diagnosis == test_pred)^2)
rmse_bagging

# Prediction for trained data result
train_pred <- predict(bagging, tumor_ensemble_train)

# RMSE on Train Data
train_rmse <- sqrt(mean(tumor_ensemble_train$diagnosis == train_pred)^2)
train_rmse

help(randomForest)

#train and test accuracy is almost similar hence it is right fit model

################################# Boosting ######################
############################## Adaboosting #############################
install.packages("adabag")
library(adabag)

tumor_ensemble_train$diagnosis <- as.factor(tumor_ensemble_train$diagnosis)

adaboost <- boosting(diagnosis ~ .,data = tumor_ensemble_train, boos = TRUE)

# Test data
adaboost_test <- predict(adaboost, tumor_ensemble_test)

table(adaboost_test$class, tumor_ensemble_test$diagnosis)
mean(adaboost_test$class == tumor_ensemble_test$diagnosis)


# Train data
adaboost_train <- predict(adaboost, tumor_ensemble_train)

table(adaboost_train$class, tumor_ensemble_train$diagnosis)
mean(adaboost_train$class == tumor_ensemble_train$diagnosis)

help(boosting)

###################### XGBoosting #########################################
install.packages("xgboost")
library(xgboost)

train_y <- tumor_ensemble_train$diagnosis == "1"

str(tumor_ensemble_train)

# create dummy variables on attributes
train_x <- model.matrix(tumor_ensemble_train$diagnosis ~  .-1, data = tumor_ensemble_train)

train_x <- train_x[, -2]
# 'n-1' dummy variables are required, hence deleting the additional variables

test_y <- tumor_ensemble_test$diagnosis == "1"

# create dummy variables on attributes
test_x <- model.matrix(tumor_ensemble_test$diagnosis ~ .-1, data = tumor_ensemble_test)
test_x <- test_x[, -2]

# DMatrix on train
Xmatrix_train <- xgb.DMatrix(data = train_x, label = train_y)
# DMatrix on test 
Xmatrix_test <- xgb.DMatrix(data = test_x, label = test_y)


# Max number of boosting iterations - nround
xg_boosting <- xgboost(data = Xmatrix_train, nround = 50,
                       objective = "multi:softmax", eta = 0.3, 
                       num_class = 2, max_depth = 100)

# Prediction for test data
xgbpred_test <- predict(xg_boosting, Xmatrix_test)
table(test_y, xgbpred_test)
mean(test_y == xgbpred_test)

# Prediction for train data
xgbpred_train <- predict(xg_boosting, Xmatrix_train)
table(train_y, xgbpred_train)
mean(train_y == xgbpred_train)

################################ voting ###############################################
#load the dataset
tumor_ensemble <- read.csv("C:\\Users\\DELL\\Downloads\\Ensemble Model\\Tumor_Ensemble.csv",stringsAsFactors = TRUE)
str(tumor_ensemble)

set.seed(12345)
Train_Test <- sample(c("Train", "Test"), nrow(tumor_ensemble), replace = TRUE, prob = c(0.7, 0.3))
tumor_ensemble_Train <- tumor_ensemble[Train_Test == "Train",]
tumor_ensemble_TestX <- within(tumor_ensemble[Train_Test == "Test", ], rm(diagnosis))
tumor_ensemble_TestY <- tumor_ensemble[Train_Test == "Test", "diagnosis"]

library(randomForest)
# Random Forest Analysis
tumor_ensemble_RF <- randomForest(diagnosis ~ ., data = tumor_ensemble_Train, keep.inbag = TRUE, ntree = 500)

# Overall class prediction (hard voting)
tumor_ensemble_RF_Test_Margin <- predict(tumor_ensemble_RF, newdata = tumor_ensemble_TestX, type = "class")

# Prediction
tumor_ensemble_RF_Test_Predict <- predict(tumor_ensemble_RF, newdata = tumor_ensemble_TestX, type = "class", predict.all = TRUE)

sum(tumor_ensemble_RF_Test_Margin == tumor_ensemble_RF_Test_Predict$aggregate)
head(tumor_ensemble_RF_Test_Margin == tumor_ensemble_RF_Test_Predict$aggregate)

# Majority Voting
dim(tumor_ensemble_RF_Test_Predict$individual)

# View(cc_RF_Test_Predict$individual) # Prediction at each tree

Row_Count_Max <- function(x) names(which.max(table(x)))

Voting_Predict <- apply(tumor_ensemble_RF_Test_Predict$individual, 1, Row_Count_Max)

head(Voting_Predict)
tail(Voting_Predict)

all(Voting_Predict == tumor_ensemble_RF_Test_Predict$aggregate)
all(Voting_Predict == tumor_ensemble_RF_Test_Margin)

mean(Voting_Predict == tumor_ensemble_TestY)

##################################### problem3 ##################################################
################################### Bagging #######################################
#load the dataset
cocoa_data <- readxl::read_excel("C:\\Users\\DELL\\Downloads\\Ensemble Model\\Coca_Rating_Ensemble.xlsx")

#remove the unwanted column
cocoa_data <- cocoa_data[, -c(1,2,4,6,8,9)]

cocoa_data$Rating <- factor(cocoa_data$Rating)

##Exploring and preparing the data ----
str(cocoa_data)

#splitting train and test data
library(caTools)
set.seed(0)
split <- sample.split(cocoa_data$Rating, SplitRatio = 0.8)
cocoa_train <- subset(cocoa_data, split == TRUE)
cocoa_test <- subset(cocoa_data, split == FALSE)

# install.packages("randomForest")
library(randomForest)

bagging <- randomForest(cocoa_train$Rating ~ .,mtree = 30,maxnodes = 4,data = cocoa_train, mtry = 17)
# bagging will take all the columns ---> mtry = all the attributes

test_pred <- predict(bagging, cocoa_test)

rmse_bagging <- sqrt(mean(cocoa_test$Rating == test_pred)^2)
rmse_bagging

# Prediction for trained data result
train_pred <- predict(bagging, cocoa_train)

# RMSE on Train Data
train_rmse <- sqrt(mean(cocoa_train$Rating == train_pred)^2)
train_rmse

help(randomForest)

#train and test accuracy is almost similar hence it is right fit model

################################# Boosting ######################
############################## Adaboosting #############################
install.packages("adabag")
library(adabag)

cocoa_train$Rating <- as.factor(cocoa_train$Rating)

adaboost <- boosting(Rating ~ .,data = cocoa_train, boos = TRUE)

# Test data
adaboost_test <- predict(adaboost, cocoa_test)

table(adaboost_test$class, cocoa_test$Rating)
mean(adaboost_test$class == cocoa_test$Rating)


# Train data
adaboost_train <- predict(adaboost, cocoa_train)

table(adaboost_train$class, cocoa_train$Rating)
mean(adaboost_train$class == cocoa_train$Rating)

help(boosting)

###################### XGBoosting #########################################
install.packages("xgboost")
library(xgboost)

train_y <- cocoa_train$Rating == "1"

str(cocoa_train)

train_x <- model.matrix(cocoa_train$Rating ~  .-1, data = cocoa_train)
train_x <- train_x[, -3]

test_y <- cocoa_test$Rating == "1"

test_x <- model.matrix(cocoa_test$Rating ~ .-1, data = cocoa_test)
test_x <- test_x[, -3]

# DMatrix on train
Xmatrix_train <- xgb.DMatrix(data = train_x, label = train_y)
# DMatrix on test 
Xmatrix_test <- xgb.DMatrix(data = test_x, label = test_y)


# Max number of boosting iterations - nround
xg_boosting <- xgboost(data = Xmatrix_train, nround = 50,
                       objective = "multi:softmax", eta = 0.3, 
                       num_class = 2, max_depth = 100)

# Prediction for test data
xgbpred_test <- predict(xg_boosting, Xmatrix_test)
table(test_y, xgbpred_test)
mean(test_y == xgbpred_test)

# Prediction for train data
xgbpred_train <- predict(xg_boosting, Xmatrix_train)
table(train_y, xgbpred_train)
mean(train_y == xgbpred_train)

################################ voting ###############################################
#load the dataset
cocoa_data <- readxl::read_excel("C:\\Users\\DELL\\Downloads\\Ensemble Model\\Coca_Rating_Ensemble.xlsx")

#remove the unwanted column
cocoa_data <- cocoa_data[, -c(1,2,4,6,8,9)]

cocoa_data$Rating <- factor(cocoa_data$Rating)

str(cocoa_data)
set.seed(12345)
Train_Test <- sample(c("Train", "Test"), nrow(cocoa_data), replace = TRUE, prob = c(0.7, 0.3))
cocoa_data_Train <- cocoa_data[Train_Test == "Train",]
cocoa_data_TestX <- within(cocoa_data[Train_Test == "Test", ], rm(Rating))
cocoa_data_TestY <- cocoa_data[Train_Test == "Test", "Rating"]

library(randomForest)
# Random Forest Analysis
cocoa_data_RF <- randomForest(Rating ~ ., data = cocoa_data_Train, keep.inbag = TRUE, ntree = 500)
?randomForest

# Overall class prediction (hard voting)
cocoa_data_RF_Test_Margin <- predict(cocoa_data_RF, newdata = cocoa_data_TestX, type = "class")

# Prediction
cocoa_data_RF_Test_Predict <- predict(cocoa_data_RF, newdata = cocoa_data_TestX, type = "class", predict.all = TRUE)

sum(cocoa_data_RF_Test_Margin == cocoa_data_RF_Test_Predict$aggregate)
head(cocoa_data_RF_Test_Margin == cocoa_data_RF_Test_Predict$aggregate)

# Majority Voting
dim(cocoa_data_RF_Test_Predict$individual)

# View(cc_RF_Test_Predict$individual) # Prediction at each tree

Row_Count_Max <- function(x) names(which.max(table(x)))

Voting_Predict <- apply(cocoa_data_RF_Test_Predict$individual, 1, Row_Count_Max)

head(Voting_Predict)
tail(Voting_Predict)

all(Voting_Predict == cocoa_data_RF_Test_Predict$aggregate)
all(Voting_Predict == cocoa_data_RF_Test_Margin)

mean(Voting_Predict == cocoa_data_TestY)

################################### problem4 #######################################
################################### Bagging #######################################
#load the dataset
password_strength <- readxl::read_excel("C:\\Users\\DELL\\Downloads\\Ensemble Model\\Ensemble_Password_Strength.xlsx")

#factor analysis to creat levels
password_strength$characters_strength <- factor(password_strength$characters_strength)

##Exploring and preparing the data ----
str(password_strength)

#splitting train and test data
library(caTools)
set.seed(0)
split <- sample.split(password_strength$characters_strength, SplitRatio = 0.8)
password_strength_train <- subset(password_strength, split == TRUE)
password_strength_test <- subset(password_strength, split == FALSE)

# install.packages("randomForest")
library(randomForest)

bagging <- randomForest(password_strength_train$characters_strength ~ .,mtree = 30,maxnodes = 4,data = password_strength_train, mtry = 17)

# bagging will take all the columns ---> mtry = all the attributes

test_pred <- predict(bagging, password_strength_test)

rmse_bagging <- sqrt(mean(password_strength_test$characters_strength == test_pred)^2)
rmse_bagging

# Prediction for trained data result
train_pred <- predict(bagging, password_strength_train)

# RMSE on Train Data
train_rmse <- sqrt(mean(password_strength_train$characters_strength == train_pred)^2)
train_rmse

help(randomForest)

#train and test accuracy is almost similar hence it is right fit model

################################# Boosting ######################
################################ XGBoosting #######################
install.packages("xgboost")
library(xgboost)

train_y <- password_strength_train$characters_strength == "1"

str(password_strength_train)

train_x <- model.matrix(password_strength_train$characters_strength ~  .-1, data = password_strength_train)
train_x <- train_x[, -2]

test_y <- password_strength_test$characters_strength == "1"

test_x <- model.matrix(password_strength_test$characters_strength ~ .-1, data = password_strength_test)
test_x <- test_x[, -2]

# DMatrix on train
Xmatrix_train <- xgb.DMatrix(data = train_x, label = train_y)
# DMatrix on test 
Xmatrix_test <- xgb.DMatrix(data = test_x, label = test_y)


# Max number of boosting iterations - nround
xg_boosting <- xgboost(data = Xmatrix_train, nround = 50,
                       objective = "multi:softmax", eta = 0.3, 
                       num_class = 2, max_depth = 100)

# Prediction for test data
xgbpred_test <- predict(xg_boosting, Xmatrix_test)
table(test_y, xgbpred_test)
mean(test_y == xgbpred_test)

# Prediction for train data
xgbpred_train <- predict(xg_boosting, Xmatrix_train)
table(train_y, xgbpred_train)
mean(train_y == xgbpred_train)

############################## Adaboosting #############################
install.packages("adabag")
library(adabag)

password_strength_train$characters_strength <- as.factor(password_strength_train$characters_strength)

adaboost <- boosting(characters_strength ~ .,data = password_strength_train, boos = TRUE)

# Test data
adaboost_test <- predict(adaboost, password_strength_test)

table(adaboost_test$class, password_strength_test$characters_strength)
mean(adaboost_test$class == password_strength_test$characters_strength)


# Train data
adaboost_train <- predict(adaboost, password_strength_train)

table(adaboost_train$class, password_strength_train$characters_strength)
mean(adaboost_train$class == password_strength_train$characters_strength)

help(boosting)

################################ voting ###############################################
#load the dataset
password_strength <- readxl::read_excel("C:\\Users\\DELL\\Downloads\\Ensemble Model\\Ensemble_Password_Strength.xlsx")

#factor analysis to creat levels
password_strength$characters_strength <- factor(password_strength$characters_strength)

##Exploring and preparing the data ----
str(password_strength)
set.seed(12345)
Train_Test <- sample(c("Train", "Test"), nrow(password_strength), replace = TRUE, prob = c(0.7, 0.3))
password_strength_Train <- password_strength[Train_Test == "Train",]
password_strength_TestX <- within(password_strength[Train_Test == "Test", ], rm(characters_strength))
password_strength_TestY <- password_strength[Train_Test == "Test", "characters_strength"]

library(randomForest)
# Random Forest Analysis
password_strength_RF <- randomForest(characters_strength ~ ., data = password_strength_Train, keep.inbag = TRUE, ntree = 500)
?randomForest

# Overall class prediction (hard voting)
password_strength_RF_Test_Margin <- predict(password_strength_RF, newdata = password_strength_TestX, type = "class")

# Prediction
password_strength_RF_Test_Predict <- predict(password_strength_RF, newdata = password_strength_TestX, type = "class", predict.all = TRUE)

sum(password_strength_RF_Test_Margin == password_strength_RF_Test_Predict$aggregate)
head(password_strength_RF_Test_Margin == password_strength_RF_Test_Predict$aggregate)

# Majority Voting
dim(password_strength_RF_Test_Predict$individual)

# View(cc_RF_Test_Predict$individual) # Prediction at each tree

Row_Count_Max <- function(x) names(which.max(table(x)))

Voting_Predict <- apply(password_strength_RF_Test_Predict$individual, 1, Row_Count_Max)

head(Voting_Predict)
tail(Voting_Predict)

all(Voting_Predict == password_strength_RF_Test_Predict$aggregate)
all(Voting_Predict == password_strength_RF_Test_Margin)

mean(Voting_Predict == password_strength_TestY)

############################# END #########################################################







