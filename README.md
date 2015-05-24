# coursera_practicalMLProject_HAR

---
title: "Pactical Machine Learning : HAR Course Project"  
author: "Angel Mak"  
date: "May 23, 2015"  
output: html_document
---

## 1. Goal

Fit a model to on-body sensor training data to predict the dumbbell lifting actions of 20 test cases.

There are 5 possible classes of dumbbell lifting actions:  
* Class A. Lifting the dumbbell correctly
* Class B. Lifting the dumbbell incorrectly by throwing the elbows to the front
* Class C. Lifting the dumbbell incorrectly by lifting the dumbbell only halfway
* Class D. Lifting the dumbbell incorrectly by lowering the dumbbell only halfway
* Class E. Lifting the dumbbell incorrectly y throwing the hips to the front

Data source
http://groupware.les.inf.puc-rio.br/har
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.


## 2. Data splitting
The data was loaded into R and split into the training and testing sets.

```{r, message=FALSE, warning=FALSE}
library(caret)
library(gdata)
data <- read.csv("/Users/angelmak/Dropbox/Course/2015_05_Coursera_PracticalMachineLearning/project_HAR/pml-training.csv")
problem <- read.csv("/Users/angelmak/Dropbox/Course/2015_05_Coursera_PracticalMachineLearning/project_HAR/pml-testing.csv")

set.seed(333)
inTrain <- createDataPartition(y=data$classe, p=0.75,list=FALSE)

training <- data[inTrain,]
testing <- data[-inTrain,]
```

## 3. Exploratory analysis

1.1 A few lines of the data were displayed to get an overall impression of the data.

```
training[1:5,]
```

1.2 The summary statistics of all the predictors were generated.
```
summary(training)
```

From the information above, the following observations were made:

1. Some columns contain a majority of NA values and should be removed from further analysis.

2. Column "X" (row number) and "user_name" are irrelevant predictors and should be removed from further analysis.

3. Timestamp data may be user/dataset-specific and its direct use may cause overfitting (see feature plot below). It should either be transformed into time period data or removed from further analysis.

```{r, message=FALSE, warning=FALSE}
col.timestamp <-matchcols(training,with=c("timestamp"))
featurePlot(x=training[,col.timestamp],y=training$classe,plot="pairs")

```

## 4. Data preprocessing

Based on the observation above, the training data was preprocessed according to the following steps:

1. Zero covariates were removed.

```{r, message=FALSE, warning=FALSE}
nsv <- nearZeroVar(training,saveMetrics=TRUE)
var <- rownames(nsv[(nsv$nzv==FALSE),])
training.var <- training[,var]
```

2. Columns which contain missing data was removed.

```{r, message=FALSE, warning=FALSE}
training.var.noNa <- training.var[,(colSums(is.na(training.var)) == 0)]
```

3. Column X, user_name and timestamp data was removed from further analysis. 

```{r, message=FALSE, warning=FALSE}
training.var.noNa.noTime <- training.var.noNa
training.var.noNa.noTime[,col.timestamp] <- list(NULL)
training.var.noNa.noTime[,'X'] <- list(NULL)
training.var.noNa.noTime[,'user_name'] <- list(NULL)
```

Below are the predictors that were used for model fitting.
```{r, message=FALSE, warning=FALSE}
names(training.var.noNa.noTime)
```

## 5. Model fitting using training data

The train and predict functions of the caret package were used.

As a first attempt, the tree method was used. Comparison of the prediction classes and the expected classes showed that this simple classification tree method failed to differentiate the 5 classes. The accurarcy was low. Boosting was used instead to fit a better model.

```{r, message=FALSE, warning=FALSE}
modelFit.rpart <- train(training.var.noNa.noTime$classe ~.,method="rpart",data=training.var.noNa.noTime)
pred.train.rpart <- predict(modelFit.rpart,newdata=training.var.noNa.noTime)
confusionMatrix(training.var.noNa.noTime$classe,pred.train.rpart)
```

Using boosting with tree method to fit a model to the training data and make prediction.

```{r, message=FALSE, warning=FALSE}
Sys.time()
modelFit.gbm <- train(training.var.noNa.noTime$classe ~.,method="gbm",data=training.var.noNa.noTime,verbose=FALSE)
Sys.time()
pred.train.gbm <- predict(modelFit.gbm,newdata=training.var.noNa.noTime)
confusionMatrix(training.var.noNa.noTime$classe,pred.train.gbm)
varImp(modelFit.gbm)
```

The top predictor (num_window) shown above seems to be correlated to the users who performed the action. Since user-specific data would probably create bias or overfitting, the column "num_window" was removed from training data set. Model fitting was rerun.

```{r, message=FALSE, warning=FALSE}
qplot(training.var.noNa$classe,training.var.noNa$num_window,data=training.var.noNa)+geom_point(aes(color=training.var.noNa$user_name))

training.var.noNa.noTime2 <- training.var.noNa.noTime
training.var.noNa.noTime2[,'num_window'] <- list(NULL)
Sys.time()
modelFit.gbm2 <- train(training.var.noNa.noTime2$classe ~.,method="gbm",data=training.var.noNa.noTime2,verbose=FALSE)
Sys.time()
pred.train.gbm2 <- predict(modelFit.gbm2,newdata=training.var.noNa.noTime2)
confusionMatrix(training.var.noNa.noTime2$classe,pred.train.gbm2)
varImp(modelFit.gbm2)
```

## 6. Evaluate model in testing data

Compare the performance of the first and second boosting with tree models to the testing set. The accuracy of the second model was lower than the first but the difference was small.

Prediction on testing set using the first boosting with tree model
```{r, message=FALSE, warning=FALSE}
testing.eq <- testing[,colnames(training.var.noNa.noTime)]
pred.test.gbm <- predict(modelFit.gbm,newdata=testing.eq)
confusionMatrix(testing.eq$classe,pred.test.gbm)
```

Prediction on testing set using the second boosting with tree model
```{r, message=FALSE, warning=FALSE}
testing.eq2 <- testing[,colnames(training.var.noNa.noTime2)]
pred.test.gbm2 <- predict(modelFit.gbm2,newdata=testing.eq2)
confusionMatrix(testing.eq2$classe,pred.test.gbm2)
```

## 7. Making predictions on the 20 test cases

The first and second boosting with tree models gave the same prediction resutls to the 20 test cases in the quiz.

Prediction using the first boosting model
```{r, message=FALSE, warning=FALSE}
problem.col <- colnames(training.var.noNa.noTime)[-54]
problem.eq <- problem[,problem.col]
pred.problem.gbm <- predict(modelFit.gbm,newdata=problem.eq)
pred.problem.gbm
```

Prediction using the second boosting model
```{r, message=FALSE, warning=FALSE}
problem.col2 <- colnames(training.var.noNa.noTime2)[-53]
problem.eq2 <- problem[,problem.col2]
pred.problem.gbm2 <- predict(modelFit.gbm2,newdata=problem.eq2)
pred.problem.gbm2
```

These predictions were submitted as answers.

