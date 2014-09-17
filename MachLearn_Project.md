---
title: "MachLearn_CourseProject"
author: "Chris Keen"
date: "Wednesday, September 10, 2014"
output:
  html_document:
    keep_md: yes
---

## Synopsis
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways(classe). I will use the data to predict the 'classe' variable for 20 different weight lifting scenarios.

The 'classe' variable has five levels (A,B,C,D,E) defined as:
- A: Exactly according to the specification
- B: Throwing the elbows to the front
- C: Lifting the dumbbell only halfway
- D: Lowering the dumbbell only halfway
- E: Throwing the hips to the front

## Data Processing
### Getting Data
All data was retreived from two links on the Coursera Practical Machine Learning website.  I downloaded the data direct into two .csv files. I labelled my data set 'niketrain' and 'niketest'.

```{r}
#download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "nike_training.csv")
#download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "nike_testing.csv")
niketrain <- read.csv('nike_training.csv',na.strings=c("NA",""))
niketest <- read.csv('nike_testing.csv',na.strings=c("NA",""))
  
```

### Cleaning Data
The two data sets contained 160 variables.  After looking at the data, it was necessary to remove all empty or "NA" columns.

```{r}
trainSumNA <- apply(niketrain,2,function(x) {sum(is.na(x))}) #Get sum of NA 
NTrain <- niketrain[,which(trainSumNA == 0)] #Remove any columns with NA

testSumNA <- apply(niketest,2,function(x) {sum(is.na(x))}) #Get sum of NA 
NTest <- niketest[,which(testSumNA == 0)] #Remove any columns with NA
  
```

The data sets now have 60 variables. I then removed all columns not required for building a prediction algorithm for the weight lifting patterns. The index, names, timestamps and windows columns do not have predictive value for this project.

```{r}
NTrain <- NTrain[,-(1:7)]  #Remove the index, names, timestamps and windows cols
NTest <- NTest[,-(1:7)]  #Remove the index, names, timestamps and windows cols
  
```

The data sets now have 53 variables of anonymous weight lifting data which will be used to build an algorithm for predicting the weight lifting 'classe'.

## Cross Validation
### Partition 'NTrain' Data into a testing and training set

I split the 'NTrain' data into 70% training set and 30% testing set. I will run different models on the 'training' set until I find the model which best predicts the 'classe' variable.  I will then run the most accurate model on the 'testing' set.  

```{r}
library(caret)
library(kernlab)
inTrain <- createDataPartition(y=NTrain$classe, p=.70, list=FALSE)
training <- NTrain[inTrain,]
testing <- NTrain[-inTrain,]
dim(training)
dim(testing)

```


### 'training' Models

The first model I tried was a Tree.  This model returned 33-49% Accuracy result.
```{r}
rpFit <- train(classe ~ ., data=training, method="rpart")
rpFit

```

The second model I tried was Random Forest.  This model returned 98-99% Accuracy
result.
```{r}
#rfFit <- train(classe ~ ., data=training, method="rf")
#rfFit

library(randomForest)
rfMod <- randomForest(classe ~ ., data=training)
rfMod

```

The third model I tried was Gradient Boosting w/ Trees.  This model returned  96% Accuracy w/ 150 trees and interaction.depth = 3.
```{r}
#gbmFit <- train(classe ~ ., data=training, method="gbm", verbose=FALSE)
#gbmFit

```

The fourth model I tried was the Linear Discriminant Analysis Model.  This model returned a 70% Accuracy result.
```{r}
modLDA <- train(classe ~ ., data=training, method="lda")
modLDA

```

The fifth model I tried was the Naive Bayes Model.  This model returned a 75% Accuracy result.
```{r}
#modNB <- train(classe ~ ., data=training, method="nb")
#modNB

```

### 'NTrain' testing set
Of the five models used above, I chose the Random Forest model to run on the testing set as it was the most accurate with 98-99%.
```{r}
NTrainpred <- predict(rfMod, newdata=testing)
confusionMatrix(NTrainpred, testing$classe)

```

## Prediction
### Predict the outcome (classe) for the 20 scenarios provided.
I used the Random Forest model to predict 'classe' variable for the 20 scenarios provided in the 'NTest' data set.

```{r}
answers <- predict(rfMod, newdata=NTest)
answers

```

Finally, I used the provided function to create text files for each of my 20 results(answers).
```{r}
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(as.character(answers))

```
