library(caret)
library(e1071)
library(astsa)
library(readxl)
library(stats)
library(factoextra)
library(ggplot2)
library(tidyverse)
library(kknn)
library(cluster)
library(GGally)
library(pROC)


#a. Data Gathering and integration
setwd("D:/E - Working/@Master - Data Science/Study/7. DSC 441 - Fundamentals of Data Science/Week 10 - 03.07.2022/HW5")
insurance = read.csv("insurance.csv")
head(insurance)

#b. Data exploration
str(insurance)
summary(insurance)
insurance %>% select(age, bmi, children, charges) %>% ggpairs()

## Convert to dataframe
df = as.data.frame(insurance)
## Create ggplot object
p = ggplot(insurance, aes(x=sex, fill=smoker))
p + geom_bar(position="stack")

p = ggplot(df, aes(x=age, fill=sex))
p + geom_bar(position="stack")

insurance %>% group_by(smoker) %>% summarise("count"=n())

ggplot(insurance, aes(x=age, y=charges, color=smoker)) + geom_point()

#c. Data cleaning
## Exclude rows of ages smaller than 20
insurance = subset(insurance, age >= 20)
p = ggplot(insurance, aes(x=age, fill=sex))
p + geom_bar(position="stack")

#d. Data preprocessing
myinsurance = insurance
##bin age groups into 5 different bins
myinsurance <- myinsurance %>%
mutate(agegroup = cut(age, breaks=c(-Inf, 29, 39, 49, 59, Inf),labels=c("twenties", "thirties", "fourties", "fifties","sixtiesplus")))
head(myinsurance)

#e. Clustering
##remove class labels
df = myinsurance
predictors <- df %>% select(-c(agegroup, region))
head(predictors)

##create dummies
dummy <- dummyVars(charges ~ ., data = predictors)
dummies <- as.data.frame(predict(dummy, newdata = predictors))

##include charges
dummies$charges = myinsurance$charges

##rename predictors
predictors <- dummies

##set seed
set.seed(123)
##find K; find the knee
fviz_nbclust(predictors, kmeans, method = "wss")
##use silhouette to find K
fviz_nbclust(predictors, kmeans, method = "silhouette")
## Fit the data
fit <- kmeans(predictors, centers = 4, nstart = 25)
## Display the kmeans object information
fit
## Display the cluster plot
fviz_cluster(fit, data = predictors)

## Calculate PCA
pca = prcomp(predictors)
## Save as dataframe
rotated_data = as.data.frame(pca$x)
## Add original labels as a reference
rotated_data$Color <- df$charges
## Plot and color by labels
ggplot(data = rotated_data, aes(x = PC1, y = PC2, col = Color)) + geom_point(alpha= 0.3)

## Assign clusters as a new column
rotated_data$Clusters = as.factor(fit$cluster)
## Plot and color by labels
ggplot(data = rotated_data, aes(x = PC1, y = PC2, col = Clusters)) + geom_point()

#f. Classification
##kNN fit
set.seed(123)
ctrl = trainControl(method="cv", number = 10)
knnFit <- train(smoker ~ ., data = myinsurance, method = "knn", trControl = ctrl, preProcess = c("center","scale"))
knnFit

##find the best k
set.seed(123)
ctrl = trainControl(method="cv", number = 10)
knnFit <- train(smoker ~ ., data = myinsurance, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 15)
knnFit
plot(knnFit)

##setup a tuneGrid with the tuning parameters
tuneGrid <- expand.grid(kmax = 3:7, kernel = c("rectangular", "cos"), distance = 1:3)
## tune and fit the model with 10-fold cross validation, standardization, and our specialized tune grid
kknn_fit <- train(smoker ~ ., data = myinsurance, method = 'kknn', trControl = ctrl, preProcess = c('center', 'scale'), tuneGrid = tuneGrid)
##Printing trained model provides report
kknn_fit

## Predict
pred_knn <- predict(kknn_fit, myinsurance)
## Generate confusion matrix
myinsurance$smoker = as.factor(myinsurance$smoker)
confusionMatrix(myinsurance$smoker, pred_knn)
## Result
knn_results = kknn_fit$results
knn_results <- knn_results %>%
  group_by(kmax, kernel) %>%
  mutate(avgacc = mean(Accuracy))
ggplot(knn_results, aes(x=kmax, y=avgacc, color=kernel)) + 
  geom_point(size=3) + geom_line()

#g. Evaluation
## Generate confusion matrix
myinsurance$smoker = as.factor(myinsurance$smoker)
cm  = confusionMatrix(myinsurance$smoker, pred_knn)

## Store the byClass object of confusion matrix as a dataframe
metrics <- as.data.frame(cm$byClass)
## View the object
metrics

## Get class probabilities for KNN
pred_prob <- predict(kknn_fit, myinsurance, type = "prob")
head(pred_prob)
## And now we can create an ROC curve for our model.
roc_obj <- roc((myinsurance$smoker), pred_prob[,1])
plot(roc_obj, print.auc=TRUE)
