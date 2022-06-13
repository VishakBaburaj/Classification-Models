#Importing libraries
library(dplyr)
library(dummies)
library(rpart)
library(caret)
library(randomForest)
library(e1071)
library(kernlab)
library(nnet)
library(NeuralNetTools)
library(ROCR)

set.seed(123)
#Importing the data
my_data = read.table("model_data.csv",
                     colClasses = c("character",
                                    "numeric",
                                    rep("factor", 4),
                                    "numeric",
                                    "factor",
                                    rep("numeric",7)),
                     header = TRUE,
                     row.names = 1,
                     sep = ",")
str(my_data)
View(my_data)
names(my_data)

#Identifying missing values in each column
summary(my_data)
apply(is.na(my_data),2,sum)
sum(is.na(my_data))

#Imputing missing values with median
data_med_impute <- preProcess(my_data, method="medianImpute")
my_data <- predict(data_med_impute, my_data)
sum(is.na(my_data))

#Subsetting the variables
model_data <- my_data %>%
  dplyr::select(Age,Gender,PrimaryInsuranceCategory, 
                Flipped, DRG01, 
                BloodPressureUpper,BloodPressureLower,BloodPressureDiff,
                Pulse,PulseOximetry,Respirations,Temperature) %>%
  filter(Flipped %in% c(0,1)) %>%
  droplevels()

str(model_data)
cluster_data_dummy <- dummy.data.frame(model_data[,c("Age",
                                                       "Gender",
                                                       "PrimaryInsuranceCategory",
                                                       "Flipped",
                                                       "DRG01","BloodPressureUpper",
                                                       "BloodPressureLower","BloodPressureDiff",
                                                       "Pulse","PulseOximetry","Respirations",
                                                       "Temperature")],
                                       names=c("Gender","PrimaryInsuranceCategory","Flipped","DRG01"))


View(cluster_data_dummy)
str(cluster_data_dummy)

#Scaling the data
cluster_data_analyzed <- scale(cluster_data_dummy, center=TRUE, scale=TRUE)
View(cluster_data_analyzed)

#Clustering
cluster_data_kmeans <- kmeans(cluster_data_analyzed, centers=3)
cluster_data_pca <- prcomp(cluster_data_analyzed, retx=TRUE)
plot(cluster_data_pca$x[,1:2], col=cluster_data_kmeans$cluster, pch=cluster_data_kmeans$cluster)

cluster_data_pca$rotation[,1:3]

# Outlier detection
boxplot(cluster_data_pca$x[,1:3])

set.seed(123)
#-------------------------------------------------
#Prediction analysis
#-------------------------------------------------
#Dividing the data into training and Testing

train_data_rows <- createDataPartition(model_data$Flipped,
                                        p = 0.7,
                                        list=FALSE)
train_data <- model_data[train_data_rows,]
test_data <- model_data[-train_data_rows,]

#Assigning Weights
summary(train_data$Flipped)
model_data_weights <- numeric(nrow(train_data))
model_data_weights[train_data$Flipped == 0] <- 1
model_data_weights[train_data$Flipped == 1] <- 2

#-------------------------------------------------
#Logistic Regression - Model 1
#-------------------------------------------------
model_data_lr <- glm(Flipped ~ ., data = train_data, weights = model_data_weights,family=binomial("logit"))
summary(model_data_lr)

#Predicting Using Logistic Regression
model_data_lr_predict <- predict(model_data_lr,
                              newdata=test_data,
                              type="response")
model_data_lr_predict_class <- character(length(model_data_lr_predict))
model_data_lr_predict_class[model_data_lr_predict < 0.5] <- 0
model_data_lr_predict_class[model_data_lr_predict >= 0.5] <- 1

#Confusion Matrix of Logistic Regression
model_data_lr_CM = table(test_data$Flipped, model_data_lr_predict_class)
model_data_lr_CM

#Misclassification
model_data_lr_MC_rate = (1-sum(diag(model_data_lr_CM))/sum(model_data_lr_CM))*100
model_data_lr_MC_rate

#-------------------------------------------------
#Classification tree model - Model 2
#-------------------------------------------------
model_data_rpart <- rpart(Flipped ~ ., data=train_data, weights = model_data_weights)

model_data_rpart_predict <- predict(model_data_rpart, newdata=test_data, type="class")

#Confusion Matrix of Classification tree model
model_data_rpart_CM = table(test_data$Flipped, model_data_rpart_predict)
model_data_rpart_CM

#Misclassification
model_data_rpart_MC_rate = (1-sum(diag(model_data_rpart_CM))/sum(model_data_rpart_CM))*100
model_data_rpart_MC_rate

#Important variables
varimp = model_data_rpart$variable.importance
varimp
plot(varimp, top = 10)
count = table(model_data$Flipped, model_data$DRG01)
barplot(count, main = "DRG01 vs Flipped", xlab = "DRG01", col = c("darkblue","red"),
        legend = rownames(count))
count2 = table(model_data$Flipped, model_data$BloodPressureLower)
barplot(count2, main = "BloodPressureLower vs Flipped", xlab = "BloodPressureLower", col = c("darkblue","red"),
        legend = rownames(count2))
#-------------------------------------------------
#Random Forest - model 3
#-------------------------------------------------
model_data_rf <- randomForest(Flipped ~ .,
                           data = train_data,
                           classwt=c(1,2),
                           importance=TRUE)
model_data_rf$importance

model_data_predict_rf <- predict(model_data_rf, newdata=test_data, type="class")

#Confusion Matrix of Random Forest model
(model_data_rf_cm <- table(test_data$Flipped, model_data_predict_rf))

#Misclassification
model_data_rf_MC_rate = (1-sum(diag(model_data_rf_cm))/sum(model_data_rf_cm))*100
model_data_rf_MC_rate

#-------------------------------------------------
#SVM and SVM rbf - model 4 and model 5
#-------------------------------------------------
train_data_dummy <- dummy.data.frame(train_data, names=c("Gender","PrimaryInsuranceCategory","DRG01"))
train_data_preprocess <- preProcess(train_data_dummy)
train_data_numeric <- predict(train_data_preprocess, train_data_dummy)
test_data_dummy <- dummy.data.frame(test_data, names=c("Gender","PrimaryInsuranceCategory","DRG01"))
test_data_numeric <- predict(train_data_preprocess, test_data_dummy)
#-------------------------------------------------
levels(train_data_numeric$Flipped) <- c("not_flipped", "flipped")
model_data_svm <- train(Flipped ~ .,
                      data=train_data_numeric,
                      method="svmLinearWeights",
                      metric="ROC",
                      trControl=trainControl(classProbs=TRUE,
                                             summaryFunction=twoClassSummary))
model_data_svm

modelLookup("svmLinearWeights")

model_data_predict_svm <- predict(model_data_svm, newdata=test_data_numeric)

#Confusion matrix
(model_data_svm_cm <- table(test_data$Flipped, model_data_predict_svm))

#Misclassification
model_data_svm_MC_rate = (1-sum(diag(model_data_svm_cm))/sum(model_data_svm_cm))*100
model_data_svm_MC_rate
#-------------------------------------------------
model_data_svm_rbf <- train(Flipped ~ .,
                          data=train_data_numeric,
                          method="svmRadialWeights",
                          metric="ROC",
                          trControl=trainControl(classProbs=TRUE,
                                                 summaryFunction=twoClassSummary))
model_data_svm_rbf

modelLookup("svmRadialWeights")

model_data_predict_svm_rbf <- predict(model_data_svm_rbf, newdata=test_data_numeric)

#Confusion matrix
(model_data_svm_rbf_cm <- table(test_data_numeric$Flipped, model_data_predict_svm_rbf))

#Misclassification
model_data_svm_rbf_MC_rate = (1-sum(diag(model_data_svm_rbf_cm))/sum(model_data_svm_rbf_cm))*100
model_data_svm_rbf_MC_rate

#-------------------------------------------------
#Neural networks - model 6
#-------------------------------------------------
model_data_weights_nn <- rep(2, nrow(train_data_numeric))
model_data_nn <- train(Flipped ~ .,
                     data=train_data_numeric,
                     method="nnet",
                     metric="ROC",
                     weights=model_data_weights_nn,
                     trControl=trainControl(classProbs=TRUE,
                                            summaryFunction=twoClassSummary))

model_data_predict_nn <- predict(model_data_nn, newdata=test_data_numeric)
(model_data_nn_cm <- table(test_data_numeric$Flipped, model_data_predict_nn))

#Misclassification
model_data_nn_MC_rate = (1-sum(diag(model_data_nn_cm))/sum(model_data_nn_cm))*100
model_data_nn_MC_rate

#-------------------------------------------------
#ROC Curve
#-------------------------------------------------
#TPR AND FPR of Logistic, Classification tree, Random Forest, SVM, SVM rbf models.

data_lr_predict <- predict(model_data_lr, test_data, type="response")
data_lr_pred <- prediction(data_lr_predict,
                           test_data$Flipped,
                           label.ordering=c(0,1))
data_lr_perf <- performance(data_lr_pred, "tpr", "fpr")

data_rpart_predict <- predict(model_data_rpart, test_data, type="prob")
data_rpart_pred <- prediction(data_rpart_predict[,2],
                              test_data$Flipped,
                              label.ordering=c(0,1))
data_rpart_perf <- performance(data_rpart_pred, "tpr", "fpr")

data_rf_predict <- predict(model_data_rf, newdata=test_data, type="prob")
data_rf_pred <- prediction(data_rf_predict[,1],
                         test_data$Flipped,
                         label.ordering=c(0,1))
data_rf_perf <- performance(data_rf_pred, "tpr", "fpr")

data_svm_predict <- predict(model_data_svm, newdata=test_data_numeric, type="prob")
data_svm_pred <- prediction(data_svm_predict[,1],
                          test_data_numeric$Flipped,
                          label.ordering=c(0,1))
data_svm_perf <- performance(data_svm_pred, "tpr", "fpr")

data_svm_rbf_predict <- predict(model_data_svm_rbf, newdata=test_data_numeric, type="prob")
data_svm_rbf_pred <- prediction(data_svm_rbf_predict[,1],
                              test_data_numeric$Flipped,
                              label.ordering=c(0,1))
data_svm_rbf_perf <- performance(data_svm_rbf_pred, "tpr", "fpr")

data_nn_predict <- predict(model_data_nn, newdata=test_data_numeric, type="prob")
data_nn_pred <- prediction(data_nn_predict[,1],
                         test_data_numeric$Flipped,
                         label.ordering=c(0,1))
data_nn_perf <- performance(data_nn_pred, "tpr", "fpr")

plot(data_lr_perf, col=1)
plot(data_rpart_perf, col=2, add=TRUE)
plot(data_rf_perf, col=3, add=TRUE)
plot(data_svm_perf, col=4, add=TRUE)
plot(data_svm_rbf_perf, col=5, add=TRUE)
plot(data_nn_perf, col=6, add=TRUE)
legend(0.7,0.35,c("LR","CT","RF", "SVM linear", "SVM RBF", "NN"), col=1:6, lwd=3)

#-------------------------------------------------
#Accuracy of each model
#-------------------------------------------------
sum(diag(model_data_lr_CM))/sum(model_data_lr_CM)*100
sum(diag(model_data_rpart_CM))/sum(model_data_rpart_CM)*100
sum(diag(model_data_rf_cm))/sum(model_data_rf_cm)*100
sum(diag(model_data_svm_cm))/sum(model_data_svm_cm)*100
sum(diag(model_data_svm_rbf_cm))/sum(model_data_svm_rbf_cm)*100
sum(diag(model_data_nn_cm))/sum(model_data_nn_cm)*100

#-------------------------------------------------
#Prediction data 
#-------------------------------------------------

#Importing the data
prediction_data = read.table("prediction_data.csv",
                     colClasses = c("character",
                                    "numeric",
                                    rep("factor", 3),
                                    rep("numeric",7)),
                     header = TRUE,
                     row.names = 1,
                     sep = ",")

View(prediction_data)

#Identifying missing values in each column
summary(prediction_data)
apply(is.na(prediction_data),2,sum)
sum(is.na(prediction_data))

#Imputing missing values with median
data_med_impute <- preProcess(prediction_data, method="medianImpute")
prediction_data <- predict(data_med_impute, prediction_data)
sum(is.na(prediction_data))

#Predicting Using SVM rbf
prediction_data_rpart_predict <- predict(model_data_rpart, newdata=prediction_data, type="class")

write.csv(prediction_data_rpart_predict,file = "pred.txt")



