install.packages("corrplot")
install.packages("caret")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("lattice")
install.packages("randomForest")
install.packages("pROC")
install.packages("klaR")
install.packages("nnet")

library(corrplot)
library(dplyr)
library(ggplot2)
library(lattice)
library(caret)
library(pROC)
library(klaR)
library(MASS)
library(e1071)
library(nnet)

# Reading the data set
data <- read.csv("wisconsin.csv", header = TRUE, sep = ",") 
str(data)
names(data)[1]="Clump.Thickness"
names(data)[2]="Uniformity.of.Cell.Size"
names(data)[3]="Uniformity.of.Cell.Shape"
names(data)[4]="Marginal.Adhesion"
names(data)[5]="Single.Epithelial.Cell.Size"
names(data)[6]="Bare.Nuclei"
names(data)[7]="Bland.Chromatin"
names(data)[8]="Normal.Nucleoli"
names(data)[9]="Mitoses"
names(data)[10]="Class"

# Checking how many empty values
sapply(data, function(x) sum(is.na(x)))

# Removing raws with empty values
data <- data[complete.cases(data), ]

# Converting bening/maligant values to 0/1
data$Class <- ifelse(data$Class == "benign", 0, 1)

# Checking 1s and 0s proportion
prop.table(table(data$Class))

# Downsampling the data set to balance the classes
data_downsampled <- data %>% group_by(Class) %>% sample_n(230)

# Splitting the downsampled data set into training and test sets
set.seed(123)
index <- createDataPartition(data_downsampled$Class, p = 0.7, list = FALSE)
train_data <- data_downsampled[index, ]
test_data <- data_downsampled[-index, ]

# Converting the Class column to a factor
train_data$Class <- as.factor(train_data$Class)
test_data$Class <- as.factor(test_data$Class)

# Checking 1s 0s proportion
prop.table(table(train_data$Class))
prop.table(table(test_data$Class))

# Calculating the correlation matrix
cor<-cor(data[,1:9])#Calculating the correlation matrix
corrplot(cor, method = "pie",shade.col = NA, tl.col ="black", tl.srt = 45, order = "AOE") 

# Observing the distribution of attributes
ggplot(stack(test_data[,1:9]),aes(x=ind,y=values))+
  geom_boxplot(position=position_dodge(0.6),
               size=0.5,
               width=0.3,
               color="blue",
               outlier.color = "red",
               notchwidth = 0.5)+xlab("Attributes")+ylab("Values")+
  ggtitle("Boxplot of Attributes")
set.seed(1234)

# Training control parameters for repeated cross-validation.
control <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

# Training the random forest model on the training data
model.rf <- train(Class ~ ., 
                  data = train_data, 
                  method = "rf",  
                  preProcess = c('center', 'scale'),
                  trControl = control)
model.rf

# Making predictions on the test data using the random forest model
predictions.rf <- predict(model.rf, test_data)

# Creating a confusion matrix to evaluate the predictions
cm.rf <- confusionMatrix(predictions.rf, test_data$Class, positive = "1")
cm.rf

# Making probability predictions on the test data using the random forest model
predictions_prob_rf <- predict(model.rf, test_data, type="prob")

# Creating a ROC curve to evaluate the probability predictions
roc_rf <- roc(test_data$Class, predictions_prob_rf$'1')

# Ploting the ROC curve
plot(roc_rf)

# Training the K-nearest neighbors model on the training data
model.knn <- train(Class ~ ., 
                   data = train_data, 
                   method = "knn",  
                   preProcess = c('center', 'scale'),
                   trControl = control)
model.knn

# Making predictions on the test data using the K-nearest neighbors model
predictions.knn <- predict(model.knn, test_data)

# Creating a confusion matrix to evaluate the predictions
cm.knn <- confusionMatrix(predictions.knn, test_data$Class, positive = "1")
cm.knn

# Making probability predictions on the test data using KNN model
predictions_prob_knn <- predict(model.knn, test_data, type="prob")

# Creating a ROC curve to evaluate the probability predictions
roc_knn <- roc(test_data$Class, predictions_prob_knn$'1')

# Ploting the ROC curve
plot(roc_knn)

# Training the MLP model on the training data
model.mlp <- train(Class ~ ., 
                   data = train_data, 
                   method = "mlp",  
                   preProcess = c('center', 'scale'),
                   trControl = control)
model.mlp

# Making predictions on the test data using the MLP model
predictions.mlp <- predict(model.mlp, test_data)

# Creating a confusion matrix to evaluate the predictions
cm.mlp <- confusionMatrix(predictions.mlp, test_data$Class, positive = "1")
cm.mlp

# Making probability predictions on the test data using the MLP model
predictions_prob_mlp <- predict(model.mlp, test_data, type="prob")

# Creating a ROC curve to evaluate the probability predictions
roc_mlp <- roc(test_data$Class, predictions_prob_mlp$'1')

# Ploting the ROC curve
plot(roc_mlp)

# Getting performance results for each model's confusion matrix
cm_list <- list(RF=cm.rf,KNN = cm.knn, MLP =cm.mlp)
cm_list_results <- sapply(cm_list, function(x) x$byClass)
cm_list_results

# Observing best model for each ROC component across models:
cm_results_max <- apply(cm_list_results, 1, which.is.max)
output_report <- data.frame(metric=names(cm_results_max), 
                            best_model=colnames(cm_list_results)[cm_results_max],
                            value=mapply(function(x,y) {cm_list_results[x,y]}, 
                                         names(cm_results_max), 
                                         cm_results_max))
rownames(output_report) <- NULL
output_report

# Creating a function to evaluate multiple models performance
compare_models <- function(model_list, control, train_data, test_data) {
  # Initializing empty list to store confusion matrices
  confusion_matrix_list <- list()
  # Iterating through each model in the model list
  for (model_name in names(model_list)) {
    # Training the model on the training data
    model <- train(Class ~ ., 
                   data = train_data, 
                   method = model_list[model_name], 
                   preProcess = c('center', 'scale'),
                   trControl = control)
    # Creating a confusion matrix to evaluate the predictions
    cm <- confusionMatrix(predictions, test_data$Class, positive = "1")
    
    # Adding the confusion matrix to the list
    confusion_matrix_list[[model_name]] <- cm
  }
  # Comparing the performance of each model using the confusion matrices
  performance_results <- sapply(confusion_matrix_list, function(x) x$byClass)
  return(performance_results)
}

# Defining the list of models to evaluate
model_list <- c("rf" = "rf", "knn" = "knn", "mlp" = "mlp")

# Calling a function
compare_models(model_list, control, train_data, test_data)
