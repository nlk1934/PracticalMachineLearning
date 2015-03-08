#https://class.coursera.org/predmachlearn-012/human_grading/view/courses/973547/assessments/4/submissions
#data preparation and Pre-Processing
library(caret)
set.seed(2015)
cwd <- 'C:\\Users\\karibuDell\\Desktop\\project\\PracticalMachineLearning'
setwd(cwd)
dat <- read.csv('data\\pml-training.csv', row.names = 1)
#https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
#https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

#remove half-NA, Zero- and Near Zero-Variance Predictors
dim(dat) #19622 rows   160 variavbles
dat <- dat[colSums(is.na(dat)) < 0.5*nrow(dat)]  #93 variavbles
nzv <- nearZeroVar(dat)
dat <- dat[, -nzv] #59 variavbles

#Identifyi and Remove Correlated Predictors
numericData <- dat[sapply(dat, is.numeric)]
descrCor <- cor(numericData)
summary(descrCor[upper.tri(descrCor)])
highlyCorDescr <- findCorrelation(descrCor, cutoff = .8)
highlyCorCol <- colnames(numericData[,highlyCorDescr])
dat <- dat[, -which(colnames(dat) %in% highlyCorCol)] #47 variavbles

#Simple Splitting Based on the Outcome by 6/4
inTraining <- createDataPartition(dat$classe, p = .6, list = FALSE)
training <- dat[ inTraining,]
testing  <- dat[-inTraining,]

#Model Training and Parameter Tuning
fitControl <- trainControl(method = "cv", number = 10)
              ##  10-fold CV

#Model List: http://topepo.github.io/caret/modelList.html
# Generalized Linear Model (glm) model,
start <- proc.time()
glmFit <- train(classe ~ ., data = training,
                method = "glm",
                trControl = fitControl,
                verbose = FALSE)
elapsed <- proc.time() - start
# glm could not be fitted

# gradient boosting machine (gbm) model,
start <- proc.time()
gbmFit <- train(classe ~ ., data = training,
                 method = "gbm",
                 trControl = fitControl,
                 verbose = FALSE)
elapsed <- proc.time() - start
gbmFit #Accuracy is 0.9974526

# Stochastic Gradient Boosting 
# 
# 11776 samples
# 46 predictor
# 5 classes: 'A', 'B', 'C', 'D', 'E' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold) 
# 
# Summary of sample sizes: 10599, 10598, 10597, 10598, 10599, 10599, ... 
# 
# Resampling results across tuning parameters:
#   
#   interaction.depth  n.trees  Accuracy   Kappa      Accuracy SD   Kappa SD    
# 1                   50      0.9994054  0.9992480  0.0006994086  0.0008845761
# 1                  100      0.9994054  0.9992480  0.0006994086  0.0008845761
# 1                  150      0.9994054  0.9992480  0.0006994086  0.0008845761
# 2                   50      0.9994904  0.9993554  0.0005939661  0.0007512831
# 2                  100      0.9995753  0.9994629  0.0004476359  0.0005661717
# 2                  150      0.9994904  0.9993554  0.0005939661  0.0007512831
# 3                   50      0.9994054  0.9992480  0.0006994086  0.0008845761
# 3                  100      0.9995754  0.9994629  0.0004475598  0.0005661135
# 3                  150      0.9995754  0.9994629  0.0004475598  0.0005661135
# 
# Tuning parameter 'shrinkage' was held constant at a value of 0.1
# Accuracy was used to select the optimal model using  the largest value.
# The final values used for the model were n.trees = 100, interaction.depth = 3 and
# shrinkage = 0.1. 
prediction_gbm <- predict(gbmFit, testing)
confusionMatrix(prediction_gbm, testing$classe)
# Overall Statistics
# 
# Accuracy : 0.9999     
# 95% CI : (0.9993, 1)
# No Information Rate : 0.2845     
# P-Value [Acc > NIR] : < 2.2e-16 
plot(gbmFit)

#Random Forest (RF)
rfFit <- train(classe ~ ., data = training,
                method = "rf",
                trControl = fitControl,
                verbose = FALSE)
rfFit #Accuracy is 0.9994904
# Random Forest 
# 
# 11776 samples
# 46 predictor
# 5 classes: 'A', 'B', 'C', 'D', 'E' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold) 
# 
# Summary of sample sizes: 10599, 10598, 10598, 10598, 10599, 10599, ... 
# 
# Resampling results across tuning parameters:
#   
#   mtry  Accuracy   Kappa      Accuracy SD   Kappa SD    
# 2    0.9928659  0.9909758  0.0028391926  0.0035917461
# 35    0.9994904  0.9993556  0.0007161565  0.0009056762
# 68    0.9992356  0.9990331  0.0007435277  0.0009403657
# 
# Accuracy was used to select the optimal model using  the largest value.
# The final value used for the model was mtry = 35. 
prediction <- predict(rfFit, testing)
confusionMatrix(prediction, testing$classe)
# Overall Statistics
# 
# Accuracy : 1          
# 95% CI : (0.9995, 1)
# No Information Rate : 0.2845     
# P-Value [Acc > NIR] : < 2.2e-16  


#Partial Least Squares (pls)
plsFit <- train(classe ~ ., data = training,
               method = "pls",
               trControl = fitControl,
               verbose = FALSE)
plsFit #Accuracy is 0.4682022 
# Partial Least Squares 
# 
# 11776 samples
# 46 predictor
# 5 classes: 'A', 'B', 'C', 'D', 'E' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold) 
# 
# Summary of sample sizes: 10599, 10599, 10598, 10600, 10598, 10598, ... 
# 
# Resampling results across tuning parameters:
#   
#   ncomp  Accuracy   Kappa      Accuracy SD   Kappa SD    
# 1      0.2843070  0.0000000  0.0002636053  0.0000000000
# 2      0.2843070  0.0000000  0.0002636053  0.0000000000
# 3      0.4681556  0.3013212  0.0003399693  0.0009783909
# 
# Accuracy was used to select the optimal model using  the largest value.
# The final value used for the model was ncomp = 3. 
prediction <- predict(plsFit, testing)
confusionMatrix(prediction, testing$classe)
# Overall Statistics
# 
# Accuracy : 0.4683          
# 95% CI : (0.4572, 0.4794)
# No Information Rate : 0.2845          
# P-Value [Acc > NIR] : < 2.2e-16   

finalTesting <- read.csv('data/pml-testing.csv', row.names = 1)
finalPrediction <- predict(gbmFit, finalTesting)
# write prediction files
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("./results/problem_id_", i, ".txt")
    write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, col.names = FALSE)
  }
}
pml_write_files(finalPrediction)

colnames(training)
final <- finalTesting[,which(colnames(finalTesting) %in% colnames(training))]
final[] <- mapply(FUN = as,final,sapply(training[,-47],class),SIMPLIFY = FALSE)
str(final)
str(training)
prediction <- predict(gbmFit, finalTesting)
table(prediction)
table(training$classe)
table(testing$classe)
table(prediction_gbm)
nrow(training) / nrow(testing)
