library(readr)
library(dplyr)
library(glmnet)
library(randomForest)
library(gridExtra)
library(ggplot2)
source('C:/Users/danny/Desktop/sta9890 project/plotfuns.R')
facebook_data <- read_csv("C:/Users/danny/Desktop/sta9890 project/facebook data.csv", col_names = FALSE)
temp = colnames(facebook_data) 
temp[54] <- 'response'
colnames(facebook_data) <- temp
colnames(facebook_data)


###################### ###################### ###################### 
###################### ###################### ###################### 
###################### ###################### ###################### 
###################### part 1,2 of the project ###################### 

## we dont have any incomplete cases --> no NAs
print(paste('Number of Incomplete Cases:',sum(!complete.cases(facebook_data))))

## number of observation
print(paste('Number of Observation:',nrow(facebook_data)))

## number of independent variables (predictors)
print(paste('Number of independent variables:', ncol(facebook_data)-1))

# library(dplyr)
equation_6_6 = function(x) {x / sqrt(mean((x - mean(x))^2))}
## using the mutate_all(function) in the dplyr library, we could pass a function to transform variable column-wisely.
data_altered = facebook_data %>% select(-response) %>%
  mutate_all(equation_6_6) %>%
  mutate(comment = facebook_data$response) 
data_altered %>% glimpse() 

## note that after alteration, variable X38 become NAs
sum(!is.na(data_altered$X38))

## dropping the column#38
dim(data_altered)
data_altered <- data_altered[,-38]
data_altered %>% glimpse() 
colnames(data_altered)
dim(data_altered)

sum(data_altered$comment > 10)  ## removing the zero comments

data_altered = data_altered[data_altered$comment > 10, ]

dim(data_altered)

# write.csv(data_altered, file='C:/Users/danny/Desktop/sta9890 project/data_altered.csv')

###################### ###################### ###################### 
###################### ###################### ###################### 
###################### ###################### ###################### 
###################### part 3 of the project ###################### 

X_all = as.matrix(data_altered[,1:52])
y_all = data_altered$comment
d = ncol(X_all)

n = nrow(data_altered)

n_train = 10*(ceiling(0.8*n)%/%10)  ## make sure it is a multiple of 10
n_test  = n - n_train
print(paste('Number of trainning data is:',n_train))
print(paste('Number of test data is:',n_test))



## create a matrix to store the shufflings for 100 simulation of Cross Validation
## this is the step to create reproducible output
set.seed(0)
i.mix = matrix(NA, nrow=100, ncol = n)
for (i in 1:100){
  i.mix[i, ] = sample(1:n)
}


## setting the lambdas for tuning 
lambda.las = c(seq(1e-1,2,length=100),seq(2.0001,10,length=100))  
lambda.rid = lambda.las*10
lambda.elast = lambda.las*2 

plot(lambda.las)

## 200 lambda values
nlam = length(lambda.las)   
nlam

#----------------------------------------------------------------------------------
#                         100 simulation of 10 fold Cross Validation
#----------------------------------------------------------------------------------
nsim = 100   ## set the number of simulation we want

# R_square   = data.frame(matrix(data=NA, nrow=400, ncol=4))
# colnames(R_square) = c('Simulation','Model','Lambda','R_square')
# colnames(R_square)

R_ridge = data.frame(matrix(data=NA, nrow=nsim, ncol=5))
R_lasso = data.frame(matrix(data=NA, nrow=nsim, ncol=5))
R_elast = data.frame(matrix(data=NA, nrow=nsim, ncol=5))
R_forest= data.frame(matrix(data=NA, nrow=nsim, ncol=5))

colnames(R_ridge) = c('lambda', 'Train_R_square','Test_R_square', 'Model', 'Time_Elapsed')
colnames(R_lasso) = c('lambda', 'Train_R_square','Test_R_square', 'Model', 'Time_Elapsed')
colnames(R_elast) = c('lambda', 'Train_R_square','Test_R_square', 'Model', 'Time_Elapsed')
colnames(R_forest) = c('lambda','Train_R_square','Test_R_square', 'Model', 'Time_Elapsed')

R_ridge$Model = 'Ridge'
R_lasso$Model = 'Lasso'
R_elast$Model = 'Elastic-Net'
R_forest$Model = 'Random Forest'

## for the final simulation residual plot
residual_rid_train   = rep(NA, n_train)
residual_las_train   = rep(NA, n_train)
residual_elast_train = rep(NA, n_train)
residual_rf_train    = rep(NA, n_train)

residual_rid_test   = rep(NA, n_test)
residual_las_test   = rep(NA, n_test)
residual_elast_test = rep(NA, n_test)
residual_rf_test    = rep(NA, n_test) 

for (i in 1:nsim){
  cat("Cross Validation Simulation",i,"\n")
  ## i.mix[i,] is the shuffling of 1:n for the ith simulation
  ## we will split the data into trainning set and test set
  ## the first n_test observation of the shuffled data will be test set
  X_test = X_all[i.mix[i,],][1:n_test, ]
  y_test = y_all[i.mix[i,]][1:n_test]
  
  X_train = X_all[i.mix[i,],][-(1:n_test), ]
  y_train = y_all[i.mix[i,]][-(1:n_test)]
  
  # These four matrices store the prediction errors for each
  # observation (along the rows), when we fit the model using
  # each value of the tuning parameter (along the columns)
  ## used to store cv error in tuning lambdas
  e.rid   = matrix(0,n_train,nlam)
  e.las   = matrix(0,n_train,nlam)
  e.elast = matrix(0,n_train,nlam)
  e.tree  = matrix(0,n_train,nlam)
  
  K = 10
  d = ceiling(length(y_train)/K)
  #_________________________________________________________________________________
  #_______________________K-fold cross validation for Ridge______________________
  #_________________________________________________________________________________
  time.start    = Sys.time()   ## time ridge for tuning
  for (k in 1:K) {
    folds = (1+(k-1)*d):(k*d);  ## [1,d],[d+1, 2d],...,[d(K-1)+1, dK]
    X.tr  = X_train[-folds, ]   # training predictors
    y.tr  = y_train[-folds]    # training responses
    X.val = X_train[folds, ] # validation predictors
    y.val = y_train[folds]  # validation responses
    
    a.rid           = glmnet(X.tr, y.tr, lambda = lambda.rid, alpha = 0)   # ridge regression model tuning
    rid.beta        = as.matrix(a.rid$beta[,nlam:1])        ## beta vector except beta_0
    yhat.rid        = X.val%*%rid.beta + a.rid$a0           ## prediction
    e.rid[folds,]   = (yhat.rid-y.val)^2                    ## error
  }
  R_ridge[i,5]  = Sys.time() - time.start  ## end time recorded
  
  #_________________________________ end of K-fold for Ridge  _______________________
  
  
  
  #_________________________________________________________________________________
  #_______________________K-fold cross validation for Lasso   ______________________
  #_________________________________________________________________________________
 
  time.start    = Sys.time()   ## time lasso for tuning
  for (k in 1:K) {
    folds = (1+(k-1)*d):(k*d);  ## [1,d],[d+1, 2d],...,[d(K-1)+1, dK]
    X.tr  = X_train[-folds, ]   # training predictors
    y.tr  = y_train[-folds]    # training responses
    X.val = X_train[folds, ] # validation predictors
    y.val = y_train[folds]  # validation responses
    
    a.las           = glmnet(X.tr, y.tr, lambda = lambda.las, alpha = 1)   # lasso regression
    las.beta        = as.matrix(a.las$beta[,nlam:1])  ## beta vector except beta_0
    yhat.las        = X.val%*%las.beta + a.las$a0    ## prediction
    e.las[folds,]   = (yhat.las-y.val)^2        ## error
  }
  R_lasso[i,5]  = Sys.time() - time.start
  
  #_________________________________ end of K-fold for lasso  _______________________
  
  
  
  #_________________________________________________________________________________
  #_________________K-fold cross validation for Elastic Net   ______________________
  #_________________________________________________________________________________
  time.start    = Sys.time()   ## start time elastic net for tuning
  for (k in 1:K) {
    cat("Fold",k,"\n")
    folds=(1+(k-1)*d):(k*d);  ## [1,d],[d+1, 2d],...,[d(K-1)+1, dK]
    X.tr  = X_train[-folds, ]   # training predictors
    y.tr  = y_train[-folds]    # training responses
    X.val = X_train[folds, ] # validation predictors
    y.val = y_train[folds]  # validation responses
    
    a.elast         = glmnet(X.tr, y.tr, lambda = lambda.elast, alpha = 0.5) # elastic-net regression
    elast.beta      = as.matrix(a.elast$beta[,nlam:1])
    yhat.elast      = X.val%*%elast.beta + a.elast$a0
    e.elast[folds,] = (yhat.elast-y.val)^2
  }
  R_elast[i,5]  = Sys.time() - time.start
  #_____________________________ end of K-fold for Elastic Net  _______________________
  
  
  
  ## 10-fold CV error or mean( MSE_i for i = 1,2,...10 where MSE_i is the i_th fold mean square error )
  cv.rid   = apply(e.rid,   2, mean)
  cv.las   = apply(e.las,   2, mean)
  cv.elast = apply(e.elast, 2, mean)
  
  ## standard error of (10-fold CV error)
  se.rid   = apply(e.rid,   2, sd)/sqrt(n_train)
  se.las   = apply(e.las,   2, sd)/sqrt(n_train)
  se.elast = apply(e.elast, 2, sd)/sqrt(n_train)
  
  i1.rid   = which.min(cv.rid)
  i1.las   = which.min(cv.las)
  i1.elast = which.min(cv.elast)
  
  i2.rid   = max( which(cv.rid   <= cv.rid[i1.rid]     + se.rid[i1.rid])   )
  i2.las   = max( which(cv.las   <= cv.las[i1.las]     + se.las[i1.las])   )
  i2.elast = max( which(cv.elast <= cv.elast[i1.elast] + se.elast[i1.elast]) )
  
  R_ridge[i,1] = lambda.rid[i1.rid]  ## records the best lambdas for each method
  R_lasso[i,1] = lambda.las[i1.las]
  R_elast[i,1] = lambda.elast[i1.elast]
  
  
  # build the models for estimation from the best lambdas
  a.rid   = glmnet(X_train, y_train, lambda = lambda.rid[i1.rid],     alpha = 0)
  a.las   = glmnet(X_train, y_train, lambda = lambda.las[i1.las],     alpha = 1)
  a.elast = glmnet(X_train, y_train, lambda = lambda.elast[i1.elast], alpha = 0.5)
  
  rid.beta   = as.matrix(a.rid$beta)
  las.beta   = as.matrix(a.las$beta)
  elast.beta = as.matrix(a.elast$beta)
  
  train_yhat.rid = X_train%*%rid.beta + a.rid$a0
  train_yhat.las = X_train%*%las.beta + a.las$a0
  train_yhat.elast = X_train%*%elast.beta + a.elast$a0
  
  test_yhat.rid = X_test%*%rid.beta + a.rid$a0
  test_yhat.las = X_test%*%las.beta + a.las$a0
  test_yhat.elast = X_test%*%elast.beta + a.elast$a0
  
  calculate_R_square = function(y, model_estimate, y_full = y_all){
    1 - mean((y - model_estimate)^2)/mean((y_all - mean(y_all))^2)
  }
  
  ## storing the trainning R^2
  R_ridge[i,2] = calculate_R_square(y = y_train, train_yhat.rid )
  R_lasso[i,2] = calculate_R_square(y = y_train, train_yhat.las )
  R_elast[i,2] = calculate_R_square(y = y_train, train_yhat.elast )
  
  ## storing the test R^2
  R_ridge[i,3] = calculate_R_square(y = y_test, test_yhat.rid)
  R_lasso[i,3] = calculate_R_square(y = y_test, test_yhat.las)
  R_elast[i,3] = calculate_R_square(y = y_test, test_yhat.elast)
  
  ## plot the CV plot for the last simulation only
  if (i==nsim){
    par(mfrow = c(1,1))
    plot.cv(cv.rid,   se.rid,   lambda.rid,   i1.rid,   i2.rid,   main='Ridge')
    plot.cv(cv.las,   se.las,   lambda.las,   i1.las,   i2.las,   main='Lasso')
    plot.cv(cv.elast, se.elast, lambda.elast, i1.elast, i2.elast, main='Elastic Net')
    
    residual_rid_train   = y_train - train_yhat.rid
    residual_las_train   = y_train - train_yhat.las
    residual_elast_train = y_train - train_yhat.elast
    
    residual_rid_test   = y_test - test_yhat.rid
    residual_las_test   = y_test - test_yhat.las
    residual_elast_test = y_test - test_yhat.elast
  }
}



##_________________________________________________________________________________
##____________________________ random forest: 100 simulation________________________
##_________________________________________________________________________________

colnames(R_forest)
dim(X_all)
length(y_all)


nsim=100
for (i in 1:nsim){
  cat("Random Forest Simulation",i,"\n")
  X_train = X_all[i.mix[i,], ][-(1:n_test),] 
  y_train = y_all[i.mix[i,] ][-(1:n_test)]
  
  X_test1 = X_all[i.mix[i,], ][1:n_test,] 
  y_test1 = y_all[i.mix[i,] ][1:n_test ]

  dim(X_train)
  dim(X_test1)
  length(y_test1)
  
  time.start       =   Sys.time() #time random forest 
  rf_model         =   randomForest( X_train, y_train, mtry = sqrt(ncol(X_train)), ntree = 15, importance = TRUE)
  train_yhat.rf    =   predict(rf_model, X_train)
  test_yhat.rf1    =   predict(rf_model, X_test1)
  
  R_forest[i,2]    =   1 - mean((y_train - train_yhat.rf)^2)  /(mean((y_all - mean(y_all))^2))
  R_forest[i,3]    =   1 - mean((y_test1  - test_yhat.rf1 )^2)/(mean((y_all - mean(y_all))^2))
  R_forest[i,5]    =   Sys.time() - time.start  ## time recorded
  
  if (i == nsim){
    residual_rf_train = (y_train - train_yhat.rf)
    residual_rf_test = (y_test1 - test_yhat.rf1)
  }
} 

hist(residual_rf_train)
hist(residual_rf_test)
###____________________________________end of random forest simulation______________________________________








## we want to combined the following for plotting
# residual_rid_train  
# residual_las_train  
# residual_elast_train
# residual_rf_train
# 
# residual_rid_test
# residual_las_test
# residual_elast_test
# residual_rf_test

residuals_all  = data.frame( 
                            Model = c(
                                          rep('Ridge', n_train),
                                          rep('Lasso', n_train),
                                          rep('Elastic Net', n_train),
                                          rep('Random Forest', n_train),
                                          rep('Ridge', n_test),
                                          rep('Lasso', n_test),
                                          rep('Elastic Net', n_test),
                                          rep('Random Forest', n_test) ),
                            Residuals = c(
                                          residual_rid_train,
                                          residual_las_train,
                                          residual_elast_train,
                                          residual_rf_train,
                                          residual_rid_test,
                                          residual_las_test,
                                          residual_elast_test,
                                          residual_rf_test
                                          ),
                            isTest   = c(
                                          rep('Train', 4*n_train),
                                          rep('Test', 4*n_test))
                            )
colnames(residuals_all)

ggplot(residuals_all)+
  aes(col = Model, x=Residuals)+
  geom_density()+
  facet_grid(.~isTest)

ggplot(residuals_all)+
  aes(col = Model, x=Residuals)+
  geom_density()+
  facet_grid(isTest~.)

ggplot(residuals_all)+
  aes(x = Model, y=Residuals)+
  geom_boxplot()+
  facet_grid(.~isTest)+
  ggtitle('Boxplot of Train and Test Residual of One Simulation')+
  theme(plot.title = element_text(hjust = 0.5))+
  ylab('Residual: y - estimate(y)')

ggplot(residuals_all)+
  aes(col = Model, x=Residuals)+
  geom_density()+
  facet_grid(isTest~.)+
  ggtitle('Aprroximate PDF of Train and Test Residual of One Simulation')+
  theme(plot.title = element_text(hjust = 0.5))+
  xlab('Residual: y - estimate(y)')

ggplot(residuals_all[residuals_all$isTest == 'Test', ])+
  aes(col = Model, x=Residuals)+
  geom_density()+
  ggtitle('Aprroximate PDF of Test Residual of One Simulation')+
  theme(plot.title = element_text(hjust = 0.5))+
  xlab('Residual: y - estimate(y)')


ggplot(residuals_all[residuals_all$isTest == 'Train', ])+
  aes(col = Model, x=Residuals)+
  geom_density()+
  ggtitle('Aprroximate PDF of Train Residual of One Simulation')+
  theme(plot.title = element_text(hjust = 0.5))+
  xlab('Residual: y - estimate(y)')

R_ridge
R_lasso
R_elast
R_forest

result0 = as.data.frame(rbind(R_ridge,R_lasso,R_elast))
result = as.data.frame(rbind(R_ridge,R_lasso,R_elast, R_forest))


## lets save the result data for analysis

# write.csv(result, file = "C:/Users/danny/Desktop/sta9890 project/result_cv_coded_reduce.csv")
# loading the result
# result = read.csv(file = "C:/Users/danny/Desktop/sta9890 project/result_cv_coded_reduce.csv")

# result[result$Model == 'Random Forest',]



# install.packages('ggplot')
library(ggplot2)


ggplot(result)+
  aes(x=Test_R_square, col=Model)+
  geom_density()+
  ggtitle('Test R^2 Density Plot')+
  theme(plot.title = element_text(hjust = 0.5))

ggplot(result)+
  aes(x=Train_R_square, col=Model)+
  geom_density()+
  ggtitle('Train R^2 Density Plot')+
  theme(plot.title = element_text(hjust = 0.5))


ggplot(result)+
  aes(y=Test_R_square, x=Model)+
  geom_boxplot()+
  ggtitle('Test R^2 Boxplot Plot')+
  theme(plot.title = element_text(hjust = 0.5))

ggplot(result)+
  aes(y=Train_R_square, x=Model)+
  geom_boxplot()+
  ggtitle('Train R^2 Boxplot Plot')+
  theme(plot.title = element_text(hjust = 0.5))
#

library(dplyr)
## average time spent
result %>% 
      group_by(Model) %>% 
      mutate(Average_Time_per_Simulation = mean(Time_Elapsed)) %>% 
      select(Model, Average_Time_per_Simulation) %>% 
      unique()
  

##time spent on tuning
ggplot(result0)+
  aes(y=Time_Elapsed, x=Model)+
  geom_boxplot()+
  ggtitle('Boxplot of Time Spent Tuning')+
  theme(plot.title = element_text(hjust = 0.5))

ggplot(result)+
  aes(x=Time_Elapsed, col=Model, fill = Model)+
  geom_density()+
  ggtitle('Density of Time Spent Tuning')+
  theme(plot.title = element_text(hjust = 0.5))

ggplot(result)+
  aes(x=Time_Elapsed, fill=Model)+
  geom_histogram()+
  ggtitle('Histogram of Time Spent Tuning')+
  theme(plot.title = element_text(hjust = 0.5))+
  facet_grid(Model~.)


## average lambda value pick by CV
library(dplyr)
result %>% 
  group_by(Model) %>% 
  mutate(mean_lambda = mean(lambda)) %>% 
  select(Model, mean_lambda) %>% unique()

ggplot(result0)+
  aes(x=lambda, fill=Model)+
  geom_density()+
  ggtitle('Distribution of Lambda Picked by CV')+
  theme(plot.title = element_text(hjust = 0.5))+
  facet_grid(Model~.)



## doing a boxplot for in one scale
colnames(result)
train_result = result[,-3]
train_result['Train_Test'] = 'train'
colnames(train_result)[2] = 'R_square'

test_result = result[,-2]
test_result['Train_Test'] = 'test'
colnames(test_result)[2] = 'R_square'

R_sq = rbind(train_result,test_result)
colnames(R_sq)

ggplot(R_sq)+
  aes(x=Model, y = R_square)+
  geom_boxplot()+
  facet_grid(.~Train_Test)+
  ggtitle('Comparison Between Test R^2 and Train R^2')+
  theme(plot.title = element_text(hjust = 0.5))

ggplot(R_sq)+
  aes(col=Model, x = R_square)+
  geom_density()+
  facet_grid(.~Train_Test)+
  ggtitle('Approximated PDF for Test R^2 and Train R^2')+
  theme(plot.title = element_text(hjust = 0.5))

























##________________________________________________________________________________
##____________________Boostrap Example____________________________________________
##________________________________________________________________________________
dim(X_all)
n = nrow(X_all)

lambda.las = c(seq(1e-1,2,length=50),seq(2.0001,10,length=50))  
lambda.rid = lambda.las*10
lambda.elast = lambda.las*2 


bootstrapSamples =     100 
p                =     ncol(X_all)
beta.rf.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.en.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples) 
beta.rid.bs      =     matrix(0, nrow = p, ncol = bootstrapSamples)
beta.las.bs      =     matrix(0, nrow = p, ncol = bootstrapSamples)


set.seed(0)  ## ensure reproducibility
indexing = matrix(NA, nrow=bootstrapSamples, ncol = n)
for (i in 1:100){
  indexing[i, ] = sample(1:n,  replace=T, size = n)
}

dim(indexing)

for (m in 1:bootstrapSamples){
  # bs_indexes       =     sample(n, replace=T)  ## this is why it is called boostrap, sampling on the original data set
  X.bs             =     X_all[indexing[m,], ]
  y.bs             =     y_all[indexing[m,]]
  
  ## fit bs rf
  rf               =     randomForest(X.bs, y.bs, mtry = sqrt(p), ntree = 15, importance = TRUE)
  beta.rf.bs[,m]   =     as.vector(rf$importance[,1])
  
  # fit bs en
  a                =     0.5 # elastic-net
  cv.fit.en        =     cv.glmnet(X.bs, y.bs, alpha = a, nfolds = 10, lambda = lambda.elast )
  fit              =     glmnet(X.bs, y.bs, alpha = a, lambda = cv.fit.en$lambda.min)  
  beta.en.bs[,m]   =     as.vector(fit$beta)
  
  # fit ridge
  a                =     0 # Ridge regression
  cv.fit.rid       =     cv.glmnet(X.bs, y.bs, alpha = a, nfolds = 10, lambda = lambda.rid )
  fit              =     glmnet(X.bs, y.bs, alpha = a, lambda = cv.fit.rid$lambda.min)  
  beta.rid.bs[,m]  =     as.vector(fit$beta)
  
  ## fit lasso
  a                =     1 # lasso
  cv.fit.las       =     cv.glmnet(X.bs, y.bs, alpha = a, nfolds = 10, lambda = lambda.las  )
  fit              =     glmnet(X.bs, y.bs, alpha = a, lambda = cv.fit.las$lambda.min)  
  beta.las.bs[,m]  =     as.vector(fit$beta)
  
  cat(sprintf("Bootstrap Sample %3.f \n", m))
}



# calculate bootstrapped standard errors / alternatively we could use qunatiles to find upper and lower bounds
rf.bs.sd    = apply(beta.rf.bs, 1, "sd")
en.bs.sd    = apply(beta.en.bs, 1, "sd")
rid.bs.sd   = apply(beta.rid.bs, 1, "sd")
las.bs.sd   = apply(beta.las.bs, 1, "sd")


# fit rf to the whole data
fit.rf           =     randomForest(X_all, y_all, mtry = sqrt(p),ntree = 15, importance = TRUE)

# fit elastic net to the whole data
a=0.5 # elastic-net
cv.fit.en        =     cv.glmnet(X_all, y_all, alpha = a, nfolds = 10,lambda = lambda.elast  )
fit.en           =     glmnet(X_all, y_all, alpha = a, lambda = cv.fit.en$lambda.min)

# fit rid to the whole data
a=0 
cv.fit.rid       =     cv.glmnet(X_all, y_all, alpha = a, nfolds = 10,lambda = lambda.rid )
fit.rid          =     glmnet(X_all, y_all, alpha = a, lambda = cv.fit.rid$lambda.min)

# fit lasso elastic net to the whole data
a=1 
cv.fit.rid       =     cv.glmnet(X_all, y_all, alpha = a, nfolds = 10, lambda = lambda.las)
fit.las          =     glmnet(X_all, y_all, alpha = a, lambda = cv.fit.las$lambda.min)




betaS.rf               =     data.frame(names(X_all[1,]),  as.vector(fit.rf$importance[,1]), 2*rf.bs.sd)
colnames(betaS.rf)     =     c( "feature", "value", "err")

betaS.en               =     data.frame(names(X_all[1,]),  as.vector(fit.en$beta), 2*en.bs.sd)
colnames(betaS.en)     =     c( "feature", "value", "err")

betaS.rid               =     data.frame(names(X_all[1,]), as.vector(fit.rid$beta), 2*rid.bs.sd)
colnames(betaS.rid)     =     c( "feature", "value", "err")

betaS.las               =     data.frame(names(X_all[1,]), as.vector(fit.las$beta), 2*las.bs.sd)
colnames(betaS.las)     =     c( "feature", "value", "err")

library(ggplot2)
rfPlot =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  ggtitle('Random Forest')+
  theme(plot.title = element_text(hjust = 0.5))

enPlot =  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)+
  ggtitle('Elastic Net')+
  theme(plot.title = element_text(hjust = 0.5))

ridPlot =  ggplot(betaS.rid, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)+
  ggtitle('Ridge')+
  theme(plot.title = element_text(hjust = 0.5))

lasPlot =  ggplot(betaS.las, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)+
  ggtitle('Lasso')+
  theme(plot.title = element_text(hjust = 0.5))


grid.arrange(rfPlot, enPlot, nrow = 2)

grid.arrange(ridPlot, lasPlot, nrow = 2)

# we need to change the order of factor levels by specifying the order explicitly.
betaS.rf$feature     =  factor(betaS.rf$feature,  levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.en$feature     =  factor(betaS.en$feature,  levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.rid$feature    =  factor(betaS.rid$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.las$feature    =  factor(betaS.las$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])


rfPlot =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  ggtitle('Random Forest')+
  theme(plot.title = element_text(hjust = 0.5))

enPlot =  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)+
  ggtitle('Elastic Net')+
  theme(plot.title = element_text(hjust = 0.5))

ridPlot =  ggplot(betaS.rid, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)+
  ggtitle('Ridge')+
  theme(plot.title = element_text(hjust = 0.5))

lasPlot =  ggplot(betaS.las, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2)+
  ggtitle('Lasso')+
  theme(plot.title = element_text(hjust = 0.5))

grid.arrange(rfPlot, enPlot, nrow = 2)
grid.arrange(ridPlot, lasPlot, nrow = 2)

grid.arrange(rfPlot, lasPlot,ridPlot, enPlot, nrow = 4)



##________________________________________________________________________________
##____________________End of Boostrap Example_____________________________________




