# ==============================
# LOGISTIC MODEL
#
# overview on logistic regression
# ===============================

#### 1. LOADING DATA #####
# The goal of our study is to analyze an individual's (Y) creditworthiness, 
# using the other covariates in the dataset (X) as input variables.

str(credit) #we can leave (1,2) as codification for bad and good since the variable is a "factor" otherwise we have to change it to (0,1)
head(credit)
summary(credit)
attach(credit) 

# defining training set and test set

set.seed(123)

indices <- sample(seq_len(nrow(credit)), 0.75 * nrow(credit))
train <- credit[indices, ]
test  <- credit[-indices, ]

# ==========================================
#### 2. LOGISTIC MODEL WITH 1 REGRESSOR ####
# ==========================================
#
# we first study the output given by 1 regressor
#glm: generalized linear models

mod1<-glm(affidabilita~saldo_conto_bin,family=binomial,credit) # y = affidabilità, x = saldo conto

## OUTPUT INTERPRETATION ##

# with summary we see the output of the mod1
summary(mod1) 

# both intercept and the regressor are significant variable at 0.05%
# saldo_conto_binpos = balance account it can be positive or negative
# 
# the output shows a logit of 1.69, meaning that having a positive balance increase
# the logit of that quantity making the person more reliable in creditworthiness

# OR calculation, represent the increase or decrease of the probability respect to the state of the varibale
exp(coef(mod1)) #5.46

# Having a positive account balance increases the probability of becoming a 'god' creditor
# by 5.46 times compared to someone with a negative balance.

## PROBABILITY ESTIMATION ##

# using the coefficients of the model we can estimate the probability of a client 
# to be considered good or bad based on the other regressor
#
# we apply the formula for the logistic regression:

P<-(exp(coef(mod1)[1]+coef(mod1)[2]))/(1+ exp(coef(mod1)[1]+coef(mod1)[2]))
P #88.32% 
# is the probability estimated by the model that a client with positive balance account is "good"
# 1 - P = 11.68% is the prob. that a client with positive balance account is "bad"

P<-(exp(coef(mod1)[1]))/(1+ exp(coef(mod1)[1])) 
# IMPORTANT: the varibale saldo_conto_binpos is (0 = negative , 1 = positive) so in the fromula 
# we do not consider the second coefficients since it would have been 1.69 * 0 beacuse
# now we want to study the probability for a negative balance account
P # 58.09%
# is the probability estimated by the model that a client with negative balance account is "good"
# 1 - P = 41.91% is the prob. that a client with negative balance account is "bad"

# it is possible to replicate this study for all the other variables

# =============================================
#### 3. LOGISTIC MODEL WITH ALL REGRESSORS ####
# =============================================
# we just use . to select all variables on the training set

modT<-glm(affidabilita~.,family=binomial,train) 
summary(modT)
# some parameters are not significant
# i run a new model with only significant parameters

# in order to reduce the regressor by keeping only the ones who are significant
# we have to run a step regression (forward, backward or both)
# based on our choice this type of regression add or remove at every iteration
# the regressor the reduce or increase the deviance

modS<-step(modT, direction = "backward")
summary(modS)

# age variable does not seem to be significant also in this case 
#
# b1 interpretation: Having a positive account balance, as opposed to a negative one, 
# increases the logit by 1.315 while holding the values of all other variables constant.
# the interpetation follows the same logic for all the other coefficients.

exp(coef(modS)) 
# b1: The probability (p) that a customer with a positive binary account balance is reliable
# is 3.726 times that of a customer with a negative account balance holding all other variables constant

# ==========================
#### 4. GOODNESS OF FIT ####
# ==========================
#
# install.packages("ResourceSelection")
library(ResourceSelection) 

## Hosmer-Lemeshow ##
#
# the null Hypothesis H0 is to have a model with a good fit of the data
# so we expect a hig p_value in order to NOT reject H0
# g = 10 is a convention corrispond to the number of groups
# fitted are the probabilities given by the model
hoslem.test(modS$y, fitted(modS), g=10) 

# p_value: 0.95 the model shows a good fit with the data

## Residuals ##

resid(modS) #deviance
resid(modS, type='pearson') #Pearson

# =======================
#### 5. ANOVA METHOD ####
# =======================
#
# This method is used to choose the best model by selecting the test we want to compute
# in this case we perform the Chi-squared test for the similarity between two models

# i create a new model with 2 regressors
# since we saw mod1 on the full dataset i use the full dataset also here
mod2<-glm(affidabilita~saldo_conto_bin+durata_prestito,family=binomial,credit) 
summary(mod2)

#anova test
anova(mod1,mod2,test="Chisq") 
# we have a small significant p_value so i reject the Hp that the models are similar and
# i chosse the one with 2 regressors
#
# another method is to choose the model with the lowest AIC

# ======================
#### 6. PREDICTIONS ####
# ======================
#
# it possible now to use the model we trained on the new dataset test
# we are going to predict the variable Y and then we will study the results

res<-predict(modS, newdata=test, type="response") # type="response" gives the probabilities 
res
#it assigns the probability to bee good or bad to every obsarvation in the test set

# ==========================
#### 7. ACCURACY MATRIX ####
# ==========================
#
# In order to study the results we create a contingent table between the actual value of the Y
# and the one predicted by our model.
#
# as default parameter to assess when a client is good or bad we use 0.5 meaning that if a client
# has a prob > 0.5 it is considered "good". Later we will see how to optimally change this value

contingent_table<-table(Actualvalue= test$affidabilita, Predictedvalue= res>0.5)
contingent_table
# it seems that our model is very good in predicting if a client is "good" but less accurate 
# in predicting if a client is bad.

accuracy<- (contingent_table[1,1]+contingent_table[2,2])/(sum(contingent_table))
accuracy #72.4%
#the overall accuracy is 72.4% 
#is given by the ratio of (true negative + true positive)/(tot observations)

# ================================
#### 8. ROC CURVE AND YOUDEN'S J ####
# ================================
#
# 
library(ROCR)

# First we estimate the prob to belonging to "good" for each observation in train set
# we find the benchmark in-sample meaning that we use the training set
# optimal would be to have a validation set
res2<- predict(modS, train, type = "response")

# The following object contains The predicted probabilities (res2) and
# The actual class labels (train$affidabilita)
#
# ROCR uses these two vectors to evaluate the classifier
# at different probability thresholds.
ROCRPred=prediction(res2, train$affidabilita)

# Calculate the ROC curve coordinates:
# TPR = True Positive Rate = Sensitivity
# FPR = False Positive Rate = 1 - Specificity
#
# Each point corresponds to a different classification
# threshold applied to the predicted probabilities.
ROCRPref<-performance(ROCRPred, "tpr", "fpr")

plot(ROCRPref, colorize=TRUE, print.cutoffs.at=seq(0.1, by=0.1)) 

## YOUDEN'S J STATISTIC ##
#
# is defined by youden_J <- tpr - fpr 
# it is used to combine sensitivity and specificity in a single indicator
# the goal is to identify the highest J that gives the better combination
# between sensitivity and specificity.

#extracting true positve and false positive 
tpr <- ROCRPref@y.values[[1]]
fpr <- ROCRPref@x.values[[1]]

# Extract the corresponding probability cutoffs
cutoffs <- ROCRPred@cutoffs[[1]]

# Calculate Youden's J
youden_J <- tpr - fpr

# Find the optimal cutoff
best_idx <- which.max(youden_J)
best_cutoff <- cutoffs[best_idx]

# Display results
best_cutoff #0.6743

youden_J[best_idx] #0.4747           #value of YOUDEN'S statistic at the best cutoff
tpr[best_idx]  #0.7408               #value true positive rate (sensitivity)
1 - fpr[best_idx] #0.7339            #value false positive rate (specificity)


# ==================================
#### 9. FINAL MATRIX EVALUATION ####
# ==================================
#
# computing the contingent table using the new benchmark values

#best_cutoff
contingent_table<-table(Actualvalue= test$affidabilita, Predictedvalue= res>0.6743)
contingent_table
accuracy<-(contingent_table[1,1]+contingent_table[2,2])/(sum(contingent_table))
accuracy #69.6% accuracy decraease but model identify better bad client


