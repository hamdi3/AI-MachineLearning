#Logistic Regression Overview
import mglearn
mglearn.plots.plot_linear_regression_wave()

#Logistic Regression
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt 
%matplotlib inline

cancer = load_breast_cancer()

X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target, stratify = cancer.target, random_state = 42)

log_reg = LogisticRegression() #with default parameters
log_reg.fit(X_train,y_train) #to train it
print("Accuracy on the Training subset : {:.3f}" .format(log_reg.score(X_train, y_train)))
print("Accuracy on testing subeset : {:.3f}".format(log_reg.score(X_test,y_test))) 
#even tho useing this one only makes a lil diffrence (About 0.01) it's still more than enough for it to be better one in the Praxis

#Modefying the Logistic Regression to get the best accuracy
#Using Regularization
log_reg100 = LogisticRegression(C=100)
log_reg100.fit(X_train,y_train)
print("Accuracy on the Training subset with C = 100 : {:.3f}" .format(log_reg100.score(X_train, y_train)))
print("Accuracy on testing subeset C = 100 : {:.3f}".format(log_reg100.score(X_test,y_test))) 
log_reg001 = LogisticRegression(C=0.001)
log_reg001.fit(X_train,y_train)
print("Accuracy on the Training subset with C = 0.001 : {:.3f}" .format(log_reg001.score(X_train, y_train)))
print("Accuracy on testing subeset C = 0.001 : {:.3f}".format(log_reg001.score(X_test,y_test))) 
#in short Increasing C would make better results

#now to see it on our modell and see the difrences with diffrent C values
plt.plot(log_reg.coef_.T , "o" , label = "C=1") #coef_.T for when it's True
plt.plot(log_reg100.coef_.T , "^" , label = "C=100")
plt.plot(log_reg001.coef_.T , "v" , label = "C=0.01")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation = 90)
plt.hlines(0,0,cancer.data.shape[1])
plt.ylim(-5,5)
plt.xlabel(" Coefficient Index")
plt.ylabel(" Coefficient Magnitude")
plt.legend()
