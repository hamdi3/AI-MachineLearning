#KNN Classifier Overview 
import mglearn
mglearn.plots.plot_knn_classification(n_neighbors=10) #usually five is the standard
#The intuition behind the KNN algorithm is one of the simplest of all the supervised machine learning algorithms. 
#It simply calculates the distance of a new data point to all other training data points. The distance can be of any type e.g Euclidean or Manhattan etc.
#It then selects the K-nearest data points,
#where K can be any integer. Finally it assigns the data point to the class to which the majority of the K data points belong.

from sklearn.datasets import load_breast_cancer #to load the database
from sklearn.neighbors import KNeighborsClassifier #to classify the k neighbores
from sklearn.model_selection import train_test_split #to spilt the data

import matplotlib.pyplot as plt 
%matplotlib inline 
#With this backend, the output of plotting commands is displayed inline within frontends like the Jupyter notebook, directly below the code cell that produced it. The resulting plots will then also be stored in the notebook document.

cancer = load_breast_cancer() #to load the dataset

#train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state = 42 )
#train_test_split splits arrays or matrices into random train and test subsets. That means that everytime you run it without specifying random_state, you will get a different result, this is expected behavior. However, if a fixed value is assigned like random_state = 42 then no matter how many times you execute your code the result would be the same .i.e, same values in train and test datasets.

knn = KNeighborsClassifier()
knn.fit(X_train, y_train) #to train our knn
print('Accuracy of KNN n-5, on the training set: {:.3f}'.format(knn.score(X_train, y_train))) #to evaluate how it did on training set
print('Accuracy of KNN n-5, on the test set: {:.3f}'.format(knn.score(X_test, y_test)))#to evaluate how it did on the test set

#Testing the K-Nearest Neighbors Algorithm to see when it's most accurate
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify = cancer.target, random_state = 66)

training_accuracy = []
test_accuracy = []

neighbos_settings = range(1,11) 
for n_neighbors in neighbos_settings :
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train,y_train)
    training_accuracy.append(clf.score(X_train,y_train))
    test_accuracy.append(clf.score(X_test,y_test))

plt.plot(neighbos_settings,training_accuracy,label=" Accuracy of the training set")
plt.plot(neighbos_settings,test_accuracy,label="Accuracy of the testing set")
plt.xlabel("Number of neighbors")
plt.ylabel("Accuracy")
plt.legend()
