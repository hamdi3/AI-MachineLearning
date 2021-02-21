#Here's an optemierung 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import metrics module for accuracy calculation

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label'] 
# load dataset
pima = pd.read_csv("diabetes.csv", header=None, names=col_names,skiprows=1) #we had to skip the first row since it starts with the colomns
#Link for the dataset "https://www.kaggle.com/uciml/pima-indians-diabetes-database"
pima.head()

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3) #criterion : This parameter allows us to use the different-different attribute selection measure.
#Supported criteria are “gini” for the Gini index and “entropy” for the information gain. 
#The maximum depth of the tree. If None, then nodes are expanded until all the leaves contain less than min_samples_split samples.
#The higher value of maximum depth causes overfitting, and a lower value causes underfitting.

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#Visualization
from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes-optimal.png')
Image(graph.create_png())
