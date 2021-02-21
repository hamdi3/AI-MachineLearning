#Here's an example on how it works
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import metrics module for accuracy calculation

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label'] 
# load dataset
pima = pd.read_csv("diabetes.csv", header=None, names=col_names,skiprows=1) #we had to skip the first row since it starts with the colomns
#Link for the dataset "https://www.kaggle.com/uciml/pima-indians-diabetes-database"
pima.head()

#split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features so basically the dependent variables
y = pima.label # Target variable

#the next step is pretty standard meaining that we should split our data in training set that will be predicted and test set with which the training set will be compared later on
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) #we need here 3 Parameters for the train_test_split() which are Features,
#Target and size (0.3 meining that 30% for testing and 70% for training) (The random_state is basically so that we get same values every time when we chose 1 as a value )

#Building a Decision Tree:
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#Visualizing
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())
