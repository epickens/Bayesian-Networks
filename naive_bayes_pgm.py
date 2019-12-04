import loader
import pandas as pd
import numpy as np
from pgmpy.models import NaiveBayes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder


col_names = pd.read_csv('data/names.csv')  # 'data/names.csv'
data = pd.read_csv('data/breast-cancer-wisconsin.data', names=col_names.columns)
data = data[data["bare_nuclei"] != '?']
data.set_index('id', inplace=True) #stop the model from using id as a node

train, test = train_test_split(data, test_size=0.2, random_state=0)
Y_test = test['class']
test = test.drop(['class'], axis=1)

#fit model
model = NaiveBayes()
model.fit(train, 'class')
print("Naive Bayes edges:        ", model.edges())

#make predictions
Y_pred = model.predict(test)

#Convert Labels so we can use sklearn function to evaluate our model
labelencoder = LabelEncoder()
Y_test = labelencoder.fit_transform(Y_test.values.ravel())
Y_pred = labelencoder.fit_transform(Y_pred.values.ravel())

# Output results
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)
print({"Accuracy": accuracy, "Precision": precision, "F1 Score": f1})