import loader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score


#load our data and break it into train test sets using our helper file
X_train, X_test, Y_train, Y_test = loader.load_data('data/names.csv', 'data/breast-cancer-wisconsin.data',
                                                    cols=True, is_wisconsin_bc=True)


#Let's build a (or a few) traditional supervised learning classifier to try and get a handle on our data
#These will be used as a baseline to compare to the Bayes Net Classifier we will be building in other files

#lets create a list of classifiers so we can compare their results at the end
classifiers = []
classifier_names = ['Logistic Regression', 'Naive Bayes', 'Random Forest']

#logistic regression
log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)
classifiers.append(log_reg)

#Naive Bayes (keep an eye on this one because it comes up in the paper)
nb = GaussianNB()
nb.fit(X_train, Y_train)
classifiers.append(nb)

#Good old random forest
rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
rf.fit(X_train, Y_train)
classifiers.append(rf)

comb_list = zip(classifiers, classifier_names)

for classifier, name in comb_list:
    Y_pred = classifier.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)
    print("Confusion matrix for {} classifier:".format(name))
    print(cm)
    print("Giving us the scores:")
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    print({"Accuracy": accuracy, "Precision": precision, "F1 Score": f1})

#The random forest classifier appears to slightly edge out the simple logistic regression in this case
#but given the straight forward-ness of logistic regression I'd prefer to proceed with it
#so let's create a final cross validated version

log_reg_cv = LogisticRegressionCV(cv=10)
log_reg_cv.fit(X_train, Y_train)
print("\n")
print("Cross validated logistic regression gives us the scores:")
Y_pred = log_reg_cv.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)
print({"Accuracy": accuracy, "Precision": precision, "F1 Score": f1})

#Based on the above results it looks like cross validated logistic regression is just as good




