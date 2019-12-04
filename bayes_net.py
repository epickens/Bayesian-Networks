import loader
import pandas as pd
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.estimators import ParameterEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import BdeuScore, K2Score, BicScore
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import ConstraintBasedEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

###
###

### Note: this file may take a while to run
### Also Note: this files prints quite a bit out of the command line
### I tried making it as easy to read as possible, but you will have to do some scrolling

###
###

col_names = pd.read_csv('data/names.csv')  # 'data/names.csv'
data = pd.read_csv('data/breast-cancer-wisconsin.data', names=col_names.columns)
data = data[data["bare_nuclei"] != '?']
data.set_index('id', inplace=True) #stop the model from using id as a node

train, test = train_test_split(data, test_size=0.2, random_state=0)
Y_test = test['class']
test = test.drop(['class'], axis=1)

#convert labels to something that can be handled be sklearn's eval functions
labelencoder = LabelEncoder()
Y_test = labelencoder.fit_transform(Y_test.values.ravel())

### Greedy Structure Learning with Hill Climbing
hc = HillClimbSearch(data, scoring_method=BicScore(train))
hc_model = hc.estimate()

### Parameter Learning with Bayesian Estimation
hc_model.fit(train, estimator=BayesianEstimator, prior_type="BDeu")
### If the following for loop is un-commented the terminal will be flooded with CPDs
"""
for cpd in best_model.get_cpds():
    print(cpd)
"""

print()

### Another Method (it will throw errors about sample size - but it still runs and shouldn't be too messed up)
###Constraint Based Structure Learning
est = ConstraintBasedEstimator(train)

skel, seperating_sets = est.estimate_skeleton(significance_level=0.01)
print("Undirected edges: ", skel.edges())

pdag = est.skeleton_to_pdag(skel, seperating_sets)
print("PDAG edges:       ", pdag.edges())

cb_model = est.pdag_to_dag(pdag)
print("DAG edges:        ", cb_model.edges())

### Parameter learning with MLE
cb_model.fit(train, estimator=MaximumLikelihoodEstimator)

#Notice the significant difference in the connections that this version produces
#Print the final significant edges learned from constraint-based learning
print("The edges learned from constraint-based learning are:")
print(est.estimate(significance_level=0.01).edges())

#Print the hill climber's edges
print("The edges learned from score-based learning (hill climbing) are:")
print(hc_model.edges())

Y_pred_hc = hc_model.predict(test)
Y_pred_cb = cb_model.predict(test)

Y_pred_hc = labelencoder.fit_transform(Y_pred_hc.values.ravel())
Y_pred_cb = labelencoder.fit_transform(Y_pred_cb.values.ravel())

# Output results {'Accuracy': 0.9708029197080292, 'Precision': 0.9423076923076923, 'F1 Score': 0.9607843137254902}
accuracy_hc = accuracy_score(Y_test, Y_pred_hc)
precision_hc = precision_score(Y_test, Y_pred_hc)
f1_hc = f1_score(Y_test, Y_pred_hc)
print("The belief network learned using hill climbing and Bayesian estimation gives us:")
print({"Accuracy": accuracy_hc, "Precision": precision_hc, "F1 Score": f1_hc})

accuracy_cb = accuracy_score(Y_test, Y_pred_cb)
precision_cb = precision_score(Y_test, Y_pred_cb)
f1_cb = f1_score(Y_test, Y_pred_cb)
print("The belief network learned using constraint-based learning and MLE gives us:")
print({"Accuracy": accuracy_cb, "Precision": precision_cb, "F1 Score": f1_cb})