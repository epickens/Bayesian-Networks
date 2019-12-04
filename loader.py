#quick helper file for loading in data
#this project specifically looks at the wisconsin breast cancer dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(col_names_dir, data_dir, cols=False, is_wisconsin_bc=False):

    if cols:
        col_names = pd.read_csv(col_names_dir)#'data/names.csv'
        data = pd.read_csv(data_dir, names=col_names.columns)#'data/breast-cancer-wisconsin.data'
    else:
        data = pd.read_csv(data_dir)

    #just a quick bit of data cleaning related to the wisconsin breast cancer data set
    #for some reason there is a random ? in the data
    #this if statement just clears that up if you use this function to load that specifc data set
    if is_wisconsin_bc:
        data = data[data["bare_nuclei"] != '?']

    #print(data.isna().sum()) #No NA values

    #Separate Training data from labels
    #assumes the final col is the label
    num_cols = data.shape[1]
    Y = data.iloc[:,-1].values
    X = data.iloc[:,1:num_cols-1].values

    #Convert Labels to either 0 or 1 (I don't want to use the default 2 or 4
    labelencoder = LabelEncoder()
    Y = labelencoder.fit_transform(Y)

    #create train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    #print(X_train)
    #Scale train/test values to keep things consistent
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, Y_train, Y_test

