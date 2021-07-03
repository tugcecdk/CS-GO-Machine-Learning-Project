
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor
import math

#--------------------------- ALGORITHMS -------------------------------#

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
def decision_tree_classifier(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier()
    model = model.fit(X_train,y_train.values.ravel())
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return y_pred, acc, model
 
# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
def random_forest_classifier(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train,y_train.values.ravel())
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return y_pred, acc, model

# KNN
from sklearn.neighbors import KNeighborsClassifier
def knn(X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return y_pred, acc, model
    
# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
def logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=len(data))
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return y_pred, acc, model

# ADABOOST CLASSIFIER
from sklearn.ensemble import AdaBoostClassifier
def adaboost_classifier(X_train, y_train, X_test, y_test):
    # Create adaboost classifer object
    model = AdaBoostClassifier(n_estimators=50, learning_rate=1)
    # Train Adaboost Classifer
    model.fit(X_train, y_train.values.ravel())
    #Predict the response for test dataset
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return y_pred, acc, model

# SVC
from sklearn.svm import SVC
def support_vector(X_train, y_train, X_test, y_test):
    model = SVC(random_state = 0) 
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return y_pred, acc, model

# NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
def naive_bayes(X_train, y_train, X_test, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return y_pred, acc, model

# Multi-Layer Perceptron
from sklearn.neural_network import MLPClassifier 
def mlp_classifier(X_train, y_train, X_test, y_test):
    model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return y_pred, acc, model
 

#--------------------------- FINDING BEST ALGORITHM WITH 10-FOLD -------------------------------#
 
    
algo_names = ["Decision Tree", "Random Forest", "KNN", "Logistic Regression", "ADA Boost", "Support Vector", "Naive Bayes", "MLP Classifier"]
    
executor = ThreadPoolExecutor(max_workers=3)
futures_list = []

def add_queue(x_train, x_test, y_train, y_test, print_stats=True):
    
    futures_list.append(executor.submit(decision_tree_classifier, x_train, x_test, y_train, y_test))
    futures_list.append(executor.submit(random_forest_classifier, x_train, x_test, y_train, y_test))
    futures_list.append(executor.submit(knn, x_train, x_test, y_train, y_test))
    futures_list.append(executor.submit(logistic_regression, x_train, x_test, y_train, y_test))
    futures_list.append(executor.submit(adaboost_classifier, x_train, x_test, y_train, y_test))
    futures_list.append(executor.submit(support_vector, x_train, x_test, y_train, y_test))
    futures_list.append(executor.submit(naive_bayes, x_train, x_test, y_train, y_test))
    futures_list.append(executor.submit(mlp_classifier, x_train, x_test, y_train, y_test))
          
def run_algs():
    results = np.zeros((8, k))
    for i, feature in enumerate(futures_list):
        predicts, acc, model = feature.result(timeout=60*60)
        results[i%8][math.floor(i/8)] = acc
        print(i+1, acc)

    return results
 
    
data = pd.read_csv("processed_data.csv", sep=",")

data["team_1"] = data["team_1"].astype('category')
data["team_2"] = data["team_2"].astype('category')
data["_map"] = data["_map"].astype('category')
data["starting_ct"] = data["starting_ct"].astype('category')
data["map_winner"] = data["map_winner"].astype('category')
print(data.dtypes)

x =  data.iloc[:, 0:-1]
y = data.iloc[:, [-1]] 
    
# split data with k-fold
k = 10
kf_indices = []
kf = KFold(n_splits=k)
indices = []
for train, test in kf.split(x):
    kf_indices.append((train, test))
        
for i, (train_idx, test_idx) in enumerate(kf_indices):
    # take k. fold values
    x_train, x_test, y_train, y_test = x.iloc[train_idx, :], x.iloc[test_idx, :], y.iloc[train_idx, :], y.iloc[test_idx, :]
    # run all algorithms at k. fold data
    add_queue(x_train, y_train, x_test, y_test, print_stats=False)
    
kfold_results = run_algs()
    
algorithms_results = np.mean(kfold_results, axis=1)


def bar_plot(values, title):
    plt.figure(figsize=(12, 10))
    plt.bar(algo_names, values)
    plt.title(title)
    plt.figtext(.8, .89, "Average of all: " + str(round(np.mean(values), 2)))
    plt.xlabel('Regression Algorithms')
    plt.ylabel('R^2 Scores')
    plt.show()
    
    
bar_plot(algorithms_results, "Accuracy Scores")

print("\n-- 10 Fold Average Accuracy Results --")
for i in range(len(algorithms_results)):
    print(algo_names[i] + ": " + str(round(np.mean(algorithms_results[i]), 4)))
