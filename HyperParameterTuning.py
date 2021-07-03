

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#---------------------------  SEARCHING FOR BETTER PARAMETERS -------------------------------#

def bar_plot(values, title, x_labels):
    plt.figure(figsize=(12, 10))
    plt.bar(range(len(values)), values)
    plt.xticks(rotation = 45)
    plt.title(title)
    plt.figtext(.8, .89, "Best of all: " + str(round(np.max(values), 3)))
    plt.xlabel('')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(x_labels)), labels=x_labels)
    for i in range(len(values)):
        plt.text(i,values[i],round(values[i], 3))
    plt.show()

data = pd.read_csv("processed_data.csv", sep=",")

data["team_1"] = data["team_1"].astype('category')
data["team_2"] = data["team_2"].astype('category')
data["_map"] = data["_map"].astype('category')
data["starting_ct"] = data["starting_ct"].astype('category')
data["map_winner"] = data["map_winner"].astype('category')
print(data.dtypes)

x =  data.iloc[:, 0:-1]
y = data.iloc[:, [-1]] 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1, random_state=0)



# ADABOOST CLASSIFIER
from sklearn.ensemble import AdaBoostClassifier
def adaboost_classifier(X_train, y_train, X_test, y_test, n_estimators=50, learning_rate=1, base_estimator_depth=1):
    # Create adaboost classifer object
    model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=base_estimator_depth), n_estimators=n_estimators, learning_rate=learning_rate, algorithm='SAMME')
    # Train Adaboost Classifer
    model.fit(X_train, y_train.values.ravel())
    #Predict the response for test dataset
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return y_pred, acc, model

n_estimators = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
learning_rate = [.0001, .0005, .001, .005, .01, .05, .1, .5, .9, 1]
base_estimator_depth = list(range(1, 11))

results = np.zeros((3, 10))

for i in range(30):
    print(i+1,"/",30)
    if int(i / 10) == 0:
        _, acc, _ = adaboost_classifier(x_train, y_train, x_test, y_test, n_estimators=n_estimators[i%10])
        results[0, i%10] = acc
    elif int(i / 10) == 1:
        _, acc, _ = adaboost_classifier(x_train, y_train, x_test, y_test, learning_rate=learning_rate[i%10])
        results[1, i%10] = acc
    else:
        _, acc, _ = adaboost_classifier(x_train, y_train, x_test, y_test, base_estimator_depth=base_estimator_depth[i%10])
        results[2, i%10] = acc

bar_plot(results[0, :], "n_estimators", n_estimators)
bar_plot(results[1, :], "learning_rate", learning_rate)
bar_plot(results[2, :], "base_estimator_depth", base_estimator_depth)


#--------------------------- COMBINING BEST PARAMETERS -------------------------------#

best_params = results.argmax(axis=1)
best_params = {"n_estimators": best_params[0], "learning_rate": best_params[1], "base_estimator_depth": best_params[2]}

accs = []
combNames = []
models = []

# n_estimators and learning_rate
_, acc, model = adaboost_classifier(x_train, y_train, x_test, y_test, 
                                         n_estimators=n_estimators[best_params["n_estimators"]], 
                                         learning_rate=learning_rate[best_params["learning_rate"]])

accs.append(acc)
combNames.append("n_estimators and learning_rate combination")
models.append(model)

# n_estimators and base_estimator_depth
_, acc, model = adaboost_classifier(x_train, y_train, x_test, y_test, 
                                         n_estimators=n_estimators[best_params["n_estimators"]], 
                                         base_estimator_depth=base_estimator_depth[best_params["base_estimator_depth"]])

accs.append(acc)
combNames.append("n_estimators and base_estimator_depth combination")
models.append(model)

# learning_rate and base_estimator_depth
_, acc, model = adaboost_classifier(x_train, y_train, x_test, y_test, 
                                         learning_rate=learning_rate[best_params["learning_rate"]],
                                         base_estimator_depth=base_estimator_depth[best_params["base_estimator_depth"]])

accs.append(acc)
combNames.append("learning_rate and base_estimator_depth combination")
models.append(model)

# all of them
_, acc, model = adaboost_classifier(x_train, y_train, x_test, y_test, 
                                         n_estimators=n_estimators[best_params["n_estimators"]], 
                                         learning_rate=learning_rate[best_params["learning_rate"]],
                                         base_estimator_depth=base_estimator_depth[best_params["base_estimator_depth"]])

accs.append(acc)
combNames.append("combination of all three parameters")
models.append(model)

bar_plot(accs, "Combination Results", combNames)

#--------------------------- PLOTTING RESULT -------------------------------#

best_model = models[np.argmax(accs)]

import random
num_true_preds = 0
for _ in range(10):
    row = data.iloc[random.randint(0, len(data)-1)]
    x = np.reshape(row.iloc[:-1].to_numpy(), (1, -1))
    y = row.iloc[-1]
    y_pred = best_model.predict(x)
    if y_pred == y:
        num_true_preds += 1
    print("\n\nRow:", row)
    print("Prediction:", y_pred[0], "Y:", y)
print("Num true preds:", num_true_preds)
    


