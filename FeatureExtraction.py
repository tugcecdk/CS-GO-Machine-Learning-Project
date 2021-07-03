

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing


data = pd.read_csv("encoded_data.csv", sep=",")
data = data.fillna(value=0)

# Removing features with low variance
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
new_data = sel.fit_transform(data)

processed_data = pd.DataFrame(new_data, columns=data.columns)

def normalize_data(x, y):
    min_max_scaler = preprocessing.MinMaxScaler()

    x_values = x.values
    x_scaled = min_max_scaler.fit_transform(x_values)

    y_values = y.values
    y_values = np.reshape(y_values, (len(y_values), 1))
    y_scaled = min_max_scaler.fit_transform(y_values)

    x = pd.DataFrame(x_scaled, columns=x.columns)
    y = pd.DataFrame(y_scaled, columns=y.columns)

    return x, y


x =  processed_data.iloc[:, 0:-1]
y = processed_data.iloc[:, [-1]] 

# normalize values
x_norm, y = normalize_data(x.iloc[:, 5:], y)
x.iloc[:, 5:] = x_norm


# L1-based feature selection
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

lsvc = LinearSVC(C=0.1, penalty="l1", dual=False).fit(x, y)
model = SelectFromModel(lsvc, prefit=True)
filtered_data = pd.DataFrame(model.transform(x))


processed_data = pd.concat([filtered_data, y], axis=1)
processed_data.columns = data.columns[1:]

processed_data.to_csv("processed_data.csv", index=False)






