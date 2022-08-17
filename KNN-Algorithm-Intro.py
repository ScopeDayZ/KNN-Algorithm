"""
-- Hunter Fritchen
-- Started: 08/11/2022
-- Last Modified: 08/11/2022
-- Intro to the KNN Algorithm
-- From: https://towardsdatascience.com/knn-algorithm-what-when-why-how-41405c16c36f
"""

import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# Read in first five rows of diabetes.csv using .head()
data = pd.read_csv("../KNN Algorithm/diabetes.csv")
data.head()

# Creat non_zero which has all data needed for predicting our outcome value
non_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in non_zero:
    data[column] = data[column].replace(0, np.NaN)
    mean = int(data[column].mean(skipna = True))
    data[column] = data[column].replace(np.NaN, mean)
    #print(data[column])

p = sns.pairplot(data, hue = 'Outcome')

# For x we take all rows and columns from 0 to 7. For y we take all rows for 8th column
# test size at 0.2 implies 20% of all data will be kept aside for later use
x = data.iloc[:, 0:8]
y = data.iloc[:,8]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)

# Feature Scaling, this standardizes different value ranges for traning the ML model
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Get value of K, need odd value so add 1 or subtract 1
math.sqrt(len(y_test))

classifier = KNeighborsClassifier(n_neighbors=13, p=2, metric='euclidean')
classifier.fit(x_train, y_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
metric_params=None, n_jobs=None, n_neighbors=13, p=2, weights='uniform')

# Predict our data using classifier predict
y_pred = classifier.predict(x_test)

# Need to evaluate model to check accuracy, use confusion matrix
# Diagonal with 86 and 30 shows correct value  and 14, 24 shows the prediction it missed
cm = confusion_matrix(y_test, y_pred)

# F1 Score
#print(f1_score(y_test, y_pred))

# Accuracy Score
#print(accuracy_score(y_test, y_pred))

# Plot the graph for data vs predicted value
plt.figure(figsize=(5, 7))

ax = sns.histplot(data['Outcome'], kde=True, color='r', label="Actual Value")
sns.histplot(y_pred, kde=True, color='b', label="Predicted Values")

plt.title('Actual vs Predicted Values for Outcome')
plt.show(block=True)



