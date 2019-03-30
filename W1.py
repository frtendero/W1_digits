"""
1. Election of dataset: Digits dataset is going to be used.
This is a classification dataset where each data-point is a 8x8 image image of a digit.
"""

from sklearn.datasets import load_digits

digits = load_digits()

"""
2. Description of the dataset
"""
# An example of the data:
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(digits.images[12])
plt.show()

# data is a dictionary. We can then navigate through the dataset using the keys
digits.keys()
X = digits['data']
y = digits['target']

# (number of observations, number of features) Note that num features is 64 because it's a 8x8 image
print(X.shape)

# classes: (each one of the 10 class is a digit from 0 to 9
classes = digits['target_names']

# number of samples per class
for i in classes:
    print("Num of digit '" + str(i) + "' samples: " + str(sum(y == i)))


"""
EXPERIMENTS
"""

# the dataset is split so that 80% of the data is used for training, and 20% for test.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

# K Nearest Neighbors Classifier is going to be used
from sklearn.neighbors import KNeighborsClassifier
myKNN = KNeighborsClassifier()

# required imports for performing grid search cross validation and stratified k fold technique
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# A param grid dictionary is created with the parameters to try in the cross-validation process
# uniform weights: all points in each neighborhood have same weight
# distance weights: points closer to the evaluated will have more influence.
param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15], 'weights': ['uniform', 'distance']}

# KNN estimator, accuracy as scoring, and the data will be split into 10 chunks
myGSCV = GridSearchCV(estimator=myKNN, param_grid=param_grid, scoring='accuracy',
                      cv=StratifiedKFold(n_splits=10, random_state=3))

# Training of the model
myGSCV.fit(X_train, y_train)

# prediction (using best_estimator_ by default)
y_pred = myGSCV.predict(X_test)

# Results
print("\nBest Estimator:\n" + str(myGSCV.best_estimator_)) # best estimator
print("\nParameters of best estimator:\n" + str(myGSCV.best_params_)) # parameters of the best estimator
print("\nTraining score: " + str(myGSCV.best_score_))  # training score for achieved with the best estimator
print("Test score: " + str(myGSCV.score(X_test, y_test)))  # test score


"""
Using Leave one out validation
"""
from sklearn.model_selection import LeaveOneOut
myLOO = GridSearchCV(estimator=myKNN, param_grid=param_grid, scoring='accuracy', cv=LeaveOneOut())

# Training of the model
myLOO.fit(X_train, y_train)

# prediction (using best_estimator_ by default)
y_predLOO = myLOO.predict(X_test)

# Results
print("\nBest Estimator:\n" + str(myLOO.best_estimator_)) # best estimator
print("\nParameters of best estimator:\n" + str(myLOO.best_params_)) # parameters of the best estimator
print("\nTraining score: " + str(myLOO.best_score_))  # training score for achieved with the best estimator
print("Test score: " + str(myLOO.score(X_test, y_test)))  # test score


