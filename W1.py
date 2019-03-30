#%%
"""
Workbook analyzing DIGITS dataset from scikit-learn using K Nearest Neighbours Classifier
"""

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

#%%
# data is a dictionary. We can then navigate through the dataset using the keys
digits.keys()
X = digits['data']
y = digits['target']

# (number of observations, number of features) Note that num features is 64 because it's a 8x8 image
print("(number of observations, number of features): "+ str(X.shape))

# classes: (each one of the 10 class is a digit from 0 to 9
classes = digits['target_names']

# number of samples per class
for i in classes:
    print("Num of digit '" + str(i) + "' samples: " + str(sum(y == i)))

#%%
"""
3. EXPERIMENTS
"""

# the dataset is split so that 80% of the data is used for training, and 20% for test.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

# K Nearest Neighbors Classifier is going to be used
from sklearn.neighbors import KNeighborsClassifier
myKNN = KNeighborsClassifier()

# required imports for performing grid search cross validation and stratified k fold technique.
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# A param grid dictionary is created with the parameters to try in the cross-validation process
param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15]}

# KNN estimator, accuracy as scoring, and the data will be split into 10 chunks, thus, it will take 10 iterations
myGSCV = GridSearchCV(estimator=myKNN, param_grid=param_grid, scoring='accuracy',
                      cv=StratifiedKFold(n_splits=10, random_state=3))

# Training of the model
myGSCV.fit(X_train, y_train)

# prediction (using best_estimator_ by default)
y_pred = myGSCV.predict(X_test)

# Results
print("\nBest Estimator:\n" + str(myGSCV.best_estimator_))  # best estimator
print("\nParameters of best estimator:\n" + str(myGSCV.best_params_))   # parameters of the best estimator
print("\nTraining score: " + str(myGSCV.best_score_))  # training score for achieved with the best estimator
print("Test score: " + str(myGSCV.score(X_test, y_test)))  # test score

#%%
"""
4. Using Leave one out validation.
Now, in each iteration of the cross validation process, all observations except one will be used for training.
This means that, in the first iteration, observation #1 is out, in second iteration, observation #2 is out, and so on.
Thus, there will be as many iterations as training samples (length of X_train), which means a high computational cost.
"""
from sklearn.model_selection import LeaveOneOut
myLOO = GridSearchCV(estimator=myKNN, param_grid=param_grid, scoring='accuracy', cv=LeaveOneOut(), verbose=2, n_jobs=-1)

# Training of the model
myLOO.fit(X_train, y_train)

# prediction (using best_estimator_ by default)
y_predLOO = myLOO.predict(X_test)

# Results
print("\nBest Estimator:\n" + str(myLOO.best_estimator_))   # best estimator
print("\nParameters of best estimator:\n" + str(myLOO.best_params_))    # parameters of the best estimator
print("\nTraining score: " + str(myLOO.best_score_))  # training score for achieved with the best estimator
print("Test score: " + str(myLOO.score(X_test, y_test)))  # test score

#%%
"""
5. Stratified K Fold.
There is a similar number of samples per each class, so, if this distribution is preserved when
splitting into training (X_train) and (X_test), stratification is not indispensable.
"""
print('Classes distribution in Training')
for i in classes:
    print("Num of digit '" + str(i) + "' samples: " + str(sum(y_train == i)))

print('\nClasses distribution in Test')
for i in classes:
    print("Num of digit '" + str(i) + "' samples: " + str(sum(y_test == i)))

# Let's try then with KFold (no Stratified)
# KNN estimator, accuracy as scoring, and the data will be split into 10 chunks, thus, it will take 10 iterations
from sklearn.model_selection import KFold

myGSCV_noStrat = GridSearchCV(estimator=myKNN, param_grid=param_grid, scoring='accuracy',
                              cv=KFold(n_splits=10, random_state=3))

# Training of the model
myGSCV_noStrat.fit(X_train, y_train)

# prediction (using best_estimator_ by default)
y_pred_noStrat = myGSCV_noStrat.predict(X_test)

# Results. It's shown how the score is not affected when not using Stratified KFOLD
print("\nBest Estimator:\n" + str(myGSCV_noStrat.best_estimator_))  # best estimator
print("\nParameters of best estimator:\n" + str(myGSCV_noStrat.best_params_))   # parameters of the best estimator
print("\nTraining score: " + str(myGSCV_noStrat.best_score_))  # training score for achieved with the best estimator
print("Test score: " + str(myGSCV_noStrat.score(X_test, y_test)))  # test score


#%%
"""
6. Distance Weights
# uniform weights (default): all points in each neighborhood have same weight (used in exercise 3)
# distance weights: points closer to the evaluated will have more influence.
"""
param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15], 'weights': ['uniform', 'distance']}
# KNN estimator, accuracy as scoring, and the data will be split into 10 chunks, thus, it will take 10 iterations
myGSCV_w = GridSearchCV(estimator=myKNN, param_grid=param_grid, scoring='accuracy',
                        cv=StratifiedKFold(n_splits=10, random_state=3))

# Training of the model
myGSCV_w.fit(X_train, y_train)

# prediction (using best_estimator_ by default)
y_pred_w = myGSCV_w.predict(X_test)

# Results. Note that distance weights are preferred
print("\nBest Estimator:\n" + str(myGSCV_w.best_estimator_))  # best estimator
print("\nParameters of best estimator:\n" + str(myGSCV_w.best_params_))   # parameters of the best estimator
print("\nTraining score: " + str(myGSCV_w.best_score_))  # training score for achieved with the best estimator
print("Test score: " + str(myGSCV_w.score(X_test, y_test)))  # test score

#%%
"""
7. Testing different metrics.
The default metric used is minkowski. Now euclidean and manhattan metrics will be taken into account, thus, they will
be included in the param grid.
Also, for minkowski, a parameter 'p' can be adjusted, so the different values will also be included.
"""
param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13],
              'weights': ['uniform', 'distance'],
              'metric': ['euclidean', 'manhattan', 'minkowski'],
              'p': [2, 3, 4, 5, 6, 7, 8, 9]}

myGSCV_m = GridSearchCV(estimator=myKNN, param_grid=param_grid, scoring='accuracy',
                        cv=StratifiedKFold(n_splits=10, random_state=3))

# Training of the model
myGSCV_m.fit(X_train, y_train)

# prediction (using best_estimator_ by default)
y_pred_m = myGSCV_m.predict(X_test)

# Results
print("\nBest Estimator:\n" + str(myGSCV_m.best_estimator_))  # best estimator
print("\nParameters of best estimator:\n" + str(myGSCV_m.best_params_))   # parameters of the best estimator
print("\nTraining score: " + str(myGSCV_m.best_score_))  # training score for achieved with the best estimator
print("Test score: " + str(myGSCV_m.score(X_test, y_test)))  # test score
