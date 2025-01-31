import pandas as pd

# Load csv file
df = pd.read_csv("/datasets/heart_failure_data_complete.csv")  
data = df.to_numpy()

# Print number of samples
print('Number of samples: ',data.shape[0])

# print
df 
----------
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Create and standardise the feature matrix, create label vector
X = StandardScaler().fit_transform(data[:,[1,3]])
y = data[:,0]
print('Features dims: {}  Labels dims: {}'.format(X.shape, y.shape))

# Plot the data
def plotData(X,y):
    plt.plot(X[y==0,0],X[y==0,1],'bo', label = 'Healthy')
    plt.plot(X[y==1,0],X[y==1,1],'rd', label = 'Mild HF')
    plt.plot(X[y==2,0],X[y==2,1],'g^', label = 'Severe HF')
    plt.legend()
    plt.title('Heart failure data')
    plt.xlabel('EF')
    plt.ylabel('QRS')
    
plotData(X,y)
------------
from sklearn.model_selection import train_test_split

# train test split
# Create feature matrix and label vector
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print('Training set size:', X_train.shape)
print('Test set size:', X_test.shape)

# plot 
def PlotData(X, y, title):
    # Plot class 0
    plt.plot(X[y == 0, 0], X[y == 0, 1], 'bo', alpha=0.75, markeredgecolor='k', label='Healthy')
    # Plot class 1
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'rd', alpha=0.75, markeredgecolor='k', label='Mild HF')
    # Plot class 2
    plt.plot(X[y == 2, 0], X[y == 2, 1], 'g^', alpha=0.75, markeredgecolor='k', label='Severe HF')
    
    plt.title(title)
    plt.xlabel('EF')
    plt.ylabel('QRS')
    plt.legend()

# Create subplots for training and test sets
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Training set plot
plt.subplot(1, 2, 1)
PlotData(X_train, y_train, 'Training Set')

# Test set plot
plt.subplot(1, 2, 2)
PlotData(X_test, y_test, 'Test Set')
--------
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create model
model = Pipeline([
    ('poly', PolynomialFeatures()),
    ('log_reg', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000))
])

# Define parameter grid
param = {
    'poly__degree': [1, 2, 3], 
    'log_reg__C': [10**-3, 10**-2, 10**-1, 1, 10, 100, 1000] 
}

# Perform grid search
g = GridSearchCV(model, param, cv=5, scoring='accuracy')
g.fit(X_train, y_train)
print('Best cross-validated accuracy:', round(g.best_score_,2))

# print hyperparameters
print('Selected parameter 1 (degree):', g.best_params_['poly__degree']) 
print('Selected parameter 2 (C):', g.best_params_['log_reg__C']) 
-------
def plotDecisionBoundary(model,X,y):
    # Create an array that represents the sampled feature space
    xx = np.linspace(-3, 3, 500) 
    yy = np.linspace(-3, 3.5, 500).T 
    xx, yy = np.meshgrid(xx, yy) 
    Feature_space = np.c_[xx.ravel(), yy.ravel()] 
    
    # predict labels
    y_pred = model.predict(Feature_space).reshape(xx.shape)

    # plot predictions
    plt.contourf(xx,yy,y_pred, cmap = 'summer')
    plotData(X, y)

    plt.xlabel('EF')
    plt.ylabel('QRS')
    plt.title('Decision Boundary and Training Data')
    plt.show()

# plot the fitted model
plotDecisionBoundary(g.best_estimator_, X_train, y_train)
--------
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.model_selection import cross_val_predict

# Calculate recalls for each class
y_pred = cross_val_predict(g.best_estimator_,X_train,y_train,cv = 5)
recalls = recall_score(y_train, y_pred, average=None)
print('Recalls: ', np.around(recalls,3))

# Calculate average recall over all classes
mean_recall_macro = recall_score(y_train, y_pred, average='macro')
print('Average recall: ', np.around(mean_recall_macro,3))

# calculate confusion matrix 
cm = confusion_matrix(y_train, y_pred)

# print confusion matrix
print('Confusion Matrix:\n', cm)

# plot confusion matrix 
plt.matshow(cm, cmap='gray')
plt.title('Confusion Matrix for Multinomial Logistic Regression')
plt.xlabel('Predicted labels', fontsize = 16)
plt.ylabel('True labels', fontsize = 16)
------
# Create and train a new improved model (Logistic Regression with class weights balanced and a penalty)
model2 = Pipeline([
    ('poly', PolynomialFeatures()),  
    ('log_reg', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, class_weight='balanced', penalty='l2'))
])


# Define parameter grid for tuning

param_grid = {
    'poly__degree': [1, 2, 3],                  
    'log_reg__C': [10**-3, 10**-2, 10**-1, 1, 10, 100, 1000]  
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(model2, param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

print('Best cross-validated accuracy:', round(grid_search.best_score_,2)) 

# Calculate recalls for each class
y_pred = cross_val_predict(grid_search.best_estimator_, X_train, y_train, cv=5)
recalls_tuned = recall_score(y_train, y_pred, average=None)
print('Recalls: ', np.around(recalls_tuned,2))

# Calculate average recall
mean_recall_macro_tuned = recall_score(y_train, y_pred, average='macro')
print('Average recall: ', np.around(mean_recall_macro_tuned,2))

# Plot the classifier 
plotDecisionBoundary(grid_search.best_estimator_, X_train,y_train) 
----------
# Amended classification
from sklearn.model_selection import cross_val_score
y_test_pred = grid_search.best_estimator_.predict(X_test)
# accuracy
cross_val_accuracy = cross_val_score(grid_search.best_estimator_, X_test, y_test, cv=2, scoring='accuracy')
print('Accuracy: ', round(cross_val_accuracy.mean(),2))

# recalls for all classes
recalls = recall_score(y_test, y_test_pred, average=None)
print('Recalls: ', np.around(recalls,2))

# average recall over all classes
average_recall = recall_score(y_test,y_test_pred,average='macro')
print('Average recall: ', np.around(average_recall,2))

# plot decision boundary
plotDecisionBoundary(grid_search.best_estimator_, X_test, y_test)
