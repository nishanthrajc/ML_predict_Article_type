from data_gen import datagen
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


X, y = datagen()

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'degree': [2, 3, 4],
    'coef0': [0.0, 0.1, 0.5]
}
# Create SVM classifier
svc = svm.SVC()

# Perform Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(svc, param_grid, cv=5)
grid_search.fit(X, y)

# Print the best hyperparameters and corresponding accuracy
print("Best hyperparameters: ", grid_search.best_params_)
print("Best accuracy: ", grid_search.best_score_)

svc_best = svm.SVC(**grid_search.best_params_)

# Perform K-fold cross-validation (e.g., K=5)
k_fold = 5
scores = cross_val_score(svc_best, X, y, cv=k_fold)

# Print accuracy for each fold and the mean accuracy
print("Accuracy for each fold:", scores)
print("Mean accuracy:", scores.mean())

