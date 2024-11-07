#importing libraries for cross validation
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, LeaveOneOut

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# function to perform k_fold cross validation
def k_fold_cross_validation(model, X_train, y_train, cv):
    k_fold = KFold(n_splits=cv, shuffle=True, random_state=0)
    scores = cross_val_score(model, X_train, y_train, cv=k_fold, scoring='accuracy')
    return scores

# function to perform stratified k_fold cross validation
def stratified_k_fold_cross_validation(model, X_train, y_train, cv):
    stratified_k_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    scores = cross_val_score(model, X_train, y_train, cv=stratified_k_fold, scoring='accuracy')
    return scores

# function to perform leave one out cross validation
def leave_one_out_cross_validation(model, X_train, y_train):
    leave_one_out = LeaveOneOut()
    scores = cross_val_score(model, X_train, y_train, cv=leave_one_out, scoring='accuracy')
    return scores

#function to perform monte carlo cross validation
def monte_carlo_cross_validation(model, X_train, y_train, cv):
    scores = []
    for i in range(cv):
        scores.append(accuracy_score(y_train, model.fit(X_train, y_train).predict(X_train)))
    return scores

#function to perform cross validation depending on user choice
def cross_validation(model, X_train, y_train, cv):
    if cv == '1':
        return k_fold_cross_validation(model, X_train, y_train, cv=5)
    elif cv == '2':
        return stratified_k_fold_cross_validation(model, X_train, y_train, cv=5)
    elif cv == '3':
        return leave_one_out_cross_validation(model, X_train, y_train)
    elif cv == '4':
        return monte_carlo_cross_validation(model, X_train, y_train, cv=5)
    else:
        raise ValueError("Invalid cross validation choice. Please choose 1, 2, 3, 4")

# function to calculate the mean and standard deviation of the scores
def calculate_mean_std(scores):
    mean = scores.mean()
    std = scores.std()
    return mean, std

# function to print the scores
def print_scores(scores):
    print(f"Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean()}")
    print(f"Standard Deviation: {scores.std()}")
    
# function to print the confusion matrix
def print_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix: \n{cm}")
    sns.heatmap(cm, annot=True)
    plt.show()
    
