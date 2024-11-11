# Importing the libraries for training the model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

# Function to train the model depending on user choice
def train_model(X_train, y_train, model_choice):
    if model_choice == '1':
        model = RandomForestClassifier(n_estimators=100, random_state=0)
    elif model_choice == '2':
        model = LogisticRegression(random_state=0)
    elif model_choice == '3':
        model = SVC(random_state=0)
    elif model_choice == '4':
        model = GaussianNB()
    elif model_choice == '5':
        model = AdaBoostClassifier(random_state=0)
    else:
        raise ValueError("Invalid model choice. Please choose 1, 2, 3, 4, 5")
    
    model.fit(X_train, y_train)
    return model
