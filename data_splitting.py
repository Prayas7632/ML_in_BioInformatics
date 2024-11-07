#importing libraries for splitting dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from normalization import normalize_data
from feature_selection import feature_selection

# function to split the dataset into train, test and blind test
def split_dataset(data, Y, test_size, random_state):
    

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=test_size, random_state=random_state)

    # Split the test set into test and blind test sets
    X_test, X_blind_test, y_test, y_blind_test = train_test_split(X_test, y_test, test_size=0.5, random_state=random_state)

    return X_train, X_test, X_blind_test, y_train, y_test, y_blind_test

