import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

def pca_feature_selection(data, n_components):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data



def feature_selection(data, target, selection_choice, **kwargs):
    if selection_choice == '1':
        return pca_feature_selection(data, **kwargs)
    else:
        raise ValueError("Invalid feature selection technique choice. Please choose 1")
