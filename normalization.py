import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer

def min_max_scaling(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def z_score_standardization(data):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    return standardized_data

def robust_scaling(data):
    scaler = RobustScaler()
    robust_scaled_data = scaler.fit_transform(data)
    return robust_scaled_data

def quantile_transformation(data):
    scaler = QuantileTransformer(output_distribution='uniform')
    quantile_transformed_data = scaler.fit_transform(data)
    return quantile_transformed_data

def normalize_data(data, normalization_choice):
    if normalization_choice == '1':
        return min_max_scaling(data)
    elif normalization_choice == '2':
        return z_score_standardization(data)
    elif normalization_choice == '3':
        return robust_scaling(data)
    elif normalization_choice == '4':
        return quantile_transformation(data)
    else:
        raise ValueError("Invalid normalization technique choice. Please choose 1, 2, 3, 4")
