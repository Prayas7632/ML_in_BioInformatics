import pandas as pd
from normalization import normalize_data
from feature_selection import feature_selection
from data_splitting import split_dataset
from model import train_model
from cross_validation import cross_validation
from cross_validation import calculate_mean_std
from cross_validation import print_scores
from cross_validation import print_confusion_matrix


def read_csv_from_terminal():
    csv_file = input("Enter the name of the CSV file : ")

    try:
        data = pd.read_csv(csv_file)

        # Assuming the id column is named "id", remove it
        if 'id' in data.columns:
            data = data.drop(columns=['id'])

        print("\nFirst few rows of the data:")
        print(data.head())

        return data

    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found. Please check the file name and try again.")

    except Exception as e:
        print(f"An error occurred: {e}")

def get_numeric_target_column(data):
    numeric_columns = data.select_dtypes(include=['number']).columns

    if len(numeric_columns) > 0:
        target_column = numeric_columns[0]
        print(f"\nAutomatically set target column to: {target_column}")
        return target_column
    else:
        print("No numeric columns found. Please ensure your dataset contains numeric columns for feature selection.")
        return None

if __name__ == "__main__":
    your_data = read_csv_from_terminal()

    if your_data is not None:
        numeric_columns = your_data.select_dtypes(include=['number']).columns
        features_only = your_data[numeric_columns]

        normalization_choice = input("\nChoose a normalization technique (1: Min-Max Scaling, 2: Z-score Standardization, 3: Robust Scaling, 4: Quantile Transformation): ")

        try:
            normalized_data = normalize_data(features_only, normalization_choice)

            normalized_data_with_non_numeric = pd.concat([your_data.drop(columns=numeric_columns), pd.DataFrame(normalized_data, columns=numeric_columns)], axis=1)

            print("\nNormalized Data:")
            print(normalized_data_with_non_numeric)
            
            # Removing diagnosis from the list of features
            Y = your_data['diagnosis']
            
            your_data = your_data.drop(columns=['diagnosis'], axis=1)
            

            selection_choice = input("\nChoose a feature selection technique (1: PCA): ")


            selected_features = feature_selection(normalized_data, your_data, selection_choice, **{'n_components': 5})
            
            print("\nSelected Features:")
            print(pd.DataFrame(selected_features))

            # Take necessary input for splitting the dataset
            test_size = float(input("\nEnter the percentage of data to be used for the test set (e.g., 0.2 for 20%): "))
            random_state = int(input("Enter a random seed for reproducibility: "))

            X_train, X_test, X_blind_test, y_train, y_test, y_blind_test = split_dataset(selected_features, Y, test_size=test_size, random_state=random_state)

            print("\nSplitting the dataset:")
            print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}, Blind Test set shape: {X_blind_test.shape}")
            
            # Take necessary input for training the model
            model_choice = input("\nChoose a model for training (1: RandomForest, 2: Logistic Regression, 3: SVM, 4: GaussianNB, 5: AdaBoost): ")
            
            # Train the model
            trained_model = train_model(X_train, y_train, model_choice)
            
            print("\nModel trained successfully!")
            
            # Take necessary input for cross-validation
            cv_choice = input("\nChoose a cross-validation technique (1: K-Fold, 2: Stratified K-Fold, 3: Leave One Out, 4: Monte Carlo): ")

            # Perform cross-validation
            cv_scores = cross_validation(trained_model, X_train, y_train, cv_choice)

            # Print cross-validation scores
            print("\nCross-Validation Results:")
            print_scores(cv_scores)

            # Calculate and print mean and standard deviation
            mean, std = calculate_mean_std(cv_scores)

            # Make predictions on the blind test set
            y_blind_test_pred = trained_model.predict(X_blind_test)

            # Print confusion matrix
            print("\nConfusion Matrix on Blind Test Set:")
            print_confusion_matrix(y_blind_test, y_blind_test_pred)

        except ValueError as e:
            print(e)
