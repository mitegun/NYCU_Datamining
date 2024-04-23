import numpy as np
import pandas as pd
# Define dictionary mapping feature indices to feature names
features = {
    0: 'AMB_TEMP', 1: 'CH4', 2: 'CO', 3: 'NMHC', 4: 'NO', 5: 'NO2', 6: 'NOx', 
    7: 'O3', 8: 'PM10', 9: 'PM2.5', 10: 'RAINFALL', 11: 'RH', 12: 'SO2', 
    13: 'THC', 14: 'WD_HR', 15: 'WIND_DIREC', 16: 'WIND_SPEED', 17: 'WS_HR'
}

# Create empty lists for each data
data_values = {
    'AMB_TEMP': np.array([]), 'CH4': np.array([]), 'CO': np.array([]), 'NMHC': np.array([]), 
    'NO': np.array([]), 'NO2': np.array([]), 'NOx': np.array([]), 'O3': np.array([]), 
    'PM10': np.array([]), 'PM2.5': np.array([]), 'RAINFALL': np.array([]), 'RH': np.array([]), 
    'SO2': np.array([]), 'THC': np.array([]), 'WD_HR': np.array([]), 'WIND_DIREC': np.array([]), 
    'WIND_SPEED': np.array([]), 'WS_HR': np.array([]), 'ERROR_AT': [[] for i in range(18)]
}

# Create empty lists for test data
test_data_values = {
    'AMB_TEMP': np.array([]), 'CH4': np.array([]), 'CO': np.array([]), 'NMHC': np.array([]), 
    'NO': np.array([]), 'NO2': np.array([]), 'NOx': np.array([]), 'O3': np.array([]), 
    'PM10': np.array([]), 'PM2.5': np.array([]), 'RAINFALL': np.array([]), 'RH': np.array([]), 
    'SO2': np.array([]), 'THC': np.array([]), 'WD_HR': np.array([]), 'WIND_DIREC': np.array([]), 
    'WIND_SPEED': np.array([]), 'WS_HR': np.array([]), 'ERROR_AT': [[] for i in range(18)]
}

# Define invalid values
invalid_value = ['#', '*', 'x', 'A']

def preprocess_data_train():
    # Load training data
    train_data = pd.read_csv('train.csv')
    # Iterate through DataFrame rows
    for index, row in train_data.iterrows():
        row_values = row.values[3:]
        row_label = str(row.values[2].replace(" ", ""))
        data_values[row_label] = np.append(data_values[row_label], row_values)
    # Retrieve the indices of invalid values in 'ERROR_AT'
    key_count = 0
    for key in data_values.keys():
        if key != 'ERROR_AT':
            for index, value in enumerate(data_values[key]):
                data_values[key][index] = str(data_values[key][index].replace(" ", ""))
                if data_values[key][index] in invalid_value:
                    data_values['ERROR_AT'][key_count].append(index)
        key_count += 1


def preprocess_data_test():
    # Load test data
    test_data = pd.read_csv('test.csv')
    # Iterate through DataFrame rows
    for index, row in test_data.iterrows():
        row_values = row.values[2:]
        row_label = str(row.values[1].replace(" ", ""))
        test_data_values[row_label] = np.append(test_data_values[row_label], row_values)
    # Retrieve the indices of invalid values in 'ERROR_AT'
    key_count = 0
    for key in test_data_values.keys():
        if key != 'ERROR_AT':
            for index, value in enumerate(test_data_values[key]):
                test_data_values[key][index] = str(test_data_values[key][index].replace(" ", ""))
                if test_data_values[key][index] in invalid_value:
                    test_data_values['ERROR_AT'][key_count].append(index)
        key_count += 1


# Interpolate missing features
def interpolate_invalid(dataset, feature_to_interpolate, degree=1):   
    for _ in feature_to_interpolate:
        dataset[features[_]] = np.where(
            np.isin(dataset[features[_]], ['x', 'A', '*', '#']), np.nan, dataset[features[_]])
        dataset[features[_]] = dataset[features[_]].astype(float)
        df = pd.DataFrame(dataset[features[_]])
        df_interpolated = df.interpolate(method='polynomial', order=degree, limit=None)       
        k = 0
        for __ in df_interpolated.values:
            dataset[features[_]][k] = __[0]
            k += 1       
        for i in range(len(dataset[features[_]])):
            if np.isnan(dataset[features[_]][i]):
                prev_index = i - 1
                next_index = i + 1
                while prev_index >= 0 and np.isnan(dataset[features[_]][prev_index]):
                    prev_index -= 1
                while next_index < len(dataset[features[_]]) and np.isnan(dataset[features[_]][next_index]):
                    next_index += 1
                if prev_index >= 0 and next_index < len(dataset[features[_]]):
                    dataset[features[_]][i] = (dataset[features[_]][prev_index] + dataset[features[_]][next_index]) / 2
                elif prev_index >= 0:
                    dataset[features[_]][i] = dataset[features[_]][prev_index]
                elif next_index < len(dataset[features[_]]):
                    dataset[features[_]][i] = dataset[features[_]][next_index]
                else:
                    dataset[features[_]][i] = 0
    return


# Normalize features in the dataset and delete outliers
def normalization(dataset, feature_to_normalize):
    for _ in feature_to_normalize:
        mean = np.mean(dataset[features[_]])
        deviation = np.std(dataset[features[_]])
        _score = (dataset[features[_]] - mean) / deviation
        outliers = np.abs(_score) > 10
        mean_non_outliers = np.mean(dataset[features[_]][~outliers])
        dataset[features[_]][outliers] = mean_non_outliers
    return

# Process selected data from the dataset
def processed_data(data_set, data_selected):
    processed_data = np.array([data_set[features[data_selected[0]]]])
    for _ in data_selected[1:]:
        processed_data = np.append(processed_data, [data_set[features[_]]], axis=0)
    processed_data = processed_data.T
    return processed_data

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None   
    def fit(self, X, y):
        n, p = X.shape
        I = np.identity(p)
        self.coef_ = np.linalg.inv(X.T @ X + self.alpha * I) @ X.T @ y   
    def predict(self, X):
        return X @ self.coef_

# Create sliding window for training data
def window_sliding_train(data, target_column_index, window_size):
    X = []
    y = []
    num_features = len(data)  # Number of explanatory variables
    for _ in range(12):
        for i in range(window_size + 480 * _, 480 * (_ + 1)):
            window_X = []
            for j in range(num_features):
                window_X.append(data[j][i - window_size:i])
            X.append(window_X)
            y.append(data[target_column_index][i])
    X = np.array(X)
    y = np.array(y)
    return X, y

# Create sliding window for testing data
def window_sliding_test(data, target_column_index, window_size):
    X = []
    y = []
    num_features = len(data)
    num_windows = len(data[0]) // window_size
    end_index = window_size - 1
    start_index = 0
    for i in range(num_windows):
        window_X = []
        for j in range(num_features):
            start_index = i * window_size
            end_index = (i + 1) * window_size
            window_X.append(data[j][start_index:end_index])
        X.append(window_X)
        y.append(data[target_column_index][start_index])
    X = np.array(X)
    y = np.array(y)
    return X, y

# Perform ridge regression
def ridge_regression():
    index_PM25 = chosen_features.index(9)  # Index of PM2.5 feature       
    # Prepare training data
    X_train = processed_data(data_values, chosen_features)
    X_train, Y_train = window_sliding_train(X_train.T, index_PM25, 9)
    X_train = X_train.reshape(X_train.shape[0], -1)    
    # Prepare testing data
    X_test = processed_data(test_data_values, chosen_features) 
    X_test, Y_test = window_sliding_test(X_test.T, index_PM25, 9)
    X_test = X_test.reshape(X_test.shape[0], -1)    
    # Perform ridge regression
    ridge_model = RidgeRegression(alpha=0.5)
    ridge_model.fit(X_train, Y_train)    
    # Make predictions
    predictions = ridge_model.predict(X_test)    
    # Format results
    results = [(f'index_{i}', prediction) for i, prediction in enumerate(predictions)]    
    # Write results to CSV file
    with open('predictions.csv', 'w') as f:
        f.write("index,answer\n")  # Write header
        for index, prediction in results:
            f.write(f"{index},{prediction}\n")  # Write predictions

""" 
# Function to Compare the impact of different amounts of training data on the PM2.5 prediction accuracy.  
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def ridge_regression():
    rmse_values = []
    for i in range(1, 12):
        index_PM25 = chosen_features.index(9)       
        X_train = processed_data(data_values, chosen_features)
        X_train, Y_train = window_sliding_train(X_train.T, index_PM25, 9, i)
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = processed_data(test_data_values, chosen_features) 
        X_test, Y_test = window_sliding_test(X_test.T, index_PM25, 9)
        X_test = X_test.reshape(X_test.shape[0], -1)
        ridge_model = RidgeRegression(alpha=0.5)
        ridge_model.fit(X_train, Y_train)
        predictions = ridge_model.predict(X_test)
        rmse = calculate_rmse(predictions, Y_test)
        rmse_values.append(rmse)
        print(f"RMSE for i={i}: {rmse}")

    # Plotting the RMSE values
    plt.plot(range(1, 12), rmse_values, marker='o')
    plt.title('RMSE vs. number of months')
    plt.xlabel('number of months')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.show() 
""" 
chosen_features = [0,2,4,7,8,9,10,11,12,15,16]
def main():
    preprocess_data_train()
    preprocess_data_test() 
    interpolate_invalid(data_values, chosen_features)
    interpolate_invalid(test_data_values, chosen_features)
    normalization(data_values, chosen_features)
    normalization(test_data_values, chosen_features)
    ridge_regression()


if __name__ == "__main__":
    main()

