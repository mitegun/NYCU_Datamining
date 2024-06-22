import csv
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

def normalize(df):
    """Normalize the dataframe using MinMaxScaler."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(df)

def calculate_distances(x_train, x_test, k, metric):
    """
    Calculate the k-nearest neighbors distances using the specified metric.
    
    Args:
        x_train (array-like): Training data.
        x_test (array-like): Test data.
        k (int): Number of neighbors to consider.
        metric (str): Distance metric to use.
        
    Returns:
        distances (array): Distances to the k-nearest neighbors.
        indices (array): Indices of the k-nearest neighbors.
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric=metric)  # Initialize NearestNeighbors with given metric
    nbrs.fit(x_train)  # Fit the model using training data
    distances, indices = nbrs.kneighbors(x_test)  # Compute the k-nearest neighbors for test data
    return distances, indices

def evaluate_metrics(x_train, x_test, k):
    """
    Evaluate different distance metrics and select the best one based on silhouette score.
    
    Args:
        x_train (array-like): Training data.
        x_test (array-like): Test data.
        k (int): Number of neighbors to consider.
        
    Returns:
        best_metric (str): The metric that resulted in the best performance.
        best_distances (array): Distances calculated using the best metric.
    """
    metrics = ['manhattan', 'euclidean', 'minkowski']  # List of metrics to evaluate
    best_score = -1  # Initialize the best score to a very low value
    best_metric = None  # Initialize the best metric as None
    best_distances = None  # Initialize the best distances as None
    for metric in metrics:  # Loop through each metric
        distances, indices = calculate_distances(x_train, x_test, k, metric)  # Calculate distances for current metric
        avg_distances = distances.mean(axis=1)  # Compute average distances
        score = silhouette_score(x_test, avg_distances)  # Compute silhouette score
        print(f'Metric: {metric}, Silhouette Score: {score}')  # Print the metric and corresponding silhouette score
        if score > best_score:  # Update the best score, metric, and distances if current score is better
            best_score = score
            best_metric = metric
            best_distances = avg_distances

    return best_metric, best_distances

def format_export_answers(answers, k_value, metric):
    """
    Format the answers and export them to a CSV file.
    
    Args:
        answers (array): Computed answers to export.
        k_value (int): The k value used in the k-nearest neighbors calculation.
        metric (str): The distance metric used.
    """
    now = datetime.now()  # Get the current date and time
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")  # Format date and time as string
    filename = f'answer_{dt_string}_{k_value}_{metric}.csv'  # Construct filename using date, k value, and metric
    fields = ["id", "outliers"]  # Define CSV field names
    rows = [[i, answers[i]] for i in range(len(answers))]  # Prepare data rows for CSV

    with open(filename, 'w', newline='') as f:  # Open file for writing
        writer = csv.writer(f)  # Create CSV writer
        writer.writerow(fields)  # Write field names
        writer.writerows(rows)  # Write data rows
    
def main():
    train_set = "training.csv"  # Define training dataset filename
    test_set = "test_X.csv"  # Define test dataset filename

    df_train = pd.read_csv(train_set)  # Load training dataset
    df_test = pd.read_csv(test_set)  # Load test dataset

    y_train = df_train.pop('lettr').values  # Separate labels from features in the training set

    df_combined = pd.concat([df_train, df_test])  # Concatenate train and test datasets for normalization
    normalized_values = normalize(df_combined.values)  # Normalize the combined dataset

    x_train = normalized_values[:len(y_train)]  # Split normalized values back into training set
    x_test = normalized_values[len(y_train):]  # Split normalized values back into test set

    k = 4  # Number of neighbors to consider for the computation of the average distance

    best_metric, avg_distances = evaluate_metrics(x_train, x_test, k)  # Evaluate metrics and select the best one

    format_export_answers(avg_distances, k, best_metric)  # Export the results using the best metric

if __name__ == '__main__':
    main()  # Run the main function
