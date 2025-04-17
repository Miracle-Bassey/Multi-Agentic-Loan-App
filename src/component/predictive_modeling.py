from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
# predictive_modeling.py


def train_predictive_model(data: pd.DataFrame, target_column: str) -> RandomForestClassifier:
    """
    Trains a RandomForestClassifier on the provided dataset.

    Parameters:
    - data: pd.DataFrame - The dataset containing features and target.
    - target_column: str - The name of the target column in the dataset.

    Returns:
    - model: RandomForestClassifier - The trained RandomForest model.
    """
    # Split the data into features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    return model
