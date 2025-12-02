# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib  # to save the model


def load_data():
    """
    Load the CSV file and return features X and target y.
    Uses ALL columns except Loan_Status as features.
    """
    # Make sure loan_data.csv is in the SAME folder as this script
    data = pd.read_csv("loan_data.csv")

    # Target (output label) - 0 or 1
    y = data["Loan_Status"]

    # Features = every column except Loan_Status
    X = data.drop(columns=["Loan_Status"])

    return X, y


def train_model(k=5):
    """
    Train a k-NN model (with StandardScaler) and save it as loan_knn_model.pkl
    """
    X, y = load_data()

    # Split data into training and testing (for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Create a pipeline: Scale features -> k-NN classifier
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ]
    )

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nModel trained with k = {k}")
    print(f"Accuracy on test data: {acc * 100:.2f}%")

    # Save the model to a file
    model_filename = "loan_knn_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}")


if __name__ == "__main__":
    # ---- Ask user for k at runtime ----
    try:
        user_k = int(input("Enter value of k (e.g., 3, 5, 7): "))
        if user_k <= 0:
            print("k must be a positive integer. Using default k = 5.")
            user_k = 5
    except ValueError:
        print("Invalid input. Using default k = 5.")
        user_k = 5

    train_model(k=user_k)
# This script trains a k-NN model for loan approval prediction and saves it as loan_knn_model.pkl.