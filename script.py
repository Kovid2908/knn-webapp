import pandas as pd
import numpy as np

rows = 500  # change to 1000 if you want more

np.random.seed(42)

data = {
    "ApplicantIncome": np.random.randint(15000, 150000, rows),
    "CoapplicantIncome": np.random.randint(0, 60000, rows),
    "LoanAmount": np.random.randint(50, 500, rows),
    "Loan_Amount_Term": np.random.choice([120, 180, 240, 300, 360], rows),
    "Credit_History": np.random.choice([1, 0], rows, p=[0.8, 0.2]),
    "Employment_Type": np.random.choice([1, 0], rows, p=[0.7, 0.3]),
    "Age": np.random.randint(21, 60, rows),
    "Dependents": np.random.randint(0, 4, rows),
    "Existing_Loans": np.random.randint(0, 4, rows),
    "Assets_Value": np.random.randint(50000, 1000000, rows),
    "Savings": np.random.randint(5000, 300000, rows),
    "Education": np.random.choice([1, 0], rows, p=[0.7, 0.3]),
    "Marital_Status": np.random.choice([1, 0], rows),
    "Residential_Area": np.random.choice([0, 1, 2], rows)
}

df = pd.DataFrame(data)

# Define loan approval condition
df["Loan_Status"] = (
    (df["ApplicantIncome"] + df["CoapplicantIncome"] > df["LoanAmount"] * 150) &
    (df["Credit_History"] == 1) &
    (df["Age"] > 23) &
    (df["Savings"] > 20000)
).astype(int)

df.to_csv("loan_data.csv", index=False)

print("Dataset generated successfully!")
# This script generates a synthetic dataset for loan approval prediction and saves it as loan_data.csv.