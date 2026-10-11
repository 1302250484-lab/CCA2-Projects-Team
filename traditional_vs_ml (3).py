# Name: Soumi Banerjee
# PRN: 1302250228
# Subject: AI ML 1 - CCA3
# Program: Traditional Programming vs Machine Learning (Loan Approval)
#
# INPUT file  : loan_applicants_input.xlsx  
#   Sheet1         -> Old data (Income, CreditScore, Approved) 
#   New_Applicants -> New applicants (Income, CreditScore) 
#
# OUTPUT file : loan_applicants_output.xlsx (program apne aap banata hai)
#   Output          -> Sheet1  Traditional vs ML results
#   New_Predictions -> New_Applicants Traditional vs ML results
#
# NOTE: Output excel file must be closed while running the program.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

INPUT_FILE = "loan_applicants_input.xlsx"
OUTPUT_FILE = "loan_applicants_output.xlsx"

# Traditional rule limits 
MIN_INCOME = 50000
MIN_CREDIT_SCORE = 650


data = pd.read_excel(INPUT_FILE, sheet_name="Sheet1")
new_data = pd.read_excel(INPUT_FILE, sheet_name="New_Applicants")

# ---------- Traditional Programming  ----------
def traditional_approval(income, credit_score):
    if income >= MIN_INCOME and credit_score >= MIN_CREDIT_SCORE:
        return 1
    return 0

def apply_traditional(df):
    return df.apply(
        lambda row: traditional_approval(row["Income"], row["CreditScore"]), axis=1
    )

# ---------- Machine Learning  ----------
X = data[["Income", "CreditScore"]]
y = data["Approved"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# ---------- Results ----------
output = data[["Income", "CreditScore", "Approved"]].copy()
output["Traditional_Result"] = apply_traditional(output)
output["ML_Result"] = model.predict(output[["Income", "CreditScore"]])

predictions = new_data[["Income", "CreditScore"]].copy()
predictions["Traditional_Result"] = apply_traditional(predictions)
predictions["ML_Result"] = model.predict(predictions[["Income", "CreditScore"]])

print("Results (Traditional vs ML):")
print(output)
print("\nNew Applicants prediction:")
print(predictions)

# ---------- Output ----------
try:
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        output.to_excel(writer, sheet_name="Output", index=False)
        predictions.to_excel(writer, sheet_name="New_Predictions", index=False)
    print("\nOutput saved:", OUTPUT_FILE)
except PermissionError:
    print("\nERROR: Output Excel file opened. ")
