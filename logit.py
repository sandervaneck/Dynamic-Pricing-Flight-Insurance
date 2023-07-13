import pandas as pd
import statsmodels.api as sm
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import matplotlib.pyplot as plt

def model_logit(train_X, train_Y, test_X, workbook, file):
    sheet = workbook.create_sheet('LOGIT')

    Y = train_Y.astype(int)
    X = sm.add_constant(train_X)  # Add a constant term to the design matrix
    X_test_set = sm.add_constant(test_X)
    # Create the logit model (logistic regression)
    logit_model = sm.Logit(Y, X)

    # Fit the logit model to the data
    logit_results = logit_model.fit()

    # Print the summary of the model
    summary_df = pd.DataFrame(logit_results.summary().tables[1].data[1:],
                              columns=logit_results.summary().tables[1].data[0])

    # Write the summary table to the Excel sheet
    for row in dataframe_to_rows(summary_df, index=False, header=True):
        sheet.append(row)

    # Predict the values of 'refund' using the trained model
    probabilities = logit_results.predict(X_test_set)
    workbook.save(file)

    return probabilities