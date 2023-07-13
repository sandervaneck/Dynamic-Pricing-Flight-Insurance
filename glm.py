import pandas as pd
import statsmodels.api as sm
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

def model_glm(train_X, train_Y, test_X, workbook, file):
    sheet = workbook.create_sheet('GLM')

    Y = train_Y
    X = sm.add_constant(train_X)  # Add a constant term to the design matrix
    X_test = sm.add_constant(test_X)  # Add a constant term to the design matrix
    glm_model = sm.GLM(Y, X, family=sm.families.Binomial())
    glm_results = glm_model.fit()

    summary_df = pd.DataFrame(glm_results.summary().tables[1].data[1:], columns=glm_results.summary().tables[1].data[0])

    # Write the summary table to the Excel sheet
    for row in dataframe_to_rows(summary_df, index=False, header=True):
        sheet.append(row)

    # Predict the values of 'refund' using the trained model
    predictions = glm_results.predict(X_test)
    workbook.save(file)
    return predictions