import numpy as np
import pandas as pd
import statsmodels.api as sm
from openpyxl.utils.dataframe import dataframe_to_rows
from sklearn.metrics import mean_squared_error


def model_glm(train_X, train_Y, test_X, test_Y, workbook, file):
    sheet = workbook.create_sheet('GLM')
    # y = train_Y
    x = sm.add_constant(train_X)  # Add a constant term to the design matrix
    x_test = sm.add_constant(test_X)  # Add a constant term to the design matrix
    glm_model = sm.GLM(train_Y, x, family=sm.families.Binomial())
    glm_results = glm_model.fit()

    summary_df = pd.DataFrame(glm_results.summary().tables[1].data[1:], columns=glm_results.summary().tables[1].data[0])

    # Write the summary table to the Excel sheet
    for row in dataframe_to_rows(summary_df, index=False, header=True):
        sheet.append(row)

    # Predict the values of 'refund' using the trained model
    predictions = glm_results.predict(x_test)

    # Calculate the RMSE
    rmse = np.sqrt(mean_squared_error(test_Y, predictions))

    # Calculate the adjusted R-squared
    n = len(test_Y)
    p = train_X.shape[1]
    r_squared = glm_results.deviance / glm_results.null_deviance
    adj_r_squared = 1 - (1 - r_squared) * ((n - 1) / (n - p - 1))

    # Create a new DataFrame for RMSE and adjusted R-squared
    extra_rows = pd.DataFrame({'Variable': ['RMSE', 'Adjusted R-squared'],
                               'Coefficient': [rmse, adj_r_squared]})

    # Concatenate the new DataFrame with the original summary_df
    summary_df = pd.concat([summary_df, extra_rows], ignore_index=True)

    # Write the updated summary table to the Excel sheet
    for row in dataframe_to_rows(summary_df, index=False, header=True):
        sheet.append(row)

    workbook.save(file)
    return predictions
