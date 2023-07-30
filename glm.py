import numpy as np
import pandas as pd
import statsmodels.api as sm
from openpyxl.utils.dataframe import dataframe_to_rows
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

def logit_p1value(model, x):

    p1 = model.predict_proba(x)
    n1 = len(p1)
    m1 = len(model.coef_[0]) + 1
    coefs = np.concatenate([model.intercept_, model.coef_[0]])
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
    answ = np.zeros((m1, m1))
    for i in range(n1):
        answ = answ + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p1[i,1] * p1[i, 0]
    vcov = np.linalg.inv(np.matrix(answ))
    se = np.sqrt(np.diag(vcov))
    t1 =  coefs/se
    p1 = (1 - norm.cdf(abs(t1))) * 2
    return p1




def logit_p1value(model, x):
    p1 = model.predict_proba(x)
    n1 = len(p1)
    m1 = len(model.coef_[0]) + 1
    coefs = np.concatenate([model.intercept_, model.coef_[0]])
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis=1))
    answ = np.zeros((m1, m1))
    for i in range(n1):
        answ = answ + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p1[i, 1] * p1[i, 0]
    vcov = np.linalg.inv(np.matrix(answ))
    se = np.sqrt(np.diag(vcov))
    t1 = coefs / se
    p1 = (1 - norm.cdf(abs(t1))) * 2
    return p1

def print_model(model, x, y, sheet, workbook, file):
    x_with_intercept = sm.add_constant(x)

    results = model.fit(x_with_intercept, y)
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]

    # Create an empty DataFrame to store the summary table
    summary_df = pd.DataFrame(
        columns=['Variable', 'Coefficient', 'P-value', 'Std. Err.'])

    # Add rows to the summary table for each variable coefficient
    for i in range(1, len(x_with_intercept.columns)):
        variable_name = x_with_intercept.columns[i]
        coefficient = coefficients[i]
        p_value = logit_p1value(model, i)
        std_err = results.bse[i]
        summary_df.loc[i - 1] = [variable_name, coefficient, p_value, std_err]

        # Manually create a row for the Intercept coefficient
    p_value_intercept = results.pvalues[0]
    std_err_intercept = results.bse[0]
    summary_df.loc[len(x_with_intercept.columns) - 1] = ['Intercept', intercept, p_value_intercept,
                                                         std_err_intercept]

    # Write the summary table to the Excel sheet
    for row in dataframe_to_rows(summary_df, index=False, header=True):
        sheet.append(row)

    workbook.save(file)


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
