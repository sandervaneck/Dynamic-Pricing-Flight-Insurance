from math import sqrt
from openpyxl.utils.dataframe import dataframe_to_rows
import pandas as pd
from sklearn.metrics import mean_squared_error

def write_model_evaluations(test_Y, models, predictions, workbook, file):
    sheet = workbook.create_sheet('Model Evaluations')

    rmse_scores = []
    for model, prediction in zip(models, predictions):
        rmse = sqrt(mean_squared_error(test_Y, prediction))
        rmse_scores.append(rmse)

    # Write the table of RMSE scores to the sheet
    rmse_table = {'Model': models, 'RMSE': rmse_scores}
    df_rmse = pd.DataFrame(rmse_table)

    sheet.append(['Model', 'RMSE'])
    for row in dataframe_to_rows(df_rmse, index=False, header=True):
        sheet.append(row)

    # Save the Excel workbook
    workbook.save(file)