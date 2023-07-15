import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, classification_report


from sklearn.metrics import mean_squared_error

def write_model_evaluations(test_Y, models, predictions, workbook, file):
    sheet = workbook.create_sheet('Model Evaluations')

    rmse_scores = []
    classification_reps = []
    for model, prediction in zip(models, predictions):
        rmse = np.sqrt(mean_squared_error(test_Y, prediction))
        rmse_scores.append(rmse)
        classification_rep = classification_report(test_Y, prediction, output_dict=True)
        classification_reps.append(classification_rep)

    # Write the table of RMSE scores to the sheet

    # Write the table of classification scores to the sheet
    classification_metrics = ['precision', 'recall', 'f1-score']
    header = ['Model'] + classification_metrics + ['RMSE']
    rows = []

    for model, classification_rep, rmse_score in zip(models, classification_reps, rmse_scores):
        row = [model]
        for metric in classification_metrics:
            score = classification_rep['weighted avg'][metric]
            row.append(score)
        row.append(rmse_score)
        rows.append(row)

    sheet.append(header)
    for row in rows:
        sheet.append(row)

    # Save the Excel workbook
    workbook.save(file)
