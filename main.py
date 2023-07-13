from parser import parse_csv
from plotter import plot_to_evaluate, plot_precision_recall_curve, plot_optimized_characteristics, plot_balancing
from paths import weather_paths, flight_data_paths
from createdf import parse_data
from scaler import my_scaler
from showDescriptives import show_descriptives
from resample import resample, resample_combination
from glm import model_glm
from probit import model_probit
from logit import model_logit
from randomForest import model_rfc, optimized_random_forest
from sklearn.model_selection import train_test_split
import pandas as pd
from math import sqrt
import openpyxl as openpyxl
from openpyxl import Workbook
from evaluation import write_model_evaluations
from modifydf import modify_df
# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    workbookDf = Workbook()
    dataFile = "Data.xlsx"
    workbookTrainDf = Workbook()
    randomForestFile = "RandomForest.xlsx"
    workbookResultsDF = Workbook()
    resultFile = "Result.xlsx"

    states = ["New York"]
        # , "Hawaii", "Colorado", "Florida"]
    weathers = []
    summarized_data = []
    variables_to_parse = ['date',  'distance', 'temp', 'dew', 'humidity', 'precip', 'windgust',
                     'windspeed', 'visibility', 'delay', 'refund', 'carrier']
    df = parse_data(flight_data_paths, summarized_data, states, weather_paths, weathers, variables_to_parse)
    df = modify_df(df)

    factors = ['scaled_carrier_score', 'distance', 'temp', 'dew', 'humidity', 'precip', 'windgust',
                     'windspeed', 'visibility']
    factors_with_date =['date', 'scaled_carrier_score', 'distance', 'temp', 'dew', 'humidity', 'precip', 'windgust', 'windspeed', 'visibility']

    show_descriptives(df, factors_with_date, factors, workbookDf, dataFile)

    factors.append('refund')
    df = df[factors]
    df = df.dropna()

    seed = 123
    train_df, aux_df = train_test_split(df, train_size=.5, random_state=seed)
    validation_df, test_df = train_test_split(aux_df, train_size=.5, random_state=seed)

    #ML Model
    train_X = train_df.loc[:, factors]
    train_Y = train_df.loc[:, 'refund']
    val_X = validation_df.loc[:, factors]
    val_Y = validation_df.loc[:, 'refund']
    test_X = test_df.loc[:, factors]
    test_Y = test_df.loc[:, 'refund']

    # Stochastic Models:
    logit_predictions = model_logit(train_X, train_Y, test_X, workbookResultsDF, resultFile)
    probit_predictions = model_probit(train_X, train_Y, test_X, workbookResultsDF, resultFile)
    glm_predictions = model_glm(train_X, train_Y, test_X, workbookResultsDF, resultFile)

    resample_combination(train_X, train_Y, workbookTrainDf, randomForestFile)

    rfc_train_model = model_rfc(train_X, train_Y)

    rfc_val_prediction = rfc_train_model.predict(val_X)
    plot_to_evaluate(val_Y, rfc_val_prediction, "RFM Validation Set", workbookTrainDf, randomForestFile)
    rfc_val_probs = rfc_train_model.predict_proba(val_X)[:, 1]
    plot_precision_recall_curve(rfc_val_probs, val_Y,'RFM Validation Set', workbookTrainDf, randomForestFile)


    rfc_optimized = optimized_random_forest(train_X, train_Y)
    plot_optimized_characteristics(rfc_optimized, val_X, workbookTrainDf, randomForestFile)
    rfc_optimized.best_estimator_.fit(train_X, train_Y)
    rfc_predictions = rfc_optimized.best_estimator_.predict(test_X)

    plot_to_evaluate(test_Y, rfc_predictions, "Evaluation on test set", workbookResultsDF, resultFile)
    #
    #Evaluation:
    models = ["RF", "GLM", "Logit", "Probit"]
    predictions = [rfc_predictions, glm_predictions, logit_predictions, probit_predictions]
    write_model_evaluations(test_Y, models, predictions, workbookResultsDF, resultFile)

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
