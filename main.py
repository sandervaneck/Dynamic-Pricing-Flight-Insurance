from parser import parse_csv
from plotter import plot_to_evaluate, plot_precision_recall_curve, plot_optimized_characteristics, plot_balancing
from paths import weather_paths, flight_data_paths
from createdf import create_flights
from scaler import my_scaler
from showDescriptives import show_descriptives
from resample import resample
from glm import model_glm
from probit import model_probit
from logit import model_logit
from randomForest import model_rfc, optimized_random_forest
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    states = ["New York"]
    weathers = []
    summarized_data = []
    variables_to_parse = ['date',  'distance', 'temp', 'dew', 'humidity', 'precip', 'windgust',
                     'windspeed', 'visibility', 'delay', 'refund', 'carrier']
    numerical_variables_parsed = ['distance', 'temp', 'dew', 'humidity', 'precip', 'windgust',
                          'windspeed', 'visibility', 'delay', 'refund']

    # Currently only parses ones without cancellations/diverted flights
    df = create_flights(flight_data_paths, summarized_data, states, weather_paths, weathers, variables_to_parse)
    print(df[numerical_variables_parsed].describe())

    plot_balancing(df, 'refund')

    df['carrier_score'] = df.groupby('carrier')['delay'].transform('mean')
    my_scaler(df,'carrier_score','scaled_carrier_score')
    columnheaders = ['date', 'scaled_carrier_score', 'distance', 'temp', 'dew', 'humidity', 'precip', 'windgust',
                     'windspeed', 'visibility', 'delay', 'refund']
    factors = ['scaled_carrier_score', 'distance', 'temp', 'dew', 'humidity', 'precip', 'windgust',
                     'windspeed', 'visibility']

    factors_with_date =['date', 'scaled_carrier_score', 'distance', 'temp', 'dew', 'humidity', 'precip', 'windgust', 'windspeed', 'visibility']
    # TODO: Print visuals to Excel
    show_descriptives(df, factors_with_date)

    df = df.dropna()
    # Separate the features and target variables
    # train_X = df.drop(columns=['refund'])
    # train_Y = df['refund']

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

    rfc_train_model = model_rfc(train_X, train_Y)

    rfc_val_prediction = rfc_train_model.predict(val_X)
    plot_to_evaluate(val_Y, rfc_val_prediction, "Random Forest Model Validation Set")
    rfc_val_probs = rfc_train_model.predict_proba(val_X)[:, 1]
    plot_precision_recall_curve(rfc_val_probs, val_Y,'Random Forest Model Validation Set')


    rfc_optimized = optimized_random_forest(train_X, train_Y)
    plot_optimized_characteristics(rfc_optimized, val_X)
    rfc_optimized.best_estimator_.fit(train_X, train_Y)
    rfc_predictions = rfc_optimized.best_estimator_.predict(test_X)
    plot_to_evaluate(test_Y, rfc_predictions, "Refund Probability evaluation on test set")

    #Stochastic Models:
    glm_predictions = model_glm(test_df, factors)
    logit_predictions = model_logit(test_df, factors)
    probit_predictions = model_probit(test_df, factors)

    #Evaluation:
    models = ["RF", "GLM", "Logit", "Probit"]
    predictions = [rfc_predictions, glm_predictions, logit_predictions, probit_predictions]
    rmse_scores = []

    for model, prediction in zip(models, predictions):
        rmse = sqrt(mean_squared_error(test_Y, prediction))
        rmse_scores.append(rmse)

    # Print the table of RMSE scores
    print("Model\t\t\tRMSE")
    print("-------------------------")
    for model, rmse in zip(models, rmse_scores):
        print(f"{model}\t\t{rmse:.4f}")

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
