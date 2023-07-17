from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from constants.states import states
from dataHandling.createdf import parse_data
from dataHandling.addCatScores import add_cat_scores
from constants.paths import rfc_factors, factors, weather_paths, df_var, flight_data_paths, dataFile, headers, num_var, cat_var, workbookDf, \
    workbookResultsDF, resultFile, workbookTrainDf, randomForestFile
from evaluation import write_model_evaluations
from glm import model_glm
from logit import model_logit
from randomForest import model_rfc, optimized_random_forest
from resample import resample_combination
from showDescriptives import show_descriptives
from excelWriters.plotter import print_df_overview, plot_to_evaluate, plot_precision_recall_curve, \
    plot_optimized_characteristics

# Press the green button in the gutter to run the script.

if __name__ == '__main__':

    weathers = []
    summarized_data = []
    df = parse_data(flight_data_paths, summarized_data, states, weather_paths, weathers, headers)
    print_df_overview(df, workbookDf, dataFile, cat_var )
    df = add_cat_scores(df)
    show_descriptives(df, cat_var, num_var, workbookDf, dataFile)
    #
    df = df[df_var]

    df = df.dropna()

    seed = 123
    train_df, aux_df = train_test_split(df, train_size=.5, random_state=seed)
    validation_df, test_df = train_test_split(aux_df, train_size=.5, random_state=seed)
    #

    train_X = train_df.loc[:, rfc_factors]
    train_X_glm = train_df.loc[:, factors]
    train_Y = train_df.loc[:, 'refund']
    val_X = validation_df.loc[:, rfc_factors]
    val_Y = validation_df.loc[:, 'refund']
    test_X_glm = test_df.loc[:, factors]
    test_X = test_df.loc[:, rfc_factors]
    test_Y = test_df.loc[:, 'refund']
    #

    # # Stochastic Models:
    # # logit_predictions = model_logit(train_X, train_Y, test_X, workbookResultsDF, resultFile)
    # # probit_predictions = model_probit(train_X, train_Y, test_X, workbookResultsDF, resultFile)
    #
    resample_combination(train_X, train_Y, workbookTrainDf, randomForestFile)
    #
    glm_predictions = model_glm(train_X_glm, train_Y, test_X_glm, test_Y, workbookResultsDF, resultFile)
    stochastic_model = LogisticRegression()
    stochastic_model.fit(train_X, train_Y)

    # Use the model to predict the target variable on the test data
    stochastic_predictions = stochastic_model.predict(test_X)

    # Calculate the classification report
    rfc_train_model = model_rfc(train_X, train_Y)
    #
    rfc_val_prediction = rfc_train_model.predict(val_X)
    plot_to_evaluate(val_Y, rfc_val_prediction, "RFM Validation Set", workbookTrainDf, randomForestFile)
    rfc_val_probs = rfc_train_model.predict_proba(val_X)[:, 1]
    plot_precision_recall_curve(rfc_val_probs, val_Y,'RFM Validation Set', workbookTrainDf, randomForestFile)
    #
    #
    rfc_optimized = optimized_random_forest(train_X, train_Y)
    plot_optimized_characteristics(rfc_optimized, val_X, workbookTrainDf, randomForestFile)
    rfc_optimized.best_estimator_.fit(train_X, train_Y)
    rfc_predictions = rfc_optimized.best_estimator_.predict(test_X)
    #
    plot_to_evaluate(test_Y, rfc_predictions, "Evaluation on test set", workbookResultsDF, resultFile)
    # #
    #Evaluation:
    models = ["RF", "Stochastic"]
    # , "Logit", "Probit"]
    predictions = [rfc_predictions, stochastic_predictions]
    # logit_predictions, probit_predictions
    write_model_evaluations(test_Y, models, predictions, workbookResultsDF, resultFile)

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
