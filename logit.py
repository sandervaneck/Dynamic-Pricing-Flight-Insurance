import pandas as pd
import statsmodels.api as sm

def model_logit(data, predictors):
    # Specify the predictors and the response variable
    #  ['scaled_carrier_score', 'distance', 'temp', 'dew', 'humidity', 'precip', 'windgust', 'windspeed', 'visibility']
    response = 'refund'

    # Filter the DataFrame to include only the necessary columns
    data_filtered = data[predictors + [response]].copy()

    # Drop rows with any NaN values
    data_filtered = data_filtered.dropna()

    # Convert the response variable to binary values (0 and 1)
    data_filtered[response] = pd.to_numeric(data_filtered[response], errors='coerce')
    data_filtered[response] = data_filtered[response].astype(int)

    # Create the design matrix
    X = data_filtered[predictors]
    X = sm.add_constant(X)  # Add a constant term to the design matrix

    # Create the logit model (logistic regression)
    logit_model = sm.Logit(data_filtered[response], X)

    # Fit the logit model to the data
    logit_results = logit_model.fit()

    # Print the summary of the model
    print(logit_results.summary())

    # Predict the probabilities of 'refund == 1' using the trained model
    probabilities = logit_results.predict(X)
    return probabilities