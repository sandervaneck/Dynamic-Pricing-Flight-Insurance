import pandas as pd
import statsmodels.api as sm

def model_glm(data, predictors):
    # Specify the predictors and the response variable
    #  ['scaled_carrier_score', 'distance', 'temp', 'dew', 'humidity', 'precip', 'windgust', 'windspeed', 'visibility']
    response = 'refund'

    # Filter the DataFrame to include only the necessary columns
    data_filtered = data[predictors + [response]].copy()


    # Drop rows with any NaN values
    data_filtered = data_filtered.dropna()

    # Convert the response variable to numeric
    data_filtered[response] = pd.to_numeric(data_filtered[response], errors='coerce')

    # Create the design matrix
    X = data_filtered[predictors]
    X = sm.add_constant(X)  # Add a constant term to the design matrix

    # Create the GLM model
    glm_model = sm.GLM(data_filtered[response], X, family=sm.families.Binomial())

    # Fit the GLM model to the data
    glm_results = glm_model.fit()

    # Print the summary of the model
    print(glm_results.summary())

    # Predict the values of 'refund' using the trained model
    predictions = glm_results.predict(X)
    return predictions