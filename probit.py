import pandas as pd
import statsmodels.api as sm

def model_probit(data, predictors):
    # Specify the predictors and the response variable
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

    # Create the probit model
    probit_model = sm.Probit(data_filtered[response], X)

    # Fit the probit model to the data
    probit_results = probit_model.fit()

    # Print the summary of the model
    print(probit_results.summary())

    # Predict the probabilities of 'refund == 1' using the trained model
    probabilities = probit_results.predict(X)
    return probabilities