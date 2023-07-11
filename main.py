from parser import parse_csv
from plotter import plot_to_evaluate
from paths import weather_paths, flight_data_paths
from createdf import create_flights
from scaler import my_scaler
from showDescriptives import show_descriptives
from resample import resample
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    states = ["New York"]
    weathers = []
    summarized_data = []
    df = create_flights(flight_data_paths, summarized_data, states, weather_paths, weathers)
    df.describe()
    plot_balancing(df, 'refund')
    df['carrier_score'] = df.groupby('carrier')['delay'].transform('mean')
    my_scaler(df,'carrier_score','scaled_carrier_score')
    columnheaders = ['date', 'scaled_carrier_score', 'distance', 'temp', 'dew', 'humidity', 'precip', 'windgust',
                     'windspeed', 'visibility', 'delay', 'refund']
    factors = ['date', 'scaled_carrier_score', 'distance', 'temp', 'dew', 'humidity', 'precip', 'windgust',
                     'windspeed', 'visibility']
    show_descriptives(df, factors)
    seed = 123
    train_df, test_df = train_test_split(df, train_size=.5, random_state=seed)
    train_df = resample(train_df)

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
