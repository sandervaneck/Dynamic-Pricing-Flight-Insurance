from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl as openpyxl
from openpyxl import Workbook

def show_descriptives(df, columnheaders):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = 'Descriptives'

    plot_descriptives_tables(df)

    refunded_df = df[df['delay'] >= 120.0][columnheaders]
    non_refunded_df = df[df['delay'] < 120.0][columnheaders]

    refunded_df['date'] = pd.to_datetime(refunded_df['date'])
    refunded_df.set_index('date', inplace=True)

    non_refunded_df['date'] = pd.to_datetime(non_refunded_df['date'])
    non_refunded_df.set_index('date', inplace=True)

    refunded_avg = refunded_df.groupby(refunded_df.index).mean()
    non_refunded_avg = non_refunded_df.groupby(non_refunded_df.index).mean()

    # Increase Agg rendering parameters
    plt.rcParams['agg.path.chunksize'] = 200
    plt.rcParams['path.simplify_threshold'] = 0.111111111111

    # Define colors for refunded and non-refunded values
    refunded_color = 'blue'
    non_refunded_color = 'red'

    # Plot line graphs for each variable
    for column in refunded_df.columns:
        if column != 'date':
            plt.figure()
            plt.plot(refunded_avg.index, refunded_avg[column], color=refunded_color, label='Refunded')
            plt.plot(non_refunded_avg.index, non_refunded_avg[column], color=non_refunded_color, label='Non-Refunded')
            plt.xlabel('Date')
            plt.ylabel(f'Average {column}')
            plt.title(f'Average {column} over time')
            temp_file = f'{column}.png'
            plt.savefig(temp_file)
            plt.close()
            img = openpyxl.drawing.image.Image(temp_file)
            sheet.add_image(img, f'A1')
    # Save the Excel spreadsheet
    workbook.save('Descriptives.xlsx')

def plot_descriptives_tables(df):
    numericvariables = ['scaled_carrier_score', 'distance', 'temp', 'dew', 'humidity', 'precip', 'windgust',
                        'windspeed', 'visibility', 'delay']

    statistics = df.groupby('refund').describe()

    statistics_filtered = statistics[numericvariables].transpose()
    statistics_filtered['delta'] = statistics_filtered.iloc[:, 0] - statistics_filtered.iloc[:, 1]

    table = tabulate(statistics_filtered, headers='keys', tablefmt='fancy_grid')
    print("Descriptives per group:")
    print(table)

