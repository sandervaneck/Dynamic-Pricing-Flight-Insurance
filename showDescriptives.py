from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl as openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from resample import plot_balancing

def show_descriptives(df, columnheaders, factors, workbook, file):
    sheet = workbook.create_sheet(title='Descriptives')
    statistics = df.groupby('refund').describe()
    statistics_filtered = statistics[factors].transpose()
    statistics_filtered['delta'] = statistics_filtered.iloc[:, 0] - statistics_filtered.iloc[:, 1]
    table = tabulate(statistics_filtered, headers='keys', tablefmt='fancy_grid')

    refunded_df = df[df['refund'] == 1][columnheaders]
    non_refunded_df = df[df['refund'] == 0][columnheaders]

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
    for i, column in enumerate(refunded_df.columns):
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
            sheet.add_image(img, f'A{i*24+1}')

    table_sheet = workbook.create_sheet(title='Table')
    for row in dataframe_to_rows(statistics_filtered, index=True, header=True):
        table_sheet.append(row)

    balance_sheet = workbook.create_sheet("Balance")
    plot_balancing(df['refund'], workbook, 1, balance_sheet, file)
