from scaler import my_scaler
import matplotlib.pyplot as plt
import openpyxl as openpyxl

def add_cat_scores(df):
    df['carrier_score'] = df.groupby('carrier')['refund'].transform('mean')
    my_scaler(df, 'carrier_score', 'scaled_carrier_score')
    df['origin_score'] = df.groupby('origin')['refund'].transform('mean')
    my_scaler(df, 'origin_score', 'scaled_origin_score')
    return df
