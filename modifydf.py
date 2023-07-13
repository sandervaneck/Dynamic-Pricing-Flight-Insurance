from scaler import my_scaler

def modify_df(df):
    df['carrier_score'] = df.groupby('carrier')['refund'].transform('mean')
    my_scaler(df, 'carrier_score', 'scaled_carrier_score')
    return df