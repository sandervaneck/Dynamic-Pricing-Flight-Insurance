from scaler import my_scaler

def add_cat_scores(df, file, workbook):
    sheet = workbook.create_sheet("Scalers")
    df['carrier_score'] = df.groupby('carrier')['refund'].transform('mean')
    my_scaler(df, 'carrier_score', 'scaled_carrier_score', 'carrier', workbook, sheet, 1, file)
    df['origin_score'] = df.groupby('origin')['refund'].transform('mean')
    my_scaler(df, 'origin_score', 'scaled_origin_score', 'origin', workbook, sheet, 20, file)
    df['time_score'] = df.groupby('time')['refund'].transform('mean')
    my_scaler(df, 'time_score', 'scaled_time_score', 'time', workbook, sheet, 40, file)
    df['weekday_score'] = df.groupby('weekday')['refund'].transform('mean')
    my_scaler(df, 'weekday_score', 'scaled_weekday_score', 'weekday', workbook, sheet, 60, file)
    return df
