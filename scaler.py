from sklearn.preprocessing import MinMaxScaler

def my_scaler(delay_df, from_col_name, to_col_name):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(pd.DataFrame(delay_df[from_col_name],index=delay_df.index))
    scaled_delay_df = pd.DataFrame(scaled, columns=['scale_temp'], index=delay_df.index)
    delay_df[to_col_name] = scaled_delay_df['scale_temp']