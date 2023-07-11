def show_descriptives(df, columnheaders):
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
            plt.legend()
            plt.show()

