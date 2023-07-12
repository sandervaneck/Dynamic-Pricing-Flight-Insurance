def resample(df):
    count_class_0, count_class_1 = train_df['refund'].value_counts()

    df_class_0 = train_df[train_df['refund'] == 0.0]
    df_class_1 = train_df[train_df['refund'] == 1.0]

    df_class_1_over = df_class_1.sample(count_class_0, replace=True)
    df_train_over = pd.concat([df_class_0, df_class_1_over], axis=0)

    df_class_0_under = df_class_0.sample(count_class_1)
    df_train_under = pd.concat([df_class_0_under, df_class_1], axis=0)

    print('Random over-sampling:')
    print(train_df['refund'].value_counts())

    print('Random over-sampling:')
    print(df_train_over['refund'].value_counts())

    print('Random under-sampling:')
    print(df_train_under['refund'].value_counts())
    return df_train_over