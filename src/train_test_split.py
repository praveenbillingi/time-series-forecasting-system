def split_data(df):

    train_size = int(len(df) * 0.8)

    train = df.iloc[:train_size]
    val = df.iloc[train_size:]

    return train, val