from prophet import Prophet

def train_prophet(train_df):

    prophet_df = train_df[['Date', 'Total']].copy()

    prophet_df.columns = ['ds', 'y']

    model = Prophet()

    model.fit(prophet_df)

    return model