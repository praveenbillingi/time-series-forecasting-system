import pandas as pd
import numpy as np
import holidays

def load_data(path):
    df = pd.read_excel(path)

    df['Date'] = pd.to_datetime(df['Date'])

    return df

def preprocess_data(df):

    # Keep only required columns
    df = df[['State', 'Date', 'Total']]

    # Remove duplicates
    df = df.drop_duplicates()

    # Sort values
    df = df.sort_values(['State', 'Date'])

    # Fill missing values
    df['Total'] = df['Total'].ffill()

    return df

def create_features(df):

    us_holidays = holidays.US()

    all_states = []

    for state in df['State'].unique():

        temp = df[df['State'] == state].copy()

        # Create continuous weekly dates
        temp = temp.set_index('Date')
        temp = temp.asfreq('W')

        temp['State'] = state

        # Fill missing sales values
        temp['Total'] = temp['Total'].interpolate()

        # Lag features
        temp['lag_1'] = temp['Total'].shift(1)
        temp['lag_7'] = temp['Total'].shift(7)
        temp['lag_30'] = temp['Total'].shift(30)

        # Rolling features
        temp['rolling_mean_4'] = temp['Total'].rolling(4).mean()
        temp['rolling_std_4'] = temp['Total'].rolling(4).std()

        # Date features
        temp['month'] = temp.index.month
        temp['week'] = temp.index.isocalendar().week.astype(int)
        temp['year'] = temp.index.year
        temp['day_of_week'] = temp.index.dayofweek

        # Holiday flag
        temp['is_holiday'] = temp.index.map(
            lambda x: 1 if x in us_holidays else 0
        )

        temp = temp.reset_index()

        all_states.append(temp)

    final_df = pd.concat(all_states)

    # Remove nulls created by lagging
    final_df = final_df.dropna()

    return final_df