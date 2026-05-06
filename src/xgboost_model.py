from xgboost import XGBRegressor

FEATURES = [
    'lag_1',
    'lag_7',
    'lag_30',
    'rolling_mean_4',
    'rolling_std_4',
    'month',
    'week',
    'year',
    'day_of_week',
    'is_holiday'
]


def train_xgboost(train_df):

    X_train = train_df[FEATURES]
    y_train = train_df['Total']

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model