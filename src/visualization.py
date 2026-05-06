import matplotlib.pyplot as plt


def plot_forecast(actual, predicted):

    plt.figure(figsize=(10, 5))

    plt.plot(actual.values, label='Actual')

    plt.plot(predicted, label='Predicted')

    plt.legend()

    plt.title('Forecast vs Actual')

    plt.savefig('outputs/forecast_plot.png')

    plt.close()