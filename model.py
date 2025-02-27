import pandas as pd
import json
from prophet import Prophet
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Caricarmento del dataset
file_path = 'EnergyLearningLocal.MonthlyScansFlattened-IT.json'
with open(file_path, 'r') as f:
    data = json.load(f)


records = []
for entry in data:
    records.append({
        'ds': pd.to_datetime(entry['timestampKey']).tz_localize(None),  # Data
        'y': entry['totalConsumption'] / 1000000,  # Conversione da MWh a TWh
    })

df = pd.DataFrame(records)

#Train-Test Split: (80%) TRAIN e (20%) TEST.

df = df.sort_values(by='ds')

split_index = int(len(df) * 0.8)
df_train = df.iloc[:split_index]
df_test = df.iloc[split_index:]

# Inizializzazione di Prophet.
model = Prophet()
model.fit(df)

# Previsione sui dati di test
forecast_test = model.predict(df_test)

# Calcolo di MAE, MSE e MAPE
mae = mean_absolute_error(df_test['y'], forecast_test['yhat'])
mse = mean_squared_error(df_test['y'], forecast_test['yhat'])

# Allineamento
df_test_reset = df_test.reset_index(drop=True)
forecast_test_reset = forecast_test.reset_index(drop=True)

# Qui rimuoviamo i valori con y = 0 per evitare divisioni per zero
valid_indices = df_test_reset['y'] != 0


y_true = df_test_reset['y'][valid_indices]
y_pred = forecast_test_reset['yhat'][valid_indices]

# Calcolo MAPE
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'MAPE: {mape:.2f}%')
future = model.make_future_dataframe(periods=12, freq='M')  # 12 mesi nel futuro
forecast = model.predict(future)


fig = go.Figure()
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Previsione'))
fig.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], mode='markers', name='Dati del Training Set', marker=dict(color='red')))
fig.add_trace(go.Scatter(x=df_test['ds'], y=df_test['y'], mode='markers', name='Dati del Testing Set', marker=dict(color='green')))
fig.update_layout(title='Previsione del Consumo Energetico in Italia (TWh)', xaxis_title='Anno', yaxis_title='Consumo (TWh)')
fig.show()


components = model.plot_components(forecast)
components.show()
