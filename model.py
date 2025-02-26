import pandas as pd
import json
from prophet import Prophet
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
# Caricare il dataset
file_path = 'EnergyLearningLocal.MonthlyScansFlattened-IT.json'
with open(file_path, 'r') as f:
    data = json.load(f)

# Creare un DataFrame
records = []
for entry in data:
    records.append({
        'ds': pd.to_datetime(entry['timestampKey']).tz_localize(None),  # Colonna data
        'y': entry['totalConsumption'] / 1000000,  # Convertire da MWh a TWh
    })

df = pd.DataFrame(records)

# Ordinare i dati cronologicamente
df = df.sort_values(by='ds')

# Dividere il dataset in training (80%) e test (20%)
split_index = int(len(df) * 0.8)
df_train = df.iloc[:split_index]
df_test = df.iloc[split_index:]

# Inizializzare Prophet e aggiungere variabili esterne
model = Prophet()
model.fit(df)

# Creare il dataframe per la previsione

# Previsione sui dati di test
forecast_test = model.predict(df_test)

# Calcolare MAE, MSE, MAPE
mae = mean_absolute_error(df_test['y'], forecast_test['yhat'])
mse = mean_squared_error(df_test['y'], forecast_test['yhat'])

# Assicurati che gli indici di df_test e forecast_test siano allineati
df_test_reset = df_test.reset_index(drop=True)
forecast_test_reset = forecast_test.reset_index(drop=True)

# Rimuovere i valori con y = 0 per evitare divisione per zero
valid_indices = df_test_reset['y'] != 0

# Ora possiamo fare il filtro su entrambi i DataFrame allineati
y_true = df_test_reset['y'][valid_indices]
y_pred = forecast_test_reset['yhat'][valid_indices]

# Calcolare MAPE solo sui dati validi
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'MAPE: {mape:.2f}%')
future = model.make_future_dataframe(periods=12, freq='M')  # 12 mesi nel futuro
forecast = model.predict(future)

# Visualizzare i risultati con Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Previsione'))
fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='markers', name='Dati Storici'))
fig.update_layout(title='Previsione del Consumo Energetico in Italia (TWh)', xaxis_title='Anno', yaxis_title='Consumo (TWh)')
fig.show()

# Visualizzare i componenti della previsione con Plotly
components = model.plot_components(forecast)
components.show()
