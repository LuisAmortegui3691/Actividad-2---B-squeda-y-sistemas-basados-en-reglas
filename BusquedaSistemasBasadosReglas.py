import tensorflow as tf
import numpy as np

# Establecer las estaciones y lineas como listas
estaciones = ['Estacion_A', 'Estacion_B', 'Estacion_C', 'Estacion_D', 'Estacion_E']
lineas = ['Linea_1', 'Linea_2']

# Crear un diccionario para mapear las estaciones y líneas a sus índices numéricos
estaciones_index = {estacion: idx for idx, estacion in enumerate(estaciones)}
lineas_index = {linea: idx for idx, linea in enumerate(lineas)}

# Datos de entrenamiento (ejemplo simplificado)
X_train = [
    [estaciones_index['Estacion_A'], estaciones_index['Estacion_B'], lineas_index['Linea_1']],
    [estaciones_index['Estacion_B'], estaciones_index['Estacion_C'], lineas_index['Linea_1']],
    [estaciones_index['Estacion_C'], estaciones_index['Estacion_D'], lineas_index['Linea_2']],
    [estaciones_index['Estacion_D'], estaciones_index['Estacion_E'], lineas_index['Linea_2']],
    # Agregar más ejemplos de rutas óptimas según sea necesario
]
y_train = [10, 15, 20, 25]  # Tiempos de viaje correspondientes en minutos

# Modelo de IA utilizando TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(estaciones) + len(lineas), output_dim=100),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenamiento del modelo
model.fit(np.array(X_train), np.array(y_train), epochs=50)

# Función para encontrar la ruta óptima utilizando el modelo de IA
def encontrar_ruta_optima_IA(origen, destino):
    tiempo_predicho = model.predict(np.array([[origen, destino, 1]]))  # Suponiendo Linea_1 como predeterminada
    return tiempo_predicho

# Ejemplo de uso
origen = estaciones_index['Estacion_A']
destino = estaciones_index['Estacion_E']
tiempo_viaje_optimo = encontrar_ruta_optima_IA(origen, destino)
print(f"El tiempo de viaje óptimo desde {estaciones[origen]} hasta {estaciones[destino]} es de aproximadamente {tiempo_viaje_optimo} minutos.")
