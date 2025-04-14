'''
13/04/2025
Act 3 - IA modelos supervisados
Linda Carolina Zambrano León
Juan Sebastián Hernández Galindo
Jose Luis Rodriguez Castillo

Aprendizaje supervisado Regresión Lineal
Este modelo aprende a predecir la cantidad de pasajeros según la hora del día
'''

# Carga los datos desde CSV
def load_supervised_dataset(file_path):
    data = []
    with open(file_path, 'r') as f:
        headers = f.readline().strip().split(',')
        for line in f:
            values = line.strip().split(',')
            record = dict(zip(headers, values))
            record['hour'] = int(record['hour'])
            record['is_holiday'] = int(record['is_holiday'])
            record['special_event'] = int(record['special_event'])
            record['passengers'] = int(record['passengers'])
            data.append(record)
    return data

# Ajusta un modelo de regresión lineal usando solo la hora como variable
def linear_regression(data):
    X = [d['hour'] for d in data]              # Variable independiente
    Y = [d['passengers'] for d in data]        # Variable dependiente
    n = len(X)

    sum_x = sum(X)
    sum_y = sum(Y)
    sum_x2 = sum(x ** 2 for x in X)
    sum_xy = sum(x * y for x, y in zip(X, Y))

    # Calcula la pendiente (a) y la intersección (b) de la recta y = ax + b
    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b = (sum_y - a * sum_x) / n
    return a, b

# Hace una predicción para una hora dada
def predict(hour, a, b):
    return a * hour + b

# =======================
# PROGRAMA PRINCIPAL
# =======================
data = load_supervised_dataset(r'RUTA DATASET') #CAMBIAR RUTA DEL DATASET PARA EL CARGUE DE LOS ARCHIVOS
a, b = linear_regression(data)

# Imprime el modelo y una predicción de ejemplo
print(f"📈 Modelo ajustado: y = {a:.2f} * hour + {b:.2f}")
print(f"Predicción para las 8 AM: {predict(8, a, b):.1f} pasajeros")

'''
El modelo estima cuantos pasajeros habra en funcion de la hora,lo cual permite identificar la cantidad de buses a usar segun la demanda de pasajeros.
Para el ejemplo:
Modelo ajustado: y = 28.32 * hour + 103.45
Predicción para las 8 AM: 330.0 pasajeros
Lo que indica que por cada hora que pasa se espera un aumento de 28 pasajeros aprox y que para la hora 0 habrian 330 pasajeros
'''

