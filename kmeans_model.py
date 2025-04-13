'''
13/04/2025
Act 3 - IA modelos supervisados
Linda Carolina Zambrano León
Juan Sebastián Hernández Galindo
Jose Luis Rodriguez Castillo

Aprendizaje no supervisado K-Means Clustering
Este modelo agrupa las estaciones de transporte en clústeres según características similares como cantidad de pasajeros y población cercana.
'''

import random
import math

# Carga datos desde archivo CSV
def load_unsupervised_dataset(file_path):
    data = []
    with open(file_path, 'r') as f:
        headers = f.readline().strip().split(',')  # Lee los nombres de las columnas
        for line in f:
            values = line.strip().split(',')  # Lee cada línea como lista
            record = dict(zip(headers, values))  # Crea un diccionario por fila
            record['avg_daily_passengers'] = int(record['avg_daily_passengers'])  # Convierte a int
            record['nearby_population'] = int(record['nearby_population'])        # Convierte a int
            data.append(record)
    return data

# Extrae solo las variables numéricas a usar como características
def extract_features(data):
    return [[d['avg_daily_passengers'], d['nearby_population']] for d in data]

# Calcula la distancia euclidiana entre dos puntos
def euclidean(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

# Selecciona k puntos iniciales aleatorios como centroides
def init_centroids(points, k):
    return random.sample(points, k)

# Asigna cada punto al centroide más cercano
def assign_clusters(points, centroids):
    clusters = [[] for _ in centroids]
    for point in points:
        distances = [euclidean(point, c) for c in centroids]
        idx = distances.index(min(distances))
        clusters[idx].append(point)
    return clusters

# Recalcula los centroides como el promedio de los puntos del clúster
def compute_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        if not cluster:
            continue
        centroid = [sum(dim)/len(cluster) for dim in zip(*cluster)]
        new_centroids.append(centroid)
    return new_centroids

# Algoritmo principal de K-Means
def kmeans(points, k, iterations=10):
    centroids = init_centroids(points, k)
    for _ in range(iterations):
        clusters = assign_clusters(points, centroids)
        centroids = compute_centroids(clusters)
    return clusters, centroids

# =======================
# PROGRAMA PRINCIPAL
# =======================
data = load_unsupervised_dataset(r'RUTA_DATASET') #CAMBIAR RUTA DEL DATASET PARA EL CARGUE DE LOS ARCHIVOS
points = extract_features(data)
clusters, centroids = kmeans(points, k=3)

# Muestra los centroides finales
print("Centroides finales:")
for c in centroids:
    print(f"→ {c}")

'''
El modelo agrupa estaciones de transporte similares, lo cual ayuda a definir zonas y asi mismo planificar los recursos.
En nuestro proyecto de transporte masivo...
Los centroides nos dirán algo como:

“Este grupo de estaciones suele tener en promedio 5000 pasajeros diarios y está rodeado de 20,000 personas.”

Lo cual nos indica como:
Planificar rutas más eficientes.
Decidir en qué estaciones invertir más recursos.
Identificar zonas con baja o alta demanda.

Los centroides se conocer como el centro geometrico de un grupo de puntos (cluster), es decir el promedio de todas las observaciones de datos
de un mismo grupo 

Ejemplo:
Suponemos que tenemos estos puntos (estaciones de transporte) en un plano 2D:
Estación A: [3000 pasajeros, 15000 personas cercanas]
Estación B: [3200 pasajeros, 16000 personas cercanas]
Estación C: [2800 pasajeros, 14000 personas cercanas]

El centroide se calcula como el promedio:
Centroide = [
    (3000 + 3200 + 2800) / 3,
    (15000 + 16000 + 14000) / 3
] = [3000, 15000]

¿Cómo se usan los centroides en K-Means?
Al inicio, el algoritmo selecciona centroides aleatorios.
Luego:
Cada punto se asigna al centroide más cercano (por distancia).
Se recalculan los centroides como el promedio de su grupo.
Repite hasta que los centroides no cambien mucho.
Así el modelo agrupa los datos automáticamente en clústeres que tienen sentido en función de la similitud.

¿Por que los resultados son diferentes cada vez?
Imagina que vas a dividir una ciudad en 3 zonas para ubicar estaciones de buses, 
pero al principio eliges 3 barrios al azar como punto de partida. Obviamente, si eliges otros 3 barrios diferentes la próxima vez, 
las zonas finales van a cambiar.

'''

