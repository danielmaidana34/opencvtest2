import cv2
import os
import numpy as np
import tensorflow as tf



# Ruta al directorio de imágenes
dir_imagenes = "/content/sample_data/dataset/hibridcows"

# Obtener una lista de las rutas de todas las imágenes en el directorio
rutas_imagenes = [os.path.join(dir_imagenes, nombre) for nombre in os.listdir(dir_imagenes)]

# Cargar cada imagen en un arreglo
imagenes = []
etiquetas = []
        
for ruta_imagen in rutas_imagenes:
   if ruta_imagen.endswith('.jpg'):
    # Leer la imagen utilizando OpenCV
    imagen = cv2.imread(ruta_imagen)
    imagen = cv2.resize(imagen, (227, 227))
    imagen = np.array(imagen).astype(float) / 255
    imagen = np.expand_dims(imagen, axis=0)
    # Agregar la imagen al arreglo
    imagenes.append(imagen)
    
# Convertir la lista de imágenes a un arreglo de NumPy
imagenes = np.array(imagenes)

# Agregar las etiquetas de las imágenes
etiquetas = [1] * len(imagenes)
etiquetas = np.array(etiquetas).astype(np.float32)
etiquetas = np.expand_dims(etiquetas, axis=1)

# Crear un objeto Dataset de TensorFlow
dataset = tf.data.Dataset.from_tensor_slices((imagenes, etiquetas))

# Mezclar el conjunto de datos
dataset = dataset.shuffle(len(imagenes))
