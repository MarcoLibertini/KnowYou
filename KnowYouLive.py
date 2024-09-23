import cv2
import face_recognition as fr
import os
import numpy as np

# Cargar las imágenes de la base de datos
ruta = 'pipol'
miPipol = []
nombresPipol = []
listaPipol = os.listdir(ruta)

for nombre in listaPipol:
    imagen = cv2.imread(os.path.join(ruta, nombre))
    miPipol.append(imagen)
    nombresPipol.append(os.path.splitext(nombre)[0])

# Codificar las imágenes en RGB
def codificar(imagenes):
    listaCodificada = []
    for imagen in imagenes:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        codificado = fr.face_encodings(imagen)[0]
        listaCodificada.append(codificado)
    return listaCodificada

# Generar la lista codificada de las imágenes
listaPipolCodificada = codificar(miPipol)

# Capturar video desde la cámara
captura = cv2.VideoCapture(0)

while True:
    # Leer cada fotograma de la cámara
    exito, imagen = captura.read()
    if not exito:
        print('No se ha podido capturar.')
        break

    # Encontrar las caras en la imagen capturada
    carasCapturadas = fr.face_locations(imagen)
    carasCapturadasCodificadas = fr.face_encodings(imagen, carasCapturadas)

    for caraCodificada, caraUbicacion in zip(carasCapturadasCodificadas, carasCapturadas):
        # Comparar la cara capturada con las imágenes de la base de datos
        coincidencias = fr.compare_faces(listaPipolCodificada, caraCodificada)
        distancias = fr.face_distance(listaPipolCodificada, caraCodificada)

        # Encontrar la coincidencia más cercana
        indiceCoincidencia = np.argmin(distancias)

        if distancias[indiceCoincidencia] > 0.6:
            nombrePersona = "Desconocido"
        else:
            nombrePersona = nombresPipol[indiceCoincidencia]

        # Dibujar un rectángulo alrededor de la cara y mostrar el nombre de la persona
        y1, x2, y2, x1 = caraUbicacion
        cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(imagen, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(imagen, nombrePersona, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # Mostrar la imagen en vivo
    cv2.imshow('Reconocimiento Facial en Vivo', imagen)

    # Si se presiona la tecla 'q', se sale del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos y cerrar ventanas
captura.release()
cv2.destroyAllWindows()