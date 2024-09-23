import cv2
import face_recognition as fr
import os
import numpy

#creamos base de datos

ruta = 'pipol'
miPipol = []
nombresPipol = []
listaPipol = os.listdir(ruta)


for nombre in listaPipol:
    #leo cada imagen del directorio
    imagen = cv2.imread(f'{ruta}\{nombre}')
    #agrego a la lista miPipol cada imagen leida
    miPipol.append(imagen)
    #agrego a la lista, los nombres de los empleados sin la extension
    #con el splittext lo q hago es solo quedarme con el texto.
    nombresPipol.append(os.path.splitext(nombre)[0])


#codificar imagenes en RGB

def codificar(imagenes):

    #creamos una lista
    listaCodificada = []

    #pasamos las imagenes a rgb
    for imagen in imagenes:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

        #encontrar el rectangulo en cada cara
        codificado = fr.face_encodings(imagen)[0]

        #agregamos imagenes codificadas a una lista
        listaCodificada.append(codificado)

    return listaCodificada


#Se genero correctamente
listaPipolCodificada = codificar(miPipol)



#tomar imagen de camara web
captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#leer imagen de la camara
# captura devuelve dos salidas

exito, imagen = captura.read()


if not exito:
    print('No se ha podido capturar.')
else:
    #reconocer cara en captura
    caraCapturada = fr.face_locations(imagen)

    #codificar cara capturada
    caraCapturadaCodificada = fr.face_encodings(imagen, caraCapturada)

    #se usa zip apra poder hacer el loop en el mismo lugar
    #buscamos coincidencias entre la cara capturada y las imagenes en la lista
    for caraCodificada, caraUbicacion in zip(caraCapturadaCodificada, caraCapturada):
        coincidencias = fr.compare_faces(listaPipolCodificada, caraCodificada)

        #guardamos las distancias
        distancias = fr.face_distance(listaPipolCodificada, caraCodificada)

        print(distancias)

        #buscamos el indice del menor valor, osea la distancia menor
        #osea la coincidencia
        indiceCoincidencia = numpy.argmin(distancias)


        #mostramos coincidencias si las hay
        if distancias[indiceCoincidencia] > 0.6:
            print('No se encontro cara conocida.')
        else:
            #buscamos el nombre de la persona encontrada
            nombrePersona = nombresPipol[indiceCoincidencia]


            #buscamos los 4 puntos de la cara encontrada
            #para crear el rectangulo
            y1, x2, y2, x1 = caraUbicacion
            #ponemos los vertices
            cv2.rectangle(imagen,
                          (x1,y1),
                          (x2,y2),
                          (0,255,0),
                          2)

            cv2.rectangle(imagen,(x1,y2 - 85), (x2,y2),(0,255,0), cv2.FILLED)

            #ponemos el texto
            cv2.putText(imagen, nombre, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255),2)


            #mostramos la persona RECONOCIDA
            cv2.imshow('Persona Capturada', imagen)


            #mantenemos la ventana abierta
            cv2.waitKey(0)

