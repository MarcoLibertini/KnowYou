import cv2
import face_recognition as fr

#cargar imagenes
foto_control = fr.load_image_file('yam.png')
foto_prueba = fr.load_image_file('yam2.png')


#pasar imagenes a rgb
foto_control = cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)
foto_prueba = cv2.cvtColor(foto_prueba, cv2.COLOR_BGR2RGB)

############################################################
#             FOTO CONTROL                #
#localizar cara control tambien marcamos el indice
lugar_cara_A = fr.face_locations(foto_control)[0]

#necesitamos codificar la cara tambien marcamos el indice
cara_codificada_A = fr.face_encodings(foto_control)[0]

#creamos rectangulo para marcar la cara
#primer parametro es la imagen, segundos parametros son la ubicacion de los vertices
#tercer parametro es el color del rectangle
#cuarto parametro es el grosos del rectangle

cv2.rectangle(foto_control,
              (lugar_cara_A[3],lugar_cara_A[0]),
              (lugar_cara_A[1], lugar_cara_A[2]),
              (0,255,0),
              2)

############################################################

############################################################
#             FOTO PRUEBA                #
#localizar cara control tambien marcamos el indice
lugar_cara_B = fr.face_locations(foto_prueba)[0]

#necesitamos codificar la cara tambien marcamos el indice
cara_codificada_B = fr.face_encodings(foto_prueba)[0]

cv2.rectangle(foto_prueba,
              (lugar_cara_B[3],lugar_cara_B[0]),
              (lugar_cara_B[1], lugar_cara_B[2]),
              (0,255,0),
              2)


############################################################


# realizar comparacion de imagentes
#primer imagen en lista, segunda a comparar, y tolerancia
resultado = fr.compare_faces([cara_codificada_A],cara_codificada_B)

#medida de la distancia
#la imagen tiene q estar en formato lista
distancia = fr.face_distance([cara_codificada_A], cara_codificada_B)

#mostrar resultado
#cv2 sirve para manipular de como manejar una imagen
#muestro un texto en el rectangulo
cv2.putText(foto_prueba,
            f'{resultado} {distancia.round(2)}',
            (50,50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0,255,0),
            2)



#mostrar imagenes
cv2.imshow('Foto Control', foto_control)
cv2.imshow('Foto Prueba', foto_prueba)

#mantener el programa abierto
cv2.waitKey(0)