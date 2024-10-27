import cv2 as cv


if __name__ == "__main__":

    # camera = cv.VideoCapture(0)
    # if not camera.isOpened():
    #     print("error camara no cargada")
    #     exit()
    # while True:
    #     ret, frame = camera.read()
    #     if not ret:
    #         print("error no se puede obtener imagen")
    #         exit()
    # cv.imshow("Ventana", frame)
    # Se crea el objeto que representa la fuente de video

    camara = cv.VideoCapture(0)

    # Si no se ha podido acceer a la fuente de video se sale del programa

    if not camara.isOpened():
        print("No es posible abrir la cámara")

        exit()

    while True:

        # Se captura la imagen frame a frame

        ret, frame = camara.read()

        # Si la captura no se ha tomado correctamente se sale del bucle

        if not ret:
            print("No es posible obtener la imagen")

            break

        # El frame se muestra en pantalla

        cv.imshow('webcam', frame)

        if cv.waitKey(1) == ord('q'):
            break

    # Se libera la cámara y se cierra la ventana

    camara.release()

    cv.destroyAllWindows()
