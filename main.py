from copyreg import pickle

import cv2 as cv2
import pathlib
import os
import xml.etree.ElementTree as ET

import numpy as np


def read_voc_xml(xmlfile: str) -> dict:
    """read the Pascal VOC XML and return (filename, object name, bounding box)
    where bounding box is a vector of (xmin, ymin, xmax, ymax). The pixel
    coordinates are 1-based.
    """
    root = ET.parse(xmlfile).getroot()
    boxes = {"filename": root.find("filename").text,
             "objects": []
             }
    for box in root.iter('object'):
        bb = box.find('bndbox')
        obj = {
            "name": box.find('name').text,
            "xmin": int(bb.find("xmin").text),
            "ymin": int(bb.find("ymin").text),
            "xmax": int(bb.find("xmax").text),
            "ymax": int(bb.find("ymax").text),
        }
        boxes["objects"].append(obj)

    return boxes


if __name__ == "__main__":
    # # Reemplaza con la ruta de tu archivo de anotaciones
    # annotations_file = "formatXMLPlate/positive.dat"
    #
    # # Verifica cada línea en el archivo de anotaciones
    # with open(annotations_file, "r") as file:
    #     lines = file.readlines()
    #     for line in lines:
    #         parts = line.strip().split()
    #         image_path = parts[0]  # Ruta de la imagen
    #         image_path = "formatXMLPlate/"+image_path
    #         if not os.path.exists(image_path):
    #             print(f"Imagen no encontrada: {image_path}")
    #             continue
    #
    #         # Lee la imagen y sus dimensiones
    #         img = cv2.imread(image_path)
    #         if img is None:
    #             print(f"No se pudo abrir la imagen: {image_path}")
    #             continue
    #
    #         img_height, img_width = img.shape[:2]
    #         num_objects = int(parts[1])
    #
    #         # Verifica cada caja delimitadora en la imagen
    #         for i in range(num_objects):
    #             x = int(parts[2 + i * 4])
    #             y = int(parts[3 + i * 4])
    #             width = int(parts[4 + i * 4])
    #             height = int(parts[5 + i * 4])
    #
    #             # Revisa si las coordenadas están fuera de los límites
    #             if x < 0 or y < 0 or x + width > img_width or y + height > img_height:
    #                 print(f"Coordenadas fuera de límites en {image_path}: ({x}, {y}, {width}, {height})")
    # Read Pascal VOC and write data
    # base_path = pathlib.Path("formatXMLPlate")
    # img_src = base_path / "images"
    # ann_src = base_path / "annotations"
    # neg_src = pathlib.Path("negatives")
    # print(img_src.exists())
    # print(img_src)
    # print(ann_src.exists())
    # print(ann_src)
    # print(neg_src.exists())
    # print(neg_src)
    #
    # negative = []
    # positive = []
    # for xmlfile in ann_src.glob("*.xml"):
    #     # load xml
    #     ann = read_voc_xml(str(xmlfile))
    #     if ann['objects'][0]['name'] == 'dog':
    #         # negative sample (dog)
    #         negative.append(str(img_src / ann['filename']))
    #     else:
    #         # positive sample (cats)
    #         bbox = []
    #         for obj in ann['objects']:
    #             x = obj['xmin']
    #             y = obj['ymin']
    #             w = obj['xmax'] - obj['xmin']
    #             h = obj['ymax'] - obj['ymin']
    #             bbox.append(f"{x} {y} {w} {h}")
    #         line = f"{str(img_src / ann['filename'])} {len(bbox)} {' '.join(bbox)}"
    #         positive.append(line)
    # for file in neg_src.glob("*.jpg"):
    #         negative.append(str(neg_src / file.name))
    #
    #
    # # write the output to `negative.dat` and `postiive.dat`
    # with open("negative.dat", "w") as fp:
    #     fp.write("\n".join(negative))
    #
    # with open("positive.dat", "w") as fp:
    #     fp.write("\n".join(positive))
    # # camera = cv.VideoCapture(0)
    # # if not camera.isOpened():
    # #     print("error camara no cargada")
    # #     exit()
    # # while True:
    # #     ret, frame = camera.read()
    # #     if not ret:
    # #         print("error no se puede obtener imagen")
    # #         exit()
    # # cv.imshow("Ventana", frame)
    # # Se crea el objeto que representa la fuente de video
    #
    # camara = cv.VideoCapture(0)
    #
    # # Si no se ha podido acceer a la fuente de video se sale del programa
    #
    # if not camara.isOpened():
    #     print("No es posible abrir la cámara")
    #
    #     exit()
    #
    # while True:
    #
    #     # Se captura la imagen frame a frame
    #
    #     ret, frame = camara.read()
    #
    #     # Si la captura no se ha tomado correctamente se sale del bucle
    #
    #     if not ret:
    #         print("No es posible obtener la imagen")
    #
    #         break
    #
    #     # El frame se muestra en pantalla
    #
    #     cv.imshow('webcam', frame)
    #
    #     if cv.waitKey(1) == ord('q'):
    #         break
    #
    # # Se libera la cámara y se cierra la ventana
    #
    # camara.release()
    #
    # cv.destroyAllWindows()

    image = 'formatXMLPlate/images/3c8e7fe1-IMG-20241109-WA0029.jpg'
    model = 'platedetc/cascade.xml'
    model2 = 'platedetc/haarcascade_licence_plate_rus_16stages.xml'
    model3 = 'platedetc/haarcascade_russian_plate_number.xml'

    classifier3 = cv2.CascadeClassifier(model3)
    classifier2 = cv2.CascadeClassifier(model2)

    img = cv2.imread(image)
    img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform object detection
    objects = classifier2.detectMultiScale(gray,
                                           scaleFactor=1.1, minNeighbors=5,
                                           minSize=(30, 30))
    numbers = classifier3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected objects
    for (x, y, w, h) in objects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        plate_region = gray[y:y + h, x:x + w]

    # Display the result
    cv2.imshow('Object Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
